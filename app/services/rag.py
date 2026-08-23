import hashlib
import json
import logging
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
import yaml
from app.schemas import QueryResponse

load_dotenv()

logger = logging.getLogger("rag_service")

FILE_PATH = "data/raw_docs/nusantaracare_panduan_operasional_internal_v2.md"
COLLECTION_NAME = "nusantaracare_docs"
TOP_K = 3


def parse_text(content: str) -> list[dict]:
    parts = content.split("---", 2)
    raw_metadata = yaml.safe_load(parts[1]) if len(parts) > 2 else {}
    body = parts[2] if len(parts) > 2 else content

    chunks = []
    current_bab = ""
    current_subbab = ""
    current_paragraphs = []
    is_active = True

    def flush_chunk():
        if current_paragraphs:
            text = "\n\n".join(current_paragraphs)
            chunk_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            chunks.append({
                "source": raw_metadata.get("source_path", FILE_PATH),
                "category": raw_metadata.get("category", "kebijakan_dan_sop_layanan_internal"),
                "bab": current_bab,
                "subbab": current_subbab,
                "text": text,
                "chunk_hash": chunk_hash,
                "doc_version": "1.4" if not is_active else str(raw_metadata.get("doc_version", "2.0")),
                "is_active": is_active,
                "effective_date": str(raw_metadata.get("effective_date", "2026-07-01"))
            })

    for block in body.split("\n\n"):
        line = block.strip()
        if not line:
            continue
        if line.startswith("## "):
            flush_chunk()
            current_bab = line
            current_subbab = ""
            current_paragraphs = []
        elif line.startswith("### "):
            flush_chunk()
            current_subbab = line
            is_active = "NONAKTIF" not in line and "v1.4" not in line
            current_paragraphs = []
        elif line.startswith("# "):
            continue
        else:
            current_paragraphs.append(line)

    flush_chunk()
    return chunks


class RAGService:
    def __init__(self):
        self.client = genai.Client()
        with open(FILE_PATH, "r", encoding="utf-8") as f:
            contents = f.read()

        self.docs = parse_text(contents)
        self.chroma = chromadb.PersistentClient(path="data/vector")
        self.collection = self.chroma.get_or_create_collection(name=COLLECTION_NAME)
        if self.collection.count() < len(self.docs) and self.docs:
            if self.collection.count() > 0:
                self.chroma.delete_collection(COLLECTION_NAME)
                self.collection = self.chroma.get_or_create_collection(name=COLLECTION_NAME)
            self.setup()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        res = self.client.models.embed_content(
            model="gemini-embedding-001",
            contents=texts
        )
        return [e.values for e in res.embeddings]

    def setup(self):
        docs = ["\n".join([f"BAB: {d['bab']}", f"SUBBAB: {d['subbab']}", f"TEKS: {d['text']}"]) for d in self.docs]
        embeddings = self.embed_texts(docs)
        metadatas = [
            {
                "source": d["source"],
                "category": d["category"],
                "bab": d["bab"],
                "subbab": d["subbab"],
                "chunk_hash": d["chunk_hash"],
                "doc_version": d["doc_version"],
                "is_active": d["is_active"],
                "effective_date": d["effective_date"]
            }
            for d in self.docs
        ]
        ids = [f"chunk_{d['chunk_hash']}" for d in self.docs]
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=docs
        )

    def query(self, question: str) -> dict:
        prompt_vec = self.embed_texts([question])
        results = self.collection.query(
            n_results=TOP_K,
            include=["metadatas", "documents", "distances"],
            query_embeddings=prompt_vec,
            where={"is_active": True}
        )

        if not results["documents"] or not results["documents"][0]:
            return {
                "answer": "tidak ditemukan dalam dokumen",
                "confidence_label": "low",
                "reason_code": "no_relevant_context",
            }

        context_blocks = [
            f"[Konteks {i+1} - Jarak: {dist:.4f}]\n{doc}"
            for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0]))
        ]
        context_str = "\n\n".join(context_blocks)

        system_instruction = (
            "Anda adalah asisten QA faktual untuk kebijakan operasional NusantaraCare.\n"
            "Jawab pertanyaan pengguna secara ringkas dan lugas hanya berdasarkan teks konteks yang diberikan.\n"
            "Jika topik atau informasi tidak ada dalam konteks, isi reason_code='no_relevant_context', "
            "confidence_label='low', dan jelaskan bahwa informasi tidak ditemukan dalam dokumen."
        )

        ai_config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0,
            response_mime_type="application/json",
            response_schema=QueryResponse
        )
        full_prompt = f"Konteks:\n{context_str}\n\nPertanyaan: {question}"

        try:
            output = self.client.models.generate_content(
                model="gemini-flash-lite-latest",
                config=ai_config,
                contents=full_prompt,
            )
            if output and output.text:
                final = json.loads(output.text)
                if final.get("reason_code") != "answered":
                    final["confidence_label"] = "low"
                return {
                    "answer": final.get("answer", "tidak ditemukan dalam dokumen"),
                    "confidence_label": final.get("confidence_label", "low"),
                    "reason_code": final.get("reason_code", "no_relevant_context"),
                }
        except Exception as e:
            logger.error(f"Generation error: {e}")

        return {
            "answer": "tidak ditemukan dalam dokumen",
            "confidence_label": "low",
            "reason_code": "no_relevant_context",
        }
