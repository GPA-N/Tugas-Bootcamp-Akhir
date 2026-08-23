# NusantaraCare RAG Service

Backend API berbasis FastAPI untuk pencarian SOP dan panduan operasional internal NusantaraCare menggunakan metode RAG (Retrieval-Augmented Generation).

- **FastAPI Cloud URL**: https://bootcamptugasakhir.fastapicloud.dev
- **Swagger UI**: https://bootcamptugasakhir.fastapicloud.dev/docs
- **GitHub Repository**: https://github.com/GPA-N/Tugas-Bootcamp-Akhir

---

## 1. Problem

Karyawan internal kesulitan mencari jawaban cepat terkait SOP operasional (reset password, penanganan insiden P1-P3, pengajuan perangkat kerja) karena dokumen panjang dan memuat bagian arsip kebijakan lama yang sudah tidak aktif.

Tujuan sistem:
- Menjawab pertanyaan operasional secara faktual berdasarkan dokumen resmi aktif v2.0.
- Mengabaikan aturan lama v1.4 yang sudah nonaktif.
- Menolak menjawab jika pertanyaan di luar konteks dokumen (`reason_code: no_relevant_context` dan `confidence_label: low`).
- Menolak input kosong dengan HTTP 400 Bad Request.

---

## 2. KB Understanding

Dokumen acuan berada di `data/raw_docs/nusantaracare_panduan_operasional_internal_v2.md`.

Poin penting dokumen:
- Dokumen memiliki metadata frontmatter (versi 2.0, tanggal efektif 2026-07-01).
- Terdapat bagian `### Arsip Kebijakan v1.4 — NONAKTIF` di akhir dokumen.
- Perbedaan aturan v1.4 vs v2.0:
  - Saluran tiket: v1.4 mengizinkan email biasa; v2.0 mewajibkan Service Portal (email hanya untuk kondisi darurat saat portal tidak dapat diakses dengan subjek `[DARURAT-PORTAL]`).
  - Pengajuan perlengkapan standar (keyboard/mouse): v1.4 minimal 3 hari kerja; v2.0 minimal 5 hari kerja sebelum tanggal kebutuhan.

---

## 3. RAG Design

- **Chunking**: Dokumen dipecah per sub-bab (`###`), menghasilkan total 40 chunk teks utuh agar konteks tabel dan aturan bersyarat tidak terputus.
- **Filter Metadata**: Chunk dari seksi v1.4 diberi metadata `is_active: False`. Query ChromaDB menerapkan filter `where={"is_active": True}` sehingga data usang v1.4 tidak pernah masuk konteks retrieval.
- **Embedding & Vector DB**: Menggunakan model `gemini-embedding-001` (3072 dimensi) dengan persistent ChromaDB di folder `data/vector`. ID chunk menggunakan prefix hash SHA-256 dari teks.
- **Structured Output**: Model `gemini-flash-lite-latest` di-generate dengan `response_schema=QueryResponse` (Pydantic) untuk memastikan output tervalidasi ke field `answer`, `confidence_label`, dan `reason_code`.

---

## 4. Arsitektur

```text
[User Request] ──> [POST /rag/] ──> [Validasi Input]
                                          │
                                          ▼
                                   [AgentRouter]
                                          │
                                          ▼
                                 [RAGService.query()]
                                          │
       ┌──────────────────────────────────┴──────────────────────────────────┐
       ▼                                                                     ▼
[gemini-embedding-001]                                           [ChromaDB Vector Query]
(Embed Query)                                                    (where is_active: True)
       │                                                                     │
       └──────────────────────────────────┬──────────────────────────────────┘
                                          │
                                          ▼
                             [Susun Konteks Dokumen]
                                          │
                                          ▼
                           [Gemini: Structured Output]
                            (response_schema: QueryResponse)
                                          │
                                          ▼
                                   [JSON Response]
```

---

## 5. Kontrak API

Endpoint utama: `POST /rag/`

### Request
```json
{
  "query": "Berapa hari minimal pengajuan perlengkapan standar seperti keyboard?"
}
```

### Response
```json
{
  "answer": "Permintaan perlengkapan standar wajib diajukan melalui Service Portal minimal 5 hari kerja sebelum tanggal Anda memerlukan perlengkapan tersebut.",
  "confidence_label": "high",
  "reason_code": "answered"
}
```

Daftar `reason_code`:
- `answered`: Terjawab dari dokumen aktif v2.0.
- `no_relevant_context`: Pertanyaan di luar cakupan dokumen atau data tidak ditemukan.
- `conflicting_sources`: Terjadi kontradiksi informasi dalam dokumen.

Endpoint health check: `GET /health` mengembalikan `{"status": "OK"}`.

---

## 6. Cara Menjalankan Lokal

### 1. Setup & Install
```bash
git clone https://github.com/GPA-N/Tugas-Bootcamp-Akhir.git
cd Tugas-Bootcamp-Akhir
uv sync
```

### 2. Environment Variable
Salin `.env.example` ke `.env` lalu masukkan API key Gemini:
```bash
cp .env.example .env
```
Isi `.env`:
```text
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Jalankan Server
```bash
uv run fastapi dev app/main.py
```
Akses Swagger UI di `http://127.0.0.1:8000/docs`.

### 4. Contoh Pengujian Curl
```bash
# Health check
curl http://127.0.0.1:8000/health

# Query in-scope
curl -X POST http://127.0.0.1:8000/rag/ \
  -H "Content-Type: application/json" \
  -d '{"query": "Apa saluran resmi untuk membuat tiket layanan internal?"}'

# Query out-of-scope
curl -X POST http://127.0.0.1:8000/rag/ \
  -H "Content-Type: application/json" \
  -d '{"query": "Berapa gaji bulanan direktur utama?"}'
```

---

## 7. Deployment

Aplikasi dideploy ke FastAPI Cloud menggunakan FastAPI CLI:

```bash
uv run fastapi login
uv run fastapi deploy
uv run fastapi cloud env set GEMINI_API_KEY
```

Endpoint deployment aktif:
- Base URL: https://bootcamptugasakhir.fastapicloud.dev
- Health check: https://bootcamptugasakhir.fastapicloud.dev/health
- Swagger Docs: https://bootcamptugasakhir.fastapicloud.dev/docs

---

## 8. Keterbatasan (jika ada)

- Batas kuota Free Tier Google Gemini API:
  - **Gemini Flash Lite** (LLM Generator): Dibatasi **15 RPM**, 250K TPM, dan 500 RPD. Batas 15 RPM ini menjadi *bottleneck* utama jika pengujian otomatis mengirimkan request berturut-turut tanpa jeda *cooldown*.
  - **Gemini Embedding 1** (Vector Embeddings): Dibatasi **100 RPM**, 30K TPM, dan 1.000 RPD.
- Database vektor ChromaDB berjalan dalam mode persistent file lokal (`data/vector`), sehingga indeks dibuat ulang saat container di-redeploy.

---

## 9. Kesimpulan & Rekomendasi

### Kesimpulan
Sistem RAG ini berhasil menjawab pertanyaan operasional SOP internal secara faktual dengan memfilter kebijakan arsip v1.4 dan menolak pertanyaan di luar konteks dokumen melalui format response Pydantic yang konsisten (`answer`, `confidence_label`, `reason_code`).

### Rekomendasi
- Menambahkan caching (misalnya Redis) untuk pertanyaan berulang agar menghemat kuota API embedding dan LLM.
- Menggunakan managed vector database (seperti Chroma Cloud atau Qdrant) untuk penyimpanan indeks vektor yang terpusat.