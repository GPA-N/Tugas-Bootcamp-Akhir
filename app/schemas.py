from typing import Literal
from pydantic import BaseModel


class QueryRequest(BaseModel):
  query: str


class QueryResponse(BaseModel):
  answer: str
  confidence_label: Literal["high", "medium", "low"]
  reason_code: Literal["answered", "no_relevant_context","conflicting_sources"]