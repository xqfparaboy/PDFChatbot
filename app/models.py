from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_indexed: int


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)


class SourceChunk(BaseModel):
    source: str
    page: int
    score: float
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
