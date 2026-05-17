from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.config import get_settings
from app.models import DocumentInfo, UploadResponse
from app.services.rag_service import get_indexed_documents, index_pdf


router = APIRouter(tags=["pdf"])


@router.post("/upload-pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A PDF filename is required",
        )

    if file.content_type != "application/pdf" and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported",
        )

    settings = get_settings()
    safe_name = Path(file.filename).name
    destination = settings.upload_dir / safe_name

    content = await file.read()
    destination.write_bytes(content)

    chunks_indexed = await index_pdf(destination, source=safe_name)
    if chunks_indexed == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No extractable text found in PDF",
        )

    return UploadResponse(
        message="PDF indexed successfully",
        filename=safe_name,
        chunks_indexed=chunks_indexed,
    )


@router.get("/documents", response_model=list[DocumentInfo])
async def documents() -> list[DocumentInfo]:
    return await get_indexed_documents()
