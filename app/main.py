from fastapi import FastAPI

from app.config import get_settings
from app.routes import chat, pdf


settings = get_settings()

app = FastAPI(title=settings.app_name)
app.include_router(pdf.router)
app.include_router(chat.router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
