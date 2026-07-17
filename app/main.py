from fastapi import FastAPI

from app.api.moderation import (
    router as moderation_router,
)
from app.api.settings import (
    router as settings_router,
)


app = FastAPI(
    title="Open Moderation API",
    description=(
        "API reutilizable para moderar contenido textual "
        "mediante modelos locales de inteligencia artificial."
    ),
    version="0.2.0",
)


@app.get(
    "/api/v1/health",
    tags=["Health"],
)
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "open-moderation-api",
    }


app.include_router(
    moderation_router,
    prefix="/api/v1",
)

app.include_router(settings_router)