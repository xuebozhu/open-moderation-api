from fastapi import APIRouter

from app.schemas.moderation import (
    ModerateRequest,
    ModerateResponse,
)
from app.services.moderation_service import (
    moderation_service,
)


router = APIRouter(
    prefix="/moderate",
    tags=["Moderation"],
)


@router.post(
    "",
    response_model=ModerateResponse,
    summary="Moderar contenido textual",
    description=(
        "Analiza un texto aplicando las reglas activas de "
        "lista blanca, lista negra, detección de datos "
        "personales y clasificación de toxicidad."
    ),
)
def moderate(
    request: ModerateRequest,
) -> ModerateResponse:
    return moderation_service.moderate(
        content=request.content,
    )