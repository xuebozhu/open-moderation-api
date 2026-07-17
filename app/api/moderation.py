from functools import lru_cache

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from app.schemas.moderation import (
    ModerateRequest,
    ModerateResponse,
)
from app.services.moderation_service import ModerationService
from app.services.pii_service import PiiService
from app.services.toxicity_service import ToxicityService


router = APIRouter(
    prefix="/moderate",
    tags=["Moderation"],
)


@lru_cache
def get_pii_service() -> PiiService:
    """
    Crea y reutiliza una única instancia del detector PII.
    """

    return PiiService()


@lru_cache
def get_toxicity_service() -> ToxicityService:
    """
    Carga y reutiliza una única instancia del modelo de toxicidad.
    """

    return ToxicityService()


@lru_cache
def get_moderation_service() -> ModerationService:
    """
    Crea el servicio coordinador utilizando las instancias
    reutilizables de PII y toxicidad.
    """

    return ModerationService(
        pii_service=get_pii_service(),
        toxicity_service=get_toxicity_service(),
    )


@router.post(
    "",
    response_model=ModerateResponse,
    status_code=status.HTTP_200_OK,
    summary="Moderar contenido textual",
    description=(
        "Modera contenido aplicando la configuración global "
        "administrada mediante los endpoints de settings."
    ),
)
def moderate(
    request: ModerateRequest,
    moderation_service: ModerationService = Depends(
        get_moderation_service
    ),
) -> ModerateResponse:
    try:
        return moderation_service.moderate(
            request.content
        )

    except (RuntimeError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(error),
        ) from error