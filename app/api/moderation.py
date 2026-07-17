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
    ModerationDecision,
)
from app.services.pii_service import PiiService
from app.services.toxicity_service import ToxicityService


router = APIRouter(
    prefix="/moderate",
    tags=["Moderation"],
)


@lru_cache
def get_toxicity_service() -> ToxicityService:
    """
    Carga y reutiliza una única instancia del modelo de toxicidad.
    """

    return ToxicityService()


@lru_cache
def get_pii_service() -> PiiService:
    """
    Crea y reutiliza el detector de datos personales.
    """

    return PiiService()


@router.post(
    "",
    response_model=ModerateResponse,
    status_code=status.HTTP_200_OK,
    summary="Moderar contenido textual",
    description=(
        "Detecta datos personales y analiza la toxicidad "
        "del contenido."
    ),
)
def moderate(
    request: ModerateRequest,
    pii_service: PiiService = Depends(get_pii_service),
    toxicity_service: ToxicityService = Depends(
        get_toxicity_service
    ),
) -> ModerateResponse:
    try:
        pii_result = pii_service.analyze(
            request.content
        )

        if pii_result.detected:
            return ModerateResponse(
                decision=ModerationDecision.REJECT,
                allowed=False,
                score=1.0,
                categories=pii_result.categories,
                reason=pii_result.reason,
                model=PiiService.MODEL_NAME,
            )

        return toxicity_service.moderate(
            request.content
        )

    except RuntimeError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(error),
        ) from error