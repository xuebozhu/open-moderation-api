from fastapi import (
    APIRouter,
    status,
)

from app.schemas.settings import ModerationSettings
from app.services.settings_service import settings_service


router = APIRouter(
    prefix="/api/v1/settings",
    tags=["Settings"],
)


@router.get(
    "",
    response_model=ModerationSettings,
    status_code=status.HTTP_200_OK,
    summary="Consultar configuración",
)
def get_settings() -> ModerationSettings:
    return settings_service.get()


@router.put(
    "",
    response_model=ModerationSettings,
    status_code=status.HTTP_200_OK,
    summary="Actualizar configuración",
)
def update_settings(
    settings: ModerationSettings,
) -> ModerationSettings:
    return settings_service.update(settings)


@router.post(
    "/reset",
    response_model=ModerationSettings,
    status_code=status.HTTP_200_OK,
    summary="Restablecer configuración",
)
def reset_settings() -> ModerationSettings:
    return settings_service.reset()