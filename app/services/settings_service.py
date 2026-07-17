from app.schemas.settings import ModerationSettings


class SettingsService:
    """
    Gestiona la configuración global de moderación.

    En esta fase se almacena en memoria. La configuración vuelve
    a los valores predeterminados cuando se reinicia la API.
    """

    def __init__(self) -> None:
        self._settings = ModerationSettings()

    def get(self) -> ModerationSettings:
        return self._settings

    def update(
        self,
        settings: ModerationSettings,
    ) -> ModerationSettings:
        self._settings = settings
        return self._settings

    def reset(self) -> ModerationSettings:
        self._settings = ModerationSettings()
        return self._settings


settings_service = SettingsService()