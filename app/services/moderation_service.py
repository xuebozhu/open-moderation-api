from app.schemas.moderation import (
    ModerateResponse,
    ModerationDecision,
)
from app.schemas.settings import ModerationSettings
from app.services.pii_service import PiiService
from app.services.settings_service import settings_service
from app.services.toxicity_service import ToxicityService


class ModerationService:
    """
    Orquesta el proceso completo de moderación.

    Orden de análisis:

    1. Whitelist.
    2. Blacklist.
    3. Información personal.
    4. Toxicidad.

    La configuración se obtiene dinámicamente en cada petición.
    """

    def __init__(self) -> None:
        self.pii_service = PiiService()
        self.toxicity_service = ToxicityService()

    def moderate(
        self,
        content: str,
    ) -> ModerateResponse:
        """
        Modera un texto utilizando la configuración activa.
        """

        settings = settings_service.get()

        normalized_content = content.casefold().strip()

        whitelist_response = self._moderate_whitelist(
            content=normalized_content,
            settings=settings,
        )

        if whitelist_response is not None:
            return whitelist_response

        blacklist_response = self._moderate_blacklist(
            content=normalized_content,
            settings=settings,
        )

        if blacklist_response is not None:
            return blacklist_response

        pii_response = self._moderate_pii(
            content=content,
            settings=settings,
        )

        if pii_response is not None:
            return pii_response

        return self._moderate_toxicity(
            content=content,
            settings=settings,
        )

    @staticmethod
    def _moderate_whitelist(
        content: str,
        settings: ModerationSettings,
    ) -> ModerateResponse | None:
        """
        Permite directamente el contenido si coincide con
        alguna expresión incluida en la whitelist.
        """

        matching_word = next(
            (
                word
                for word in settings.whitelist
                if word.casefold().strip()
                and word.casefold().strip() in content
            ),
            None,
        )

        if matching_word is None:
            return None

        return ModerateResponse(
            decision=ModerationDecision.APPROVE,
            allowed=True,
            score=0.0,
            categories=[],
            reason=(
                "El contenido coincide con una expresión "
                "incluida en la lista blanca."
            ),
            model="whitelist-rule",
        )

    @staticmethod
    def _moderate_blacklist(
        content: str,
        settings: ModerationSettings,
    ) -> ModerateResponse | None:
        """
        Rechaza directamente el contenido si coincide con
        alguna expresión incluida en la blacklist.
        """

        matching_word = next(
            (
                word
                for word in settings.blacklist
                if word.casefold().strip()
                and word.casefold().strip() in content
            ),
            None,
        )

        if matching_word is None:
            return None

        return ModerateResponse(
            decision=ModerationDecision.REJECT,
            allowed=False,
            score=1.0,
            categories=[],
            reason=(
                "El contenido contiene una expresión "
                "incluida en la lista negra."
            ),
            model="blacklist-rule",
        )

    def _moderate_pii(
        self,
        content: str,
        settings: ModerationSettings,
    ) -> ModerateResponse | None:
        """
        Ejecuta la detección de información personal si el
        módulo PII está activado.
        """

        if not settings.pii.enabled:
            return None

        pii_result = self.pii_service.analyze(content)

        if not pii_result.detected:
            return None

        enabled_categories = self._get_enabled_pii_categories(
            settings
        )

        filtered_categories = [
            category
            for category in pii_result.categories
            if self._is_pii_category_enabled(
                label=category.label,
                enabled_categories=enabled_categories,
            )
        ]

        if not filtered_categories:
            return None

        return ModerateResponse(
            decision=ModerationDecision.REJECT,
            allowed=False,
            score=1.0,
            categories=filtered_categories,
            reason=pii_result.reason,
            model=PiiService.MODEL_NAME,
        )

    def _moderate_toxicity(
        self,
        content: str,
        settings: ModerationSettings,
    ) -> ModerateResponse:
        """
        Ejecuta el modelo de toxicidad utilizando el umbral
        dinámico configurado desde el dashboard.
        """

        if not settings.toxicity.enabled:
            return ModerateResponse(
                decision=ModerationDecision.APPROVE,
                allowed=True,
                score=0.0,
                categories=[],
                reason=(
                    "El análisis de toxicidad está "
                    "desactivado."
                ),
                model=ToxicityService.MODEL_NAME,
            )

        toxicity_result = self.toxicity_service.analyze(
            content=content,
            threshold=settings.toxicity.threshold,
        )

        decision = (
            ModerationDecision.REJECT
            if toxicity_result.detected
            else ModerationDecision.APPROVE
        )

        return ModerateResponse(
            decision=decision,
            allowed=not toxicity_result.detected,
            score=round(
                toxicity_result.score,
                6,
            ),
            categories=toxicity_result.categories,
            reason=toxicity_result.reason,
            model=toxicity_result.model,
        )

    @staticmethod
    def _get_enabled_pii_categories(
        settings: ModerationSettings,
    ) -> dict[str, bool]:
        """
        Convierte la configuración PII de Pydantic en un
        diccionario sencillo.
        """

        categories = settings.pii.categories

        if hasattr(categories, "model_dump"):
            return categories.model_dump()

        if isinstance(categories, dict):
            return categories

        return {}

    @staticmethod
    def _is_pii_category_enabled(
        label: str,
        enabled_categories: dict[str, bool],
    ) -> bool:
        """
        Relaciona las etiquetas internas de PiiService con
        las categorías expuestas en el dashboard.
        """

        label_to_setting = {
            "email-address": "email",
            "phone-number": "phone",
            "spanish-national-id": "dni",
            "spanish-foreigner-id": "nie",
            "bank-account": "iban",
            "payment-card": "credit_card",
            "credential": "credential",
            "person": "person",
            "location": "location",
        }

        setting_name = label_to_setting.get(label)

        if setting_name is None:
            return True

        return enabled_categories.get(
            setting_name,
            True,
        )


moderation_service = ModerationService()