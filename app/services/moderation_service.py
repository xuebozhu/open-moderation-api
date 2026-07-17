from app.schemas.moderation import (
    CategoryResult,
    ModerateResponse,
    ModerationDecision,
)
from app.schemas.settings import (
    ModerationSettings,
    PiiCategorySettings,
)
from app.services.pii_service import PiiService
from app.services.settings_service import settings_service
from app.services.toxicity_service import ToxicityService


class ModerationService:
    """
    Coordina todo el proceso de moderación.

    Orden de ejecución:

    1. Recuperar la configuración global.
    2. Comprobar lista blanca.
    3. Comprobar lista negra.
    4. Ejecutar el detector PII si está activado.
    5. Ejecutar el detector de toxicidad si está activado.
    6. Devolver APPROVE o REJECT.
    """

    def __init__(
        self,
        pii_service: PiiService,
        toxicity_service: ToxicityService,
    ) -> None:
        self.pii_service = pii_service
        self.toxicity_service = toxicity_service

    def moderate(
        self,
        content: str,
    ) -> ModerateResponse:
        settings = settings_service.get()

        normalized_content = self._normalize(content)

        # La whitelist utiliza coincidencia exacta para evitar que una
        # palabra común permita saltarse toda la moderación.
        if self._matches_whitelist(
            normalized_content,
            settings.whitelist,
        ):
            return self._build_policy_response(
                decision=ModerationDecision.APPROVE,
                label="whitelist",
                reason=(
                    "El contenido completo coincide con una entrada "
                    "de la lista blanca."
                ),
            )

        if self._matches_blacklist(
            normalized_content,
            settings.blacklist,
        ):
            return self._build_policy_response(
                decision=ModerationDecision.REJECT,
                label="blacklist",
                reason=(
                    "El contenido contiene una expresión incluida "
                    "en la lista negra."
                ),
            )

        pii_response = self._moderate_pii(
            content=content,
            settings=settings,
        )

        if pii_response is not None:
            return pii_response

        if settings.toxicity.enabled:
            return self.toxicity_service.moderate(
                content=content,
                threshold=settings.toxicity.threshold,
            )

        return ModerateResponse(
            decision=ModerationDecision.APPROVE,
            allowed=True,
            score=0.0,
            categories=[],
            reason=(
                "El contenido ha sido aprobado porque no se ha "
                "detectado información personal y el módulo de "
                "toxicidad está desactivado."
            ),
            model="moderation-settings",
        )

    def _moderate_pii(
        self,
        content: str,
        settings: ModerationSettings,
    ) -> ModerateResponse | None:
        """
        Ejecuta PII y conserva únicamente las categorías que estén
        activadas en la configuración.
        """

        if not settings.pii.enabled:
            return None

        pii_result = self.pii_service.analyze(content)

        enabled_detections = [
            detection
            for detection in pii_result.categories
            if self._is_pii_category_enabled(
                label=detection.label,
                categories=settings.pii.categories,
            )
        ]

        if not enabled_detections:
            return None

        highest_score = max(
            detection.score
            for detection in enabled_detections
        )

        return ModerateResponse(
            decision=ModerationDecision.REJECT,
            allowed=False,
            score=highest_score,
            categories=enabled_detections,
            reason=(
                "El contenido parece incluir una categoría de "
                "información personal o financiera que está "
                "activada en la configuración."
            ),
            model=PiiService.MODEL_NAME,
        )

    @staticmethod
    def _normalize(value: str) -> str:
        """
        Normaliza textos para realizar comparaciones sin distinguir
        mayúsculas, minúsculas ni espacios exteriores.
        """

        return value.strip().casefold()

    @classmethod
    def _matches_whitelist(
        cls,
        normalized_content: str,
        whitelist: list[str],
    ) -> bool:
        """
        La lista blanca requiere coincidencia con el contenido
        completo.

        Ejemplo:
            whitelist = ["hola mundo"]

            "Hola mundo"         -> coincide
            "Hola mundo 1234"    -> no coincide
        """

        normalized_whitelist = {
            cls._normalize(entry)
            for entry in whitelist
            if entry.strip()
        }

        return normalized_content in normalized_whitelist

    @classmethod
    def _matches_blacklist(
        cls,
        normalized_content: str,
        blacklist: list[str],
    ) -> bool:
        """
        La lista negra comprueba si alguna expresión configurada
        está contenida en el texto.
        """

        return any(
            cls._normalize(entry) in normalized_content
            for entry in blacklist
            if entry.strip()
        )

    @staticmethod
    def _is_pii_category_enabled(
        label: str,
        categories: PiiCategorySettings,
    ) -> bool:
        """
        Relaciona las etiquetas internas de PiiService con los
        parámetros configurables expuestos al dashboard.
        """

        category_mapping = {
            "spanish-national-id": categories.dni,
            "spanish-foreigner-id": categories.nie,
            "email-address": categories.email,
            "phone-number": categories.phone,
            "bank-account": categories.iban,
            "payment-card": categories.credit_card,
            "credential": categories.credential,
            "person": categories.person,
            "location": categories.location,
        }

        return category_mapping.get(label, False)

    @staticmethod
    def _build_policy_response(
        decision: ModerationDecision,
        label: str,
        reason: str,
    ) -> ModerateResponse:
        allowed = decision == ModerationDecision.APPROVE
        score = 0.0 if allowed else 1.0

        return ModerateResponse(
            decision=decision,
            allowed=allowed,
            score=score,
            categories=[
                CategoryResult(
                    label=label,
                    score=1.0,
                )
            ],
            reason=reason,
            model="configurable-policy-rules",
        )