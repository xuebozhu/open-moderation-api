from typing import Any

from transformers import pipeline

from app.schemas.moderation import (
    CategoryResult,
    ModerateResponse,
    ModerationDecision,
)


MODEL_NAME = (
    "gravitee-io/"
    "distilbert-multilingual-toxicity-classifier"
)


class ToxicityService:
    """
    Servicio encargado de analizar la toxicidad de un texto.

    El umbral de rechazo se recibe dinámicamente desde la
    configuración global de moderación.
    """

    DEFAULT_REJECT_THRESHOLD = 0.80

    def __init__(self) -> None:
        self.classifier = self._load_model()

    @staticmethod
    def _load_model() -> Any:
        """
        Carga el modelo y su tokenizador.

        La primera vez puede descargar los archivos desde
        Hugging Face. Después se reutilizan desde la caché local.
        """

        return pipeline(
            task="text-classification",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            device=-1,
        )

    def moderate(
        self,
        content: str,
        threshold: float = DEFAULT_REJECT_THRESHOLD,
    ) -> ModerateResponse:
        """
        Analiza el texto aplicando el umbral indicado.

        Args:
            content: Texto que se quiere analizar.
            threshold: Puntuación mínima para rechazar el contenido.

        Returns:
            Resultado completo de la moderación.

        Raises:
            ValueError: Si el umbral no está entre 0 y 1.
            RuntimeError: Si el modelo devuelve un resultado inválido.
        """

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "El umbral de toxicidad debe estar entre 0 y 1."
            )

        prediction = self._predict(content)

        toxic_score = self._extract_toxic_score(
            prediction
        )

        decision = self._get_decision(
            toxic_score=toxic_score,
            threshold=threshold,
        )

        rounded_score = round(toxic_score, 6)

        return ModerateResponse(
            decision=decision,
            allowed=(
                decision == ModerationDecision.APPROVE
            ),
            score=rounded_score,
            categories=[
                CategoryResult(
                    label="toxicity",
                    score=rounded_score,
                )
            ],
            reason=self._get_reason(
                decision=decision,
                threshold=threshold,
            ),
            model=MODEL_NAME,
        )

    def _predict(
        self,
        content: str,
    ) -> dict[str, Any]:
        """
        Envía el texto al modelo y devuelve su predicción.
        """

        predictions = self.classifier(
            content,
            truncation=True,
        )

        if not predictions:
            raise RuntimeError(
                "El modelo no ha devuelto ninguna predicción."
            )

        prediction = predictions[0]

        if not isinstance(prediction, dict):
            raise RuntimeError(
                "El modelo ha devuelto un formato inesperado."
            )

        return prediction

    @staticmethod
    def _extract_toxic_score(
        prediction: dict[str, Any],
    ) -> float:
        """
        Convierte la respuesta del modelo en una escala común:

        0.0 = toxicidad muy baja
        1.0 = toxicidad muy alta
        """

        label = str(
            prediction.get("label", "")
        ).strip().lower()

        score = float(
            prediction.get("score", 0.0)
        )

        toxic_labels = {
            "toxic",
            "toxicity",
            "label_1",
        }

        non_toxic_labels = {
            "not-toxic",
            "not_toxic",
            "non-toxic",
            "non_toxic",
            "label_0",
        }

        if label in toxic_labels:
            return score

        if label in non_toxic_labels:
            return 1.0 - score

        raise RuntimeError(
            f"Etiqueta desconocida devuelta por el modelo: {label}"
        )

    @staticmethod
    def _get_decision(
        toxic_score: float,
        threshold: float,
    ) -> ModerationDecision:
        """
        Convierte la puntuación en una decisión binaria usando
        el umbral configurado.
        """

        if toxic_score >= threshold:
            return ModerationDecision.REJECT

        return ModerationDecision.APPROVE

    @staticmethod
    def _get_reason(
        decision: ModerationDecision,
        threshold: float,
    ) -> str:
        formatted_threshold = round(threshold, 3)

        if decision == ModerationDecision.REJECT:
            return (
                "El contenido supera el umbral de toxicidad "
                f"configurado ({formatted_threshold})."
            )

        return (
            "El contenido no supera el umbral de toxicidad "
            f"configurado ({formatted_threshold})."
        )