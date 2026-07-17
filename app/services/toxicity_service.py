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
    """

    REJECT_THRESHOLD = 0.80

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
    ) -> ModerateResponse:
        """
        Ejecuta el proceso completo de moderación.
        """

        prediction = self._predict(content)

        toxic_score = self._extract_toxic_score(
            prediction
        )

        decision = self._get_decision(
            toxic_score
        )

        return ModerateResponse(
            decision=decision,
            allowed=(
                decision == ModerationDecision.APPROVE
            ),
            score=round(toxic_score, 6),
            categories=[
                CategoryResult(
                    label="toxicity",
                    score=round(toxic_score, 6),
                )
            ],
            reason=self._get_reason(decision),
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

    @classmethod
    def _get_decision(
        cls,
        toxic_score: float,
    ) -> ModerationDecision:
        """
        Convierte la puntuación en una decisión binaria.
        """

        if toxic_score >= cls.REJECT_THRESHOLD:
            return ModerationDecision.REJECT

        return ModerationDecision.APPROVE

    @staticmethod
    def _get_reason(
        decision: ModerationDecision,
    ) -> str:
        if decision == ModerationDecision.REJECT:
            return (
                "El contenido supera el umbral de toxicidad."
            )

        return (
            "No se ha detectado toxicidad suficiente "
            "para bloquear el contenido."
        )