import logging
import time
from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from app.schemas.moderation import CategoryResult


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToxicityAnalysis:
    detected: bool
    score: float
    categories: list[CategoryResult]
    reason: str
    model: str


class ToxicityService:
    """
    Analiza la toxicidad de un texto mediante un modelo Transformer.

    Además, registra en consola:

    - Texto recibido.
    - Tokens.
    - Identificadores de tokens.
    - Máscara de atención.
    - Logits.
    - Probabilidades.
    - Score de toxicidad.
    - Umbral.
    - Decisión.
    - Tiempo de inferencia.
    """

    MODEL_NAME = (
        "gravitee-io/"
        "distilbert-multilingual-toxicity-classifier"
    )

    MAX_LENGTH = 512
    MAX_LOGGED_TOKENS = 80

    def __init__(self) -> None:
        logger.info(
            "[TOXICITY] Cargando tokenizer: %s",
            self.MODEL_NAME,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME
        )

        logger.info(
            "[TOXICITY] Cargando modelo: %s",
            self.MODEL_NAME,
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(self.MODEL_NAME)
        )

        # Modo inferencia: desactiva dropout.
        self.model.eval()

        logger.info(
            "[TOXICITY] Etiquetas id2label: %s",
            self.model.config.id2label,
        )

        logger.info(
            "[TOXICITY] Etiquetas label2id: %s",
            self.model.config.label2id,
        )

        logger.info(
            "[TOXICITY] Modelo cargado correctamente."
        )

    def analyze(
        self,
        content: str,
        threshold: float,
    ) -> ToxicityAnalysis:
        """
        Analiza un texto y compara su score de toxicidad
        con el umbral configurado.
        """

        start_time = time.perf_counter()

        # 1. Tokenización.
        encoded = self.tokenizer(
            content,
            return_tensors="pt",
            truncation=True,
            max_length=self.MAX_LENGTH,
            padding=False,
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        token_ids = input_ids[0].tolist()

        tokens = self.tokenizer.convert_ids_to_tokens(
            token_ids
        )

        # 2. Inferencia.
        #
        # inference_mode evita calcular gradientes, ya que
        # aquí no estamos entrenando el modelo.
        with torch.inference_mode():
            outputs = self.model(**encoded)

        # Para una única entrada:
        # outputs.logits tiene forma [1, número_de_clases].
        logits_tensor = outputs.logits[0]

        # 3. Conversión de logits a probabilidades.
        probabilities_tensor = torch.softmax(
            logits_tensor,
            dim=-1,
        )

        logits = logits_tensor.tolist()
        probabilities = probabilities_tensor.tolist()

        # 4. Asociar cada posición con su etiqueta.
        scores_by_label: dict[str, float] = {}

        for class_id, probability in enumerate(
            probabilities
        ):
            label = self._get_label(class_id)

            scores_by_label[
                self._normalize_label(label)
            ] = float(probability)

        # 5. Extraer específicamente el score tóxico.
        toxic_score = self._extract_toxic_score(
            scores_by_label
        )

        # 6. Aplicar la política configurable.
        detected = toxic_score >= threshold

        decision = (
            "REJECT"
            if detected
            else "ALLOW"
        )

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        # 7. Mostrar trazabilidad.
        self._log_analysis(
            content=content,
            tokens=tokens,
            input_ids=token_ids,
            attention_mask=attention_mask[0].tolist(),
            logits=logits,
            probabilities=probabilities,
            threshold=threshold,
            toxic_score=toxic_score,
            decision=decision,
            elapsed_ms=elapsed_ms,
        )

        if detected:
            return ToxicityAnalysis(
                detected=True,
                score=toxic_score,
                categories=[
                    CategoryResult(
                        label="toxicity",
                        score=toxic_score,
                    )
                ],
                reason=(
                    "El contenido supera el umbral "
                    "de toxicidad configurado."
                ),
                model=self.MODEL_NAME,
            )

        return ToxicityAnalysis(
            detected=False,
            score=toxic_score,
            categories=[],
            reason=(
                "El contenido no supera el umbral "
                "de toxicidad configurado."
            ),
            model=self.MODEL_NAME,
        )

    def _get_label(
        self,
        class_id: int,
    ) -> str:
        """
        Obtiene la etiqueta asociada a una salida del modelo.
        """

        id2label = self.model.config.id2label

        # Algunas configuraciones usan claves enteras y otras
        # pueden exponerlas como cadenas.
        return str(
            id2label.get(
                class_id,
                id2label.get(
                    str(class_id),
                    f"LABEL_{class_id}",
                ),
            )
        )

    @staticmethod
    def _normalize_label(label: str) -> str:
        """
        Normaliza variantes de escritura de etiquetas.
        """

        return (
            label
            .casefold()
            .strip()
            .replace("_", "-")
            .replace(" ", "-")
        )

    @staticmethod
    def _extract_toxic_score(
        scores_by_label: dict[str, float],
    ) -> float:
        """
        Busca la etiqueta correspondiente a toxicidad.
        """

        toxic_labels = (
            "toxic",
            "toxicity",
            "label-1",
        )

        for label in toxic_labels:
            if label in scores_by_label:
                return scores_by_label[label]

        raise RuntimeError(
            "No se ha encontrado la clase de toxicidad. "
            "Etiquetas disponibles: "
            f"{list(scores_by_label.keys())}"
        )

    def _log_analysis(
        self,
        *,
        content: str,
        tokens: list[str],
        input_ids: list[int],
        attention_mask: list[int],
        logits: list[float],
        probabilities: list[float],
        threshold: float,
        toxic_score: float,
        decision: str,
        elapsed_ms: float,
    ) -> None:
        """
        Registra en consola las principales etapas
        de la clasificación.
        """

        logged_tokens = tokens[
            :self.MAX_LOGGED_TOKENS
        ]

        logged_input_ids = input_ids[
            :self.MAX_LOGGED_TOKENS
        ]

        logged_attention_mask = attention_mask[
            :self.MAX_LOGGED_TOKENS
        ]

        was_truncated = (
            len(tokens) > self.MAX_LOGGED_TOKENS
        )

        suffix = (
            " ..."
            if was_truncated
            else ""
        )

        logger.info(
            "[TOXICITY] Texto recibido: %s",
            content,
        )

        logger.info(
            "[TOXICITY] Tokens: %s%s",
            logged_tokens,
            suffix,
        )

        logger.info(
            "[TOXICITY] Input IDs: %s%s",
            logged_input_ids,
            suffix,
        )

        logger.info(
            "[TOXICITY] Attention mask: %s%s",
            logged_attention_mask,
            suffix,
        )

        rounded_logits = [
            round(value, 4)
            for value in logits
        ]

        logger.info(
            "[TOXICITY] Logits: %s",
            rounded_logits,
        )

        logger.info(
            "[TOXICITY] Probabilidades:"
        )

        for class_id, probability in enumerate(
            probabilities
        ):
            label = self._get_label(class_id)

            logger.info(
                "  %-12s = %.4f",
                label,
                probability,
            )

        logger.info(
            "[TOXICITY] Score tóxico: %.4f",
            toxic_score,
        )

        logger.info(
            "[TOXICITY] Threshold: %.4f",
            threshold,
        )

        logger.info(
            "[TOXICITY] Comparación: %.4f %s %.4f",
            toxic_score,
            ">=" if decision == "REJECT" else "<",
            threshold,
        )

        logger.info(
            "[TOXICITY] Decisión: %s",
            decision,
        )

        logger.info(
            "[TOXICITY] Tiempo de inferencia: %.2f ms",
            elapsed_ms,
        )