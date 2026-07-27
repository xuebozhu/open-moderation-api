import re
from dataclasses import dataclass

from app.schemas.moderation import CategoryResult


@dataclass(frozen=True)
class SpamAnalysis:
    detected: bool
    score: float
    categories: list[CategoryResult]
    reason: str
    model: str


class SpamService:
    """
    Detecta posibles mensajes de spam mediante reglas
    configurables.

    Analiza:

    - Número excesivo de enlaces.
    - Uso excesivo de mayúsculas.
    - Repetición excesiva de caracteres.
    - Repetición excesiva de una misma palabra.
    - Presencia de expresiones promocionales.

    No almacena el contenido analizado.
    """

    MODEL_NAME = "rule-based-spam-detector-v1"

    URL_PATTERN = re.compile(
        r"""
        (?:
            https?://[^\s]+
            |
            www\.[^\s]+
        )
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    WORD_PATTERN = re.compile(
        r"\b[\wáéíóúüñ]+\b",
        re.IGNORECASE,
    )

    def analyze(
        self,
        content: str,
        threshold: float,
        max_urls: int,
        max_uppercase_ratio: float,
        max_repeated_characters: int,
        max_repeated_word_count: int,
        promotional_terms: list[str],
    ) -> SpamAnalysis:
        """
        Analiza el texto y calcula una puntuación de spam
        comprendida entre 0 y 1.
        """

        normalized_content = content.casefold().strip()

        url_count = self._count_urls(content)

        uppercase_ratio = self._calculate_uppercase_ratio(
            content
        )

        repeated_characters = (
            self._has_repeated_characters(
                content=content,
                maximum=max_repeated_characters,
            )
        )

        repeated_words = self._find_repeated_words(
            content=content,
            maximum=max_repeated_word_count,
        )

        matching_promotional_terms = (
            self._find_promotional_terms(
                content=normalized_content,
                promotional_terms=promotional_terms,
            )
        )

        score = self._calculate_score(
            url_count=url_count,
            max_urls=max_urls,
            uppercase_ratio=uppercase_ratio,
            max_uppercase_ratio=max_uppercase_ratio,
            repeated_characters=repeated_characters,
            repeated_words=repeated_words,
            promotional_term_count=len(
                matching_promotional_terms
            ),
        )

        detected = score >= threshold

        categories = self._build_categories(
            url_count=url_count,
            max_urls=max_urls,
            uppercase_ratio=uppercase_ratio,
            max_uppercase_ratio=max_uppercase_ratio,
            repeated_characters=repeated_characters,
            repeated_words=repeated_words,
            promotional_terms=matching_promotional_terms,
        )

        if detected:
            reason = (
                "El contenido supera el umbral de spam "
                "configurado."
            )
        else:
            reason = (
                "El contenido no supera el umbral de spam "
                "configurado."
            )

        return SpamAnalysis(
            detected=detected,
            score=score,
            categories=categories,
            reason=reason,
            model=self.MODEL_NAME,
        )

    @classmethod
    def _count_urls(
        cls,
        content: str,
    ) -> int:
        return len(
            cls.URL_PATTERN.findall(content)
        )

    @classmethod
    def _calculate_uppercase_ratio(
        cls,
        content: str,
    ) -> float:

        content_without_urls = cls.URL_PATTERN.sub(
            "",
            content,
        )

        words = re.findall(
            r"\b[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+\b",
            content_without_urls,
        )

        if not words:
            return 0.0

        uppercase_words = sum(
            1
            for word in words
            if len(word) >= 2 and word.isupper()
        )

        return uppercase_words / len(words)

    @staticmethod
    def _has_repeated_characters(
        content: str,
        maximum: int,
    ) -> bool:
        """
        Devuelve True cuando un carácter aparece repetido
        más veces que el máximo permitido.

        Por ejemplo, con maximum=4:

        holaaaaa -> detectado
        holaaa   -> permitido
        """

        repetition_limit = maximum + 1

        pattern = re.compile(
            rf"(.)\1{{{repetition_limit - 1},}}",
            re.IGNORECASE,
        )

        return bool(pattern.search(content))

    @classmethod
    def _find_repeated_words(
        cls,
        content: str,
        maximum: int,
    ) -> list[str]:
        words = [
            word.casefold()
            for word in cls.WORD_PATTERN.findall(content)
            if len(word) >= 3
        ]

        word_counts: dict[str, int] = {}

        for word in words:
            word_counts[word] = (
                word_counts.get(word, 0) + 1
            )

        return [
            word
            for word, count in word_counts.items()
            if count > maximum
        ]

    @staticmethod
    def _find_promotional_terms(
        content: str,
        promotional_terms: list[str],
    ) -> list[str]:
        matches: list[str] = []

        for term in promotional_terms:
            normalized_term = term.casefold().strip()

            if (
                normalized_term
                and normalized_term in content
            ):
                matches.append(term)

        return matches

    @staticmethod
    def _calculate_score(
        url_count: int,
        max_urls: int,
        uppercase_ratio: float,
        max_uppercase_ratio: float,
        repeated_characters: bool,
        repeated_words: list[str],
        promotional_term_count: int,
    ) -> float:
        """
        Calcula una puntuación acumulativa.

        Los pesos suman como máximo 1:
        - Enlaces: 0.30
        - Mayúsculas: 0.20
        - Caracteres repetidos: 0.20
        - Palabras repetidas: 0.20
        - Expresiones promocionales: 0.10
        """

        score = 0.0

        if url_count > max_urls:
            score += 0.30

        if uppercase_ratio >= max_uppercase_ratio:
            score += 0.20

        if repeated_characters:
            score += 0.20

        if repeated_words:
            score += 0.20

        if promotional_term_count > 0:
            score += 0.10

        return round(
            min(score, 1.0),
            6,
        )

    @staticmethod
    def _build_categories(
        url_count: int,
        max_urls: int,
        uppercase_ratio: float,
        max_uppercase_ratio: float,
        repeated_characters: bool,
        repeated_words: list[str],
        promotional_terms: list[str],
    ) -> list[CategoryResult]:
        categories: list[CategoryResult] = []

        if url_count > max_urls:
            categories.append(
                CategoryResult(
                    label="spam-excessive-links",
                    score=1.0,
                )
            )

        if uppercase_ratio >= max_uppercase_ratio:
            categories.append(
                CategoryResult(
                    label="spam-excessive-uppercase",
                    score=round(
                        uppercase_ratio,
                        6,
                    ),
                )
            )

        if repeated_characters:
            categories.append(
                CategoryResult(
                    label="spam-repeated-characters",
                    score=1.0,
                )
            )

        if repeated_words:
            categories.append(
                CategoryResult(
                    label="spam-repeated-words",
                    score=1.0,
                )
            )

        if promotional_terms:
            categories.append(
                CategoryResult(
                    label="spam-promotional-content",
                    score=1.0,
                )
            )

        return categories