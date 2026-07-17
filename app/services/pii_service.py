import re
from dataclasses import dataclass

import spacy
from spacy.language import Language

from app.schemas.moderation import CategoryResult


@dataclass(frozen=True)
class PiiAnalysis:
    detected: bool
    categories: list[CategoryResult]
    reason: str


class PiiService:
    """
    Detecta información personal identificable mediante:

    - Expresiones regulares para datos estructurados.
    - spaCy para reconocer personas y ubicaciones.
    - Reglas de contexto para reducir falsos positivos.

    No almacena ni devuelve el valor personal detectado.
    """

    MODEL_NAME = "hybrid-regex-spacy-pii-detector-v1"

    ADDRESS_CONTEXTS = (
        "vive en",
        "vivo en",
        "vivimos en",
        "reside en",
        "resido en",
        "residimos en",
        "mi dirección",
        "su dirección",
        "dirección de",
        "direccion de",
        "domicilio",
        "calle",
        "avenida",
        "plaza",
        "paseo",
        "carretera",
        "camino",
        "urbanización",
        "urbanizacion",
        "portal",
        "piso",
    )

    PERSONAL_CONTEXTS = (
        "vive en",
        "vivo con",
        "reside en",
        "resido con",
        "su dirección",
        "mi dirección",
        "dirección de",
        "direccion de",
        "su domicilio",
        "mi domicilio",
        "domicilio de",
        "su teléfono",
        "mi teléfono",
        "teléfono de",
        "telefono de",
        "su móvil",
        "mi móvil",
        "móvil de",
        "movil de",
        "su correo",
        "mi correo",
        "correo de",
        "su email",
        "mi email",
        "email de",
        "su dni",
        "mi dni",
        "dni de",
        "contraseña de",
        "password de",
        "clave de",
        "pin de",
        "token de",
    )

    DNI_CONTEXT_PATTERN = re.compile(
        r"""
        \b
        (?:dni|documento\s+de\s+identidad)
        \b
        [^\d]{0,40}
        \d{8}
        [a-z]?
        \b
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    PASSWORD_PATTERN = re.compile(
        r"""
        \b
        (?:contraseña|password|clave|pin|token|secret)
        \b
        .{0,50}?
        (?:
            es
            |
            sea
            |
            vale
            |
            :
            |
            =
        )
        \s*
        [A-Za-z0-9!@#$%^&*()_\-+=.]{4,}
        \b
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    NIE_PATTERN = re.compile(
        r"\b[xyz]\d{7}[a-z]\b",
        re.IGNORECASE,
    )

    EMAIL_PATTERN = re.compile(
        r"""
        \b
        [a-z0-9._%+-]+
        @
        [a-z0-9.-]+
        \.[a-z]{2,}
        \b
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    PHONE_CONTEXT_PATTERN = re.compile(
        r"""
        \b
        (?:
            tel[eé]fono
            |
            m[oó]vil
            |
            contacto
            |
            whatsapp
            |
            bizum
            |
            llamar
            |
            ll[aá]mame
            |
            escr[ií]beme
        )
        \b
        [^\d+]{0,30}
        (?:\+34[\s.-]?)?
        [6789]
        (?:[\s.-]?\d){8}
        \b
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    IBAN_PATTERN = re.compile(
        r"""
        \b
        ES
        \d{2}
        (?:[\s-]?\d{4}){5}
        \b
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    CREDIT_CARD_PATTERN = re.compile(
        r"""
        \b
        (?:\d[ -]*?){13,19}
        \b
        """,
        re.VERBOSE,
    )

    def __init__(self) -> None:
        self.nlp: Language = spacy.load(
            "es_core_news_md"
        )

    def analyze(self, content: str) -> PiiAnalysis:
        detections: list[CategoryResult] = []

        self._append_if_matches(
            content,
            self.DNI_CONTEXT_PATTERN,
            "spanish-national-id",
            detections,
        )

        self._append_if_matches(
            content,
            self.PASSWORD_PATTERN,
            "credential",
            detections,
        )

        self._append_if_matches(
            content,
            self.NIE_PATTERN,
            "spanish-foreigner-id",
            detections,
        )

        self._append_if_matches(
            content,
            self.EMAIL_PATTERN,
            "email-address",
            detections,
        )

        self._append_if_matches(
            content,
            self.PHONE_CONTEXT_PATTERN,
            "phone-number",
            detections,
        )

        self._append_if_matches(
            content,
            self.IBAN_PATTERN,
            "bank-account",
            detections,
        )

        self._append_if_matches(
            content,
            self.CREDIT_CARD_PATTERN,
            "payment-card",
            detections,
        )

        self._analyze_named_entities(
            content,
            detections,
        )

        if detections:
            return PiiAnalysis(
                detected=True,
                categories=detections,
                reason=(
                    "El contenido parece incluir información "
                    "personal o financiera."
                ),
            )

        return PiiAnalysis(
            detected=False,
            categories=[],
            reason="No se han detectado datos personales.",
        )

    def _analyze_named_entities(
        self,
        content: str,
        detections: list[CategoryResult],
    ) -> None:
        document = self.nlp(content)

        normalized_content = content.casefold()

        has_address_context = any(
            context in normalized_content
            for context in self.ADDRESS_CONTEXTS
        )

        has_personal_context = any(
            context in normalized_content
            for context in self.PERSONAL_CONTEXTS
        )

        detected_labels = {
            detection.label
            for detection in detections
        }

        for entity in document.ents:
            category: str | None = None

            if (
                entity.label_ == "PER"
                and has_personal_context
            ):
                category = "person"

            elif (
                entity.label_ == "LOC"
                and has_address_context
            ):
                category = "location"

            if category is None:
                continue

            if category in detected_labels:
                continue

            detections.append(
                CategoryResult(
                    label=category,
                    score=1.0,
                )
            )

            detected_labels.add(category)

    @staticmethod
    def _append_if_matches(
        content: str,
        pattern: re.Pattern[str],
        label: str,
        detections: list[CategoryResult],
    ) -> None:
        if pattern.search(content):
            detections.append(
                CategoryResult(
                    label=label,
                    score=1.0,
                )
            )