from pydantic import BaseModel, Field


class ToxicitySettings(BaseModel):
    enabled: bool = True

    threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description=(
            "Puntuación mínima de toxicidad necesaria "
            "para rechazar el contenido."
        ),
    )


class PiiCategorySettings(BaseModel):
    email: bool = True
    phone: bool = True
    dni: bool = True
    nie: bool = True
    iban: bool = True
    credit_card: bool = True
    credential: bool = True
    person: bool = True
    location: bool = True


class PiiSettings(BaseModel):
    enabled: bool = True

    categories: PiiCategorySettings = Field(
        default_factory=PiiCategorySettings
    )


class SpamSettings(BaseModel):
    enabled: bool = True

    threshold: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description=(
            "Puntuación mínima de spam necesaria "
            "para rechazar el contenido."
        ),
    )

    max_urls: int = Field(
        default=2,
        ge=0,
        le=20,
        description=(
            "Número máximo de enlaces permitidos."
        ),
    )

    max_uppercase_ratio: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description=(
            "Proporción máxima permitida de letras "
            "mayúsculas."
        ),
    )

    max_repeated_characters: int = Field(
        default=4,
        ge=1,
        le=50,
        description=(
            "Número máximo permitido de repeticiones "
            "consecutivas de un carácter."
        ),
    )

    max_repeated_word_count: int = Field(
        default=4,
        ge=1,
        le=50,
        description=(
            "Número máximo de veces que puede repetirse "
            "una misma palabra."
        ),
    )

    promotional_terms: list[str] = Field(
        default_factory=lambda: [
            "compra ahora",
            "oferta exclusiva",
            "haz clic aquí",
            "gana dinero",
            "dinero rápido",
            "promoción limitada",
        ],
        max_length=500,
        description=(
            "Expresiones promocionales utilizadas "
            "para detectar posibles mensajes de spam."
        ),
    )


class ModerationSettings(BaseModel):
    toxicity: ToxicitySettings = Field(
        default_factory=ToxicitySettings
    )

    pii: PiiSettings = Field(
        default_factory=PiiSettings
    )

    spam: SpamSettings = Field(
        default_factory=SpamSettings
    )

    blacklist: list[str] = Field(
        default_factory=list,
        max_length=500,
        description="Expresiones que provocan el rechazo.",
    )

    whitelist: list[str] = Field(
        default_factory=list,
        max_length=500,
        description=(
            "Contenidos completos que se aprueban directamente."
        ),
    )