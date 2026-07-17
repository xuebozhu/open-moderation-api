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


class ModerationSettings(BaseModel):
    toxicity: ToxicitySettings = Field(
        default_factory=ToxicitySettings
    )

    pii: PiiSettings = Field(
        default_factory=PiiSettings
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