from enum import StrEnum

from pydantic import BaseModel, Field


class ModerationDecision(StrEnum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"


class ModerateRequest(BaseModel):
    content: str = Field(
        min_length=1,
        max_length=5000,
        description="Texto que se quiere analizar.",
        examples=["Hola, espero que tengas un buen día."],
    )


class CategoryResult(BaseModel):
    label: str = Field(
        description="Categoría analizada."
    )

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Puntuación obtenida entre 0 y 1.",
    )


class ModerateResponse(BaseModel):
    decision: ModerationDecision

    allowed: bool = Field(
        description=(
            "Indica si el contenido puede publicarse."
        )
    )

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Puntuación general de toxicidad.",
    )

    categories: list[CategoryResult]

    reason: str = Field(
        description="Explicación resumida de la decisión."
    )

    model: str = Field(
        description="Modelo utilizado para analizar el texto."
    )