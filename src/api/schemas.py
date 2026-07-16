from pydantic import BaseModel, field_validator

class PredictRequest(BaseModel):
    """Entrada: lista de closes brutos (não escalados)."""
    closes: list[float]

    @field_validator("closes")
    @classmethod
    def validar_tamanho(cls, v):
        # o modelo exige exatamente n_steps valores — validamos aqui
        if len(v) !=60:
            raise ValueError(f"Esperados exatamente 60 closes, recebidos {len(v)}")
        return v


class PredictResponse(BaseModel):
    """Saída: previsão em reais."""
    prediction: float
    ticker: str    