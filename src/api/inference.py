import json
import torch
import joblib
import numpy as np
from model import ModeloLSTM

# ---- STARTUP: carrega artefatos uma vez ----
with open("../artifacts/config.json") as f:
    config = json.load(f)

N_STEPS = config["n_steps"]
TICKER = config["ticker"]

modelo = ModeloLSTM()
modelo.load_state_dict(torch.load("../artifacts/model.pt"))
modelo.eval()

scaler = joblib.load("../artifacts/scaler.pkl")

def prever(closes: list[float]) -> float:
    """Recebe 60 closes brutos, devolve previsão em reais."""
    arr = np.array(closes).reshape(-1,1)
    arr_esc = scaler.transform(arr)
    arr_tensor = torch.from_numpy(arr_esc).float().reshape(1, N_STEPS, 1)
    
    with torch.no_grad():
        pred = modelo(arr_tensor)
    
    pred_numpy = pred.numpy()
    pred_currency = scaler.inverse_transform(pred_numpy)

    return float(pred_currency[0][0])