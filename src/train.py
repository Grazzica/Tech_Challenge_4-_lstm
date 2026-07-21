import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
import joblib
from data import baixar_dados, dividir_treino_teste, ajustar_scaler, criar_janelas
from model import ModeloLSTM

    
# baixar e dividir dados
df = baixar_dados()
treino_df, teste_df = dividir_treino_teste(df, frac_treino=0.8)

# scaler
scaler = ajustar_scaler(treino_df)
treino_esc = scaler.transform(treino_df[["Close"]])
teste_esc = scaler.transform(teste_df[["Close"]])

# Criação das Janelas

X_train, y_train = criar_janelas(treino_esc, n_steps = 60)
X_test, y_test = criar_janelas(teste_esc, n_steps = 60)

X_train_t = torch.from_numpy(X_train).float()
y_train_t = torch.from_numpy(y_train).float().unsqueeze(1)

X_test_t = torch.from_numpy(X_test).float()
y_test_t = torch.from_numpy(y_test).float().unsqueeze(1)

modelo = ModeloLSTM()
criterio = nn.MSELoss() #função perda
otimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)

n_epochs = 100

# Treinamento do modelo

modelo.train()
for epoch in range(n_epochs):
    otimizador.zero_grad()
    previsao = modelo(X_train_t)
    perda = criterio(previsao, y_train_t)
    perda.backward()
    otimizador.step()
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} / loss: {perda.item()}")

# Avaliação do modelo
modelo.eval()
with torch.no_grad():
    pred_esc = modelo(X_test_t)         

pred_esc = pred_esc.numpy()                  # tensor --> Array NumPy
y_test_2d = y_test.reshape(-1,1)

pred = scaler.inverse_transform(pred_esc)
real = scaler.inverse_transform(y_test_2d)

MAE = np.mean(np.abs(pred - real))
MSE = np.mean((pred - real) ** 2)
RMSE = np.sqrt(MSE)
MAPE = np.mean(np.abs((pred - real) / real)) * 100

print(f"MAE: {MAE}, RMSE: {RMSE}, MAPE: {MAPE}")

plt.figure(figsize=(14, 6))
plt.plot(real, label="Real", color="black", linewidth=1.5)
plt.plot(pred, label="Previsto", color="red", linewidth=1, alpha=0.8)
plt.title("VALE3 — Preço de fechamento: Real vs. Previsto (conjunto de teste)")
plt.xlabel("Dia (pregão no período de teste)")
plt.ylabel("Preço (R$)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.text(-0.5, 48, f"MAE: {MAE}\nMSE: {RMSE}\nMAPE: {MAPE}" )
plt.savefig("../artifacts/previsao_vs_real.png", dpi=120)  # salva para o vídeo/README
plt.show()  # abre a janela interativa

torch.save(modelo.state_dict(), "../artifacts/model.pt")
joblib.dump(scaler, "../artifacts/scaler.pkl")

config = {
    "ticker": "VALE3.SA",
    "feature": "Close",
    "n_steps": 60,
    "n_features": 1,
    "horizon": 1,
}
with open("../artifacts/config.json", "w") as f:
    json.dump(config, f, indent=2)

print("Artefatos salvos em ../artifacts/")