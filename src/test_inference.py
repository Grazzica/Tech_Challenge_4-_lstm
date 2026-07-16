from api.inference import prever
from data import baixar_dados

# pega dados reais da VALE3 e usa os últimos 60 closes do período usado no teste para validar operação do inference.py
df = baixar_dados("VALE3.SA")
ultimos_60 = df["Close"].tail(60).tolist()

print("Últimos 60 closes (amostra):", ultimos_60[:3], "...")
print("Quantidade:", len(ultimos_60))

resultado = prever(ultimos_60)
print(f"\nPrevisão para o próximo pregão: R$ {resultado:.2f}")