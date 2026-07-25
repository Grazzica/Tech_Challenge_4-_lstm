"""
Demonstração da API em produção contra dados reais não vistos no treino.
Para cada data-alvo, busca os 60 pregões anteriores, envia à API,
e compara a previsão com o fechamento real do dia.
"""
import requests
import yfinance as yf
import pandas as pd

API_URL = "http://50.17.38.186:8000/predict"
TICKER = "VALE3.SA"
N_STEPS = 60

# datas-alvo que queremos prever (dias reais, fora do treino)
DATAS_ALVO = ["2025-06-16", "2026-06-16"]


def baixar_janela(data_alvo):
    """Busca dados suficientes para ter 60 pregoes ANTES da data-alvo + o dia."""
    fim = pd.to_datetime(data_alvo)
    # margem generosa de dias corridos para garantir 60 pregoes (feriados/fins de semana)
    inicio = fim - pd.Timedelta(days=130)
    df = yf.download(TICKER, start=inicio, end=fim + pd.Timedelta(days=1), progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Close"]].dropna()


def testar_data(data_alvo):
    df = baixar_janela(data_alvo)

    # a data-alvo precisa existir no historico (pode cair em fim de semana/feriado)
    if data_alvo not in df.index.strftime("%Y-%m-%d").tolist():
        print(f"\n[{data_alvo}] nao e pregao (fim de semana/feriado). Pulando.")
        return

    # posicao da data-alvo na serie
    idx = df.index.strftime("%Y-%m-%d").tolist().index(data_alvo)

    # precisa de 60 pregoes ANTES dela
    if idx < N_STEPS:
        print(f"\n[{data_alvo}] pregoes insuficientes antes da data. Pulando.")
        return

    # os 60 closes anteriores (entrada) e o real do dia (alvo)
    closes_entrada = df["Close"].iloc[idx - N_STEPS:idx].tolist()
    real = df["Close"].iloc[idx]

    # chama a API
    resposta = requests.post(API_URL, json={"closes": closes_entrada})
    resposta.raise_for_status()
    previsto = resposta.json()["prediction"]

    # resultado
    erro = abs(previsto - real)
    erro_pct = erro / real * 100
    print(f"\n=== Previsao para {data_alvo} ===")
    print(f"Ultimos closes usados (3 de 60): ...{[round(c,2) for c in closes_entrada[-3:]]}")
    print(f"Previsto: R$ {previsto:.2f}")
    print(f"Real:     R$ {real:.2f}")
    print(f"Erro:     R$ {erro:.2f}  ({erro_pct:.2f}%)")


if __name__ == "__main__":
    print(f"Testando API em producao: {API_URL}")
    for data in DATAS_ALVO:
        testar_data(data)