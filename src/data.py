"""Coleta e pré-processamento dos dados da VALE3."""
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def baixar_dados(ticker="VALE3.SA", start="2018-01-01", end="2024-12-31"):
    """Baixa histórico e retorna DataFrame só com a coluna Close."""
    df = yf.download(ticker, start=start, end=end, progress=False)
    # yfinance às vezes retorna colunas multi-nível; achatamos
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].dropna()  # univariado: só o fechamento
    return df


def dividir_treino_teste(df, frac_treino=0.8):
    """Divide CRONOLOGICAMENTE — série temporal não pode ser embaralhada."""
    n = int(len(df) * frac_treino)
    return df.iloc[:n], df.iloc[n:]


def ajustar_scaler(treino):
    """Fit do MinMax SÓ no treino. O scaler nunca pode ver o teste."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(treino[["Close"]])
    return scaler


def criar_janelas(serie_escalada, n_steps=60):
    """
    Transforma a série 1D em pares (X, y) para a LSTM.
    X: janela de n_steps preços | y: o preço do dia seguinte.
    """
    X, y = [], []
    for i in range(n_steps, len(serie_escalada)):
        X.append(serie_escalada[i - n_steps:i, 0])  # 60 dias anteriores
        y.append(serie_escalada[i, 0])              # dia atual (alvo)
    X = np.array(X)
    y = np.array(y)
    # LSTM exige formato 3D: (amostras, n_steps, n_features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y