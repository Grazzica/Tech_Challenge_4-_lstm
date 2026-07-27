# Tech Challenge Fase 4 — Previsão de Preços com LSTM

Pipeline completo de deep learning para previsão do preço de fechamento da ação **VALE3** (B3), desde a coleta de dados até o deploy do modelo em uma API RESTful containerizada.

O projeto implementa uma rede neural **LSTM (Long Short-Term Memory)** para capturar padrões temporais na série histórica de preços, servida via **FastAPI** e empacotada com **Docker**.

---

## Arquitetura

```mermaid
flowchart TB
    subgraph Treino[Pipeline de Treino - offline]
        YF[yfinance<br/>VALE3 2018-2024]
        DATA[data.py<br/>coleta + preproc]
        TRAIN[train.py<br/>treino LSTM + avaliacao]
        YF --> DATA --> TRAIN
    end

    subgraph Artefatos[Artefatos - contrato]
        MODEL[(model.pt<br/>pesos treinados)]
        SCALER[(scaler.pkl<br/>MinMax do treino)]
        CONFIG[(config.json<br/>n_steps, feature)]
    end

    subgraph API[FastAPI - servico online]
        MAIN[main.py<br/>rotas + middleware]
        INFER[inference.py<br/>carga + predicao]
        SCHEMA[schemas.py<br/>validacao Pydantic]
        MAIN --> INFER
        MAIN --> SCHEMA
    end

    subgraph Cliente
        USER[Usuario / HTTP Client]
    end

    subgraph Container[Docker]
        DOCKERFILE[Dockerfile<br/>python:3.11-slim]
    end

    TRAIN --> MODEL
    TRAIN --> SCALER
    TRAIN --> CONFIG

    MODEL --> INFER
    SCALER --> INFER
    CONFIG --> INFER

    USER --> MAIN
    API -.empacotado em.-> DOCKERFILE
```

---

## Fluxo de predição

```mermaid
sequenceDiagram
    participant U as Usuario
    participant API as FastAPI (/predict)
    participant S as Scaler
    participant M as Modelo LSTM

    U->>API: POST /predict {closes: [60 valores]}
    API->>API: valida entrada (60 closes)
    API->>S: transform (escala para [0,1])
    S-->>API: sequencia escalada
    API->>M: predict (tensor 1,60,1)
    M-->>API: previsao escalada
    API->>S: inverse_transform (volta para R$)
    S-->>API: previsao em reais
    API-->>U: {prediction: 59.55, ticker: VALE3.SA}
```

---

## Estrutura do projeto

```
tc4-vale3/
├── src/
│   ├── data.py            # coleta (yfinance) e pre-processamento
│   ├── model.py           # definicao da classe ModeloLSTM
│   ├── train.py           # treino, avaliacao e salvamento de artefatos
│   ├── demo_teste.py      # testa a API em producao contra dados reais (2025/2026)
│   └── api/
│       ├── __init__.py
│       ├── main.py        # app FastAPI, rotas e middleware de monitoramento
│       ├── inference.py   # carga dos artefatos e logica de predicao
│       └── schemas.py     # modelos Pydantic (entrada/saida)
├── artifacts/
│   ├── model.pt           # pesos do modelo treinado
│   ├── scaler.pkl         # scaler MinMax ajustado no treino
│   └── config.json        # metadados (n_steps, feature, ticker)
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

---

## Responsabilidade dos módulos

| Módulo | Responsabilidade |
|--------|------------------|
| `data.py` | Baixa o histórico da VALE3 via yfinance; divide treino/teste cronologicamente; ajusta o scaler apenas no treino; cria as janelas deslizantes para a LSTM |
| `model.py` | Define a classe `ModeloLSTM` (camada LSTM + camada linear). Compartilhada entre treino e API |
| `train.py` | Executa o pipeline de treino, avalia com MAE/RMSE/MAPE, gera o gráfico real vs. previsto e salva os três artefatos |
| `api/schemas.py` | Define e valida o contrato de entrada (60 closes) e saída (previsão + ticker) via Pydantic |
| `api/inference.py` | Carrega os artefatos uma única vez no startup e expõe a função `prever()` que executa todo o fluxo de inferência |
| `api/main.py` | Expõe as rotas HTTP e mede o tempo de resposta de cada requisição (monitoramento) |

---

## Decisões técnicas

| Decisão | Escolha | Justificativa |
|---------|---------|---------------|
| Feature | Univariado (`Close`) | Cumpre o objetivo (prever fechamento) com o menor risco de overfitting |
| Janela (`n_steps`) | 60 pregões | ~3 meses de contexto — suficiente sem sequência longa demais |
| Arquitetura | 1 camada LSTM (50 unidades) | Empilhar não melhora em sinal univariado e aumenta overfitting |
| Normalização | MinMax `[0,1]`, ajustado só no treino | Mantém as ativações na faixa útil e evita vazamento temporal |
| Divisão treino/teste | Cronológica (80/20) | Série temporal não pode ser embaralhada — evita usar o futuro para prever o passado |
| Função de perda | MSE | Pune erros grandes, adequado a previsão de preço |
| Otimizador | Adam (`lr=0.001`) | Padrão robusto para convergência estável |

---

## Resultados

Métricas no conjunto de teste (des-normalizadas, em reais):

| Métrica | Valor |
|---------|-------|
| MAE | R$ 1,72 |
| RMSE | R$ 2,20 |
| MAPE | 2,70% |

![Real vs Previsto](artifacts/previsao_vs_real.png)

**Leitura:** o modelo captura bem a tendência de médio prazo, mas suaviza a volatilidade diária de curto prazo. Um MAPE de ~2,7% indica erro médio inferior a 3% do valor do ativo — resultado sólido para previsão univariada. Observa-se um leve atraso na resposta do modelo, característico de previsão baseada em janela histórica de um único sinal (o próprio preço), o que impõe cautela na interpretação do poder preditivo para horizontes de negociação.

### Validação com dados reais (fora do treino)

O script `src/demo_teste.py` consome a **API em produção** e compara a previsão com o fechamento real de pregões que o modelo nunca viu (o treino usou dados até 2024):

| Data-alvo | Previsto | Real | Erro |
|-----------|----------|------|------|
| 2025-06-16 | R$ 49,27 | R$ 49,42 | 0,29% |
| 2026-06-16 | R$ 73,71 | R$ 81,44 | 9,50% |

O caso de 2025 (preços dentro da faixa histórica de treino) apresenta erro mínimo. Já o de 2026, quando a VALE3 ultrapassou o máximo histórico do período de treino (~R$ 73), apresenta erro maior — o modelo **extrapola mal fora da faixa em que foi treinado**, tendendo a subestimar valores acima do teto conhecido. Este comportamento é esperado e ilustra um princípio central de ML em produção: modelos são confiáveis dentro da distribuição de dados vista no treino, exigindo retreino periódico com dados recentes.

---

## Como executar

A API já está hospedada e disponível para consumo (ver seção [Deploy](#deploy)) — **não é necessário rodar nada localmente para testá-la**. As opções abaixo são para quem deseja reproduzir o projeto na própria máquina.

### Pré-requisitos
- Python 3.11+
- Docker (para execução containerizada)

### Opção A — Local (desenvolvimento)

```bash
# ambiente virtual
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# treinar o modelo (gera os artefatos)
cd src
python3 train.py

# subir a API
uvicorn api.main:app --reload
```

### Opção B — Docker (recomendado)

```bash
# construir a imagem
docker build -t tc4-vale3 .

# rodar o container
docker run -p 8000:8000 tc4-vale3
```

A API fica disponível em `http://localhost:8000`.

---

## Uso da API

**API em produção:** `http://50.17.38.186:8000`

Documentação interativa (Swagger) gerada automaticamente: **`http://50.17.38.186:8000/docs`**

> Para uso local, substitua o endereço por `http://localhost:8000`.

### Rotas

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/` | Informações da API |
| `GET` | `/health` | Health check (usado por orquestradores) |
| `GET` | `/metrics` | Utilização de recursos: CPU, memória e uptime |
| `POST` | `/predict` | Recebe 60 closes brutos e retorna a previsão do próximo pregão |

### Exemplo de requisição

```bash
curl -X POST "http://50.17.38.186:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"closes": [58.2, 58.5, 57.9, ... 60 valores ...]}'
```

### Resposta

```json
{
  "prediction": 59.55,
  "ticker": "VALE3.SA"
}
```

O tempo de processamento de cada requisição é registrado em log e devolvido no header `X-Process-Time-ms`.

### Validação automatizada

Para validar a API em produção contra dados reais não vistos no treino:

```bash
cd src
python3 demo_teste.py
```

O script busca os 60 pregões anteriores a cada data-alvo, envia à API e compara a previsão com o fechamento real do dia.

---

## Monitoramento

O monitoramento cobre os dois eixos exigidos — tempo de resposta e utilização de recursos:

- **Latência:** um middleware mede o tempo de processamento de todas as requisições, registrando no log do servidor e expondo o valor no header `X-Process-Time-ms` de cada resposta.
- **Recursos:** o endpoint `GET /metrics` reporta o consumo de CPU e memória (do sistema e do processo da API) e o uptime da aplicação, via `psutil`.
- **Disponibilidade:** o endpoint `GET /health` permite verificação de disponibilidade por ferramentas externas e orquestradores.

---

## Stack

- **Modelagem:** PyTorch (LSTM)
- **Dados:** yfinance, pandas, NumPy, scikit-learn (MinMaxScaler)
- **API:** FastAPI, Pydantic, Uvicorn
- **Monitoramento:** psutil (CPU/memória), middleware de latência
- **Serialização:** joblib (scaler), state_dict (modelo)
- **Containerização:** Docker

---

## Limitações e próximos passos

- **Univariado:** o modelo usa apenas o preço de fechamento. Incluir volume, indicadores técnicos ou variáveis macroeconômicas pode reduzir o atraso da previsão.
- **Faixa de treino:** a confiabilidade cai para preços que extrapolem a faixa histórica de 2018–2024 (o scaler foi ajustado nesse período).
- **Um ticker:** o modelo e o scaler são específicos da VALE3. Outros ativos exigem novo treino.
- **Evolução possível:** prever retorno (variação percentual) em vez de preço absoluto, para forçar o modelo a aprender além da tendência trivial.

---

## Deploy

- API disponível em: http://50.17.38.186:8000/
- Documentação interativa: http://50.17.38.186:8000/docs
- Vídeo de apresentação disponível em: https://youtu.be/nCs6wzhf9Aw

---

## Autores

**Adriano Cabrera**

- LinkedIn: https://www.linkedin.com/in/adriano-cabrera-b7b680a7/
- GitHub: https://github.com/cabrpin

**Caio Grazzini**

- LinkedIn: https://www.linkedin.com/in/caiograzzini/
- GitHub: https://github.com/Grazzica/

**Fabrício Batista Dias**

- LinkedIn: https://www.linkedin.com/in/fabriciobdias/
- GitHub: https://github.com/DiasFabricio

---

**Desenvolvido como parte do Tech Challenge — Pós-Graduação em Machine Learning Engineering — FIAP 2025/2026**
