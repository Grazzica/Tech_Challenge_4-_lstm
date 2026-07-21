import os
import time
import logging
import psutil
from fastapi import FastAPI
from api.schemas import PredictRequest, PredictResponse
from api.inference import prever, TICKER

# ---- logging: base do monitoramento ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tc4-api")

app = FastAPI(title="TC4 - Previsão VALE3", version="1.0")

# processo atual (para medir consumo da própria API) e marco de subida
processo = psutil.Process(os.getpid())
inicio_app = time.time()


# ---- middleware: mede tempo de resposta de TODA requisição ----
@app.middleware("http")
async def medir_tempo(request, call_next):
    inicio = time.time()
    response = await call_next(request)
    duracao = time.time() - inicio
    logger.info(f"{request.method} {request.url.path} - {duracao*1000:.1f}ms")
    response.headers["X-Process-Time-ms"] = f"{duracao*1000:.1f}"
    return response


# ---- ROTA 1: health check ----
@app.get("/health")
def health_check():
    return {"status": "ok"}

# ---- ROTA: métricas de recursos (monitoramento de CPU/memória) ----
@app.get("/metrics")
def metrics():
    mem = psutil.virtual_memory()
    return {
        "sistema": {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memoria_total_mb": round(mem.total / 1024 / 1024, 1),
            "memoria_usada_mb": round(mem.used / 1024 / 1024, 1),
            "memoria_percent": mem.percent,
        },
        "processo_api": {
            "cpu_percent": processo.cpu_percent(interval=0.1),
            "memoria_rss_mb": round(processo.memory_info().rss / 1024 / 1024, 1),
        },
        "uptime_segundos": round(time.time() - inicio_app, 1),
    }

# ---- ROTA 2: predição ----
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
   resultado =  prever(request.closes)
   return PredictResponse(prediction = resultado, ticker=TICKER)

# ---- ROTA RAIZ ----
@app.get("/")
def home():
    return {"api": "TC4 - Previsão VALE3", "docs": "/docs", "health": "/health"}