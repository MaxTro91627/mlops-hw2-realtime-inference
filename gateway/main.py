"""Real-time inference gateway: CV (Triton) + LLM (vLLM) за JWT, с метриками.

Ручки — по API_CONTRACT.md:
    GET  /health, /metrics
    GET  /triton/health, /vllm/health
    GET  /metrics/triton, /metrics/vllm, /metrics/node
    GET  /info    (Bearer)
    POST /vision  (Bearer, multipart `file`)
    POST /chat    (Bearer, JSON → SSE)

Принципы:
  * ничего блокирующего в event loop: препроцессинг картинки идёт в пуле потоков,
    клиент Triton асинхронный (aio), vLLM — через httpx.AsyncClient;
  * ошибки бэкендов — честными HTTP-кодами (для /chat статус vLLM проверяется ДО
    начала стрима);
  * метрики бэкендов отдаются сырыми, без изменений.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import jwt
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image, UnidentifiedImageError
from prometheus_fastapi_instrumentator import Instrumentator
from tritonclient.http import InferInput, InferRequestedOutput
from tritonclient.http.aio import InferenceServerClient

log = logging.getLogger("gateway")

# --------------------------------------------------------------------------- конфиг

JWT_SECRET = os.environ.get("JWT_SECRET", "")
CV_MODEL = os.environ.get("CV_MODEL", "resnet50")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen2.5-1.5B-Instruct")

TRITON_URL = os.environ.get("TRITON_URL", "triton:8000")  # host:port (http)
TRITON_METRICS_URL = os.environ.get("TRITON_METRICS_URL", "http://triton:8002/metrics")
VLLM_URL = os.environ.get("VLLM_URL", "http://vllm:8000").rstrip("/")
VLLM_METRICS_URL = os.environ.get("VLLM_METRICS_URL", f"{VLLM_URL}/metrics")
NODE_METRICS_URL = os.environ.get("NODE_METRICS_URL", "http://node-exporter:9100/metrics")

GPU_NAME = os.environ.get("GPU_NAME", "unknown")
STUDENT_NAME = os.environ.get("STUDENT_NAME", "")
STUDENT_EMAIL = os.environ.get("STUDENT_EMAIL", "")

CLASSES = [
    line.rstrip("\n")
    for line in (Path(__file__).parent / "imagenet_classes.txt").read_text(encoding="utf-8").splitlines()
]

# ------------------------------------------------------------------ препроцессинг
# Параметры соответствуют ResNet50_Weights.IMAGENET1K_V1.transforms():
# resize(256, bilinear) по короткой стороне → center crop 224 → /255 → normalize.
# У других вариантов (efficientnet_b0: 256/bicubic/224, mobilenet_v3_large: 256/bilinear/224,
# convnext_tiny: 236/bicubic/224) параметры другие — здесь они зафиксированы под resnet50.
RESIZE_SIZE = 256
CROP_SIZE = 224
RESAMPLE = Image.BILINEAR
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

# «Огромные» картинки не должны ронять Pillow защитой от decompression bomb.
Image.MAX_IMAGE_PIXELS = None


def preprocess(data: bytes) -> np.ndarray:
    """bytes → float32 [3,224,224]. Бросает ValueError, если это не картинка."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError(f"not an image: {exc}") from exc

    # Прозрачность (RGBA/LA/P) → белый фон не нужен: эталон делает convert("RGB").
    img = img.convert("RGB")

    w, h = img.size
    if w <= h:
        new_w, new_h = RESIZE_SIZE, max(1, int(RESIZE_SIZE * h / w))
    else:
        new_w, new_h = max(1, int(RESIZE_SIZE * w / h)), RESIZE_SIZE
    img = img.resize((new_w, new_h), RESAMPLE)

    left = int(round((new_w - CROP_SIZE) / 2.0))
    top = int(round((new_h - CROP_SIZE) / 2.0))
    img = img.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))

    arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0  # CHW
    return np.ascontiguousarray((arr - MEAN) / STD, dtype=np.float32)


def softmax_top5(logits: np.ndarray) -> list[dict[str, Any]]:
    z = logits.astype(np.float64)
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    idx = np.argsort(-p)[:5]
    return [
        {"class_id": int(i), "label": CLASSES[int(i)], "score": float(p[i])}
        for i in idx
    ]


# ----------------------------------------------------------------------- lifespan


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    if not JWT_SECRET:
        log.warning("JWT_SECRET не задан — все защищённые ручки будут отвечать 401")
    app.state.triton = InferenceServerClient(url=TRITON_URL, conn_limit=100, conn_timeout=10.0)
    app.state.http = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=30.0),
        limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
    )
    try:
        yield
    finally:
        await app.state.triton.close()
        await app.state.http.aclose()


app = FastAPI(title="Real-time Inference Gateway", lifespan=lifespan)

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

# --------------------------------------------------------------------------- auth

_bearer = HTTPBearer(auto_error=False)


def require_auth(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> dict:
    """JWT HS256. Чужая подпись, alg=none, мусор, просроченный — всё 401."""
    if creds is None or not JWT_SECRET:
        raise HTTPException(status_code=401, detail="missing or invalid token")
    try:
        # algorithms=["HS256"] жёстко запрещает alg=none и подмену алгоритма.
        return jwt.decode(creds.credentials, JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="missing or invalid token")


# ------------------------------------------------------------------------- health


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/triton/health")
async def triton_health(request: Request) -> JSONResponse:
    try:
        ready = await request.app.state.triton.is_server_ready()
        model_ready = ready and await request.app.state.triton.is_model_ready(CV_MODEL)
    except Exception:  # noqa: BLE001 — любой сбой = бэкенд недоступен
        ready = model_ready = False
    ok = bool(ready and model_ready)
    return JSONResponse(
        {"backend": "triton", "ready": ok, "model": CV_MODEL},
        status_code=200 if ok else 503,
    )


@app.get("/vllm/health")
async def vllm_health(request: Request) -> JSONResponse:
    try:
        r = await request.app.state.http.get(f"{VLLM_URL}/v1/models", timeout=5.0)
        r.raise_for_status()
        models = [m["id"] for m in r.json().get("data", [])]
        ok = bool(models)
    except Exception:  # noqa: BLE001
        models, ok = [], False
    return JSONResponse(
        {"backend": "vllm", "ready": ok, "models": models},
        status_code=200 if ok else 503,
    )


# ------------------------------------------------------------- проброс метрик


async def _proxy_metrics(request: Request, url: str) -> Response:
    try:
        r = await request.app.state.http.get(url, timeout=10.0)
        r.raise_for_status()
    except Exception:  # noqa: BLE001
        raise HTTPException(status_code=502, detail="backend metrics unavailable")
    return Response(content=r.content, media_type="text/plain")


@app.get("/metrics/triton")
async def metrics_triton(request: Request) -> Response:
    return await _proxy_metrics(request, TRITON_METRICS_URL)


@app.get("/metrics/vllm")
async def metrics_vllm(request: Request) -> Response:
    return await _proxy_metrics(request, VLLM_METRICS_URL)


@app.get("/metrics/node")
async def metrics_node(request: Request) -> Response:
    return await _proxy_metrics(request, NODE_METRICS_URL)


# ---------------------------------------------------------------------------- info


@app.get("/info")
def info(_: dict = Depends(require_auth)) -> dict:
    return {
        "student": {"name": STUDENT_NAME, "email": STUDENT_EMAIL},
        "models": {"vision": CV_MODEL, "llm": LLM_MODEL},
        "gpu": GPU_NAME,
        "version": "1.0",
    }


# -------------------------------------------------------------------------- vision


@app.post("/vision")
async def vision(
    request: Request,
    file: UploadFile = File(...),
    _: dict = Depends(require_auth),
) -> dict:
    t0 = time.perf_counter()

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")

    # Препроцессинг — CPU-работа: в пул потоков, чтобы не блокировать event loop.
    try:
        tensor = await asyncio.to_thread(preprocess, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    inp = InferInput("INPUT__0", [1, 3, CROP_SIZE, CROP_SIZE], "FP32")
    inp.set_data_from_numpy(tensor[None, ...], binary_data=True)
    out = InferRequestedOutput("OUTPUT__0", binary_data=True)

    try:
        result = await request.app.state.triton.infer(CV_MODEL, [inp], outputs=[out])
        logits = result.as_numpy("OUTPUT__0")[0]
    except Exception as exc:  # noqa: BLE001
        log.warning("triton infer failed: %s", exc)
        raise HTTPException(status_code=502, detail="triton unavailable")

    return {
        "model": CV_MODEL,
        "predictions": softmax_top5(logits),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


# ---------------------------------------------------------------------------- chat

_PASS_THROUGH = (
    "max_tokens", "temperature", "top_p", "top_k", "stop", "seed", "n",
    "presence_penalty", "frequency_penalty", "stop_token_ids",
)
_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _error_from_vllm(status: int, body: bytes) -> HTTPException:
    """4xx vLLM (напр. промпт длиннее контекста) → 400; 5xx → 502."""
    detail: Any
    try:
        detail = json.loads(body)
    except ValueError:
        detail = body.decode("utf-8", "replace")[:500]
    code = 400 if 400 <= status < 500 else 502
    return HTTPException(status_code=code, detail=detail)


async def _sse_relay(resp: httpx.Response) -> AsyncIterator[str]:
    """Прокси SSE vLLM → клиент; гарантирует финальный `data: [DONE]`."""
    done = False
    try:
        async for line in resp.aiter_lines():
            if line.startswith("data:") and line[5:].strip() == "[DONE]":
                done = True
                yield "data: [DONE]\n\n"
                break
            if line.startswith("data:"):
                yield line + "\n\n"
            # пустые строки и комментарии (`:`) vLLM опускаем: разделитель добавляем сами
    except (httpx.HTTPError, asyncio.CancelledError) as exc:
        if isinstance(exc, asyncio.CancelledError):
            raise
        # Поток оборвался уже после первого байта — код сменить нельзя.
        err = {"error": {"message": "upstream stream interrupted", "type": "bad_gateway"}}
        yield f"data: {json.dumps(err)}\n\n"
    finally:
        await resp.aclose()
    if not done:
        yield "data: [DONE]\n\n"


@app.post("/chat")
async def chat(request: Request, _: dict = Depends(require_auth)):
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object expected")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="`messages` must be a non-empty list")

    stream = bool(body.get("stream", False))
    payload: dict[str, Any] = {"model": LLM_MODEL, "messages": messages, "stream": stream}
    for key in _PASS_THROUGH:
        if key in body:
            payload[key] = body[key]

    client: httpx.AsyncClient = request.app.state.http
    url = f"{VLLM_URL}/v1/chat/completions"

    if not stream:
        try:
            r = await client.post(url, json=payload)
        except httpx.HTTPError:
            raise HTTPException(status_code=502, detail="vllm unavailable")
        if r.status_code != 200:
            raise _error_from_vllm(r.status_code, r.content)
        return Response(content=r.content, media_type="application/json")

    # Стрим: сначала открываем запрос и проверяем статус, только потом отдаём байты.
    try:
        req = client.build_request("POST", url, json=payload)
        resp = await client.send(req, stream=True)
    except httpx.HTTPError:
        raise HTTPException(status_code=502, detail="vllm unavailable")

    if resp.status_code != 200:
        body_bytes = await resp.aread()
        await resp.aclose()
        raise _error_from_vllm(resp.status_code, body_bytes)

    return StreamingResponse(_sse_relay(resp), media_type="text/event-stream", headers=_SSE_HEADERS)
