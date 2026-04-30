"""
main.py — FastAPI backend server
Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import baseline_model
import optimized_model
from optimized_model import QuantMode
from metrics import collect_metrics, warmup, model_memory_mb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set DEV_MODE=1 to skip model loading (useful for testing the API without a GPU/checkpoint)
DEV_MODE = os.getenv("DEV_MODE", "0") == "1"

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

class ModelState:
    baseline_tokenizer: Optional[Any]
    baseline_model: Optional[Any]
    opt_tokenizer: Optional[Any]
    opt_model: Optional[Any]
    current_quant_mode: QuantMode
    baseline_memory_mb: float
    opt_memory_mb: float

    def __init__(self):
        self.baseline_tokenizer = None
        self.baseline_model = None
        self.opt_tokenizer = None
        self.opt_model = None
        self.current_quant_mode: QuantMode = "int4"
        self.baseline_memory_mb: float = 0.0
        self.opt_memory_mb: float = 0.0

state = ModelState()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")


# ---------------------------------------------------------------------------
# Lifespan: load baseline at startup (optimized loaded on first request)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    if DEV_MODE:
        logger.warning("DEV_MODE=1 — skipping model loading. /infer/* will return 503.")
    else:
        logger.info("Loading baseline model...")
        try:
            state.baseline_tokenizer, state.baseline_model = baseline_model.load_model(device=DEVICE)
            warmup(baseline_model.generate, state.baseline_tokenizer, state.baseline_model)
            state.baseline_memory_mb = model_memory_mb(state.baseline_model)
            logger.info(f"Baseline model ready ({state.baseline_memory_mb:.0f} MB).")
        except Exception as e:
            logger.warning(f"Baseline model NOT loaded (will return 503): {e}")

        logger.info(f"Loading optimized model (mode={state.current_quant_mode})...")
        try:
            requested_mode = state.current_quant_mode
            state.opt_tokenizer, state.opt_model = optimized_model.load_model(
                quant_mode=requested_mode, device=DEVICE
            )
            warmup(optimized_model.generate, state.opt_tokenizer, state.opt_model)
            state.opt_memory_mb = model_memory_mb(state.opt_model)
            logger.info(f"Optimized model ready ({state.opt_memory_mb:.0f} MB, mode: {state.current_quant_mode}).")
        except Exception as e:
            logger.warning(f"Optimized model NOT loaded (will return 503): {e}")

    yield  # server runs here

    # cleanup
    del state.baseline_model, state.opt_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="HPC Model Comparison API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class InferRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2048)
    max_new_tokens: int = Field(default=128, ge=1, le=512)


class MetricsResponse(BaseModel):
    output_text: str
    latency_ms: float
    throughput_tps: float
    gpu_memory_mb: float
    perplexity: Optional[float]
    gflops_per_tok: Optional[float]
    quant_mode: str          # which mode was used ("fp16" for baseline)


class QuantSwitchRequest(BaseModel):
    quant_mode: Literal["int4_base", "int4"]


class StatusResponse(BaseModel):
    baseline_loaded: bool
    optimized_loaded: bool
    current_quant_mode: str
    device: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/status", response_model=StatusResponse)
def get_status():
    return StatusResponse(
        baseline_loaded=state.baseline_model is not None,
        optimized_loaded=state.opt_model is not None,
        current_quant_mode=state.current_quant_mode,
        device=DEVICE,
    )


@app.post("/infer/baseline", response_model=MetricsResponse)
def infer_baseline(req: InferRequest):
    if state.baseline_model is None:
        raise HTTPException(status_code=503, detail="Baseline model not loaded")
    try:
        result = collect_metrics(
            generate_fn=baseline_model.generate,
            tokenizer=state.baseline_tokenizer,
            model=state.baseline_model,
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
        )
        result["quant_mode"] = "bf16"
        result["gpu_memory_mb"] = state.baseline_memory_mb
        return MetricsResponse(**result)
    except Exception as e:
        logger.exception("Baseline inference failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/infer/optimized", response_model=MetricsResponse)
def infer_optimized(req: InferRequest):
    if state.opt_model is None:
        raise HTTPException(status_code=503, detail="Optimized model not loaded")
    try:
        result = collect_metrics(
            generate_fn=optimized_model.generate,
            tokenizer=state.opt_tokenizer,
            model=state.opt_model,
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
        )
        result["quant_mode"] = state.current_quant_mode
        result["gpu_memory_mb"] = state.opt_memory_mb
        return MetricsResponse(**result)
    except Exception as e:
        logger.exception("Optimized inference failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/switch_quant", response_model=StatusResponse)
def switch_quant(req: QuantSwitchRequest):
    """
    Reload the optimized model with a different quantization mode.
    Called when the user toggles INT4 / INT8 / FP16 in the UI.
    """
    if req.quant_mode == state.current_quant_mode:
        return get_status()  # nothing to do

    logger.info(f"Switching optimized model: {state.current_quant_mode} → {req.quant_mode}")
    try:
        # free old model first
        del state.opt_model
        state.opt_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        state.opt_tokenizer, state.opt_model = optimized_model.load_model(
            quant_mode=req.quant_mode, device=DEVICE
        )
        warmup(optimized_model.generate, state.opt_tokenizer, state.opt_model)
        state.opt_memory_mb = model_memory_mb(state.opt_model)
        state.current_quant_mode = req.quant_mode
        logger.info(f"Switch complete (actual mode: {state.current_quant_mode}).")
        return get_status()
    except Exception as e:
        logger.exception("Model switch failed")
        raise HTTPException(status_code=500, detail=str(e))
