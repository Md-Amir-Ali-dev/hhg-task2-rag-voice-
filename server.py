from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import os
import json
from dotenv import load_dotenv

load_dotenv(override=True)

from retriever import FastRetriever
from llm import FastLLM
from stt import SarvamSTTClient
import app as app_pipeline


# ── Lifespan (modern FastAPI startup) ─────────────────────────────────────────
retriever: FastRetriever | None = None
llm_real: FastLLM | None = None
llm_mock: FastLLM | None = None
stt_client: SarvamSTTClient | None = None


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    global retriever, llm_real, llm_mock, stt_client
    print("Loading Voice RAG pipeline components...")
    retriever = FastRetriever()
    llm_real = FastLLM(use_mock=False)
    llm_mock = FastLLM(use_mock=True)
    stt_client = SarvamSTTClient()
    print("Voice RAG Pipeline ready!")
    yield


app = FastAPI(
    title="Voice RAG API — Powered by Sarvam AI & Groq",
    version="2.0.0",
    lifespan=lifespan
)

# Allow React dev server, production origin, and any custom origin set via env
_extra_origin = os.getenv("ALLOWED_ORIGIN", "")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        *( [_extra_origin] if _extra_origin else [] ),
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ── Schemas ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    mock: bool = True  # default to mock until user provides API keys


class QueryResponse(BaseModel):
    query: str
    answer: str
    off_topic: bool
    grounded: bool
    confidence_note: str
    chunk_sources: list[dict]
    metrics: dict


# ── Helper ─────────────────────────────────────────────────────────────────────

async def _run_query(query_text: str, use_mock: bool, initial_stt_ms: float = 0.0) -> QueryResponse:
    """Shared pipeline execution for text and voice queries."""
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not ready yet")

    total_start = time.time()

    # 1. Retrieval
    retrieval_result = retriever.retrieve(query_text)
    retrieval_ms = retrieval_result.latency * 1000

    # 2. LLM Generation with Harness
    llm_client = llm_mock if use_mock else llm_real
    generation_result = await llm_client.generate_answer(
        query=query_text,
        context=retrieval_result.context,
        off_topic=retrieval_result.off_topic,
        unsafe=retrieval_result.unsafe
    )

    total_e2e_ms = initial_stt_ms + ((time.time() - total_start) * 1000)

    metrics = {
        "stt_ms": round(initial_stt_ms, 1),
        "retrieval_ms": round(retrieval_ms, 1),
        "llm_ttft_ms": round(generation_result.ttft * 1000, 1),
        "llm_total_ms": round(generation_result.latency * 1000, 1),
        "total_e2e_ms": round(total_e2e_ms, 1),
        "best_sim_score": round(float(retrieval_result.best_score), 3),
        "retries_used": generation_result.retries_used
    }

    return QueryResponse(
        query=query_text,
        answer=generation_result.answer,
        off_topic=retrieval_result.off_topic or retrieval_result.unsafe,
        grounded=generation_result.grounded,
        confidence_note=generation_result.confidence_note,
        chunk_sources=retrieval_result.chunk_sources,
        metrics=metrics,
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "retriever_ready": retriever is not None and retriever.ready,
        "stt_provider": "Sarvam AI (saaras:v3)",
        "llm_provider": "Groq (llama-3.1-8b-instant)",
        "chunking_strategies": ["semantic", "fixed_sliding", "metadata_aware"],
        "latency_target_ms": 200
    }


@app.post("/api/query", response_model=QueryResponse)
async def run_query(req: QueryRequest):
    """Text-based query endpoint."""
    return await _run_query(req.query, req.mock)


@app.post("/api/voice", response_model=QueryResponse)
async def run_voice_query(
    file: UploadFile = File(...),
    mock: bool = True,
):
    """
    Voice query endpoint.
    Transcribes audio via Sarvam AI STT, then executes full RAG pipeline.
    """
    stt_start = time.time()
    audio_bytes = await file.read()

    async def byte_gen():
        chunk_size = 4096
        for i in range(0, len(audio_bytes), chunk_size):
            yield audio_bytes[i: i + chunk_size]

    filename = file.filename or "recording.webm"
    content_type = file.content_type or "audio/webm"
    
    stt_client_inst = SarvamSTTClient(use_mock=mock)
    transcript = await stt_client_inst.transcribe_stream(byte_gen(), filename=filename, content_type=content_type)
    stt_ms = (time.time() - stt_start) * 1000

    if not transcript or "STT error" in transcript:
        raise HTTPException(status_code=422, detail=transcript or "Could not transcribe audio via Sarvam STT.")

    return await _run_query(transcript, mock, initial_stt_ms=stt_ms)


@app.get("/api/benchmark")
async def get_benchmark_report():
    """Returns saved P50/P70/P100 latency analytics report, or runs one."""
    if os.path.exists("latency_report.json"):
        try:
            with open("latency_report.json", "r") as f:
                return json.load(f)
        except Exception:
            pass
    # Generate benchmark report
    report = await app_pipeline.run_benchmark(num_runs=15, use_mock=True)
    return report
