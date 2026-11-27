# app/main.py
# Owner: Aadhishree

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import (
    FeasibilityRequest, FeasibilityResponse,
    ExtractionRequest, ExtractionResponse,
    TrendRequest, TrendResponse,
    ChatRequest, ChatResponse,
    PhysioRequest, PhysioResponse,
    IntegrativeRequest, IntegrativeResponse,
    VoiceRequest, VoiceResponse,
)

app = FastAPI(
    title="PUNARVAVA Backend API",
    description="AI-Assisted Post-Hospitalisation Recovery — BioForge 2026",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok", "app": "punarvava-backend", "version": "1.0.0"}


# ── Aadhishree's endpoints ────────────────────────────────────────────────────
@app.post("/check-feasibility", response_model=FeasibilityResponse)
def check_feasibility(req: FeasibilityRequest):
    from app.scheduler import evaluate_feasibility
    return evaluate_feasibility(req)


@app.post("/analyze-trends", response_model=TrendResponse)
def analyze_trends(req: TrendRequest):
    from app.anomaly_model import analyze_symptom_trends
    return analyze_symptom_trends(req)


# ── Adit's endpoints ──────────────────────────────────────────────────────────
@app.post("/extract-discharge", response_model=ExtractionResponse)
def extract_discharge(req: ExtractionRequest):
    from app.vault_rag import extract_tasks_from_text
    return extract_tasks_from_text(req)


@app.post("/chat", response_model=ChatResponse)
def chat_companion(req: ChatRequest):
    from app.vault_rag import chat_with_companion
    return chat_with_companion(req)


@app.post("/maatri/voice", response_model=VoiceResponse)
def maatri_voice(req: VoiceRequest):
    from app.voice_maatri import generate_voice_payload
    return generate_voice_payload(req)


# ── Gaurika's endpoints ───────────────────────────────────────────────────────
@app.post("/evaluate-physio", response_model=PhysioResponse)
def evaluate_physio(req: PhysioRequest):
    from app.physio_eval import score_knee_angle
    return score_knee_angle(req)


@app.post("/check-integrative", response_model=IntegrativeResponse)
def check_integrative(req: IntegrativeRequest):
    from app.ayush_rules import check_schedule_overlaps
    return check_schedule_overlaps(req)