# app/schemas.py
# Owner: Aadhishree
# DO NOT change field names without a PR reviewed by all three

from pydantic import BaseModel
from typing import List, Optional


class RecoveryTask(BaseModel):
    id: str
    name: str
    start_time: str            # "HH:MM" 24-hour
    duration_minutes: int
    needs_caregiver: bool
    needs_transport: bool
    source_text: str           # original OCR text — NEVER invent this
    confirmed: bool = False    # MUST be False until human confirms


class PatientConstraints(BaseModel):
    caregiver_start: str       # "HH:MM"
    caregiver_end: str         # "HH:MM"
    can_travel_alone: bool
    mobility: str              # "independent" | "walker_assisted" | "wheelchair"


class FeasibilityRequest(BaseModel):
    tasks: List[RecoveryTask]
    constraints: PatientConstraints


class ConflictDetail(BaseModel):
    task_id: str
    task_name: str
    reason: str
    action: str                # e.g. "ROUTE_TO_CARECONNECT"


class FeasibilityResponse(BaseModel):
    status: str                # "FEASIBLE" | "INFEASIBLE" | "UNKNOWN"
    conflicts: List[ConflictDetail]
    preserved_tasks: List[RecoveryTask]   # tasks are NEVER deleted


class ExtractionRequest(BaseModel):
    document_id: str
    text: str


class ExtractionResponse(BaseModel):
    source_document_id: str
    tasks: List[RecoveryTask]
    human_confirmation_required: bool = True  # ALWAYS True


class SymptomCheckin(BaseModel):
    day: int
    pain: float                # 0.0–10.0
    temperature: float         # Celsius
    sleep_hours: float
    mobility_score: float      # 0.0–10.0
    vomiting_flag: int         # 0 or 1
    headache_flag: int         # 0 or 1


class TrendRequest(BaseModel):
    patient_id: str
    check_ins: List[SymptomCheckin]


class TrendResponse(BaseModel):
    patient_id: str
    anomaly_detected: bool
    changed_features: List[str]
    severity: str              # "NORMAL" | "REVIEW_NEEDED" | "URGENT_ESCALATE"


class ChatRequest(BaseModel):
    confirmed_discharge_text: str
    alerts: Optional[dict] = {}
    question: str


class ChatResponse(BaseModel):
    answer: str
    route_to_careconnect: bool
    disclaimer: str = "This is a prototype recovery coordination tool, not a clinical device."


class PhysioRequest(BaseModel):
    knee_angle_degrees: float


class PhysioResponse(BaseModel):
    score: float               # 0–100
    label: str                 # "LOW" | "OKAY" | "GOOD" | "PERFECT"
    note: str = "Prototype movement-quality signal only. Not a medical diagnosis."


class MedicationEntry(BaseModel):
    name: str
    system: str                # "allopathy" | "ayurveda" | "homeopathy" | "yoga"
    time: str                  # "HH:MM"
    duration_minutes: int


class IntegrativeRequest(BaseModel):
    allopathy: List[MedicationEntry]
    ayush: List[MedicationEntry]


class OverlapDetail(BaseModel):
    item1: str
    item2: str
    overlap_type: str          # "time_overlap" | "proximity_flag"
    recommendation: str


class IntegrativeResponse(BaseModel):
    overlaps: List[OverlapDetail]
    review_flag: bool
    note: str = "AYUSH coordination only. Not clinical interchangeability."


class VoiceRequest(BaseModel):
    text: str
    tone: str = "gentle"       # "gentle" | "clear" | "calm"


class VoiceResponse(BaseModel):
    tts_text: str
    voice_params: dict