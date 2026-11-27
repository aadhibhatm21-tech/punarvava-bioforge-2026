import os
import re
from typing import List, Optional
from app.schemas import (ExtractionRequest, ExtractionResponse, RecoveryTask,
                         ChatRequest, ChatResponse)

# ---------- OCR (image -> raw text) ----------
def ocr_image_to_text(image_path: str) -> str:
    import pytesseract                       # imported here so the app still runs without it
    from PIL import Image
    pytesseract.pytesseract.tesseract_cmd = os.getenv(
        "TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    return pytesseract.image_to_string(Image.open(image_path))

# ---------- Extraction (raw text -> candidate tasks) ----------
DEFAULT_DURATION_MIN = 15   # ASSUMPTION, not from the document. See section 8.

# "08:00 AM - Morning medicine"  ->  groups: hour, minute, AM/PM, task text
LINE_RE = re.compile(r'^\s*(\d{1,2})[:.](\d{2})\s*(AM|PM)\s*[-–—]\s*(.+?)\s*$', re.IGNORECASE)
CAREGIVER_RE = re.compile(r'\bcaregiver present\b|\bassisted\b|\bwith (?:a )?caregiver\b', re.IGNORECASE)
NO_CAREGIVER_RE = re.compile(r'\b(?:no|without)\s+caregiver\b', re.IGNORECASE)
TRANSPORT_RE = re.compile(r'\b(?:clinic|hospital|transport|appointment|follow-up|visit)\b', re.IGNORECASE)
DURATION_RE = re.compile(r'(\d+)\s*min', re.IGNORECASE)

def _to_24h(hour: int, minute: int, ampm: str) -> Optional[str]:
    if not (1 <= hour <= 12 and 0 <= minute <= 59):
        return None                           # OCR garbage like "27:99 PM" -> reject, don't guess
    ampm = ampm.upper()
    if ampm == "PM" and hour != 12: hour += 12
    if ampm == "AM" and hour == 12: hour = 0
    return f"{hour:02d}:{minute:02d}"

def _extract_tasks(text: str, document_id: str) -> List[RecoveryTask]:
    tasks = []
    for line in text.splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue                          # lines that don't look like a timed task are skipped
        hh, mm, ampm, name = int(m[1]), int(m[2]), m[3], m[4]
        start = _to_24h(hh, mm, ampm)
        if start is None:
            continue
        dur = DURATION_RE.search(name)
        tasks.append(RecoveryTask(
            id=f"{document_id}-task-{len(tasks)+1:03d}",
            name=name,
            start_time=start,
            duration_minutes=int(dur[1]) if dur else DEFAULT_DURATION_MIN,
            needs_caregiver=bool(CAREGIVER_RE.search(name)) and not NO_CAREGIVER_RE.search(name),
            needs_transport=bool(TRANSPORT_RE.search(name)),
            source_text=line.strip(),         # verbatim original line, always
            confirmed=False,                  # a human flips this, never code
        ))
    return tasks

def extract_tasks_from_text(req: ExtractionRequest) -> ExtractionResponse:
    return ExtractionResponse(
        source_document_id=req.document_id,
        tasks=_extract_tasks(req.text, req.document_id),
        human_confirmation_required=True,
    )

# ---------- Chat (rules first, optional LLM, output guard always) ----------
UNSAFE_OUTPUT_RE = re.compile(
    r'\bdiagnos|\byou (?:have|are suffering)\b|\b(?:increase|decrease|double|halve|skip|stop taking)\b.{0,30}\b(?:dose|medicine|tablet|medication)\b'
    r'|\bsafe to (?:take|combine|mix)\b|\binteract', re.IGNORECASE)
FALLBACK = ("I can only help with your confirmed recovery plan. "
            "For medical questions, please consult your doctor or tap CareConnect.")

def _rule_based_answer(req: ChatRequest) -> Optional[ChatResponse]:
    q = req.question.lower()
    alerts = req.alerts or {}
    if alerts.get("anomaly_detected") or alerts.get("status") == "INFEASIBLE":
        return ChatResponse(answer="I've noticed something in your plan that needs a professional's review. Please tap CareConnect.", route_to_careconnect=True)
    if any(w in q for w in ["conflict", "problem", "schedule"]):
        return ChatResponse(answer="Let's check your plan with a professional. Please tap CareConnect.", route_to_careconnect=True)
    if any(w in q for w in ["medicine", "medication", "tablet", "pill"]):
        return ChatResponse(answer="Please follow the timings in your confirmed plan. For any change, ask your doctor.", route_to_careconnect=False)
    return None

def _llm_answer(req: ChatRequest) -> Optional[str]:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    try:
        from google import genai              # NEW SDK: pip install google-genai (verify in Google's docs)
        client = genai.Client(api_key=key)
        prompt = (
            "You are a recovery-plan assistant. Use ONLY the confirmed text below. "
            "Never diagnose, never change doses/timings, never comment on AYUSH remedies. "
            "If unsure say: I am not sure - please consult your doctor. Be brief and warm.\n\n"
            f"CONFIRMED TEXT:\n{req.confirmed_discharge_text}\n\nQUESTION:\n{req.question}")
        return client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),   # confirm current name in docs
            contents=prompt).text
    except Exception as e:
        print(f"LLM error, using fallback: {e}")
        return None

def chat_with_companion(req: ChatRequest) -> ChatResponse:
    rule = _rule_based_answer(req)
    if rule:
        return rule
    answer = _llm_answer(req)
    if not answer or UNSAFE_OUTPUT_RE.search(answer):   # prompt rules are requests; this check enforces them
        answer = FALLBACK
    route = "careconnect" in answer.lower()
    return ChatResponse(answer=answer, route_to_careconnect=route)