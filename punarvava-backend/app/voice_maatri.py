import re
from app.schemas import VoiceRequest, VoiceResponse

# rate: 1.0 = normal speed. These map directly onto expo-speech options on the phone.
TONE_CONFIGS = {
    "gentle": {"rate": 0.85, "pitch": 1.0, "language": "en-IN"},
    "clear":  {"rate": 1.0,  "pitch": 1.0, "language": "en-IN"},
    "calm":   {"rate": 0.80, "pitch": 0.9, "language": "en-IN"},
}
FORBIDDEN_RE = re.compile(
    r'\byou (?:are diagnosed|have (?:an?|the) (?!appointment|session|visit|dose|check))\b|\byour condition is\b|\bthis (?:indicates|means you have)\b',
    re.IGNORECASE)
SAFE_REPLACEMENT = "Please consult your doctor for any clinical assessment."
DISCLAIMER = " For any medical questions, please consult your doctor."

def generate_voice_payload(req: VoiceRequest) -> VoiceResponse:
    tone = req.tone if req.tone in TONE_CONFIGS else "gentle"
    text = SAFE_REPLACEMENT if FORBIDDEN_RE.search(req.text) else req.text.strip()
    return VoiceResponse(tts_text=text + DISCLAIMER, voice_params=TONE_CONFIGS[tone])