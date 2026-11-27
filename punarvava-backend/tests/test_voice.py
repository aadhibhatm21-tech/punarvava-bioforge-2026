# test_voice.py
from app.schemas import VoiceRequest
from app.voice_maatri import generate_voice_payload

def test_disclaimer_appended():
    r = generate_voice_payload(VoiceRequest(text="Time for your medicine."))
    assert r.tts_text.endswith("please consult your doctor.")

def test_diagnostic_text_is_replaced():
    r = generate_voice_payload(VoiceRequest(text="You have an infection."))
    assert "infection" not in r.tts_text

def test_harmless_you_have_is_kept():
    r = generate_voice_payload(VoiceRequest(text="You have an appointment at 10."))
    assert "appointment" in r.tts_text

def test_unknown_tone_falls_back():
    r = generate_voice_payload(VoiceRequest(text="Hi", tone="angry"))
    assert r.voice_params["rate"] == 0.85