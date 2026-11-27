from fastapi import FastAPI
from app.schemas import *
from app.vault_rag import extract_tasks_from_text, chat_with_companion
from app.voice_maatri import generate_voice_payload

app = FastAPI(title="Adit dev server")

@app.post("/extract-discharge", response_model=ExtractionResponse)
def extract(req: ExtractionRequest): return extract_tasks_from_text(req)

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest): return chat_with_companion(req)

@app.post("/maatri/voice", response_model=VoiceResponse)
def voice(req: VoiceRequest): return generate_voice_payload(req)