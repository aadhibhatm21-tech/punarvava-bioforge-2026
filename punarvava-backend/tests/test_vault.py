from app.schemas import ExtractionRequest
from app.vault_rag import extract_tasks_from_text

MEERA = """08:00 AM - Morning prescribed medicine, after food.
10:30 AM - Assisted knee physiotherapy, 45 minutes, caregiver present.
04:00 PM - Wound dressing check.
Caregiver available: 08:00 AM to 11:30 AM."""

def run(text): return extract_tasks_from_text(ExtractionRequest(document_id="t", text=text))

def test_meera_extracts_three_tasks_all_unconfirmed():
    r = run(MEERA)
    assert len(r.tasks) == 3
    assert all(t.confirmed is False for t in r.tasks)
    assert r.human_confirmation_required is True

def test_times_and_flags():
    t = run(MEERA).tasks
    assert [x.start_time for x in t] == ["08:00", "10:30", "16:00"]
    assert t[1].needs_caregiver and t[1].duration_minutes == 45

def test_source_text_is_verbatim():
    for task in run(MEERA).tasks:
        assert task.source_text in MEERA

def test_garbage_time_is_dropped_not_guessed():
    assert run("27:99 PM - mystery").tasks == []