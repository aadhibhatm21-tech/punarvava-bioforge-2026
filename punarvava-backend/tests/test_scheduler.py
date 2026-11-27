# tests/test_scheduler.py — Owner: Aadhishree
from app.schemas import (
    RecoveryTask, PatientConstraints, FeasibilityRequest,
    SymptomCheckin, TrendRequest
)
from app.scheduler import evaluate_feasibility
from app.anomaly_model import analyze_symptom_trends


def test_feasible_schedule():
    req = FeasibilityRequest(
        tasks=[
            RecoveryTask(
                id="t1", name="Morning Physio", start_time="09:00",
                duration_minutes=45, needs_caregiver=True, needs_transport=False,
                source_text="09:00 physio", confirmed=True
            )
        ],
        constraints=PatientConstraints(
            caregiver_start="08:00", caregiver_end="12:00",
            can_travel_alone=False, mobility="walker_assisted"
        )
    )
    res = evaluate_feasibility(req)
    assert res.status == "FEASIBLE"
    assert len(res.conflicts) == 0
    assert len(res.preserved_tasks) == 1


def test_infeasible_caregiver_window_preserves_task():
    req = FeasibilityRequest(
        tasks=[
            RecoveryTask(
                id="t2", name="Afternoon Physio", start_time="15:00",
                duration_minutes=45, needs_caregiver=True, needs_transport=False,
                source_text="15:00 physio", confirmed=True
            )
        ],
        constraints=PatientConstraints(
            caregiver_start="08:00", caregiver_end="11:30",
            can_travel_alone=False, mobility="walker_assisted"
        )
    )
    res = evaluate_feasibility(req)
    assert res.status == "INFEASIBLE"
    assert res.conflicts[0].action == "ROUTE_TO_CARECONNECT"
    assert len(res.preserved_tasks) == 1


def test_infeasible_transport_restriction():
    req = FeasibilityRequest(
        tasks=[
            RecoveryTask(
                id="t3", name="Clinic Follow-up", start_time="10:00",
                duration_minutes=60, needs_caregiver=True, needs_transport=True,
                source_text="clinic visit", confirmed=True
            )
        ],
        constraints=PatientConstraints(
            caregiver_start="08:00", caregiver_end="13:00",
            can_travel_alone=False, mobility="wheelchair"
        )
    )
    res = evaluate_feasibility(req)
    assert res.status == "INFEASIBLE"
    assert any("transport" in c.reason.lower() for c in res.conflicts)
    assert len(res.preserved_tasks) == 1


def test_multiple_tasks_never_deleted_on_conflict():
    req = FeasibilityRequest(
        tasks=[
            RecoveryTask(
                id="t1", name="Medicine", start_time="08:00",
                duration_minutes=10, needs_caregiver=False, needs_transport=False,
                source_text="08:00 med", confirmed=True
            ),
            RecoveryTask(
                id="t2", name="Late Physio", start_time="16:00",
                duration_minutes=30, needs_caregiver=True, needs_transport=False,
                source_text="16:00 physio", confirmed=True
            ),
        ],
        constraints=PatientConstraints(
            caregiver_start="08:00", caregiver_end="12:00",
            can_travel_alone=True, mobility="independent"
        )
    )
    res = evaluate_feasibility(req)
    assert res.status == "INFEASIBLE"
    assert len(res.preserved_tasks) == 2


def test_urgent_red_flag_headache_vomiting():
    req = TrendRequest(
        patient_id="patient-001",
        check_ins=[
            SymptomCheckin(
                day=9, pain=8.0, temperature=39.0, sleep_hours=4.0,
                mobility_score=3.0, vomiting_flag=1, headache_flag=1
            )
        ]
    )
    res = analyze_symptom_trends(req)
    assert res.anomaly_detected is True
    assert res.severity == "URGENT_ESCALATE"


def test_normal_recovery_trend():
    req = TrendRequest(
        patient_id="patient-008",
        check_ins=[
            SymptomCheckin(
                day=1, pain=3.0, temperature=36.8, sleep_hours=7.5,
                mobility_score=5.0, vomiting_flag=0, headache_flag=0
            ),
            SymptomCheckin(
                day=2, pain=2.5, temperature=36.7, sleep_hours=8.0,
                mobility_score=5.5, vomiting_flag=0, headache_flag=0
            )
        ]
    )
    res = analyze_symptom_trends(req)
    assert res.anomaly_detected is False
    assert res.severity == "NORMAL"


def test_lstm_autoencoder_detects_multiday_drift():
    req = TrendRequest(
        patient_id="patient-002",
        check_ins=[
            SymptomCheckin(
                day=7, pain=6.5, temperature=37.8, sleep_hours=4.2,
                mobility_score=3.2, vomiting_flag=0, headache_flag=0
            ),
            SymptomCheckin(
                day=8, pain=7.0, temperature=38.0, sleep_hours=3.8,
                mobility_score=3.0, vomiting_flag=0, headache_flag=0
            ),
            SymptomCheckin(
                day=9, pain=6.8, temperature=37.9, sleep_hours=4.0,
                mobility_score=3.1, vomiting_flag=0, headache_flag=0
            ),
        ]
    )
    res = analyze_symptom_trends(req)
    assert res.anomaly_detected is True
    assert res.severity in ("REVIEW_NEEDED", "URGENT_ESCALATE")
    assert any("LSTM Autoencoder" in msg for msg in res.changed_features)