# app/anomaly_model.py — Owner: Aadhishree

from app.schemas import TrendRequest, TrendResponse

def _check_deterministic_red_flags(check_ins) -> tuple:
    reasons = []
    max_severity = "NORMAL"
    sorted_cins = sorted(check_ins, key=lambda c: c.day)

    for c in sorted_cins:
        if c.headache_flag == 1 and c.vomiting_flag == 1:
            reasons.append(
                f"Day {c.day}: Headache and vomiting co-occurring — "
                "route to CareConnect for professional review."
            )
            max_severity = "URGENT_ESCALATE"

    high_temp_days = [c.day for c in sorted_cins if c.temperature > 38.5]
    if len(high_temp_days) >= 2:
        for i in range(len(high_temp_days) - 1):
            if high_temp_days[i + 1] - high_temp_days[i] == 1:
                reasons.append(
                    f"Temperature above 38.5°C on days "
                    f"{high_temp_days[i]} and {high_temp_days[i+1]}."
                )
                if max_severity == "NORMAL":
                    max_severity = "REVIEW_NEEDED"

    pains = [(c.day, c.pain) for c in sorted_cins]
    streak = 1
    for i in range(1, len(pains)):
        if pains[i][1] > pains[i - 1][1]:
            streak += 1
            if streak >= 3:
                reasons.append(
                    f"Pain score increasing for {streak} consecutive days "
                    f"(ending day {pains[i][0]}) — review needed."
                )
                if max_severity == "NORMAL":
                    max_severity = "REVIEW_NEEDED"
                break
        else:
            streak = 1

    mobilities = [(c.day, c.mobility_score) for c in sorted_cins]
    streak = 1
    for i in range(1, len(mobilities)):
        if mobilities[i][1] < mobilities[i - 1][1]:
            streak += 1
            if streak >= 3:
                reasons.append(
                    f"Mobility declining for {streak} consecutive days "
                    f"(ending day {mobilities[i][0]}) — physiotherapy review needed."
                )
                if max_severity == "NORMAL":
                    max_severity = "REVIEW_NEEDED"
                break
        else:
            streak = 1

    return bool(reasons), reasons, max_severity


def analyze_symptom_trends(req: TrendRequest) -> TrendResponse:
    anomaly, reasons, severity = _check_deterministic_red_flags(req.check_ins)
    return TrendResponse(
        patient_id=req.patient_id,
        anomaly_detected=anomaly,
        changed_features=reasons,
        severity=severity
    )