# generate_data.py — Owner: Aadhishree
import json
import csv
import random
import os

random.seed(42)


def make_elderly_scenario(scenario_id: int, plant_conflict: bool) -> dict:
    physio_time = "15:00" if plant_conflict else "09:00"
    return {
        "scenario_id": f"elderly-{scenario_id:03d}",
        "pathway": "older_adult_post_surgical",
        "has_planted_conflict": plant_conflict,
        "constraints": {
            "caregiver_start": "08:00",
            "caregiver_end": "11:30",
            "can_travel_alone": False,
            "mobility": "walker_assisted",
        },
        "tasks": [
            {
                "id": f"task-med-{scenario_id}",
                "name": "Morning prescribed medicine",
                "start_time": "08:00",
                "duration_minutes": 10,
                "needs_caregiver": False,
                "needs_transport": False,
                "source_text": "08:00 AM morning prescribed medicine after food",
                "confirmed": True,
            },
            {
                "id": f"task-physio-{scenario_id}",
                "name": "Assisted knee physiotherapy",
                "start_time": physio_time,
                "duration_minutes": 45,
                "needs_caregiver": True,
                "needs_transport": False,
                "source_text": "10:30 AM assisted knee physiotherapy 45 minutes caregiver present",
                "confirmed": True,
            },
            {
                "id": f"task-wound-{scenario_id}",
                "name": "Wound dressing check",
                "start_time": "16:00",
                "duration_minutes": 20,
                "needs_caregiver": False,
                "needs_transport": False,
                "source_text": "04:00 PM wound dressing check",
                "confirmed": True,
            },
        ],
    }


def make_postnatal_scenario(scenario_id: int, plant_conflict: bool) -> dict:
    visit_time = "14:00" if plant_conflict else "10:00"
    return {
        "scenario_id": f"postnatal-{scenario_id:03d}",
        "pathway": "postnatal_recovery",
        "has_planted_conflict": plant_conflict,
        "constraints": {
            "caregiver_start": "09:00",
            "caregiver_end": "13:00",
            "can_travel_alone": False,
            "mobility": "walker_assisted",
        },
        "tasks": [
            {
                "id": f"task-feed-{scenario_id}",
                "name": "Newborn feeding schedule",
                "start_time": "08:00",
                "duration_minutes": 30,
                "needs_caregiver": False,
                "needs_transport": False,
                "source_text": "08:00 AM newborn feeding every 2-3 hours",
                "confirmed": True,
            },
            {
                "id": f"task-visit-{scenario_id}",
                "name": "Paediatrician follow-up visit",
                "start_time": visit_time,
                "duration_minutes": 60,
                "needs_caregiver": True,
                "needs_transport": True,
                "source_text": "paediatrician follow-up at clinic",
                "confirmed": True,
            },
        ],
    }


scenarios = []
for i in range(1, 16):
    scenarios.append(make_elderly_scenario(i, i <= 7))
for i in range(1, 16):
    scenarios.append(make_postnatal_scenario(i, i <= 7))

os.makedirs("data", exist_ok=True)
with open("data/synthetic_scenarios.json", "w") as f:
    json.dump(scenarios, f, indent=2)
print(f"Generated {len(scenarios)} scenarios -> data/synthetic_scenarios.json")

patients = [f"patient-{i:03d}" for i in range(1, 11)]
rows = []
for pid in patients:
    is_deteriorating = int(pid.split("-")[1]) <= 5
    for day in range(1, 15):
        if is_deteriorating and 8 <= day <= 10:
            pain = round(random.uniform(6.5, 9.0), 1)
            temp = round(random.uniform(38.1, 39.5), 1)
            sleep = round(random.uniform(3.0, 5.0), 1)
            mobility = round(random.uniform(2.0, 4.5), 1)
            vomit = 1 if day == 9 else 0
            headache = 1 if day == 9 else 0
        else:
            pain = round(max(0.5, random.gauss(3.0 - day * 0.1, 0.5)), 1)
            temp = round(random.uniform(36.5, 37.3), 1)
            sleep = round(random.uniform(6.5, 8.5), 1)
            mobility = round(min(10.0, 4.0 + day * 0.3 + random.gauss(0, 0.4)), 1)
            vomit = headache = 0
        rows.append(
            {
                "patient_id": pid,
                "day": day,
                "pain": pain,
                "temperature": temp,
                "sleep_hours": sleep,
                "mobility_score": mobility,
                "vomiting_flag": vomit,
                "headache_flag": headache,
            }
        )

with open("data/checkin_timeseries.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"Generated {len(rows)} check-in rows -> data/checkin_timeseries.csv")
print("Planted deterioration: patients 001-005, days 8-10")
print("All data is 100% synthetic for hackathon testing only.")