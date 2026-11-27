# app/scheduler.py — Owner: Aadhishree

from ortools.sat.python import cp_model
from app.schemas import FeasibilityRequest, FeasibilityResponse, ConflictDetail


def _time_to_minutes(t: str) -> int:
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def evaluate_feasibility(req: FeasibilityRequest) -> FeasibilityResponse:
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()

    c = req.constraints
    cg_start = _time_to_minutes(c.caregiver_start)
    cg_end = _time_to_minutes(c.caregiver_end)
    conflicts = []

    for task in req.tasks:
        t_start = _time_to_minutes(task.start_time)
        t_end = t_start + task.duration_minutes

        if task.needs_caregiver:
            if not (t_start >= cg_start and t_end <= cg_end):
                conflicts.append(ConflictDetail(
                    task_id=task.id,
                    task_name=task.name,
                    reason=(
                        f"Task requires caregiver assistance. "
                        f"Task runs {task.start_time} for {task.duration_minutes} min, "
                        f"but caregiver is only available "
                        f"{c.caregiver_start}-{c.caregiver_end}."
                    ),
                    action="ROUTE_TO_CARECONNECT"
                ))

        if task.needs_transport and not c.can_travel_alone:
            conflicts.append(ConflictDetail(
                task_id=task.id,
                task_name=task.name,
                reason=(
                    f"Task requires transport, but patient mobility is "
                    f"'{c.mobility}' and cannot travel alone."
                ),
                action="ROUTE_TO_CARECONNECT"
            ))

    if conflicts:
        dummy = model.NewBoolVar("forced_infeasible")
        model.AddBoolAnd([dummy.Not()])

    solver.Solve(model)
    result_status = "INFEASIBLE" if conflicts else "FEASIBLE"

    return FeasibilityResponse(
        status=result_status,
        conflicts=conflicts,
        preserved_tasks=req.tasks
    )