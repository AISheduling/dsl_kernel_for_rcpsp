"""
Parsers for RCPSP benchmark formats → SchedulingProblem DSL (schema v0.1)

Supported formats:
  • .sm   – PSPLIB single-mode (j30, j60, j90, j120)
  • .mm   – PSPLIB multi-mode (m-RCPSP)
  • .rcp  – RCP format (Patterson-style, compact)
  • .msrcp – Multi-Skill RCP (extended RCP with workforce/skill modules)

Usage:
    from parsers import parse_sm, parse_mm, parse_rcp, parse_msrcp
    problem = parse_sm("j302_1.sm")
    print(problem.model_dump_json(indent=2))
"""

from __future__ import annotations

import re
from pathlib import Path

# DSL-схема
from src.dsl_schema import (
    SchedulingProblem,
    Task,
    Resource,
    Dependency,
    ProjectMeta,
    ProjectExt,
    TaskExt,
    ResourceExt,
    RCPSPTaskExt,
    RCPSPResourceExt,
    RCPSPMode,
    RCPSPModeRequirement,
    RCPSPProjectExt,
)


# Helpers
def _problem_id(path: str) -> str:
    return Path(path).stem.upper()

def _task_id(jobnr: int, n_jobs: int) -> str:
    """Map job number to a string ID; first/last job → START/END."""
    if jobnr == 1:
        return "START"
    if jobnr == n_jobs:
        return "END"
    return f"T{jobnr}"

def _renewable_resource(rid: str, capacity: int) -> Resource:
    return Resource(
        id=rid,
        capacity=capacity,
        extensions=ResourceExt(rcpsp=RCPSPResourceExt(type="renewable")),
    )

def _nonrenewable_resource(rid: str, capacity: int) -> Resource:
    return Resource(
        id=rid,
        capacity=capacity,
        extensions=ResourceExt(rcpsp=RCPSPResourceExt(type="non_renewable")),
    )

def _dummy_mode(mode_id: str = "M1") -> RCPSPMode:
    return RCPSPMode(mode_id=mode_id, duration=0, requirements=[])


# Parser 1 – .sm  (PSPLIB single-mode)

def parse_sm(path: str) -> SchedulingProblem:
    """
    Parse a PSPLIB .sm file (single-mode RCPSP).
    Handles j30, j60, j90, j120 and similar PSPLIB benchmark sets.

    Format sections:
      - RESOURCES: counts of R (renewable), N (non-renewable), D (doubly constrained)
      - PRECEDENCE RELATIONS: jobnr → successors
      - REQUESTS/DURATIONS: jobnr mode duration R1..Rn [N1..Nn]
      - RESOURCEAVAILABILITIES: capacities for each resource
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    # Resource counts
    n_renewable = int(re.search(r"renewable\s*:\s*(\d+)", text).group(1))
    n_nonrenewable_match = re.search(r"nonrenewable\s*:\s*(\d+)", text)
    n_nonrenewable = int(n_nonrenewable_match.group(1)) if n_nonrenewable_match else 0

    # Total job count
    n_jobs_match = re.search(r"jobs\s*\(incl.*?\)\s*:\s*(\d+)", text)
    n_jobs = int(n_jobs_match.group(1))

    # Precedence section
    prec_section = _extract_section(lines, "PRECEDENCE RELATIONS")
    successors: dict[int, list[int]] = {}
    for line in prec_section:
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            jobnr = int(parts[0])
            # parts[1] = #modes, parts[2] = #successors, rest = successor ids
            n_succ = int(parts[2])
            succs = [int(x) for x in parts[3:3 + n_succ]]
            successors[jobnr] = succs
        except ValueError:
            continue

    # Requests/Durations section
    req_section = _extract_section(lines, "REQUESTS/DURATIONS")
    # Each row: jobnr mode duration R1 R2 ... [N1 N2 ...]
    durations: dict[int, int] = {}
    req_per_job: dict[int, list[int]] = {}  # resource quantities indexed by job
    for line in req_section:
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            jobnr = int(parts[0])
            # mode = int(parts[1])  # always 1 for .sm
            duration = int(parts[2])
            quantities = [int(x) for x in parts[3:]]
            durations[jobnr] = duration
            req_per_job[jobnr] = quantities
        except ValueError:
            continue

    # Resource availabilities
    avail_section = _extract_section(lines, "RESOURCEAVAILABILITIES")
    capacities = _parse_capacities(avail_section)

    # Build resources
    resources: list[Resource] = []
    for i in range(n_renewable):
        resources.append(_renewable_resource(f"R{i+1}", capacities[i] if i < len(capacities) else 0))
    for i in range(n_nonrenewable):
        idx = n_renewable + i
        resources.append(_nonrenewable_resource(f"N{i+1}", capacities[idx] if idx < len(capacities) else 0))

    # Resource ids in order used in req_per_job
    resource_ids = [f"R{i+1}" for i in range(n_renewable)] + [f"N{i+1}" for i in range(n_nonrenewable)]

    # Build tasks
    tasks: list[Task] = []

    # Predecessor map: for each job, which jobs point to it
    predecessors: dict[int, list[int]] = {j: [] for j in successors}
    for job, succs in successors.items():
        for s in succs:
            if s not in predecessors:
                predecessors[s] = []
            predecessors[s].append(job)

    for jobnr in sorted(successors.keys()):
        tid = _task_id(jobnr, n_jobs)
        dur = durations.get(jobnr, 0)
        qty = req_per_job.get(jobnr, [])

        requirements = [
            RCPSPModeRequirement(resource_id=resource_ids[i], quantity=qty[i])
            for i in range(len(resource_ids))
            if i < len(qty) and qty[i] > 0
        ]
        mode = RCPSPMode(mode_id="M1", duration=dur, requirements=requirements)

        deps = [
            Dependency(task_id=_task_id(p, n_jobs), type="FS")
            for p in predecessors.get(jobnr, [])
        ]

        tasks.append(Task(
            id=tid,
            dependencies=deps,
            extensions=TaskExt(rcpsp=RCPSPTaskExt(modes=[mode])),
        ))

    return SchedulingProblem(
        problem_id=_problem_id(path),
        domain="rcpsp",
        description=f"PSPLIB single-mode RCPSP parsed from {Path(path).name}",
        project=ProjectMeta(name=_problem_id(path), objective="minimize_makespan"),
        extensions=ProjectExt(rcpsp=RCPSPProjectExt()),
        resources=resources,
        tasks=tasks,
    )


# Parser 2 – .mm  (PSPLIB multi-mode, m-RCPSP)

def parse_mm(path: str) -> SchedulingProblem:
    """
    Parse a PSPLIB .mm file (multi-mode RCPSP / m-RCPSP).

    Differences from .sm:
      - Each job can have multiple modes (varying duration + resource usage)
      - Has both renewable (R) and non-renewable (N) resources
      - REQUESTS/DURATIONS block has multiple lines per job
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    n_renewable = int(re.search(r"renewable\s*:\s*(\d+)", text).group(1))
    n_nonrenewable_match = re.search(r"nonrenewable\s*:\s*(\d+)", text)
    n_nonrenewable = int(n_nonrenewable_match.group(1)) if n_nonrenewable_match else 0

    n_jobs_match = re.search(r"jobs\s*\(incl.*?\)\s*:\s*(\d+)", text)
    n_jobs = int(n_jobs_match.group(1))

    # Precedence section
    prec_section = _extract_section(lines, "PRECEDENCE RELATIONS")
    successors: dict[int, list[int]] = {}
    n_modes_per_job: dict[int, int] = {}
    for line in prec_section:
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            jobnr = int(parts[0])
            n_modes = int(parts[1])
            n_succ = int(parts[2])
            succs = [int(x) for x in parts[3:3 + n_succ]]
            successors[jobnr] = succs
            n_modes_per_job[jobnr] = n_modes
        except ValueError:
            continue

    # Requests/Durations section
    # Format per mode row:
    #   jobnr  mode  dur  R1  R2  ...  N1  N2  ...
    # Continuation rows (same job, next mode) may omit jobnr or repeat it.
    # Use raw lines to detect tab-indented continuation rows
    raw_lines_mm = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    req_raw: list[tuple[bool, str]] = []  # (is_continuation, stripped_line)
    in_req = False
    for raw_line in raw_lines_mm:
        stripped = raw_line.strip()
        if not in_req:
            if "REQUESTS/DURATIONS" in stripped:
                in_req = True
            continue
        if stripped.startswith("***") or (not stripped):
            if in_req and stripped.startswith("***"):
                break
            continue
        if stripped.startswith("jobnr") or stripped.startswith("---"):
            continue
        # A continuation line starts with whitespace (tab or spaces) before the mode number
        is_continuation = raw_line.startswith(("\t", " "))
        req_raw.append((is_continuation, stripped))

    job_modes: dict[int, list[tuple[str, int, list[int]]]] = {}
    current_job = None
    for is_continuation, line in req_raw:
        parts = line.split()
        if not parts:
            continue
        try:
            nums = [int(p) for p in parts]
        except ValueError:
            continue
        if is_continuation and current_job is not None:
            # continuation: mode  dur  R1 R2 N1 N2
            mode_id = f"M{nums[0]}"
            duration = nums[1]
            quantities = nums[2:]
            job_modes[current_job].append((mode_id, duration, quantities))
        else:
            # new job row: jobnr  mode  dur  R1 R2 N1 N2
            current_job = nums[0]
            if current_job not in job_modes:
                job_modes[current_job] = []
            mode_id = f"M{nums[1]}"
            duration = nums[2]
            quantities = nums[3:]
            job_modes[current_job].append((mode_id, duration, quantities))

    # Resource availabilities
    avail_section = _extract_section(lines, "RESOURCE AVAILABILITIES")
    capacities = _parse_capacities(avail_section)

    # Build resources
    resources: list[Resource] = []
    for i in range(n_renewable):
        resources.append(_renewable_resource(f"R{i+1}", capacities[i] if i < len(capacities) else 0))
    for i in range(n_nonrenewable):
        idx = n_renewable + i
        resources.append(_nonrenewable_resource(f"N{i+1}", capacities[idx] if idx < len(capacities) else 0))

    resource_ids = [f"R{i+1}" for i in range(n_renewable)] + [f"N{i+1}" for i in range(n_nonrenewable)]

    # Predecessors
    predecessors: dict[int, list[int]] = {j: [] for j in successors}
    for job, succs in successors.items():
        for s in succs:
            if s not in predecessors:
                predecessors[s] = []
            predecessors[s].append(job)

    # Build tasks
    tasks: list[Task] = []
    for jobnr in sorted(successors.keys()):
        tid = _task_id(jobnr, n_jobs)

        modes: list[RCPSPMode] = []
        for mode_id, duration, quantities in job_modes.get(jobnr, []):
            requirements = [
                RCPSPModeRequirement(resource_id=resource_ids[i], quantity=quantities[i])
                for i in range(len(resource_ids))
                if i < len(quantities) and quantities[i] > 0
            ]
            modes.append(RCPSPMode(mode_id=mode_id, duration=duration, requirements=requirements))

        if not modes:
            modes = [_dummy_mode()]

        deps = [
            Dependency(task_id=_task_id(p, n_jobs), type="FS")
            for p in predecessors.get(jobnr, [])
        ]

        tasks.append(Task(
            id=tid,
            dependencies=deps,
            extensions=TaskExt(rcpsp=RCPSPTaskExt(modes=modes)),
        ))

    return SchedulingProblem(
        problem_id=_problem_id(path),
        domain="rcpsp",
        description=f"PSPLIB multi-mode RCPSP (m-RCPSP) parsed from {Path(path).name}",
        project=ProjectMeta(name=_problem_id(path), objective="minimize_makespan"),
        extensions=ProjectExt(rcpsp=RCPSPProjectExt()),
        resources=resources,
        tasks=tasks,
    )


# Parser 3 – .rcp  (Patterson / RCP format)

def parse_rcp(path: str) -> SchedulingProblem:
    """
    Parse a Patterson-style .rcp file (compact RCPSP format).

    Line layout:
      Line 1:  n_jobs  n_resources
      Line 2:  capacity_R1  capacity_R2  ...
      Then n_jobs blocks, one per job:
        dur  R1_req  R2_req  ...  n_successors  succ1  succ2  ...
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    # Normalize: strip \r, collapse blank lines
    raw_lines = [l.strip() for l in text.splitlines() if l.strip()]

    tokens = []
    for l in raw_lines:
        tokens.extend(l.split())

    idx = 0

    def next_int() -> int:
        nonlocal idx
        val = int(tokens[idx])
        idx += 1
        return val

    n_jobs = next_int()
    n_res = next_int()

    capacities = [next_int() for _ in range(n_res)]

    # Per-job data: (duration, [req per resource], [successor job numbers])
    job_data: list[tuple[int, list[int], list[int]]] = []
    for _ in range(n_jobs):
        dur = next_int()
        reqs = [next_int() for _ in range(n_res)]
        n_succ = next_int()
        succs = [next_int() for _ in range(n_succ)]
        job_data.append((dur, reqs, succs))

    # Resources
    resources = [_renewable_resource(f"R{i+1}", capacities[i]) for i in range(n_res)]
    resource_ids = [f"R{i+1}" for i in range(n_res)]

    # Predecessor map
    predecessors: dict[int, list[int]] = {j+1: [] for j in range(n_jobs)}
    for jobnr, (_, _, succs) in enumerate(job_data, start=1):
        for s in succs:
            if s not in predecessors:
                predecessors[s] = []
            predecessors[s].append(jobnr)

    # Tasks
    tasks: list[Task] = []
    for jobnr, (dur, reqs, _) in enumerate(job_data, start=1):
        tid = _task_id(jobnr, n_jobs)

        requirements = [
            RCPSPModeRequirement(resource_id=resource_ids[i], quantity=reqs[i])
            for i in range(n_res)
            if reqs[i] > 0
        ]
        mode = RCPSPMode(mode_id="M1", duration=dur, requirements=requirements)

        deps = [
            Dependency(task_id=_task_id(p, n_jobs), type="FS")
            for p in predecessors.get(jobnr, [])
        ]

        tasks.append(Task(
            id=tid,
            dependencies=deps,
            extensions=TaskExt(rcpsp=RCPSPTaskExt(modes=[mode])),
        ))

    return SchedulingProblem(
        problem_id=_problem_id(path),
        domain="rcpsp",
        description=f"Patterson-style RCP parsed from {Path(path).name}",
        project=ProjectMeta(name=_problem_id(path), objective="minimize_makespan"),
        extensions=ProjectExt(rcpsp=RCPSPProjectExt()),
        resources=resources,
        tasks=tasks,
    )

# Parser 4 – .msrcp

def parse_msrcp(path: str) -> SchedulingProblem:
    """
    Parse a Multi-Skill RCP (.msrcp) file.

    Sections (marked by \\* Name *\\):
      Project Module – n_jobs  n_skills  n_workers  n_periods
      Workforce Module     – worker × skill binary availability matrix
      Workforce Module with Skill Levels – worker × skill level matrix
      Skill Requirements Module – job × skill requirement matrix
      Skill Level Requirements Module – per-job skill-level lists (complex)
      Cost Module          – salary and cost-per-unit arrays
      Common Resource Usage Module – n_common  +  worker × period usage matrix
      Rework Module        – quality/rework probability matrix

    We map:
      • Each worker → a renewable resource with capacity 1
      • Each skill → a "virtual" renewable resource (aggregated demand per job)
      • Precedence: taken from Project Module's successor list
      • Durations: taken from Project Module's duration column
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = [l.rstrip() for l in text.splitlines()]

    # Split into named sections
    sections: dict[str, list[str]] = {}
    current_name = "__header__"
    sections[current_name] = []
    section_re = re.compile(r"\\[*]\s*(.*?)\s*[*]\\")

    for line in lines:
        m = section_re.search(line)
        if m:
            current_name = m.group(1).strip()
            sections[current_name] = []
        else:
            sections[current_name].append(line)

    def section_tokens(name: str) -> list[int]:
        raw = sections.get(name, [])
        out = []
        for l in raw:
            for tok in l.split():
                try:
                    out.append(int(tok))
                except ValueError:
                    pass
        return out

    def section_floats(name: str) -> list[float]:
        raw = sections.get(name, [])
        out = []
        for l in raw:
            for tok in l.split():
                try:
                    out.append(float(tok))
                except ValueError:
                    pass
        return out

    # Project Module
    proj_tokens = section_tokens("Project Module")
    # Format: n_jobs  n_skills  n_workers  n_periods
    # Then: horizon (single number on next non-empty line, sometimes twice)
    # Then: per-job rows: duration  n_successors  succ1 succ2 ...
    n_jobs, n_skills, n_workers, n_periods = proj_tokens[0], proj_tokens[1], proj_tokens[2], proj_tokens[3]

    # Parse the per-job rows from raw section lines (preserve structure)
    proj_lines = [l for l in sections.get("Project Module", []) if l.strip()]
    # First line: header numbers
    # Next lines that are just a single integer: horizon values (skip)
    # Then job rows

    job_rows_raw: list[list[int]] = []
    header_consumed = False
    for line in proj_lines:
        parts = line.split()
        if not parts:
            continue
        ints = []
        for p in parts:
            try:
                ints.append(int(p))
            except ValueError:
                break
        if len(ints) != len(parts):
            continue  # non-integer line
        if not header_consumed:
            header_consumed = True  # skip the "n_jobs n_skills n_workers n_periods" line
            continue
        if len(ints) == 1:
            continue  # horizon line
        job_rows_raw.append(ints)

    # Each job row: duration  n_successors  succ1  succ2  ...
    job_durations: list[int] = []
    job_successors: list[list[int]] = []
    for row in job_rows_raw:
        dur = row[0]
        n_succ = row[1]
        succs = row[2:2 + n_succ]
        job_durations.append(dur)
        job_successors.append(succs)

    if len(job_durations) != n_jobs:
        # Fallback: just use what we parsed
        n_jobs = len(job_durations)

    # Skill Requirements Module
    # n_jobs × n_skills matrix: skill_req[job_idx][skill_idx]
    skill_req_tokens = section_tokens("Skill Requirements Module")
    skill_req: list[list[int]] = []
    for j in range(n_jobs):
        row = skill_req_tokens[j * n_skills:(j + 1) * n_skills]
        skill_req.append(row)

    # Workforce Module (binary availability)
    workforce_tokens = section_tokens("Workforce Module")
    # n_workers × n_skills binary matrix
    worker_skill: list[list[int]] = []
    for w in range(n_workers):
        row = workforce_tokens[w * n_skills:(w + 1) * n_skills]
        worker_skill.append(row)

    # Build resources
    # Strategy: one resource per skill (capacity = number of workers with that skill)
    skill_capacities = [0] * n_skills
    for w in range(n_workers):
        for s in range(n_skills):
            if s < len(worker_skill[w]) and worker_skill[w][s]:
                skill_capacities[s] += 1

    resources: list[Resource] = []
    for s in range(n_skills):
        resources.append(Resource(
            id=f"SKILL{s+1}",
            name=f"Skill {s+1}",
            capacity=skill_capacities[s],
            extensions=ResourceExt(rcpsp=RCPSPResourceExt(type="renewable")),
        ))

    resource_ids = [f"SKILL{s+1}" for s in range(n_skills)]

    # Predecessor map
    predecessors: dict[int, list[int]] = {j+1: [] for j in range(n_jobs)}
    for jobnr, succs in enumerate(job_successors, start=1):
        for s in succs:
            if s not in predecessors:
                predecessors[s] = []
            predecessors[s].append(jobnr)

    # Build tasks
    tasks: list[Task] = []
    for jobnr in range(1, n_jobs + 1):
        tid = _task_id(jobnr, n_jobs)
        dur = job_durations[jobnr - 1] if jobnr - 1 < len(job_durations) else 0
        reqs_row = skill_req[jobnr - 1] if jobnr - 1 < len(skill_req) else []

        requirements = [
            RCPSPModeRequirement(resource_id=resource_ids[s], quantity=reqs_row[s])
            for s in range(n_skills)
            if s < len(reqs_row) and reqs_row[s] > 0
        ]
        mode = RCPSPMode(mode_id="M1", duration=dur, requirements=requirements)

        deps = [
            Dependency(task_id=_task_id(p, n_jobs), type="FS")
            for p in predecessors.get(jobnr, [])
        ]

        tasks.append(Task(
            id=tid,
            dependencies=deps,
            extensions=TaskExt(rcpsp=RCPSPTaskExt(modes=[mode])),
        ))

    return SchedulingProblem(
        problem_id=_problem_id(path),
        domain="rcpsp",
        description=f"Multi-Skill RCP parsed from {Path(path).name}",
        project=ProjectMeta(name=_problem_id(path), objective="minimize_makespan"),
        extensions=ProjectExt(rcpsp=RCPSPProjectExt()),
        resources=resources,
        tasks=tasks,
    )


# Internal utilities

def _extract_section(lines: list[str], header: str) -> list[str]:
    """
    Return lines between the row containing `header` and the next
    separator line (all stars) or end of file.
    """
    result = []
    inside = False
    header_lower = header.lower()
    for line in lines:
        stripped = line.strip()
        if not inside:
            if header_lower in stripped.lower():
                inside = True
            continue
        # Stop at separator lines or empty section headers
        if stripped.startswith("***") or (stripped.startswith("\\*") and inside):
            break
        if stripped:
            result.append(stripped)
    return result


def _parse_capacities(avail_lines: list[str]) -> list[int]:
    """
    Extract integers from resource availability lines.
    Skips header rows with letters (e.g. 'R 1  R 2  N 1').
    """
    for line in avail_lines:
        parts = line.split()
        ints = []
        try:
            ints = [int(p) for p in parts]
        except ValueError:
            continue
        if ints:
            return ints
    return []


# CLI demo

if __name__ == "__main__":
    import sys
    import json

    parsers = {
        ".sm": parse_sm,
        ".mm": parse_mm,
        ".rcp": parse_rcp,
        ".msrcp": parse_msrcp,
    }

    if len(sys.argv) < 2:
        print("Usage: python parsers.py <file.[sm|mm|rcp|msrcp]>")
        sys.exit(1)

    fpath = sys.argv[1]
    ext = Path(fpath).suffix.lower()
    parser = parsers.get(ext)
    if not parser:
        print(f"Unknown extension: {ext}. Supported: {list(parsers)}")
        sys.exit(1)

    problem = parser(fpath)
    print(problem.model_dump_json(indent=2, exclude_none=True))