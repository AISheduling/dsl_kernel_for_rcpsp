# Datasets

This document describes the benchmark datasets used for evaluating the parser pipeline. All datasets come from the project scheduling literature and cover two problem families: the Multi-Mode Resource-Constrained Project Scheduling Problem (MMRCPSP / MRCPSP) and the Multi-Skilled RCPSP (MSRCPSP), as well as the classic single-mode RCPSP.

---

## MMLIB (Multi-Mode RCPSP Library)

**Problem:** Multi-Mode Resource-Constrained Project Scheduling Problem (MMRCPSP / MRCPSP). The goal is to minimize total project makespan subject to renewable and non-renewable resource constraints, where each activity can be executed in one of several alternative modes that differ in duration and resource requirements.

**File format:** `.mm` (PSPLIB multi-mode format)

**Reference:**
> Van Peteghem, V. and Vanhoucke, M. (2014). An experimental investigation of metaheuristics for the multi-mode resource-constrained project scheduling problem on new dataset instances. *European Journal of Operational Research*, 235(1), 62–72. [doi:10.1016/j.ejor.2013.10.012](https://doi.org/10.1016/j.ejor.2013.10.012)

**Subset used:** `MMLIB50` - instances with 50 activities

**Best known solutions:** Available via the SolutionsUpdate tool provided alongside the dataset.

**More information:** [https://www.projectmanagement.ugent.be/research/project_scheduling/multi_mode](https://www.projectmanagement.ugent.be/research/project_scheduling/multi_mode)

---

## MSLIB (Multi-Skilled RCPSP Library)

**Problem:** Multi-Skilled Resource-Constrained Project Scheduling Problem (MSRCPSP). An extension of the classic RCPSP where resources (workers) possess skills at varying proficiency levels, and tasks require specific skills rather than generic resource units.

**File format:** `.msrcp` (multi-skill RCP format)

**Subset used:** `MSLIB1` - basic set for MSRCPSP research with 6600 instances

**References:**
> Snauwaert, J., & Vanhoucke, M. (2022). Mathematical formulations for project scheduling problems with categorical and hierarchical skills. *Computers and Industrial Engineering*, 169, 108147. [doi:10.1016/j.cie.2022.108147](https://doi.org/10.1016/j.cie.2022.108147)

> Snauwaert, J., & Vanhoucke, M. (2023). A classification and new benchmark instances for the multi-skilled resource-constrained project scheduling problem. *European Journal of Operational Research*, 307(1), 1–19. [doi:10.1016/j.ejor.2022.05.049](https://doi.org/10.1016/j.ejor.2022.05.049)

**Download:** [https://github.com/mariovanhoucke/MSLIB-Multi-Skilled-Resource-Constrained-Project-Library/releases/latest](https://github.com/mariovanhoucke/MSLIB-Multi-Skilled-Resource-Constrained-Project-Library/releases/latest)

**More information:** [https://www.projectmanagement.ugent.be/research/data](https://www.projectmanagement.ugent.be/research/data)

---

## PSPLIB j30 (Single-Mode RCPSP)

**Problem:** Classic Resource-Constrained Project Scheduling Problem (RCPSP) with a single execution mode per activity. The benchmark j30 set contains instances with 30 activities (excluding dummy start and finish nodes) and 4 renewable resources.

**File format:** `.sm` and `.rcp` (PSPLIB single-mode / Patterson compact formats)

**Download:** [https://www.om-db.wi.tum.de/psplib/getdata.php?mode=sm](https://www.om-db.wi.tum.de/psplib/getdata.php?mode=sm)

**More information:** [https://www.om-db.wi.tum.de/psplib/](https://www.om-db.wi.tum.de/psplib/)

---

## Summary

| Dataset | Format | Problem type | Instances used |
|---|---|---|---|
| MMLIB50 | `.mm` | MMRCPSP | Subset for benchmarking |
| MSLIB1 | `.msrcp` | MSRCPSP | Subset for benchmarking |
| PSPLIB j30 | `.sm`, `.rcp` | RCPSP | Subset for benchmarking |