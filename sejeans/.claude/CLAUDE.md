# Project: Lightweight Anomaly Detection for EV Motor-Reducer and Battery

## Goal

Build a lightweight ML pipeline that detects anomalies in EV drivetrain components (motor-reducer and battery) and runs inference on in-vehicle edge devices in real time.

## Scope

- **Motor-reducer**: 5-class classification — NORMAL / ECC10 / ECC20 / DEMAG / REDUC
- **Battery**: 3-class classification — NORMAL / Cell Voltage Fault / Cell Deviation Fault
- **Output**: Real-time monitoring dashboard with anomaly scores and diagnostic reports

---

## Data
All sample data is under `샘플데이터/Sample/`.
The sample data is for a pilot study to determine the feasibility of this project. Once the pilot test is complete, we will upload the full 600GB dataset to validate the model.

### Motor-Reducer

| Item | Detail |
|---|---|
| Vehicles | IONIQ, KONA, NIRO |
| Format | PNG (STFT spectrogram images) |
| Channels(sensors) | Current_U, Vib_Motor, Vib_TM |
| Volume | 550,800 samples |
| Classes | NORMAL, ECC10, ECC20, DEMAG, REDUC |

- Images are 2D spectrograms produced by applying Short-Time Fourier Transform (STFT) to raw time-series signals.
- **ECC10/ECC20**: shaft misalignment (eccentricity) at two severity levels
- **DEMAG**: magnet demagnetization fault
- **REDUC**: reducer gear fault

### Battery

| Item | Detail |
|---|---|
| Vehicles | IONIQ, KONA, NIRO |
| Format | NPY (numerical matrix) |
| Structure | rows = time steps, columns = cell index |
| Channels | 96 or 98 cell voltages + voltage deviations |
| Volume | 220,320 samples |
| Classes | NORMAL, Cell Voltage Fault, Cell Deviation Fault |

- **Cell Voltage Fault**: one or more individual cell voltages exceed threshold
- **Cell Deviation Fault**: inter-cell voltage imbalance is abnormally large (individual values may be within range)

---

## Agents

Custom agents live in `.claude/agents/`. Invoke them for their respective tasks:

| Agent | Responsibility |
|---|---|
| `prd-writer` | Write/update PRD documents in `.claude/prd/` |
| `plan-writer` | Write/update implementation plan in `.claude/plan/` |
| `todo-manager` | Manage task list in `.claude/todo/` |
| `code-implementer` | Implement code from TODO tasks |

# Workflow Orchestration
## 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

## 2. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

## 3. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

## 4. Core Principles
- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.