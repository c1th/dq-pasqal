# DuQuantum × Pasqal — Track A: capacity-feasible clustering on a neutral-atom register

Team Alex and Bob (Alex Wang, Christopher Thai). Tier S, emulated on Pulser's `AnalogDevice`.

## Environment
Python 3.10+.

    python -m venv .venv
    .venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
    pip install -r requirements.txt     # pinned versions (pulser, scipy, numpy, cplex, docplex, ...)

## Headline result — one command

    python reproduce.py

Regenerates the atom registers and runs the emulation (about 6–8 minutes), writing `results.csv`. Expected summary
(tier S, 20 instances, 8 atoms, 1000 shots, baseline 4 µs pulse):

| Method | Feasible shots (raw) | Best shot optimal | P(reach optimum) |
|---|---|---|---|
| Ladder layout | 96.1% | 15/20 | 0.323 |
| Coefficient matching | 68.9% | 16/20 | 0.292 |
| Weighted log fit | 70.3% | 16/20 | 0.261 |
| Uniform random | 2.9% | 20/20 | 0.008 |

`python reproduce.py --all` additionally runs the instance export, the penalty-weight study and the CPLEX cross-check.
The coefficient-matching and log fits depend on the optimizer's local minimum and can differ slightly with other
`numpy`/`scipy` versions; use the pinned environment.

## Documents
The report and the model write-up are in [`docs/`](docs/), as compiled PDFs alongside their LaTeX sources:

| Document | PDF | LaTeX source |
|---|---|---|
| Report (4 pages max) | [`docs/DuQuantum_Pasqal_Report.pdf`](docs/DuQuantum_Pasqal_Report.pdf) | [`docs/report.tex`](docs/report.tex) |
| Model write-up (formulations, derivations, variable-count table) | [`docs/DuQuantum_Pasqal_Model_Writeup.pdf`](docs/DuQuantum_Pasqal_Model_Writeup.pdf) | [`docs/model_writeup.tex`](docs/model_writeup.tex) |

## Deliverables
| Deliverable | Where |
|---|---|
| Model write-up (formulations, variable-count table) | `docs/DuQuantum_Pasqal_Model_Writeup.pdf` (source `docs/model_writeup.tex`) |
| Report (4 pages max) | `docs/DuQuantum_Pasqal_Report.pdf` (source `docs/report.tex`) |
| Instances: generator, validator | `instances.py` (`generate(tier, seed)`, `validate(inst)`) |
| Instances: seeds, certificates, reference values | `instances_tierS.json` (seeds 0–99; `python export_instances.py`) |
| Results, one row per (instance, method, seed) | `results.csv` |

## Pipeline
| Step | Script | Output |
|---|---|---|
| Instances | `instances.py` (self-test: `python instances.py`), `export_instances.py` | `instances_tierS.json` |
| 1. Penalty weights (exhaustive, 100 instances) | `step1_qubo_check.py` | `step1_results.csv` |
| 2. Coefficient matching and weighted log fit | `step2_embed_check.py` | `step2_results.csv`, `step2_positions/` |
| 2b. Ladder layout | `step2b_ladder_embed.py` | `step2b_results.csv`, `step2b_positions/` |
| 3. Emulation, decoding, repair, metrics | `step3_emulate.py` | `results.csv` |
| CPLEX vs brute force | `cplex_check.py` | `cplex_check.csv` |

Seeds: tier S, seeds 0–99 for Step 1, seeds 0–19 for Steps 2–3 and the CPLEX check. Chosen weights:
λ₂ = 10, A = Γ_max/2.

## Notes
- Emulation uses free-form atom positions; `AnalogDevice` requires a register layout for QPU submission.
- Use of AI is described in the report (Sec. 7).