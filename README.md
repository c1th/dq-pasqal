# DuQuantum × Pasqal — Track A (Team Alex and Bob)

## Environment
Python 3.10+.
    python -m venv .venv
    .venv\Scripts\Activate.ps1      # macOS/Linux: source .venv/bin/activate
    pip install -r requirements.txt

## Step 1: tier-S QUBO validation (classical, exhaustive)
    python instances.py             # self-tests of the generator and validator
    python step1_qubo_check.py      # writes step1_results.csv

Settings (top of step1_qubo_check.py): tier S, N_SEEDS = 100 (seeds 0–99),
LAMBDA2S = [6, 8, 10, 12, 15], A modes rigorous / tight / gamma_half / gamma_quarter.
Instances: instances.generate(tier, seed) with default load-ratio band [0.45, 0.95];
each instance ships an exact bin-packing certificate checked by instances.validate.

## Steps 2–3: embedding and first emulation (tier S)
    python step2_embed_check.py      # coefficient matching; writes step2_positions/, step2_results.csv
    python step2b_ladder_embed.py    # ladder layout (K'=2); writes step2b_positions/, step2b_results.csv
    python step3_emulate.py          # baseline pulse, 1000 shots; writes results.csv

### Result
Chosen weights: lambda_2 = 10, A = Gamma_max / 2 (validated empirically, not proven).
On 100 tier-S instances, the QUBO ground state satisfies the assignment constraint (C1)
on every instance, identical to the rigorous bound. At lambda_2 = 10 it is the constrained
optimum on 79, feasible but suboptimal on 19, and over capacity on 2. No lambda_2 in
{6, 8, 10, 12, 15} removes all overloads. A = Gamma_max / 4 violates (C1) for lambda_2 >= 12.
A = Gamma_max / 2 lies below the proven bound on 100 of 100 instances.
