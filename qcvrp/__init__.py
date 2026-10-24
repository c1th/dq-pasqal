"""qcvrp — CVRP on Pasqal neutral-atom analog devices (DuQuantum 2026, Pasqal challenge).

Pipeline (cluster-first / route-second):

    instances   -> seeded, certified-feasible CVRP instances (tiers S/M/L)
    reference   -> proven-optimal references (Held-Karp tour table + set-partition MILP)
    models      -> clustering model -> wanted couplings J, one-hot groups, decoder, repair
    embedding   -> atom positions matching C6/r^6 to lambda*J, with fidelity report
    pulses      -> pulser sequences on AnalogDevice (adiabatic baseline)
    sampling    -> run an emulator, get bitstring counts
    metrics     -> gaps, feasibility rate, P(reach), results.csv rows
    pipeline    -> glue: one (instance, model, seed) -> results row
"""
