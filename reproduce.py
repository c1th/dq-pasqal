"""One-command reproduction.

    python reproduce.py          headline result: tier-S emulation table (results.csv), about 6-8 minutes
    python reproduce.py --all    every step: instances, weights, embeddings, emulation, CPLEX cross-check

The headline steps regenerate the atom positions (Steps 2 and 2b) and then emulate them (Step 3).
"""
import sys
import step2_embed_check, step2b_ladder_embed, step3_emulate

if __name__ == "__main__":
    if "--all" in sys.argv:
        import instances, export_instances, step1_qubo_check, cplex_check
        print("== instances: self-tests and export =="); export_instances.main()
        print("\n== Step 1: tier-S penalty weights =="); step1_qubo_check.main()
    print("\n== Step 2: coefficient matching and weighted log fit =="); step2_embed_check.main()
    print("\n== Step 2b: ladder layout =="); step2b_ladder_embed.main()
    print("\n== Step 3: emulation (headline result) =="); step3_emulate.main()
    if "--all" in sys.argv:
        print("\n== CPLEX cross-check =="); cplex_check.main()