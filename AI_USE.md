# AI use disclosure (required by the brief)

Team: Christopher Thai, Alex Wang, Kevin Nguyen.

Claude (Anthropic; Claude Max plans of two team members) was used throughout the weekend:
brainstorming the formulation space, first drafts of the derivations (what the register minimises;
the one-hot-manifold argument that linear terms are constants and the load penalty collapses to a
d_i d_j repulsion), most of the code in `qcvrp/` and `scripts/`, and the first draft of the report.

What was checked rather than trusted:
* the device envelope and the pulser API, by running `scripts/check_pulser.py` against pulser 1.9.1
  (pulser's own validation caught two errors in AI-written code: a 4.99 um atom pair and a missing
  trap layout);
* every embeddability claim, by brute-forcing the realised register cost
  (`embedding.classical_ground_states`, `embedding.manifold_check`) and comparing with emulation;
* every reference optimum, by a second exact method (brute-force enumeration on S and M;
  independent dynamic programme on L, `scripts/crosscheck_L.py`);
* every derivation in the report, re-done by hand on a whiteboard by the person who presents it.

Who did what (fill in before submission):
* Christopher Thai — [emulation runs on the Mac (QuTiP, pulse sweep), decisions on model and
  report, ...]
* Alex Wang — [...]
* Kevin Nguyen — [...]

Where AI output was rejected or rewritten: [list, e.g. the first selection rule let a classical
completion solve S-0001 with zero covered customers; it was replaced by the raw-feasible rule].
