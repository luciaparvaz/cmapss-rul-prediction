"""Single feature/modeling/statistics package for NASA C-MAPSS FD001.

Replaces the duplicated, mutually incompatible logic that used to live across
scripts/phase1..phase7. The whole pipeline (load -> features -> modeling ->
statistical evaluation) lives here and runs with a single command:
`python train.py` from the repo root.
"""
