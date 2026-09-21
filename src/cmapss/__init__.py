"""Paquete único de features/modelado/estadística para NASA C-MAPSS FD001.

Sustituye la lógica duplicada e incompatible que existía en scripts/fase1..fase7.
Todo el pipeline (carga -> features -> modelado -> evaluación estadística) vive
aquí y se ejecuta con un único comando: `python train.py` desde la raíz del repo.
"""
