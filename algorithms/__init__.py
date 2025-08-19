"""
Módulo de algoritmos para el problema MDVSP.
Contiene implementaciones de heurísticas constructivas y herramientas de experimentación.
"""

from .solution_model import SolucionMDVSP, Ruta
from .concurrent_schedule import ConcurrentScheduleAlgorithm
from .experiment_runner import ExperimentRunner

__all__ = [
    'SolucionMDVSP',
    'Ruta',
    'ConcurrentScheduleAlgorithm',
    'ExperimentRunner'
]

__version__ = '1.0.0'