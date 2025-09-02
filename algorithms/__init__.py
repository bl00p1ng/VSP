"""
Módulo de algoritmos para problemas MDVSP y VSP.
Contiene implementaciones de heurísticas constructivas y herramientas de experimentación.
"""

from .solution_model import SolucionMDVSP, Ruta
from .concurrent_schedule import ConcurrentScheduleAlgorithm
from .experiment_runner import ExperimentRunner
from .vsp_solution_model import SolucionVSP, RutaVSP
from .vsp_constructive import VSPConstructiveAlgorithm

__all__ = [
    'SolucionMDVSP',
    'Ruta',
    'ConcurrentScheduleAlgorithm',
    'ExperimentRunner',
    'SolucionVSP',
    'RutaVSP',
    'VSPConstructiveAlgorithm'
]

__version__ = '1.1.0'