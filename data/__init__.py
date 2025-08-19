"""
Módulo de datos para el problema MDVSP.
Proporciona funcionalidades para carga y gestión de instancias.
"""

from .mdvsp_data_model import MDVSPData, Viaje, Deposito
from .mdvsp_data_loader import MDVSPDataLoader

__all__ = [
    'MDVSPData',
    'Viaje', 
    'Deposito',
    'MDVSPDataLoader'
]

__version__ = '1.0.0'