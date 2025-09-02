"""
Módulo de datos para problemas MDVSP y VSP.
Proporciona funcionalidades para carga y gestión de instancias.
"""

from .mdvsp_data_model import MDVSPData, Viaje, Deposito
from .mdvsp_data_loader import MDVSPDataLoader

from .vsp_data_model import VSPData, Servicio, DepositoVSP
from .vsp_data_loader import VSPDataLoader

__all__ = [
    'MDVSPData',
    'Viaje', 
    'Deposito',
    'MDVSPDataLoader',
    'VSPData',
    'Servicio',
    'DepositoVSP',
    'VSPDataLoader'
]

__version__ = '1.1.0'