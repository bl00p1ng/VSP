"""
Módulo que define las estructuras de datos para el problema MDVSP.
Contiene las clases que representan viajes, depósitos y la instancia completa del problema.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class Viaje:
    """Representa un viaje individual en el problema MDVSP."""
    
    id_viaje: int
    tiempo_inicio: int
    tiempo_fin: int
    
    def __post_init__(self) -> None:
        """Valida la consistencia de los tiempos del viaje."""
        if self.tiempo_inicio > self.tiempo_fin:
            raise ValueError(f"Viaje {self.id_viaje}: tiempo inicio ({self.tiempo_inicio}) "
                           f"mayor que tiempo fin ({self.tiempo_fin})")


@dataclass
class Deposito:
    """Representa un depósito con su flota de vehículos disponibles."""
    
    id_deposito: int
    numero_vehiculos: int
    
    def __post_init__(self) -> None:
        """Valida que el número de vehículos sea positivo."""
        if self.numero_vehiculos <= 0:
            raise ValueError(f"Depósito {self.id_deposito}: número de vehículos debe ser positivo")


@dataclass
class MDVSPData:
    """
    Estructura principal que contiene todos los datos de una instancia MDVSP.
    Gestiona la información de depósitos, viajes y matriz de costos.
    """
    
    nombre_archivo_instancia: str
    numero_depositos: int
    numero_viajes: int
    numero_total_vehiculos: int
    depositos: List[Deposito]
    viajes: List[Viaje]
    matriz_viajes: np.ndarray
    
    COSTO_INFACTIBLE: float = 100000000.0
    
    def __post_init__(self) -> None:
        """Valida la consistencia de los datos cargados."""
        self._validar_dimensiones()
        self._validar_matriz_costos()
    
    def _validar_dimensiones(self) -> None:
        """Valida que las dimensiones sean consistentes."""
        if self.numero_depositos <= 0:
            raise ValueError("Número de depósitos debe ser positivo")
        
        if self.numero_viajes <= 0:
            raise ValueError("Número de viajes debe ser positivo")
        
        if len(self.depositos) != self.numero_depositos:
            raise ValueError("Número de depósitos no coincide con la lista de depósitos")
        
        if len(self.viajes) != self.numero_viajes:
            raise ValueError("Número de viajes no coincide con la lista de viajes")
        
        vehiculos_total = sum(deposito.numero_vehiculos for deposito in self.depositos)
        if vehiculos_total != self.numero_total_vehiculos:
            raise ValueError("Total de vehículos no coincide con la suma por depósitos")
    
    def _validar_matriz_costos(self) -> None:
        """Valida las dimensiones y propiedades de la matriz de costos."""
        dimension_esperada = self.numero_viajes + self.numero_depositos
        
        if self.matriz_viajes.shape != (dimension_esperada, dimension_esperada):
            raise ValueError(f"Matriz de costos debe ser {dimension_esperada}x{dimension_esperada}")
        
        # Valida que la matriz tenga valores no negativos (excepto infactibles)
        costos_validos = (self.matriz_viajes >= 0) | (self.matriz_viajes == self.COSTO_INFACTIBLE)
        if not np.all(costos_validos):
            raise ValueError("Matriz de costos contiene valores inválidos")
    
    def obtener_costo(self, origen: int, destino: int) -> float:
        """
        Obtiene el costo de transición entre dos nodos.
        
        Args:
            origen: Índice del nodo origen
            destino: Índice del nodo destino
            
        Returns:
            Costo de la transición o COSTO_INFACTIBLE si no es factible
        """
        dimension = self.numero_viajes + self.numero_depositos
        
        if not (0 <= origen < dimension and 0 <= destino < dimension):
            raise IndexError("Índices de origen o destino fuera de rango")
        
        return self.matriz_viajes[origen, destino]
    
    def es_factible(self, origen: int, destino: int) -> bool:
        """
        Verifica si una transición entre dos nodos es factible.
        
        Args:
            origen: Índice del nodo origen
            destino: Índice del nodo destino
            
        Returns:
            True si la transición es factible, False en caso contrario
        """
        return self.obtener_costo(origen, destino) != self.COSTO_INFACTIBLE
    
    def calcular_estadisticas_factibilidad(self) -> dict:
        """
        Calcula estadísticas sobre la factibilidad de las transiciones.
        
        Returns:
            Diccionario con estadísticas de aristas factibles e infactibles
        """
        total_aristas = self.matriz_viajes.size
        aristas_infactibles = np.sum(self.matriz_viajes == self.COSTO_INFACTIBLE)
        aristas_factibles = total_aristas - aristas_infactibles
        
        return {
            'total_aristas': total_aristas,
            'aristas_factibles': aristas_factibles,
            'aristas_infactibles': aristas_infactibles,
            'porcentaje_infactibles': (aristas_infactibles / total_aristas) * 100.0
        }
    
    def obtener_resumen(self) -> str:
        """
        Genera un resumen textual de la instancia del problema.
        
        Returns:
            String con información resumida de la instancia
        """
        estadisticas = self.calcular_estadisticas_factibilidad()
        
        resumen = [
            f"Instancia MDVSP: {self.nombre_archivo_instancia}",
            f"Depósitos: {self.numero_depositos}",
            f"Viajes: {self.numero_viajes}",
            f"Vehículos totales: {self.numero_total_vehiculos}",
            f"Aristas factibles: {estadisticas['aristas_factibles']} "
            f"({100 - estadisticas['porcentaje_infactibles']:.2f}%)",
            f"Aristas infactibles: {estadisticas['aristas_infactibles']} "
            f"({estadisticas['porcentaje_infactibles']:.2f}%)"
        ]
        
        return "\n".join(resumen)