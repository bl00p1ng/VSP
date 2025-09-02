"""
Módulo que define las estructuras de datos para el problema VSP (Vehicle Scheduling Problem).
Contiene las clases que representan servicios, depósitos y la instancia completa del problema.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple
import numpy as np


@dataclass
class Servicio:
    """Representa un servicio individual en el problema VSP."""
    
    id_servicio: int
    tiempo_inicio: int
    tiempo_fin: int
    ubicacion_inicio: Optional[str] = None
    ubicacion_fin: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Valida la consistencia de los tiempos del servicio."""
        if self.tiempo_inicio >= self.tiempo_fin:
            raise ValueError(f"Servicio {self.id_servicio}: tiempo inicio ({self.tiempo_inicio}) "
                           f"debe ser menor que tiempo fin ({self.tiempo_fin})")
    
    def duracion(self) -> int:
        """Retorna la duración del servicio en unidades de tiempo."""
        return self.tiempo_fin - self.tiempo_inicio
    
    def se_traslapa_con(self, otro_servicio: 'Servicio') -> bool:
        """
        Verifica si este servicio se traslapa temporalmente con otro.
        
        Args:
            otro_servicio: Servicio a comparar
            
        Returns:
            True si hay traslape temporal
        """
        return not (self.tiempo_fin <= otro_servicio.tiempo_inicio or 
                   otro_servicio.tiempo_fin <= self.tiempo_inicio)
    
    def puede_preceder_a(self, otro_servicio: 'Servicio') -> bool:
        """
        Verifica si este servicio puede preceder temporalmente a otro.
        
        Args:
            otro_servicio: Servicio que seguiría a este
            
        Returns:
            True si este servicio puede ejecutarse antes que el otro
        """
        return self.tiempo_fin <= otro_servicio.tiempo_inicio


@dataclass
class DepositoVSP:
    """Representa el depósito central en el problema VSP."""
    
    id_deposito: int
    numero_vehiculos: int
    nombre_deposito: Optional[str] = None
    ubicacion: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Valida que el número de vehículos sea positivo."""
        if self.numero_vehiculos <= 0:
            raise ValueError(f"Depósito {self.id_deposito}: número de vehículos debe ser positivo")


@dataclass  
class VSPData:
    """
    Estructura principal que contiene todos los datos de una instancia VSP.
    Gestiona servicios, depósito y matriz de costos con restricciones de conexión.
    """
    
    nombre_instancia: str
    numero_servicios: int
    deposito: DepositoVSP
    servicios: List[Servicio]
    matriz_costos: np.ndarray
    
    # Constantes del modelo VSP
    COSTO_INFACTIBLE: float = 100000000.0
    COSTO_PROHIBIDO: float = 0.0
    
    # Estructuras para optimización
    servicios_ordenados_por_inicio: List[int] = field(default_factory=list)
    servicios_ordenados_por_fin: List[int] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Valida la consistencia de los datos y construye estructuras auxiliares."""
        self._validar_datos()
        self._construir_estructuras_optimizacion()
        self._aplicar_restricciones_conexion()
    
    def _validar_datos(self) -> None:
        """Valida la consistencia de los datos cargados."""
        if self.numero_servicios <= 0:
            raise ValueError("Número de servicios debe ser positivo")
        
        if len(self.servicios) != self.numero_servicios:
            raise ValueError("Número de servicios no coincide con la lista de servicios")
        
        # Valida matriz de costos (servicios + depósito)
        dimension_esperada = self.numero_servicios + 1  # +1 para el depósito
        if self.matriz_costos.shape != (dimension_esperada, dimension_esperada):
            raise ValueError(f"Matriz debe ser {dimension_esperada}x{dimension_esperada}")
        
        # Reporta traslapes temporales pero no interrumpe la ejecución
        # Los traslapes se manejan correctamente en _construir_matriz_vsp marcando conexiones como infactibles
        traslapes_detectados = 0
        for i in range(len(self.servicios)):
            for j in range(i + 1, len(self.servicios)):
                if self.servicios[i].se_traslapa_con(self.servicios[j]):
                    traslapes_detectados += 1
        
        if traslapes_detectados > 0:
            print(f"Detectados {traslapes_detectados} pares de servicios con traslapes temporales "
                  f"(se marcarán como conexiones infactibles)")
    
    def _construir_estructuras_optimizacion(self) -> None:
        """Construye estructuras auxiliares para optimización de consultas."""
        # Ordena servicios por tiempo de inicio
        self.servicios_ordenados_por_inicio = sorted(
            range(self.numero_servicios),
            key=lambda i: self.servicios[i].tiempo_inicio
        )
        
        # Ordena servicios por tiempo de finalización
        self.servicios_ordenados_por_fin = sorted(
            range(self.numero_servicios),
            key=lambda i: self.servicios[i].tiempo_fin
        )
    
    def _aplicar_restricciones_conexion(self) -> None:
        """
        Aplica restricciones de conexión basadas en la matriz de costos.
        Marca como infactibles las conexiones con costo 0 o 100000000.
        """
        restricciones_aplicadas = 0
        
        for i in range(self.matriz_costos.shape[0]):
            for j in range(self.matriz_costos.shape[1]):
                costo_ij = self.matriz_costos[i, j]
                
                # Aplica restricciones bidireccionales
                if (costo_ij == self.COSTO_PROHIBIDO or 
                    costo_ij >= self.COSTO_INFACTIBLE):
                    
                    # Marca ambas direcciones como infactibles
                    self.matriz_costos[i, j] = self.COSTO_INFACTIBLE
                    self.matriz_costos[j, i] = self.COSTO_INFACTIBLE
                    restricciones_aplicadas += 1
        
        print(f"Restricciones de conexión aplicadas: {restricciones_aplicadas} pares bidireccionales")
    
    def es_conexion_factible(self, servicio_origen: int, servicio_destino: int) -> bool:
        """
        Verifica si la conexión entre dos servicios es factible.
        
        Args:
            servicio_origen: Índice del servicio origen
            servicio_destino: Índice del servicio destino
            
        Returns:
            True si la conexión es factible
        """
        if not (0 <= servicio_origen < self.numero_servicios and 
                0 <= servicio_destino < self.numero_servicios):
            raise IndexError("Índices de servicios fuera de rango")
        
        if servicio_origen == servicio_destino:
            return False
        
        # Verifica restricciones de matriz de costos
        costo = self.matriz_costos[servicio_origen, servicio_destino]
        if costo == self.COSTO_INFACTIBLE or costo == self.COSTO_PROHIBIDO:
            return False
        
        # Verifica restricciones temporales
        servicio_o = self.servicios[servicio_origen]
        servicio_d = self.servicios[servicio_destino]
        
        # No debe haber traslapes y debe respetar precedencia temporal
        return servicio_o.puede_preceder_a(servicio_d) and not servicio_o.se_traslapa_con(servicio_d)
    
    def obtener_costo_conexion(self, servicio_origen: int, servicio_destino: int) -> float:
        """
        Obtiene el costo de conexión entre dos servicios.
        
        Args:
            servicio_origen: Índice del servicio origen
            servicio_destino: Índice del servicio destino
            
        Returns:
            Costo de la conexión (100000000.0 si es infactible)
        """
        if not (0 <= servicio_origen <= self.numero_servicios and 
                0 <= servicio_destino <= self.numero_servicios):
            raise IndexError("Índices fuera de rango")
        
        return float(self.matriz_costos[servicio_origen, servicio_destino])
    
    def obtener_costo_desde_deposito(self, servicio_destino: int) -> float:
        """
        Obtiene el costo de conexión desde el depósito hacia un servicio.
        
        Args:
            servicio_destino: Índice del servicio destino
            
        Returns:
            Costo desde el depósito al servicio
        """
        if not (0 <= servicio_destino < self.numero_servicios):
            raise IndexError(f"Índice de servicio fuera de rango: {servicio_destino}")
        
        # El depósito está en el índice número_servicios
        indice_deposito = self.numero_servicios
        return float(self.matriz_costos[indice_deposito, servicio_destino])
    
    def obtener_costo_hacia_deposito(self, servicio_origen: int) -> float:
        """
        Obtiene el costo de conexión desde un servicio hacia el depósito.
        
        Args:
            servicio_origen: Índice del servicio origen
            
        Returns:
            Costo desde el servicio al depósito
        """
        if not (0 <= servicio_origen < self.numero_servicios):
            raise IndexError(f"Índice de servicio fuera de rango: {servicio_origen}")
        
        # El depósito está en el índice número_servicios
        indice_deposito = self.numero_servicios
        return float(self.matriz_costos[servicio_origen, indice_deposito])
    
    def obtener_servicios_conectables_desde(self, servicio_origen: int) -> List[int]:
        """
        Obtiene todos los servicios que pueden conectarse desde un servicio dado.
        
        Args:
            servicio_origen: Índice del servicio origen
            
        Returns:
            Lista de índices de servicios factibles como destino
        """
        if not (0 <= servicio_origen <= self.numero_servicios):
            raise IndexError(f"Índice de servicio fuera de rango: {servicio_origen}")
        
        servicios_conectables = []
        
        # Si es el depósito, puede conectar a cualquier servicio
        if servicio_origen == self.numero_servicios:  # Depósito
            for j in range(self.numero_servicios):
                if self.matriz_costos[servicio_origen, j] < self.COSTO_INFACTIBLE:
                    servicios_conectables.append(j)
        else:
            # Para servicios, incluye otros servicios y el depósito
            for j in range(self.numero_servicios + 1):
                if j != servicio_origen and self.matriz_costos[servicio_origen, j] < self.COSTO_INFACTIBLE:
                    servicios_conectables.append(j)
        
        return servicios_conectables
    
    def obtener_servicios_que_conectan_a(self, servicio_destino: int) -> List[int]:
        """
        Obtiene todos los servicios que pueden conectar hacia un servicio dado.
        
        Args:
            servicio_destino: Índice del servicio destino
            
        Returns:
            Lista de índices de servicios que pueden conectar al destino
        """
        if not (0 <= servicio_destino <= self.numero_servicios):
            raise IndexError(f"Índice de servicio fuera de rango: {servicio_destino}")
        
        servicios_conectores = []
        
        # Verifica todos los posibles orígenes (servicios + depósito)
        for i in range(self.numero_servicios + 1):
            if i != servicio_destino and self.matriz_costos[i, servicio_destino] < self.COSTO_INFACTIBLE:
                servicios_conectores.append(i)
        
        return servicios_conectores
    
    def obtener_estadisticas(self) -> dict:
        """
        Calcula estadísticas detalladas de la instancia VSP.
        
        Returns:
            Diccionario con estadísticas detalladas
        """
        # Calcula conexiones factibles
        conexiones_factibles = 0
        conexiones_totales = self.numero_servicios * (self.numero_servicios - 1)
        
        for i in range(self.numero_servicios):
            for j in range(self.numero_servicios):
                if i != j and self.es_conexion_factible(i, j):
                    conexiones_factibles += 1
        
        # Calcula ventana temporal
        tiempo_min = min(servicio.tiempo_inicio for servicio in self.servicios)
        tiempo_max = max(servicio.tiempo_fin for servicio in self.servicios)
        
        # Calcula duraciones
        duraciones = [servicio.duracion() for servicio in self.servicios]
        
        return {
            'numero_servicios': self.numero_servicios,
            'numero_vehiculos_disponibles': self.deposito.numero_vehiculos,
            'conexiones_factibles': conexiones_factibles,
            'conexiones_totales': conexiones_totales,
            'porcentaje_factibilidad': (conexiones_factibles / conexiones_totales) * 100.0 if conexiones_totales > 0 else 0,
            'ventana_temporal': (tiempo_min, tiempo_max),
            'duracion_total': tiempo_max - tiempo_min,
            'duracion_promedio_servicio': np.mean(duraciones),
            'duracion_minima_servicio': min(duraciones),
            'duracion_maxima_servicio': max(duraciones)
        }
    
    def obtener_resumen(self) -> str:
        """
        Genera un resumen textual de la instancia VSP.
        
        Returns:
            String con información resumida
        """
        stats = self.obtener_estadisticas()
        
        resumen = f"""
=== Instancia VSP: {self.nombre_instancia} ===
Servicios: {stats['numero_servicios']}
Vehículos disponibles: {stats['numero_vehiculos_disponibles']}
Ventana temporal: [{stats['ventana_temporal'][0]}, {stats['ventana_temporal'][1]}]
Duración total: {stats['duracion_total']} unidades

Factibilidad:
  - Conexiones factibles: {stats['conexiones_factibles']:,} / {stats['conexiones_totales']:,}
  - Porcentaje factible: {stats['porcentaje_factibilidad']:.2f}%

Servicios:
  - Duración promedio: {stats['duracion_promedio_servicio']:.1f}
  - Rango duración: [{stats['duracion_minima_servicio']}, {stats['duracion_maxima_servicio']}]
        """.strip()
        
        return resumen