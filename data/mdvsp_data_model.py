"""
Módulo que define las estructuras de datos para el problema MDVSP.
Contiene las clases que representan viajes, depósitos y la instancia completa del problema.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import numpy as np


@dataclass
class Viaje:
    """Representa un viaje individual en el problema MDVSP."""
    
    id_viaje: int
    tiempo_inicio: int
    tiempo_fin: int
    lugar_inicio: Optional[str] = None
    lugar_final: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Valida la consistencia de los tiempos del viaje."""
        if self.tiempo_inicio > self.tiempo_fin:
            raise ValueError(f"Viaje {self.id_viaje}: tiempo inicio ({self.tiempo_inicio}) "
                           f"mayor que tiempo fin ({self.tiempo_fin})")
    
    def duracion(self) -> int:
        """Retorna la duración del viaje en unidades de tiempo."""
        return self.tiempo_fin - self.tiempo_inicio
    
    def es_compatible_temporalmente(self, otro_viaje: 'Viaje') -> bool:
        """
        Verifica si este viaje es temporalmente compatible con otro viaje.
        
        Args:
            otro_viaje: Viaje a comparar
            
        Returns:
            True si este viaje puede ejecutarse después del otro
        """
        return self.tiempo_inicio >= otro_viaje.tiempo_fin


@dataclass
class Deposito:
    """Representa un depósito con su flota de vehículos disponibles."""
    
    id_deposito: int
    numero_vehiculos: int
    nombre_deposito: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Valida que el número de vehículos sea positivo."""
        if self.numero_vehiculos <= 0:
            raise ValueError(f"Depósito {self.id_deposito}: número de vehículos debe ser positivo")
    
    def tiene_vehiculos_disponibles(self) -> bool:
        """Verifica si el depósito tiene vehículos disponibles."""
        return self.numero_vehiculos > 0


@dataclass
class MDVSPData:
    """
    Estructura principal que contiene todos los datos de una instancia MDVSP.
    Gestiona la información de depósitos, viajes y matriz de costos con restricciones temporales.
    """
    
    nombre_archivo_instancia: str
    numero_depositos: int
    numero_viajes: int
    numero_total_vehiculos: int
    depositos: List[Deposito]
    viajes: List[Viaje]
    matriz_viajes: np.ndarray
    
    # Constantes del modelo
    COSTO_INFACTIBLE: float = 100000000.0
    
    # Estructuras adicionales para optimización
    codigos_puntos_cambio: Dict[str, int] = field(default_factory=dict)
    tiempos_desplazamiento_puntos_cambio: Optional[np.ndarray] = None
    distancias_entre_puntos_cambio: Optional[np.ndarray] = None
    
    def __post_init__(self) -> None:
        """Valida la consistencia de los datos cargados."""
        self._validar_dimensiones()
        self._validar_matriz_costos()
        self._inicializar_estructuras_optimizacion()
    
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
    
    def _inicializar_estructuras_optimizacion(self) -> None:
        """Inicializa estructuras adicionales para optimización de consultas."""
        # Precalcula índices de depósitos y viajes para acceso rápido
        self._indices_depositos = list(range(self.numero_viajes, self.numero_viajes + self.numero_depositos))
        self._indices_viajes = list(range(self.numero_viajes))
    
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
    
    def es_factible_temporalmente(self, viaje_origen: int, viaje_destino: int) -> bool:
        """
        Verifica si dos viajes son factibles temporalmente considerando tiempos y desplazamiento.
        
        Args:
            viaje_origen: Índice del viaje origen
            viaje_destino: Índice del viaje destino
            
        Returns:
            True si es factible realizar viaje_destino después de viaje_origen
        """
        if not (0 <= viaje_origen < self.numero_viajes and 0 <= viaje_destino < self.numero_viajes):
            raise IndexError("Índices de viajes fuera de rango")
        
        # Obtiene los objetos de viaje
        v_origen = self.viajes[viaje_origen]
        v_destino = self.viajes[viaje_destino]
        
        # Obtiene tiempo de desplazamiento de la matriz
        tiempo_desplazamiento = self.matriz_viajes[viaje_origen, viaje_destino]
        
        # Verifica restricción temporal: tiempo_fin_origen + tiempo_desplazamiento <= tiempo_inicio_destino
        if tiempo_desplazamiento == self.COSTO_INFACTIBLE:
            return False
        
        return v_origen.tiempo_fin + tiempo_desplazamiento <= v_destino.tiempo_inicio
    
    def obtener_viajes_compatibles(self, viaje_origen: int) -> List[int]:
        """
        Obtiene todos los viajes que pueden ejecutarse después del viaje dado.
        
        Args:
            viaje_origen: Índice del viaje origen
            
        Returns:
            Lista de índices de viajes compatibles
        """
        compatibles = []
        
        for viaje_destino in range(self.numero_viajes):
            if viaje_destino != viaje_origen and self.es_factible_temporalmente(viaje_origen, viaje_destino):
                compatibles.append(viaje_destino)
        
        return compatibles
    
    def obtener_deposito_mas_cercano(self, viaje_id: int) -> int:
        """
        Obtiene el depósito más cercano a un viaje específico.
        
        Args:
            viaje_id: Índice del viaje
            
        Returns:
            Índice del depósito más cercano
        """
        if not (0 <= viaje_id < self.numero_viajes):
            raise IndexError("Índice de viaje fuera de rango")
        
        mejor_deposito = 0
        mejor_costo = float('inf')
        
        for deposito_id in range(self.numero_depositos):
            indice_deposito = self.numero_viajes + deposito_id
            costo_ida = self.matriz_viajes[indice_deposito, viaje_id]
            costo_vuelta = self.matriz_viajes[viaje_id, indice_deposito]
            
            if costo_ida != self.COSTO_INFACTIBLE and costo_vuelta != self.COSTO_INFACTIBLE:
                costo_total = costo_ida + costo_vuelta
                if costo_total < mejor_costo:
                    mejor_costo = costo_total
                    mejor_deposito = deposito_id
        
        return mejor_deposito
    
    def calcular_estadisticas_factibilidad(self) -> dict:
        """
        Calcula estadísticas sobre la factibilidad de las transiciones.
        
        Returns:
            Diccionario con estadísticas de aristas factibles e infactibles
        """
        total_aristas = self.matriz_viajes.size
        aristas_infactibles = np.sum(self.matriz_viajes == self.COSTO_INFACTIBLE)
        aristas_factibles = total_aristas - aristas_infactibles
        
        # Estadísticas detalladas por tipo de transición
        estadisticas_detalladas = self._calcular_estadisticas_detalladas()
        
        return {
            'total_aristas': total_aristas,
            'aristas_factibles': aristas_factibles,
            'aristas_infactibles': aristas_infactibles,
            'porcentaje_infactibles': (aristas_infactibles / total_aristas) * 100.0,
            'estadisticas_detalladas': estadisticas_detalladas
        }
    
    def _calcular_estadisticas_detalladas(self) -> dict:
        """
        Calcula estadísticas detalladas por tipo de transición.
        
        Returns:
            Diccionario con estadísticas por categoría
        """
        stats = {
            'deposito_a_viaje': {'factibles': 0, 'infactibles': 0},
            'viaje_a_deposito': {'factibles': 0, 'infactibles': 0},
            'viaje_a_viaje': {'factibles': 0, 'infactibles': 0},
            'deposito_a_deposito': {'factibles': 0, 'infactibles': 0}
        }
        
        # Analiza cada tipo de transición
        for i in range(self.matriz_viajes.shape[0]):
            for j in range(self.matriz_viajes.shape[1]):
                costo = self.matriz_viajes[i, j]
                es_factible = costo != self.COSTO_INFACTIBLE
                
                # Clasifica el tipo de transición
                if i < self.numero_viajes and j < self.numero_viajes:
                    # Viaje a viaje
                    if es_factible:
                        stats['viaje_a_viaje']['factibles'] += 1
                    else:
                        stats['viaje_a_viaje']['infactibles'] += 1
                        
                elif i < self.numero_viajes and j >= self.numero_viajes:
                    # Viaje a depósito
                    if es_factible:
                        stats['viaje_a_deposito']['factibles'] += 1
                    else:
                        stats['viaje_a_deposito']['infactibles'] += 1
                        
                elif i >= self.numero_viajes and j < self.numero_viajes:
                    # Depósito a viaje
                    if es_factible:
                        stats['deposito_a_viaje']['factibles'] += 1
                    else:
                        stats['deposito_a_viaje']['infactibles'] += 1
                        
                else:
                    # Depósito a depósito
                    if es_factible:
                        stats['deposito_a_deposito']['factibles'] += 1
                    else:
                        stats['deposito_a_deposito']['infactibles'] += 1
        
        return stats
    
    def obtener_ventana_temporal_global(self) -> Tuple[int, int]:
        """
        Obtiene la ventana temporal global de todos los viajes.
        
        Returns:
            Tuple con (tiempo_inicio_minimo, tiempo_fin_maximo)
        """
        if not self.viajes:
            return (0, 0)
        
        tiempo_min = min(viaje.tiempo_inicio for viaje in self.viajes)
        tiempo_max = max(viaje.tiempo_fin for viaje in self.viajes)
        
        return (tiempo_min, tiempo_max)
    
    def obtener_viajes_en_ventana_temporal(self, tiempo_inicio: int, tiempo_fin: int) -> List[int]:
        """
        Obtiene viajes que se ejecutan dentro de una ventana temporal específica.
        
        Args:
            tiempo_inicio: Inicio de la ventana temporal
            tiempo_fin: Fin de la ventana temporal
            
        Returns:
            Lista de índices de viajes en la ventana
        """
        viajes_en_ventana = []
        
        for i, viaje in enumerate(self.viajes):
            # Verifica si hay solapamiento con la ventana temporal
            if (viaje.tiempo_inicio <= tiempo_fin and viaje.tiempo_fin >= tiempo_inicio):
                viajes_en_ventana.append(i)
        
        return viajes_en_ventana
    
    def validar_secuencia_viajes(self, secuencia_viajes: List[int]) -> bool:
        """
        Valida si una secuencia de viajes es factible temporalmente.
        
        Args:
            secuencia_viajes: Lista ordenada de índices de viajes
            
        Returns:
            True si la secuencia es factible
        """
        if len(secuencia_viajes) <= 1:
            return True
        
        for i in range(len(secuencia_viajes) - 1):
            viaje_actual = secuencia_viajes[i]
            viaje_siguiente = secuencia_viajes[i + 1]
            
            if not self.es_factible_temporalmente(viaje_actual, viaje_siguiente):
                return False
        
        return True
    
    def obtener_costo_secuencia(self, secuencia_viajes: List[int], deposito_origen: int) -> Optional[float]:
        """
        Calcula el costo total de una secuencia de viajes desde un depósito.
        
        Args:
            secuencia_viajes: Lista ordenada de índices de viajes
            deposito_origen: Índice del depósito de origen
            
        Returns:
            Costo total o None si la secuencia es infactible
        """
        if deposito_origen >= self.numero_depositos:
            raise IndexError("Índice de depósito fuera de rango")
        
        if not secuencia_viajes:
            return 0.0
        
        # Valida que la secuencia sea factible
        if not self.validar_secuencia_viajes(secuencia_viajes):
            return None
        
        costo_total = 0.0
        indice_deposito = self.numero_viajes + deposito_origen
        
        # Costo desde depósito al primer viaje
        primer_viaje = secuencia_viajes[0]
        costo_inicial = self.matriz_viajes[indice_deposito, primer_viaje]
        if costo_inicial == self.COSTO_INFACTIBLE:
            return None
        costo_total += costo_inicial
        
        # Costos entre viajes consecutivos
        for i in range(len(secuencia_viajes) - 1):
            viaje_actual = secuencia_viajes[i]
            viaje_siguiente = secuencia_viajes[i + 1]
            
            costo_transicion = self.matriz_viajes[viaje_actual, viaje_siguiente]
            if costo_transicion == self.COSTO_INFACTIBLE:
                return None
            costo_total += costo_transicion
        
        # Costo desde último viaje al depósito
        ultimo_viaje = secuencia_viajes[-1]
        costo_final = self.matriz_viajes[ultimo_viaje, indice_deposito]
        if costo_final == self.COSTO_INFACTIBLE:
            return None
        costo_total += costo_final
        
        return costo_total
    
    def obtener_resumen(self) -> str:
        """
        Genera un resumen textual de la instancia del problema.
        
        Returns:
            String con información resumida de la instancia
        """
        estadisticas = self.calcular_estadisticas_factibilidad()
        ventana_temporal = self.obtener_ventana_temporal_global()
        
        resumen = f"""
=== Instancia MDVSP: {self.nombre_archivo_instancia} ===
Depósitos: {self.numero_depositos}
Viajes: {self.numero_viajes}
Total vehículos: {self.numero_total_vehiculos}
Ventana temporal: [{ventana_temporal[0]}, {ventana_temporal[1]}]

Matriz de costos:
  - Dimensión: {self.matriz_viajes.shape[0]}x{self.matriz_viajes.shape[1]}
  - Aristas factibles: {estadisticas['aristas_factibles']:,}
  - Aristas infactibles: {estadisticas['aristas_infactibles']:,}
  - Porcentaje infactible: {estadisticas['porcentaje_infactibles']:.2f}%

Depósitos:"""
        
        for deposito in self.depositos:
            resumen += f"\n  - Depósito {deposito.id_deposito}: {deposito.numero_vehiculos} vehículos"
        
        return resumen.strip()
    
    def exportar_matriz_csv(self, archivo_salida: str = None) -> None:
        """
        Exporta la matriz de costos a un archivo CSV.
        
        Args:
            archivo_salida: Nombre del archivo de salida (opcional)
        """
        if archivo_salida is None:
            archivo_salida = f"{self.nombre_archivo_instancia}_matriz_costos.csv"
        
        try:
            np.savetxt(archivo_salida, self.matriz_viajes, delimiter=';', fmt='%.0f')
            print(f"Matriz exportada a: {archivo_salida}")
        except IOError as e:
            print(f"Error exportando matriz: {e}")
    
    def obtener_estadisticas_rendimiento(self) -> dict:
        """
        Obtiene estadísticas útiles para análisis de rendimiento de algoritmos.
        
        Returns:
            Diccionario con métricas de rendimiento
        """
        estadisticas_fact = self.calcular_estadisticas_factibilidad()
        ventana_temporal = self.obtener_ventana_temporal_global()
        
        # Calcula densidad de la matriz
        total_elementos = self.matriz_viajes.size
        elementos_factibles = estadisticas_fact['aristas_factibles']
        densidad = elementos_factibles / total_elementos if total_elementos > 0 else 0
        
        # Analiza distribución temporal
        duraciones_viajes = [viaje.duracion() for viaje in self.viajes]
        duracion_promedio = np.mean(duraciones_viajes) if duraciones_viajes else 0
        
        return {
            'densidad_matriz': densidad,
            'complejidad_temporal': ventana_temporal[1] - ventana_temporal[0],
            'duracion_promedio_viajes': duracion_promedio,
            'ratio_vehiculos_viajes': self.numero_total_vehiculos / self.numero_viajes,
            'factibilidad_por_deposito': self._calcular_factibilidad_por_deposito()
        }
    
    def _calcular_factibilidad_por_deposito(self) -> dict:
        """
        Calcula métricas de factibilidad específicas por depósito.
        
        Returns:
            Diccionario con factibilidad por depósito
        """
        factibilidad_depositos = {}
        
        for deposito_id in range(self.numero_depositos):
            indice_deposito = self.numero_viajes + deposito_id
            
            # Cuenta viajes accesibles desde este depósito
            viajes_accesibles = 0
            for viaje_id in range(self.numero_viajes):
                if self.matriz_viajes[indice_deposito, viaje_id] != self.COSTO_INFACTIBLE:
                    viajes_accesibles += 1
            
            factibilidad_depositos[deposito_id] = {
                'viajes_accesibles': viajes_accesibles,
                'porcentaje_accesible': (viajes_accesibles / self.numero_viajes) * 100.0
            }
        
        return factibilidad_depositos