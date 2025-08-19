"""
Modelos de solución para el problema MDVSP.
Define las estructuras para representar rutas de vehículos y soluciones completas.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set
import time


@dataclass
class Ruta:
    """Representa una ruta asignada a un vehículo específico."""
    
    id_vehiculo: int
    id_deposito: int
    viajes: List[int] = field(default_factory=list)
    costo_total: float = 0.0
    tiempo_inicio: Optional[int] = None
    tiempo_fin: Optional[int] = None
    
    def agregar_viaje(self, id_viaje: int, costo_adicional: float, 
                     tiempo_inicio_viaje: int, tiempo_fin_viaje: int) -> None:
        """
        Agrega un viaje a la ruta actualizando costos y tiempos.
        
        Args:
            id_viaje: Identificador del viaje a agregar
            costo_adicional: Costo de agregar este viaje a la ruta
            tiempo_inicio_viaje: Tiempo de inicio del viaje
            tiempo_fin_viaje: Tiempo de fin del viaje
        """
        self.viajes.append(id_viaje)
        self.costo_total += costo_adicional
        
        # Actualiza ventana temporal de la ruta
        if self.tiempo_inicio is None or tiempo_inicio_viaje < self.tiempo_inicio:
            self.tiempo_inicio = tiempo_inicio_viaje
        if self.tiempo_fin is None or tiempo_fin_viaje > self.tiempo_fin:
            self.tiempo_fin = tiempo_fin_viaje
    
    def es_vacia(self) -> bool:
        """Verifica si la ruta está vacía."""
        return len(self.viajes) == 0
    
    def numero_viajes(self) -> int:
        """Retorna el número de viajes en la ruta."""
        return len(self.viajes)
    
    def obtener_resumen(self) -> str:
        """
        Genera un resumen textual de la ruta.
        
        Returns:
            String con información de la ruta
        """
        if self.es_vacia():
            return f"Ruta vacía (Vehículo {self.id_vehiculo}, Depósito {self.id_deposito})"
        
        return (f"Vehículo {self.id_vehiculo} (Dep. {self.id_deposito}): "
               f"{self.numero_viajes()} viajes, Costo: {self.costo_total:.0f}, "
               f"Tiempo: [{self.tiempo_inicio}-{self.tiempo_fin}]")


@dataclass
class SolucionMDVSP:
    """
    Representa una solución completa del problema MDVSP.
    Contiene todas las rutas de vehículos y métricas de calidad.
    """
    
    nombre_instancia: str
    rutas: List[Ruta] = field(default_factory=list)
    viajes_asignados: Set[int] = field(default_factory=set)
    costo_total: float = 0.0
    numero_vehiculos_usados: int = 0
    tiempo_construccion: float = 0.0
    es_factible: bool = False
    
    def agregar_ruta(self, ruta: Ruta) -> None:
        """
        Agrega una ruta a la solución.
        
        Args:
            ruta: Ruta a agregar
        """
        self.rutas.append(ruta)
        
        if not ruta.es_vacia():
            self.numero_vehiculos_usados += 1
            self.costo_total += ruta.costo_total
            self.viajes_asignados.update(ruta.viajes)
    
    def calcular_metricas(self, numero_total_viajes: int) -> None:
        """
        Calcula métricas de calidad de la solución.
        
        Args:
            numero_total_viajes: Número total de viajes en la instancia
        """
        # Recalcula totales
        self.costo_total = sum(ruta.costo_total for ruta in self.rutas)
        self.numero_vehiculos_usados = sum(1 for ruta in self.rutas if not ruta.es_vacia())
        
        # Verifica factibilidad
        self.es_factible = len(self.viajes_asignados) == numero_total_viajes
    
    def obtener_rutas_activas(self) -> List[Ruta]:
        """
        Obtiene solo las rutas que tienen viajes asignados.
        
        Returns:
            Lista de rutas no vacías
        """
        return [ruta for ruta in self.rutas if not ruta.es_vacia()]
    
    def obtener_gap(self, mejor_conocido: Optional[float] = None) -> Optional[float]:
        """
        Calcula el GAP respecto al mejor valor conocido.
        
        Args:
            mejor_conocido: Mejor valor conocido para la instancia
            
        Returns:
            GAP en porcentaje o None si no hay referencia
        """
        if mejor_conocido is None or mejor_conocido == 0:
            return None
        
        return ((self.costo_total - mejor_conocido) / mejor_conocido) * 100.0
    
    def obtener_estadisticas(self) -> dict:
        """
        Calcula estadísticas detalladas de la solución.
        
        Returns:
            Diccionario con estadísticas
        """
        rutas_activas = self.obtener_rutas_activas()
        
        if not rutas_activas:
            return {
                'numero_rutas_activas': 0,
                'viajes_por_ruta_promedio': 0,
                'viajes_por_ruta_max': 0,
                'viajes_por_ruta_min': 0,
                'costo_por_vehiculo_promedio': 0,
                'utilizacion_vehiculos': 0
            }
        
        viajes_por_ruta = [ruta.numero_viajes() for ruta in rutas_activas]
        costos_por_ruta = [ruta.costo_total for ruta in rutas_activas]
        
        return {
            'numero_rutas_activas': len(rutas_activas),
            'viajes_por_ruta_promedio': sum(viajes_por_ruta) / len(viajes_por_ruta),
            'viajes_por_ruta_max': max(viajes_por_ruta),
            'viajes_por_ruta_min': min(viajes_por_ruta),
            'costo_por_vehiculo_promedio': sum(costos_por_ruta) / len(costos_por_ruta),
            'utilizacion_vehiculos': (len(rutas_activas) / len(self.rutas)) * 100.0
        }
    
    def obtener_resumen(self) -> str:
        """
        Genera un resumen textual completo de la solución.
        
        Returns:
            String con información detallada de la solución
        """
        estadisticas = self.obtener_estadisticas()
        
        resumen = [
            f"=== Solución MDVSP: {self.nombre_instancia} ===",
            f"Factible: {'✓ Sí' if self.es_factible else '✗ No'}",
            f"Costo Total: {self.costo_total:.0f}",
            f"Vehículos Usados: {self.numero_vehiculos_usados} / {len(self.rutas)}",
            f"Viajes Asignados: {len(self.viajes_asignados)}",
            f"Tiempo Construcción: {self.tiempo_construccion:.4f}s",
            "",
            f"Estadísticas:",
            f"  - Rutas Activas: {estadisticas['numero_rutas_activas']}",
            f"  - Viajes/Ruta (prom): {estadisticas['viajes_por_ruta_promedio']:.1f}",
            f"  - Viajes/Ruta (rango): [{estadisticas['viajes_por_ruta_min']}-{estadisticas['viajes_por_ruta_max']}]",
            f"  - Costo/Vehículo (prom): {estadisticas['costo_por_vehiculo_promedio']:.0f}",
            f"  - Utilización: {estadisticas['utilizacion_vehiculos']:.1f}%"
        ]
        
        return "\n".join(resumen)
    
    def obtener_detalle_rutas(self, mostrar_max: int = 10) -> str:
        """
        Genera un detalle de las rutas activas.
        
        Args:
            mostrar_max: Número máximo de rutas a mostrar
            
        Returns:
            String con detalle de rutas
        """
        rutas_activas = self.obtener_rutas_activas()
        
        if not rutas_activas:
            return "No hay rutas activas."
        
        # Ordena por costo descendente
        rutas_ordenadas = sorted(rutas_activas, key=lambda r: r.costo_total, reverse=True)
        
        detalle = [f"Top {min(mostrar_max, len(rutas_ordenadas))} rutas por costo:"]
        
        for i, ruta in enumerate(rutas_ordenadas[:mostrar_max]):
            detalle.append(f"  {i+1:2d}. {ruta.obtener_resumen()}")
            if len(ruta.viajes) <= 10:
                detalle.append(f"      Viajes: {ruta.viajes}")
            else:
                detalle.append(f"      Viajes: {ruta.viajes[:5]} ... {ruta.viajes[-5:]}")
        
        if len(rutas_ordenadas) > mostrar_max:
            detalle.append(f"  ... y {len(rutas_ordenadas) - mostrar_max} rutas más")
        
        return "\n".join(detalle)