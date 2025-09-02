"""
Modelos de solución para el problema VSP (Vehicle Scheduling Problem).
Define las estructuras para representar rutas de vehículos y soluciones completas VSP.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple
import time


@dataclass
class RutaVSP:
    """Representa una ruta de vehículo en el problema VSP."""
    
    id_vehiculo: int
    servicios: List[int] = field(default_factory=list)
    costo_total: float = 0.0
    tiempo_inicio_ruta: Optional[int] = None
    tiempo_fin_ruta: Optional[int] = None
    
    def agregar_servicio(self, id_servicio: int, costo_adicional: float,
                        tiempo_inicio_servicio: int, tiempo_fin_servicio: int) -> None:
        """
        Agrega un servicio a la ruta actualizando costos y tiempos.
        
        Args:
            id_servicio: Identificador del servicio a agregar
            costo_adicional: Costo de agregar este servicio a la ruta
            tiempo_inicio_servicio: Tiempo de inicio del servicio
            tiempo_fin_servicio: Tiempo de fin del servicio
        """
        self.servicios.append(id_servicio)
        self.costo_total += costo_adicional
        
        # Actualiza ventana temporal de la ruta
        if self.tiempo_inicio_ruta is None or tiempo_inicio_servicio < self.tiempo_inicio_ruta:
            self.tiempo_inicio_ruta = tiempo_inicio_servicio
        if self.tiempo_fin_ruta is None or tiempo_fin_servicio > self.tiempo_fin_ruta:
            self.tiempo_fin_ruta = tiempo_fin_servicio
    
    def insertar_servicio(self, posicion: int, id_servicio: int, incremento_costo: float,
                         tiempo_inicio_servicio: int, tiempo_fin_servicio: int) -> None:
        """
        Inserta un servicio en una posición específica de la ruta.
        
        Args:
            posicion: Posición donde insertar el servicio (0 = al inicio)
            id_servicio: Identificador del servicio
            incremento_costo: Incremento de costo por la inserción
            tiempo_inicio_servicio: Tiempo de inicio del servicio
            tiempo_fin_servicio: Tiempo de fin del servicio
        """
        self.servicios.insert(posicion, id_servicio)
        self.costo_total += incremento_costo
        
        # Actualiza ventana temporal si es necesario
        if self.tiempo_inicio_ruta is None or tiempo_inicio_servicio < self.tiempo_inicio_ruta:
            self.tiempo_inicio_ruta = tiempo_inicio_servicio
        if self.tiempo_fin_ruta is None or tiempo_fin_servicio > self.tiempo_fin_ruta:
            self.tiempo_fin_ruta = tiempo_fin_servicio
    
    def es_vacia(self) -> bool:
        """Verifica si la ruta está vacía."""
        return len(self.servicios) == 0
    
    def numero_servicios(self) -> int:
        """Retorna el número de servicios en la ruta."""
        return len(self.servicios)
    
    def duracion_total(self) -> int:
        """
        Calcula la duración total de la ruta desde el primer al último servicio.
        
        Returns:
            Duración en unidades de tiempo o 0 si la ruta está vacía
        """
        if self.es_vacia() or self.tiempo_inicio_ruta is None or self.tiempo_fin_ruta is None:
            return 0
        return self.tiempo_fin_ruta - self.tiempo_inicio_ruta
    
    def obtener_ultimo_servicio(self) -> Optional[int]:
        """
        Obtiene el último servicio de la ruta.
        
        Returns:
            ID del último servicio o None si la ruta está vacía
        """
        return self.servicios[-1] if not self.es_vacia() else None
    
    def obtener_primer_servicio(self) -> Optional[int]:
        """
        Obtiene el primer servicio de la ruta.
        
        Returns:
            ID del primer servicio o None si la ruta está vacía
        """
        return self.servicios[0] if not self.es_vacia() else None
    
    def contiene_servicio(self, id_servicio: int) -> bool:
        """
        Verifica si la ruta contiene un servicio específico.
        
        Args:
            id_servicio: ID del servicio a buscar
            
        Returns:
            True si la ruta contiene el servicio
        """
        return id_servicio in self.servicios
    
    def obtener_resumen(self) -> str:
        """
        Genera un resumen textual de la ruta.
        
        Returns:
            String con información de la ruta
        """
        if self.es_vacia():
            return f"Ruta vacía (Vehículo {self.id_vehiculo})"
        
        servicios_str = " -> ".join(map(str, self.servicios))
        return (f"Vehículo {self.id_vehiculo}: {self.numero_servicios()} servicios "
               f"[{servicios_str}], Costo: {self.costo_total:.0f}, "
               f"Duración: {self.duracion_total()}")


@dataclass
class SolucionVSP:
    """
    Representa una solución completa del problema VSP.
    Contiene todas las rutas de vehículos y métricas de calidad.
    """
    
    nombre_instancia: str
    rutas: List[RutaVSP] = field(default_factory=list)
    servicios_asignados: Set[int] = field(default_factory=set)
    costo_total: float = 0.0
    numero_vehiculos_usados: int = 0
    tiempo_construccion: float = 0.0
    es_factible: bool = False
    
    # Métricas específicas del VSP
    numero_servicios_total: int = 0
    makespan: int = 0  # Tiempo total desde el primer al último servicio
    
    def agregar_ruta(self, ruta: RutaVSP) -> None:
        """
        Agrega una ruta a la solución.
        
        Args:
            ruta: Ruta a agregar
        """
        self.rutas.append(ruta)
        
        if not ruta.es_vacia():
            self.numero_vehiculos_usados += 1
            self.costo_total += ruta.costo_total
            self.servicios_asignados.update(ruta.servicios)
    
    def crear_nueva_ruta(self, id_vehiculo: int) -> RutaVSP:
        """
        Crea una nueva ruta vacía y la agrega a la solución.
        
        Args:
            id_vehiculo: ID del vehículo para la nueva ruta
            
        Returns:
            Nueva ruta creada
        """
        nueva_ruta = RutaVSP(id_vehiculo=id_vehiculo)
        self.rutas.append(nueva_ruta)
        return nueva_ruta
    
    def calcular_metricas(self, numero_servicios_instancia: int) -> None:
        """
        Calcula métricas de calidad de la solución VSP.
        
        Args:
            numero_servicios_instancia: Número total de servicios en la instancia
        """
        self.numero_servicios_total = numero_servicios_instancia
        
        # Recalcula totales
        self.costo_total = sum(ruta.costo_total for ruta in self.rutas)
        self.numero_vehiculos_usados = sum(1 for ruta in self.rutas if not ruta.es_vacia())
        
        # Verifica factibilidad (todos los servicios asignados)
        self.es_factible = len(self.servicios_asignados) == numero_servicios_instancia
        
        # Calcula makespan
        self._calcular_makespan()
    
    def _calcular_makespan(self) -> None:
        """Calcula el makespan (tiempo total) de la solución."""
        if not self.rutas:
            self.makespan = 0
            return
        
        tiempos_inicio = []
        tiempos_fin = []
        
        for ruta in self.rutas:
            if not ruta.es_vacia() and ruta.tiempo_inicio_ruta is not None and ruta.tiempo_fin_ruta is not None:
                tiempos_inicio.append(ruta.tiempo_inicio_ruta)
                tiempos_fin.append(ruta.tiempo_fin_ruta)
        
        if tiempos_inicio and tiempos_fin:
            self.makespan = max(tiempos_fin) - min(tiempos_inicio)
        else:
            self.makespan = 0
    
    def obtener_rutas_activas(self) -> List[RutaVSP]:
        """
        Obtiene solo las rutas que tienen servicios asignados.
        
        Returns:
            Lista de rutas no vacías
        """
        return [ruta for ruta in self.rutas if not ruta.es_vacia()]
    
    def obtener_servicios_no_asignados(self, numero_servicios_total: int) -> Set[int]:
        """
        Obtiene los servicios que no han sido asignados a ninguna ruta.
        
        Args:
            numero_servicios_total: Número total de servicios en la instancia
            
        Returns:
            Conjunto de IDs de servicios no asignados
        """
        todos_los_servicios = set(range(numero_servicios_total))
        return todos_los_servicios - self.servicios_asignados
    
    def obtener_utilizacion_vehiculos(self, vehiculos_disponibles: int) -> float:
        """
        Calcula el porcentaje de utilización de vehículos.
        
        Args:
            vehiculos_disponibles: Número total de vehículos disponibles
            
        Returns:
            Porcentaje de utilización (0-100)
        """
        if vehiculos_disponibles <= 0:
            return 0.0
        return (self.numero_vehiculos_usados / vehiculos_disponibles) * 100.0
    
    def obtener_eficiencia_promedio(self) -> float:
        """
        Calcula la eficiencia promedio de las rutas (servicios por vehículo).
        
        Returns:
            Número promedio de servicios por vehículo usado
        """
        if self.numero_vehiculos_usados <= 0:
            return 0.0
        return len(self.servicios_asignados) / self.numero_vehiculos_usados
    
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
    
    def validar_solucion(self, instancia_vsp) -> Tuple[bool, List[str]]:
        """
        Valida la consistencia de la solución VSP.
        
        Args:
            instancia_vsp: Instancia VSP para validación
            
        Returns:
            Tuple (es_valida, lista_errores)
        """
        errores = []
        
        # Verifica que todos los servicios estén asignados
        if len(self.servicios_asignados) != instancia_vsp.numero_servicios:
            servicios_faltantes = self.obtener_servicios_no_asignados(instancia_vsp.numero_servicios)
            errores.append(f"Servicios no asignados: {servicios_faltantes}")
        
        # Verifica que no haya servicios duplicados
        servicios_en_rutas = []
        for ruta in self.rutas:
            servicios_en_rutas.extend(ruta.servicios)
        
        if len(servicios_en_rutas) != len(set(servicios_en_rutas)):
            errores.append("Hay servicios duplicados en múltiples rutas")
        
        # Verifica factibilidad de cada ruta
        for i, ruta in enumerate(self.rutas):
            if ruta.es_vacia():
                continue
            
            es_factible, mensaje = instancia_vsp.validar_secuencia_servicios(ruta.servicios)
            if not es_factible:
                errores.append(f"Ruta {i} infactible: {mensaje}")
        
        # Verifica límites de vehículos
        if self.numero_vehiculos_usados > instancia_vsp.deposito.numero_vehiculos:
            errores.append(f"Excede vehículos disponibles: {self.numero_vehiculos_usados} > {instancia_vsp.deposito.numero_vehiculos}")
        
        return len(errores) == 0, errores
    
    def obtener_estadisticas_detalladas(self) -> dict:
        """
        Calcula estadísticas detalladas de la solución.
        
        Returns:
            Diccionario con estadísticas completas
        """
        rutas_activas = self.obtener_rutas_activas()
        
        if not rutas_activas:
            return {
                'numero_rutas_activas': 0,
                'servicios_por_ruta': [],
                'costos_por_ruta': [],
                'duraciones_por_ruta': [],
                'eficiencia_promedio': 0.0,
                'costo_promedio_por_servicio': 0.0,
                'utilizacion_tiempo': 0.0
            }
        
        servicios_por_ruta = [ruta.numero_servicios() for ruta in rutas_activas]
        costos_por_ruta = [ruta.costo_total for ruta in rutas_activas]
        duraciones_por_ruta = [ruta.duracion_total() for ruta in rutas_activas]
        
        return {
            'numero_rutas_activas': len(rutas_activas),
            'servicios_por_ruta': {
                'promedio': sum(servicios_por_ruta) / len(servicios_por_ruta),
                'minimo': min(servicios_por_ruta),
                'maximo': max(servicios_por_ruta)
            },
            'costos_por_ruta': {
                'promedio': sum(costos_por_ruta) / len(costos_por_ruta),
                'minimo': min(costos_por_ruta),
                'maximo': max(costos_por_ruta)
            },
            'duraciones_por_ruta': {
                'promedio': sum(duraciones_por_ruta) / len(duraciones_por_ruta),
                'minimo': min(duraciones_por_ruta),
                'maximo': max(duraciones_por_ruta)
            },
            'eficiencia_promedio': self.obtener_eficiencia_promedio(),
            'costo_promedio_por_servicio': self.costo_total / len(self.servicios_asignados) if self.servicios_asignados else 0,
            'makespan': self.makespan
        }
    
    def obtener_resumen(self) -> str:
        """
        Genera un resumen textual de la solución.
        
        Returns:
            String con información resumida de la solución
        """
        resumen = f"""
=== Solución VSP: {self.nombre_instancia} ===
Factible: {'Sí' if self.es_factible else 'No'}
Vehículos usados: {self.numero_vehiculos_usados}
Servicios asignados: {len(self.servicios_asignados)} / {self.numero_servicios_total}
Costo total: {self.costo_total:.0f}
Makespan: {self.makespan}
Tiempo construcción: {self.tiempo_construccion:.4f}s

Eficiencia: {self.obtener_eficiencia_promedio():.2f} servicios/vehículo
        """.strip()
        
        return resumen
    
    def exportar_solucion(self, archivo_salida: str = None) -> None:
        """
        Exporta la solución a un archivo de texto.
        
        Args:
            archivo_salida: Nombre del archivo de salida (opcional)
        """
        if archivo_salida is None:
            archivo_salida = f"solucion_vsp_{self.nombre_instancia}.txt"
        
        try:
            with open(archivo_salida, 'w', encoding='utf-8') as archivo:
                archivo.write(self.obtener_resumen())
                archivo.write("\n\n=== RUTAS DETALLADAS ===\n")
                
                for i, ruta in enumerate(self.obtener_rutas_activas()):
                    archivo.write(f"\n{ruta.obtener_resumen()}\n")
                
                archivo.write(f"\n=== ESTADÍSTICAS ===\n")
                stats = self.obtener_estadisticas_detalladas()
                for clave, valor in stats.items():
                    archivo.write(f"{clave}: {valor}\n")
            
            print(f"Solución exportada a: {archivo_salida}")
            
        except IOError as e:
            print(f"Error exportando solución: {e}")