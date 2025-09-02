"""
Algoritmo constructivo optimizado para el problema VSP (Vehicle Scheduling Problem).
Implementa una heurística greedy que minimiza el número de vehículos y el costo total.
"""

import time
from typing import List, Tuple, Optional
from memory_profiler import profile
import numpy as np

from data.vsp_data_model import VSPData, Servicio
from algorithms.vsp_solution_model import SolucionVSP, RutaVSP


class VSPConstructiveAlgorithm:
    """
    Algoritmo constructivo para VSP basado en estrategia greedy con múltiples criterios.
    Prioriza minimizar vehículos usados y luego minimizar costo total.
    """
    
    def __init__(self) -> None:
        """Inicializa el algoritmo constructivo VSP."""
        self.estadisticas = {
            'servicios_procesados': 0,
            'rutas_creadas': 0,
            'insercciones_realizadas': 0,
            'evaluaciones_factibilidad': 0
        }
    
    @profile
    def resolver(self, instancia: VSPData, estrategia: str = "tiempo_inicio") -> SolucionVSP:
        """
        Resuelve una instancia VSP usando el algoritmo constructivo.
        
        Args:
            instancia: Datos de la instancia VSP
            estrategia: Estrategia de ordenamiento ("tiempo_inicio", "tiempo_fin", "duracion")
            
        Returns:
            Solución VSP construida
        """
        inicio_tiempo = time.perf_counter()
        
        print(f"=== Iniciando VSP Constructivo ===")
        print(f"Instancia: {instancia.nombre_instancia}")
        print(f"Servicios: {instancia.numero_servicios}")
        print(f"Vehículos disponibles: {instancia.deposito.numero_vehiculos}")
        print(f"Estrategia: {estrategia}")
        
        # Reinicia estadísticas
        self._reiniciar_estadisticas()
        
        # Crea solución vacía
        solucion = SolucionVSP(nombre_instancia=instancia.nombre_instancia)
        
        # Ordena servicios según estrategia
        servicios_ordenados = self._ordenar_servicios(instancia, estrategia)
        
        # Procesa cada servicio
        for id_servicio in servicios_ordenados:
            self._procesar_servicio(instancia, solucion, id_servicio)
            self.estadisticas['servicios_procesados'] += 1
        
        # Calcula métricas finales
        tiempo_total = time.perf_counter() - inicio_tiempo
        solucion.tiempo_construccion = tiempo_total
        solucion.calcular_metricas(instancia.numero_servicios)
        
        # Imprime resultados
        self._imprimir_resultados(solucion, tiempo_total)
        
        return solucion
    
    def _reiniciar_estadisticas(self) -> None:
        """Reinicia las estadísticas del algoritmo."""
        for clave in self.estadisticas:
            self.estadisticas[clave] = 0
    
    def _ordenar_servicios(self, instancia: VSPData, estrategia: str) -> List[int]:
        """
        Ordena los servicios según la estrategia especificada.
        
        Args:
            instancia: Instancia VSP
            estrategia: Estrategia de ordenamiento
            
        Returns:
            Lista de IDs de servicios ordenados
        """
        servicios = instancia.servicios
        
        if estrategia == "tiempo_inicio":
            # Ordena por tiempo de inicio (Earliest Start Time First)
            return sorted(range(len(servicios)), key=lambda i: servicios[i].tiempo_inicio)
        
        elif estrategia == "tiempo_fin":
            # Ordena por tiempo de fin (Earliest Finish Time First)
            return sorted(range(len(servicios)), key=lambda i: servicios[i].tiempo_fin)
        
        elif estrategia == "duracion":
            # Ordena por duración (Shortest Processing Time First)
            return sorted(range(len(servicios)), key=lambda i: servicios[i].duracion())
        
        elif estrategia == "mixta":
            # Estrategia mixta: tiempo inicio + duración
            return sorted(range(len(servicios)), 
                         key=lambda i: (servicios[i].tiempo_inicio, servicios[i].duracion()))
        
        else:
            raise ValueError(f"Estrategia desconocida: {estrategia}")
    
    def _procesar_servicio(self, instancia: VSPData, solucion: SolucionVSP, id_servicio: int) -> None:
        """
        Procesa un servicio individual intentando asignarlo a la mejor ruta disponible.
        
        Args:
            instancia: Instancia VSP
            solucion: Solución en construcción
            id_servicio: ID del servicio a procesar
        """
        servicio = instancia.servicios[id_servicio]
        
        # Busca la mejor opción de asignación entre rutas existentes
        mejor_opcion = self._encontrar_mejor_asignacion(instancia, solucion, id_servicio)
        
        if mejor_opcion is not None:
            # Asigna a ruta existente
            id_ruta, posicion, costo_incremental = mejor_opcion
            self._asignar_a_ruta_existente(instancia, solucion, id_servicio, id_ruta, posicion, costo_incremental)
            self.estadisticas['insercciones_realizadas'] += 1
        
        else:
            # Crea nueva ruta
            self._crear_nueva_ruta(instancia, solucion, id_servicio)
            self.estadisticas['rutas_creadas'] += 1
    
    def _encontrar_mejor_asignacion(self, instancia: VSPData, solucion: SolucionVSP, 
                                   id_servicio: int) -> Optional[Tuple[int, int, float]]:
        """
        Encuentra la mejor asignación para un servicio entre rutas existentes.
        
        Args:
            instancia: Instancia VSP
            solucion: Solución actual
            id_servicio: ID del servicio a asignar
            
        Returns:
            Tuple (id_ruta, posicion, costo_incremental) o None si no es factible
        """
        mejor_opcion = None
        mejor_costo = float('inf')
        
        # Evalúa cada ruta existente
        for i, ruta in enumerate(solucion.rutas):
            if ruta.es_vacia():
                continue
            
            # Encuentra la mejor posición en esta ruta
            opcion_ruta = self._evaluar_insercion_en_ruta(instancia, ruta, id_servicio)
            
            if opcion_ruta is not None:
                posicion, costo_incremental = opcion_ruta
                
                # Aplica criterio de selección: minimiza costo incremental
                if costo_incremental < mejor_costo:
                    mejor_costo = costo_incremental
                    mejor_opcion = (i, posicion, costo_incremental)
        
        return mejor_opcion
    
    def _evaluar_insercion_en_ruta(self, instancia: VSPData, ruta: RutaVSP, 
                                  id_servicio: int) -> Optional[Tuple[int, float]]:
        """
        Evalúa las posibles inserciones de un servicio en una ruta específica.
        
        Args:
            instancia: Instancia VSP
            ruta: Ruta donde evaluar la inserción
            id_servicio: ID del servicio a insertar
            
        Returns:
            Tuple (mejor_posicion, costo_incremental) o None si no es factible
        """
        servicio = instancia.servicios[id_servicio]
        mejor_posicion = None
        mejor_costo = float('inf')
        
        # Evalúa inserción en cada posición posible
        for posicion in range(len(ruta.servicios) + 1):
            costo_incremental = self._calcular_costo_insercion(instancia, ruta, id_servicio, posicion)
            self.estadisticas['evaluaciones_factibilidad'] += 1
            
            if costo_incremental is not None and costo_incremental < mejor_costo:
                # Verifica factibilidad temporal de la inserción
                if self._es_insercion_factible(instancia, ruta, id_servicio, posicion):
                    mejor_costo = costo_incremental
                    mejor_posicion = posicion
        
        return (mejor_posicion, mejor_costo) if mejor_posicion is not None else None
    
    def _calcular_costo_insercion(self, instancia: VSPData, ruta: RutaVSP, 
                                 id_servicio: int, posicion: int) -> Optional[float]:
        """
        Calcula el costo incremental de insertar un servicio en una posición específica.
        
        Args:
            instancia: Instancia VSP
            ruta: Ruta donde insertar
            id_servicio: ID del servicio a insertar
            posicion: Posición de inserción (0 = al inicio)
            
        Returns:
            Costo incremental o None si no es factible
        """
        if posicion < 0 or posicion > len(ruta.servicios):
            return None
        
        # Caso: inserción al final de la ruta
        if posicion == len(ruta.servicios):
            return self._calcular_costo_insercion_final(instancia, ruta, id_servicio)
        
        # Caso: inserción al inicio de la ruta  
        elif posicion == 0:
            return self._calcular_costo_insercion_inicio(instancia, ruta, id_servicio)
        
        # Caso: inserción en el medio de la ruta
        else:
            return self._calcular_costo_insercion_medio(instancia, ruta, id_servicio, posicion)
    
    def _calcular_costo_insercion_final(self, instancia: VSPData, ruta: RutaVSP, 
                                       id_servicio: int) -> Optional[float]:
        """Calcula costo de inserción al final de una ruta."""
        if ruta.es_vacia():
            # Ruta vacía: depósito -> servicio -> depósito
            costo_ida = instancia.obtener_costo_desde_deposito(id_servicio)
            costo_vuelta = instancia.obtener_costo_hacia_deposito(id_servicio)
            
            if (costo_ida >= instancia.COSTO_INFACTIBLE or 
                costo_vuelta >= instancia.COSTO_INFACTIBLE):
                return None
            
            return costo_ida + costo_vuelta
        
        else:
            # Inserción después del último servicio
            ultimo_servicio = ruta.obtener_ultimo_servicio()
            
            # Costo original: último_servicio -> depósito
            costo_original = instancia.obtener_costo_hacia_deposito(ultimo_servicio)
            
            # Costo nuevo: último_servicio -> nuevo_servicio -> depósito
            costo_conexion = instancia.obtener_costo_conexion(ultimo_servicio, id_servicio)
            costo_regreso = instancia.obtener_costo_hacia_deposito(id_servicio)
            
            if (costo_conexion >= instancia.COSTO_INFACTIBLE or 
                costo_regreso >= instancia.COSTO_INFACTIBLE or
                costo_original >= instancia.COSTO_INFACTIBLE):
                return None
            
            return costo_conexion + costo_regreso - costo_original
    
    def _calcular_costo_insercion_inicio(self, instancia: VSPData, ruta: RutaVSP, 
                                        id_servicio: int) -> Optional[float]:
        """Calcula costo de inserción al inicio de una ruta."""
        primer_servicio = ruta.obtener_primer_servicio()
        
        # Costo original: depósito -> primer_servicio
        costo_original = instancia.obtener_costo_desde_deposito(primer_servicio)
        
        # Costo nuevo: depósito -> nuevo_servicio -> primer_servicio
        costo_ida = instancia.obtener_costo_desde_deposito(id_servicio)
        costo_conexion = instancia.obtener_costo_conexion(id_servicio, primer_servicio)
        
        if (costo_ida >= instancia.COSTO_INFACTIBLE or 
            costo_conexion >= instancia.COSTO_INFACTIBLE or
            costo_original >= instancia.COSTO_INFACTIBLE):
            return None
        
        return costo_ida + costo_conexion - costo_original
    
    def _calcular_costo_insercion_medio(self, instancia: VSPData, ruta: RutaVSP, 
                                       id_servicio: int, posicion: int) -> Optional[float]:
        """Calcula costo de inserción en el medio de una ruta."""
        servicio_anterior = ruta.servicios[posicion - 1]
        servicio_siguiente = ruta.servicios[posicion]
        
        # Costo original: servicio_anterior -> servicio_siguiente
        costo_original = instancia.obtener_costo_conexion(servicio_anterior, servicio_siguiente)
        
        # Costo nuevo: servicio_anterior -> nuevo_servicio -> servicio_siguiente
        costo_entrada = instancia.obtener_costo_conexion(servicio_anterior, id_servicio)
        costo_salida = instancia.obtener_costo_conexion(id_servicio, servicio_siguiente)
        
        if (costo_entrada >= instancia.COSTO_INFACTIBLE or 
            costo_salida >= instancia.COSTO_INFACTIBLE or
            costo_original >= instancia.COSTO_INFACTIBLE):
            return None
        
        return costo_entrada + costo_salida - costo_original
    
    def _es_insercion_factible(self, instancia: VSPData, ruta: RutaVSP, 
                              id_servicio: int, posicion: int) -> bool:
        """
        Verifica si la inserción de un servicio en una posición es temporalmente factible.
        
        Args:
            instancia: Instancia VSP
            ruta: Ruta donde insertar
            id_servicio: ID del servicio a insertar
            posicion: Posición de inserción
            
        Returns:
            True si la inserción es factible
        """
        servicio = instancia.servicios[id_servicio]
        
        # Verifica con servicio anterior (si existe)
        if posicion > 0:
            servicio_anterior = ruta.servicios[posicion - 1]
            if not instancia.es_conexion_factible(servicio_anterior, id_servicio):
                return False
        
        # Verifica con servicio siguiente (si existe)
        if posicion < len(ruta.servicios):
            servicio_siguiente = ruta.servicios[posicion]
            if not instancia.es_conexion_factible(id_servicio, servicio_siguiente):
                return False
        
        return True
    
    def _asignar_a_ruta_existente(self, instancia: VSPData, solucion: SolucionVSP, 
                                 id_servicio: int, id_ruta: int, posicion: int, 
                                 costo_incremental: float) -> None:
        """
        Asigna un servicio a una ruta existente en la posición especificada.
        
        Args:
            instancia: Instancia VSP
            solucion: Solución en construcción
            id_servicio: ID del servicio a asignar
            id_ruta: ID de la ruta destino
            posicion: Posición de inserción
            costo_incremental: Costo incremental de la asignación
        """
        ruta = solucion.rutas[id_ruta]
        servicio = instancia.servicios[id_servicio]
        
        # Inserta el servicio en la ruta
        ruta.insertar_servicio(
            posicion=posicion,
            id_servicio=id_servicio,
            incremento_costo=costo_incremental,
            tiempo_inicio_servicio=servicio.tiempo_inicio,
            tiempo_fin_servicio=servicio.tiempo_fin
        )
        
        # Actualiza la solución
        solucion.servicios_asignados.add(id_servicio)
        solucion.costo_total += costo_incremental
    
    def _crear_nueva_ruta(self, instancia: VSPData, solucion: SolucionVSP, id_servicio: int) -> None:
        """
        Crea una nueva ruta con el servicio dado.
        
        Args:
            instancia: Instancia VSP
            solucion: Solución en construcción
            id_servicio: ID del servicio para la nueva ruta
        """
        # Verifica que haya vehículos disponibles
        if len(solucion.rutas) >= instancia.deposito.numero_vehiculos:
            raise RuntimeError(f"No hay vehículos disponibles para nueva ruta (servicio {id_servicio})")
        
        # Crea nueva ruta
        id_vehiculo = len(solucion.rutas)
        nueva_ruta = RutaVSP(id_vehiculo=id_vehiculo)
        
        # Calcula costo de la nueva ruta: depósito -> servicio -> depósito
        costo_ida = instancia.obtener_costo_desde_deposito(id_servicio)
        costo_vuelta = instancia.obtener_costo_hacia_deposito(id_servicio)
        
        if (costo_ida >= instancia.COSTO_INFACTIBLE or 
            costo_vuelta >= instancia.COSTO_INFACTIBLE):
            raise RuntimeError(f"Servicio {id_servicio} no puede conectarse con depósito")
        
        costo_total_ruta = costo_ida + costo_vuelta
        servicio = instancia.servicios[id_servicio]
        
        # Agrega el servicio a la nueva ruta
        nueva_ruta.agregar_servicio(
            id_servicio=id_servicio,
            costo_adicional=costo_total_ruta,
            tiempo_inicio_servicio=servicio.tiempo_inicio,
            tiempo_fin_servicio=servicio.tiempo_fin
        )
        
        # Agrega la ruta a la solución
        solucion.rutas.append(nueva_ruta)
        solucion.servicios_asignados.add(id_servicio)
        solucion.costo_total += costo_total_ruta
    
    def _imprimir_resultados(self, solucion: SolucionVSP, tiempo_total: float) -> None:
        """
        Imprime los resultados del algoritmo constructivo.
        
        Args:
            solucion: Solución construida
            tiempo_total: Tiempo total de construcción
        """
        print(f"\n=== Resultados VSP Constructivo ===")
        print(f"Factible: {'Sí' if solucion.es_factible else 'No'}")
        print(f"Vehículos usados: {solucion.numero_vehiculos_usados}")
        print(f"Servicios asignados: {len(solucion.servicios_asignados)}")
        print(f"Costo total: {solucion.costo_total:.0f}")
        print(f"Makespan: {solucion.makespan}")
        print(f"Tiempo construcción: {tiempo_total:.4f}s")
        print(f"Eficiencia: {solucion.obtener_eficiencia_promedio():.2f} servicios/vehículo")
        
        print(f"\n=== Estadísticas del Algoritmo ===")
        for clave, valor in self.estadisticas.items():
            print(f"{clave}: {valor:,}")
        
        print(f"\n=== Rutas Construidas ===")
        for i, ruta in enumerate(solucion.obtener_rutas_activas()):
            print(f"Ruta {i}: {ruta.obtener_resumen()}")
    
    def resolver_con_multiples_estrategias(self, instancia: VSPData) -> SolucionVSP:
        """
        Resuelve la instancia con múltiples estrategias y retorna la mejor solución.
        
        Args:
            instancia: Instancia VSP a resolver
            
        Returns:
            Mejor solución encontrada entre todas las estrategias
        """
        estrategias = ["tiempo_inicio", "tiempo_fin", "duracion", "mixta"]
        mejor_solucion = None
        mejor_costo = float('inf')
        
        print(f"=== Ejecutando Múltiples Estrategias ===")
        
        for estrategia in estrategias:
            print(f"\n--- Estrategia: {estrategia} ---")
            
            try:
                solucion = self.resolver(instancia, estrategia)
                
                # Evalúa la solución (prioriza vehículos, luego costo)
                criterio_evaluacion = (solucion.numero_vehiculos_usados, solucion.costo_total)
                criterio_mejor = (mejor_solucion.numero_vehiculos_usados, mejor_solucion.costo_total) if mejor_solucion else (float('inf'), float('inf'))
                
                if criterio_evaluacion < criterio_mejor:
                    mejor_solucion = solucion
                    mejor_costo = solucion.costo_total
                    print(f"✓ Nueva mejor solución encontrada!")
                
            except Exception as e:
                print(f"✗ Error con estrategia {estrategia}: {e}")
        
        if mejor_solucion:
            print(f"\n=== Mejor Solución Final ===")
            print(f"Estrategia ganadora: {estrategias[0]}")  # Simplificado
            print(f"Vehículos: {mejor_solucion.numero_vehiculos_usados}")
            print(f"Costo: {mejor_solucion.costo_total:.0f}")
        
        return mejor_solucion