"""
Implementación del algoritmo Concurrent Schedule para el problema MDVSP.
Algoritmo constructivo que asigna viajes a vehículos de manera secuencial.
"""

import time
from typing import List, Tuple, Optional, Set
from memory_profiler import profile
import numpy as np

from data.mdvsp_data_model import MDVSPData
from .solution_model import SolucionMDVSP, Ruta


class ConcurrentScheduleAlgorithm:
    """
    Implementación del algoritmo Concurrent Schedule para MDVSP.
    Construye soluciones asignando viajes de manera greedy considerando costos y factibilidad temporal.
    """
    
    def __init__(self, verbose: bool = True) -> None:
        """
        Inicializa el algoritmo.
        
        Args:
            verbose: Si debe mostrar información de progreso
        """
        self.verbose = verbose
        self.estadisticas_ejecucion = {}
    
    @profile
    def resolver(self, instancia: MDVSPData) -> SolucionMDVSP:
        """
        Resuelve una instancia MDVSP usando Concurrent Schedule.
        
        Args:
            instancia: Datos de la instancia a resolver
            
        Returns:
            Solución construida por el algoritmo
        """
        inicio_tiempo = time.perf_counter()
        
        if self.verbose:
            print(f"\n=== Resolviendo {instancia.nombre_archivo_instancia} ===")
            print(f"Depósitos: {instancia.numero_depositos}, Viajes: {instancia.numero_viajes}")
        
        # Inicializa estructuras de datos
        solucion = self._inicializar_solucion(instancia)
        viajes_pendientes = set(range(instancia.numero_viajes))
        
        # Estadísticas para debugging
        iteraciones = 0
        asignaciones_exitosas = 0
        
        # Ciclo principal: asigna viajes hasta que no queden pendientes
        while viajes_pendientes:
            iteraciones += 1
            
            # Encuentra la mejor asignación viaje-vehículo
            mejor_asignacion = self._encontrar_mejor_asignacion(
                instancia, solucion, viajes_pendientes
            )
            
            if mejor_asignacion is None:
                # No se puede asignar ningún viaje más
                if self.verbose:
                    print(f"  ⚠️  No se pueden asignar {len(viajes_pendientes)} viajes restantes")
                break
            
            # Realiza la asignación
            id_viaje, id_ruta, costo_asignacion = mejor_asignacion
            self._asignar_viaje_a_ruta(instancia, solucion, id_viaje, id_ruta, costo_asignacion)
            
            viajes_pendientes.remove(id_viaje)
            asignaciones_exitosas += 1
            
            if self.verbose and (asignaciones_exitosas % 20 == 0):
                print(f"  Asignados: {asignaciones_exitosas}/{instancia.numero_viajes}")
        
        # Finaliza la solución
        tiempo_total = time.perf_counter() - inicio_tiempo
        solucion.tiempo_construccion = tiempo_total
        solucion.calcular_metricas(instancia.numero_viajes)
        
        # Guarda estadísticas
        self.estadisticas_ejecucion[instancia.nombre_archivo_instancia] = {
            'iteraciones': iteraciones,
            'asignaciones_exitosas': asignaciones_exitosas,
            'tiempo_ejecucion': tiempo_total,
            'viajes_no_asignados': len(viajes_pendientes)
        }
        
        if self.verbose:
            print(f"  ✓ Completado en {tiempo_total:.4f}s")
            print(f"  Costo: {solucion.costo_total:.0f}, Vehículos: {solucion.numero_vehiculos_usados}")
        
        return solucion
    
    def _inicializar_solucion(self, instancia: MDVSPData) -> SolucionMDVSP:
        """
        Inicializa la estructura de solución con rutas vacías.
        
        Args:
            instancia: Datos de la instancia
            
        Returns:
            Solución inicializada con rutas vacías
        """
        solucion = SolucionMDVSP(nombre_instancia=instancia.nombre_archivo_instancia)
        
        # Crea una ruta por cada vehículo disponible
        id_vehiculo_global = 0
        for deposito in instancia.depositos:
            for _ in range(deposito.numero_vehiculos):
                ruta = Ruta(
                    id_vehiculo=id_vehiculo_global,
                    id_deposito=deposito.id_deposito
                )
                solucion.agregar_ruta(ruta)
                id_vehiculo_global += 1
        
        return solucion
    
    def _encontrar_mejor_asignacion(self, instancia: MDVSPData, solucion: SolucionMDVSP,
                                   viajes_pendientes: Set[int]) -> Optional[Tuple[int, int, float]]:
        """
        Encuentra la mejor asignación viaje-vehículo basada en costo mínimo.
        
        Args:
            instancia: Datos de la instancia
            solucion: Solución actual
            viajes_pendientes: Conjunto de viajes aún no asignados
            
        Returns:
            Tupla (id_viaje, id_ruta, costo) de la mejor asignación o None si no hay factibles
        """
        mejor_asignacion = None
        menor_costo = float('inf')
        
        # Evalúa cada viaje pendiente
        for id_viaje in viajes_pendientes:
            viaje = instancia.viajes[id_viaje]
            
            # Evalúa cada ruta (vehículo)
            for id_ruta, ruta in enumerate(solucion.rutas):
                costo_asignacion = self._calcular_costo_asignacion(
                    instancia, ruta, viaje
                )
                
                if costo_asignacion is not None and costo_asignacion < menor_costo:
                    menor_costo = costo_asignacion
                    mejor_asignacion = (id_viaje, id_ruta, costo_asignacion)
        
        return mejor_asignacion
    
    def _calcular_costo_asignacion(self, instancia: MDVSPData, ruta: Ruta,
                                  viaje) -> Optional[float]:
        """
        Calcula el costo de asignar un viaje a una ruta específica.
        
        Args:
            instancia: Datos de la instancia
            ruta: Ruta donde se evaluaría la asignación
            viaje: Viaje a evaluar
            
        Returns:
            Costo de la asignación o None si no es factible
        """
        # Índices en la matriz de costos
        indice_viaje = viaje.id_viaje
        indice_deposito = instancia.numero_viajes + ruta.id_deposito
        
        if ruta.es_vacia():
            # Ruta vacía: costo desde depósito al viaje + viaje al depósito
            costo_ida = instancia.obtener_costo(indice_deposito, indice_viaje)
            costo_vuelta = instancia.obtener_costo(indice_viaje, indice_deposito)
            
            # Verifica factibilidad
            if (costo_ida == instancia.COSTO_INFACTIBLE or 
                costo_vuelta == instancia.COSTO_INFACTIBLE):
                return None
            
            return costo_ida + costo_vuelta
        
        else:
            # Ruta con viajes: busca la mejor posición para insertar
            return self._calcular_mejor_insercion(instancia, ruta, viaje)
    
    def _calcular_mejor_insercion(self, instancia: MDVSPData, ruta: Ruta,
                                 viaje) -> Optional[float]:
        """
        Calcula el mejor costo de inserción de un viaje en una ruta existente.
        
        Args:
            instancia: Datos de la instancia
            ruta: Ruta existente
            viaje: Viaje a insertar
            
        Returns:
            Mejor costo de inserción o None si no es factible
        """
        mejor_costo = float('inf')
        indice_nuevo_viaje = viaje.id_viaje
        indice_deposito = instancia.numero_viajes + ruta.id_deposito
        
        # Evalúa inserción al inicio de la ruta
        primer_viaje = ruta.viajes[0]
        costo_inicio = (
            instancia.obtener_costo(indice_deposito, indice_nuevo_viaje) +
            instancia.obtener_costo(indice_nuevo_viaje, primer_viaje) -
            instancia.obtener_costo(indice_deposito, primer_viaje)
        )
        
        # Verifica factibilidad temporal al inicio
        if (self._es_factible_temporalmente(instancia, viaje, ruta.viajes[0], posicion='antes') and
            costo_inicio != instancia.COSTO_INFACTIBLE):
            mejor_costo = min(mejor_costo, costo_inicio)
        
        # Evalúa inserción entre viajes consecutivos
        for i in range(len(ruta.viajes) - 1):
            viaje_anterior = ruta.viajes[i]
            viaje_siguiente = ruta.viajes[i + 1]
            
            costo_insercion = (
                instancia.obtener_costo(viaje_anterior, indice_nuevo_viaje) +
                instancia.obtener_costo(indice_nuevo_viaje, viaje_siguiente) -
                instancia.obtener_costo(viaje_anterior, viaje_siguiente)
            )
            
            # Verifica factibilidad temporal
            if (self._es_factible_temporalmente(instancia, viaje, viaje_anterior, posicion='despues') and
                self._es_factible_temporalmente(instancia, viaje, viaje_siguiente, posicion='antes') and
                costo_insercion != instancia.COSTO_INFACTIBLE):
                mejor_costo = min(mejor_costo, costo_insercion)
        
        # Evalúa inserción al final de la ruta
        ultimo_viaje = ruta.viajes[-1]
        costo_final = (
            instancia.obtener_costo(ultimo_viaje, indice_nuevo_viaje) +
            instancia.obtener_costo(indice_nuevo_viaje, indice_deposito) -
            instancia.obtener_costo(ultimo_viaje, indice_deposito)
        )
        
        # Verifica factibilidad temporal al final
        if (self._es_factible_temporalmente(instancia, viaje, ultimo_viaje, posicion='despues') and
            costo_final != instancia.COSTO_INFACTIBLE):
            mejor_costo = min(mejor_costo, costo_final)
        
        return mejor_costo if mejor_costo != float('inf') else None
    
    def _es_factible_temporalmente(self, instancia: MDVSPData, viaje_nuevo,
                                  viaje_referencia, posicion: str) -> bool:
        """
        Verifica si es factible insertar un viaje respecto a otro temporalmente.
        
        Args:
            instancia: Datos de la instancia
            viaje_nuevo: Viaje a insertar
            viaje_referencia: Viaje de referencia (índice)
            posicion: 'antes' o 'despues' respecto al viaje de referencia
            
        Returns:
            True si la inserción es factible temporalmente
        """
        viaje_ref_obj = instancia.viajes[viaje_referencia]
        
        if posicion == 'antes':
            # El nuevo viaje debe terminar antes de que empiece el de referencia
            return viaje_nuevo.tiempo_fin <= viaje_ref_obj.tiempo_inicio
        elif posicion == 'despues':
            # El nuevo viaje debe empezar después de que termine el de referencia
            return viaje_nuevo.tiempo_inicio >= viaje_ref_obj.tiempo_fin
        
        return False
    
    def _asignar_viaje_a_ruta(self, instancia: MDVSPData, solucion: SolucionMDVSP,
                             id_viaje: int, id_ruta: int, costo_asignacion: float) -> None:
        """
        Asigna un viaje a una ruta específica.
        
        Args:
            instancia: Datos de la instancia
            solucion: Solución donde realizar la asignación
            id_viaje: ID del viaje a asignar
            id_ruta: ID de la ruta destino
            costo_asignacion: Costo de la asignación
        """
        ruta = solucion.rutas[id_ruta]
        viaje = instancia.viajes[id_viaje]
        
        if ruta.es_vacia():
            # Primera asignación a la ruta
            ruta.agregar_viaje(id_viaje, costo_asignacion, viaje.tiempo_inicio, viaje.tiempo_fin)
        else:
            # Inserción en ruta existente - encuentra la mejor posición
            mejor_posicion = self._encontrar_mejor_posicion_insercion(instancia, ruta, viaje)
            
            # Inserta el viaje en la posición óptima
            ruta.viajes.insert(mejor_posicion, id_viaje)
            ruta.costo_total += costo_asignacion
            
            # Actualiza ventana temporal
            if viaje.tiempo_inicio < ruta.tiempo_inicio:
                ruta.tiempo_inicio = viaje.tiempo_inicio
            if viaje.tiempo_fin > ruta.tiempo_fin:
                ruta.tiempo_fin = viaje.tiempo_fin
        
        # Actualiza la solución
        solucion.viajes_asignados.add(id_viaje)
    
    def _encontrar_mejor_posicion_insercion(self, instancia: MDVSPData, ruta: Ruta, viaje) -> int:
        """
        Encuentra la mejor posición para insertar un viaje en una ruta.
        
        Args:
            instancia: Datos de la instancia
            ruta: Ruta donde insertar
            viaje: Viaje a insertar
            
        Returns:
            Índice de la mejor posición de inserción
        """
        mejor_posicion = 0
        menor_incremento = float('inf')
        
        # Evalúa todas las posiciones posibles
        for pos in range(len(ruta.viajes) + 1):
            incremento_costo = self._calcular_incremento_costo_posicion(
                instancia, ruta, viaje, pos
            )
            
            if incremento_costo is not None and incremento_costo < menor_incremento:
                menor_incremento = incremento_costo
                mejor_posicion = pos
        
        return mejor_posicion
    
    def _calcular_incremento_costo_posicion(self, instancia: MDVSPData, ruta: Ruta,
                                          viaje, posicion: int) -> Optional[float]:
        """
        Calcula el incremento de costo de insertar en una posición específica.
        
        Args:
            instancia: Datos de la instancia
            ruta: Ruta destino
            viaje: Viaje a insertar
            posicion: Posición de inserción (0 = inicio, len(viajes) = final)
            
        Returns:
            Incremento de costo o None si no es factible
        """
        indice_nuevo_viaje = viaje.id_viaje
        indice_deposito = instancia.numero_viajes + ruta.id_deposito
        
        if posicion == 0:
            # Inserción al inicio
            if len(ruta.viajes) == 0:
                return 0.0  # Ya calculado en _calcular_costo_asignacion
            
            primer_viaje = ruta.viajes[0]
            if not self._es_factible_temporalmente(instancia, viaje, primer_viaje, 'antes'):
                return None
            
            return (instancia.obtener_costo(indice_deposito, indice_nuevo_viaje) +
                   instancia.obtener_costo(indice_nuevo_viaje, primer_viaje) -
                   instancia.obtener_costo(indice_deposito, primer_viaje))
        
        elif posicion == len(ruta.viajes):
            # Inserción al final
            ultimo_viaje = ruta.viajes[-1]
            if not self._es_factible_temporalmente(instancia, viaje, ultimo_viaje, 'despues'):
                return None
            
            return (instancia.obtener_costo(ultimo_viaje, indice_nuevo_viaje) +
                   instancia.obtener_costo(indice_nuevo_viaje, indice_deposito) -
                   instancia.obtener_costo(ultimo_viaje, indice_deposito))
        
        else:
            # Inserción en el medio
            viaje_anterior = ruta.viajes[posicion - 1]
            viaje_siguiente = ruta.viajes[posicion]
            
            if (not self._es_factible_temporalmente(instancia, viaje, viaje_anterior, 'despues') or
                not self._es_factible_temporalmente(instancia, viaje, viaje_siguiente, 'antes')):
                return None
            
            return (instancia.obtener_costo(viaje_anterior, indice_nuevo_viaje) +
                   instancia.obtener_costo(indice_nuevo_viaje, viaje_siguiente) -
                   instancia.obtener_costo(viaje_anterior, viaje_siguiente))
    
    def obtener_estadisticas_algoritmo(self) -> dict:
        """
        Obtiene estadísticas de ejecución del algoritmo.
        
        Returns:
            Diccionario con estadísticas por instancia
        """
        return self.estadisticas_ejecucion.copy()