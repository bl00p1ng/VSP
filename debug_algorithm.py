"""
Script de debugging para identificar problemas en el algoritmo Concurrent Schedule.
Analiza paso a paso una instancia simple para encontrar errores.
"""

import sys
from pathlib import Path

# Agrega directorios al path
sys.path.append(str(Path(__file__).parent))

from data.mdvsp_data_loader import MDVSPDataLoader
from algorithms.concurrent_schedule import ConcurrentScheduleAlgorithm
import numpy as np


def debug_instancia_simple():
    """Analiza una instancia simple paso a paso."""
    
    print("=== DEBUGGING DEL ALGORITMO CONCURRENT SCHEDULE ===\n")
    
    # Cargar instancia más simple
    cargador = MDVSPDataLoader("fischetti")
    instancias = cargador.obtener_instancias_disponibles()
    
    if not instancias:
        print("No se encontraron instancias")
        return
    
    # Usa la primera instancia disponible
    nombre_instancia = instancias[0]
    print(f"Analizando instancia: {nombre_instancia}")
    
    instancia = cargador.cargar_instancia(nombre_instancia)
    
    # 1. VERIFICAR DATOS DE ENTRADA
    print("\n1. VERIFICACIÓN DE DATOS DE ENTRADA")
    print("-" * 50)
    print(f"Depósitos: {instancia.numero_depositos}")
    print(f"Viajes: {instancia.numero_viajes}")
    print(f"Total vehículos: {instancia.numero_total_vehiculos}")
    
    print(f"\nPrimeros 5 viajes:")
    for i in range(min(5, len(instancia.viajes))):
        viaje = instancia.viajes[i]
        print(f"  Viaje {i}: inicio={viaje.tiempo_inicio}, fin={viaje.tiempo_fin}")
    
    print(f"\nDepósitos y vehículos:")
    for deposito in instancia.depositos:
        print(f"  Depósito {deposito.id_deposito}: {deposito.numero_vehiculos} vehículos")
    
    # 2. VERIFICAR MATRIZ DE COSTOS
    print(f"\n2. VERIFICACIÓN DE MATRIZ DE COSTOS")
    print("-" * 50)
    dimension = instancia.numero_viajes + instancia.numero_depositos
    print(f"Dimensión matriz: {dimension}x{dimension}")
    
    # Muestra esquina superior izquierda
    print(f"\nEsquina superior izquierda (primeros 5x5):")
    submatriz = instancia.matriz_viajes[:5, :5]
    for i in range(submatriz.shape[0]):
        fila = " ".join(f"{val:>8.0f}" for val in submatriz[i])
        print(f"  [{i}] {fila}")
    
    # Verifica costos depósito-viaje
    print(f"\nCostos desde depósitos a primeros viajes:")
    for dep in range(instancia.numero_depositos):
        indice_dep = instancia.numero_viajes + dep
        for viaje in range(min(3, instancia.numero_viajes)):
            costo = instancia.matriz_viajes[indice_dep, viaje]
            factible = "Factible" if costo != instancia.COSTO_INFACTIBLE else "Infactible"
            print(f"  Depósito {dep} -> Viaje {viaje}: {costo:>10.0f} ({factible})")
    
    # 3. PROBAR ALGORITMO PASO A PASO
    print(f"\n3. EJECUCIÓN PASO A PASO DEL ALGORITMO")
    print("-" * 50)
    
    algoritmo_debug = ConcurrentScheduleDebug()
    solucion = algoritmo_debug.resolver_con_debug(instancia, max_viajes=10)
    
    # 4. VERIFICAR RESULTADOS
    print(f"\n4. ANÁLISIS DE RESULTADOS")
    print("-" * 50)
    print(f"Viajes asignados: {len(solucion.viajes_asignados)}/{instancia.numero_viajes}")
    print(f"Costo total: {solucion.costo_total:,.0f}")
    print(f"Vehículos usados: {solucion.numero_vehiculos_usados}")
    print(f"Factible: {solucion.es_factible}")
    
    rutas_activas = solucion.obtener_rutas_activas()
    print(f"\nDetalles de rutas activas:")
    for i, ruta in enumerate(rutas_activas[:5]):
        print(f"  Ruta {i}: {ruta.obtener_resumen()}")
        if ruta.viajes:
            # Verifica cálculo de costo manual
            costo_manual = calcular_costo_ruta_manual(instancia, ruta)
            print(f"    Costo calculado: {ruta.costo_total:.0f}")
            print(f"    Costo manual: {costo_manual:.0f}")
            print(f"    ¿Coinciden? {'Sí' if abs(ruta.costo_total - costo_manual) < 1 else 'NO'}")


def calcular_costo_ruta_manual(instancia, ruta):
    """Calcula manualmente el costo de una ruta para verificación."""
    if not ruta.viajes:
        return 0.0
    
    costo_total = 0.0
    indice_deposito = instancia.numero_viajes + ruta.id_deposito
    
    # Costo: depósito -> primer viaje
    primer_viaje = ruta.viajes[0]
    costo_total += instancia.matriz_viajes[indice_deposito, primer_viaje]
    
    # Costos entre viajes consecutivos
    for i in range(len(ruta.viajes) - 1):
        viaje_actual = ruta.viajes[i]
        viaje_siguiente = ruta.viajes[i + 1]
        costo_total += instancia.matriz_viajes[viaje_actual, viaje_siguiente]
    
    # Costo: último viaje -> depósito
    ultimo_viaje = ruta.viajes[-1]
    costo_total += instancia.matriz_viajes[ultimo_viaje, indice_deposito]
    
    return costo_total


class ConcurrentScheduleDebug:
    """Versión de debugging del algoritmo con trazas detalladas."""
    
    def resolver_con_debug(self, instancia, max_viajes=None):
        """Resuelve con debugging limitado a max_viajes para análisis."""
        
        from algorithms.solution_model import SolucionMDVSP, Ruta
        
        print("Inicializando solución...")
        solucion = self._inicializar_solucion(instancia)
        
        viajes_pendientes = set(range(instancia.numero_viajes))
        if max_viajes:
            viajes_pendientes = set(list(viajes_pendientes)[:max_viajes])
        
        print(f"Viajes a procesar: {len(viajes_pendientes)}")
        
        iteracion = 0
        while viajes_pendientes and iteracion < 20:  # Límite para debugging
            iteracion += 1
            print(f"\n--- Iteración {iteracion} ---")
            print(f"Viajes pendientes: {sorted(list(viajes_pendientes))}")
            
            # Encuentra mejor asignación
            mejor_asignacion = self._encontrar_mejor_asignacion_debug(
                instancia, solucion, viajes_pendientes
            )
            
            if mejor_asignacion is None:
                print("❌ No se encontró asignación factible")
                break
            
            id_viaje, id_ruta, costo = mejor_asignacion
            print(f"✅ Mejor asignación: Viaje {id_viaje} -> Ruta {id_ruta} (Costo: {costo:.0f})")
            
            # Realiza asignación
            self._asignar_viaje_debug(instancia, solucion, id_viaje, id_ruta, costo)
            viajes_pendientes.remove(id_viaje)
        
        # Finalizar solución
        solucion.calcular_metricas(instancia.numero_viajes)
        return solucion
    
    def _inicializar_solucion(self, instancia):
        """Inicializa solución con rutas vacías."""
        from algorithms.solution_model import SolucionMDVSP, Ruta
        
        solucion = SolucionMDVSP(nombre_instancia=instancia.nombre_archivo_instancia)
        
        id_vehiculo = 0
        for deposito in instancia.depositos:
            for _ in range(deposito.numero_vehiculos):
                ruta = Ruta(id_vehiculo=id_vehiculo, id_deposito=deposito.id_deposito)
                solucion.agregar_ruta(ruta)
                id_vehiculo += 1
        
        print(f"Inicializadas {len(solucion.rutas)} rutas vacías")
        return solucion
    
    def _encontrar_mejor_asignacion_debug(self, instancia, solucion, viajes_pendientes):
        """Encuentra mejor asignación con debugging."""
        
        mejor_asignacion = None
        menor_costo = float('inf')
        evaluaciones = 0
        
        print(f"Evaluando asignaciones...")
        
        for id_viaje in list(viajes_pendientes)[:3]:  # Solo primeros 3 para debugging
            viaje = instancia.viajes[id_viaje]
            print(f"  Viaje {id_viaje} (tiempo: {viaje.tiempo_inicio}-{viaje.tiempo_fin})")
            
            for id_ruta in range(min(5, len(solucion.rutas))):  # Solo primeras 5 rutas
                ruta = solucion.rutas[id_ruta]
                evaluaciones += 1
                
                costo = self._calcular_costo_asignacion_debug(instancia, ruta, viaje)
                
                if costo is not None:
                    print(f"    -> Ruta {id_ruta} (Dep {ruta.id_deposito}): {costo:.0f}")
                    
                    if costo < menor_costo:
                        menor_costo = costo
                        mejor_asignacion = (id_viaje, id_ruta, costo)
                else:
                    print(f"    -> Ruta {id_ruta} (Dep {ruta.id_deposito}): INFACTIBLE")
        
        print(f"Total evaluaciones: {evaluaciones}")
        return mejor_asignacion
    
    def _calcular_costo_asignacion_debug(self, instancia, ruta, viaje):
        """Calcula costo con debugging detallado."""
        
        indice_viaje = viaje.id_viaje
        indice_deposito = instancia.numero_viajes + ruta.id_deposito
        
        if ruta.es_vacia():
            # Ruta vacía: depósito -> viaje -> depósito
            costo_ida = instancia.matriz_viajes[indice_deposito, indice_viaje]
            costo_vuelta = instancia.matriz_viajes[indice_viaje, indice_deposito]
            
            if (costo_ida == instancia.COSTO_INFACTIBLE or 
                costo_vuelta == instancia.COSTO_INFACTIBLE):
                return None
            
            costo_total = costo_ida + costo_vuelta
            return costo_total
        
        else:
            # Ruta con viajes: inserción al final por simplicidad
            ultimo_viaje = ruta.viajes[-1]
            
            # Verificar factibilidad temporal
            ultimo_viaje_obj = instancia.viajes[ultimo_viaje]
            if viaje.tiempo_inicio < ultimo_viaje_obj.tiempo_fin:
                return None  # No factible temporalmente
            
            # Calcular costo de inserción al final
            costo_original = instancia.matriz_viajes[ultimo_viaje, indice_deposito]
            costo_nuevo = (instancia.matriz_viajes[ultimo_viaje, indice_viaje] +
                          instancia.matriz_viajes[indice_viaje, indice_deposito])
            
            if (instancia.matriz_viajes[ultimo_viaje, indice_viaje] == instancia.COSTO_INFACTIBLE or
                instancia.matriz_viajes[indice_viaje, indice_deposito] == instancia.COSTO_INFACTIBLE):
                return None
            
            incremento = costo_nuevo - costo_original
            return incremento
    
    def _asignar_viaje_debug(self, instancia, solucion, id_viaje, id_ruta, costo):
        """Asigna viaje con debugging."""
        
        ruta = solucion.rutas[id_ruta]
        viaje = instancia.viajes[id_viaje]
        
        print(f"    Asignando viaje {id_viaje} a ruta {id_ruta}")
        print(f"    Ruta antes: {ruta.viajes}")
        
        if ruta.es_vacia():
            ruta.agregar_viaje(id_viaje, costo, viaje.tiempo_inicio, viaje.tiempo_fin)
        else:
            # Insertar al final por simplicidad en debugging
            ruta.viajes.append(id_viaje)
            ruta.costo_total += costo
            
            if viaje.tiempo_inicio < ruta.tiempo_inicio:
                ruta.tiempo_inicio = viaje.tiempo_inicio
            if viaje.tiempo_fin > ruta.tiempo_fin:
                ruta.tiempo_fin = viaje.tiempo_fin
        
        print(f"    Ruta después: {ruta.viajes}")
        print(f"    Costo ruta: {ruta.costo_total:.0f}")
        
        solucion.viajes_asignados.add(id_viaje)


if __name__ == "__main__":
    debug_instancia_simple()