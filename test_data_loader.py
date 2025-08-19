"""
Script de prueba para validar el cargador de datos MDVSP.
Demuestra el uso del cargador y verifica el rendimiento de carga.
"""

import sys
import time
from pathlib import Path
from memory_profiler import profile

# Agrega el directorio padre al path para importar módulos
sys.path.append(str(Path(__file__).parent))

from data.mdvsp_data_loader import MDVSPDataLoader
from data.mdvsp_data_model import MDVSPData


def main() -> None:
    """Función principal que demuestra el uso del cargador de datos."""
    
    print("=== Prueba del Cargador de Datos MDVSP ===\n")
    
    try:
        # Inicializa el cargador
        cargador = MDVSPDataLoader("fischetti")
        print(f"Cargador inicializado para directorio: {cargador.directorio_instancias}")
        
        # Obtiene instancias disponibles
        instancias_disponibles = cargador.obtener_instancias_disponibles()
        print(f"\nInstancias disponibles: {len(instancias_disponibles)}")
        
        if not instancias_disponibles:
            print("No se encontraron instancias en el directorio.")
            print("Ejecuta 'python diagnose_format.py' para analizar el formato de archivos.")
            return
        
        for instancia in instancias_disponibles[:5]:  # Muestra las primeras 5
            print(f"  - {instancia}")
        
        if len(instancias_disponibles) > 5:
            print(f"  ... y {len(instancias_disponibles) - 5} más")
        
        # Prueba carga de una instancia específica
        instancia_prueba = instancias_disponibles[0]
        print(f"\n=== Cargando instancia: {instancia_prueba} ===")
        
        inicio_tiempo = time.perf_counter()
        datos_instancia = cargador.cargar_instancia(instancia_prueba)
        tiempo_carga = time.perf_counter() - inicio_tiempo
        
        print(f"Tiempo de carga: {tiempo_carga:.4f} segundos")
        print("\n" + datos_instancia.obtener_resumen())
        
        # Valida integridad
        print(f"\n=== Validación de Integridad ===")
        es_valida = cargador.validar_integridad_instancia(datos_instancia)
        print(f"Instancia válida: {'✓ Sí' if es_valida else '✗ No'}")
        
        # Muestra ejemplos de uso de la API
        print(f"\n=== Ejemplos de Uso de la API ===")
        mostrar_ejemplos_api(datos_instancia)
        
        # Prueba de rendimiento con múltiples instancias
        print(f"\n=== Prueba de Rendimiento ===")
        ejecutar_prueba_rendimiento(cargador, instancias_disponibles[:3])
        
    except FileNotFoundError as e:
        print(f"Error de archivos: {str(e)}")
        print("Asegúrate de que el directorio 'fischetti' existe y contiene archivos .cst y .tim")
        sys.exit(1)
    except Exception as e:
        print(f"Error durante la prueba: {str(e)}")
        print("\nEjecuta 'python diagnose_format.py' para analizar el formato de archivos.")
        sys.exit(1)


def mostrar_ejemplos_api(datos: MDVSPData) -> None:
    """
    Demuestra el uso de la API de los datos cargados.
    
    Args:
        datos: Instancia de datos MDVSP cargada
    """
    # Ejemplo de acceso a datos básicos
    print(f"Número de depósitos: {datos.numero_depositos}")
    print(f"Número de viajes: {datos.numero_viajes}")
    print(f"Total de vehículos: {datos.numero_total_vehiculos}")
    
    # Ejemplo de acceso a depósitos
    print(f"\nDepósitos:")
    for deposito in datos.depositos:
        print(f"  Depósito {deposito.id_deposito}: {deposito.numero_vehiculos} vehículos")
    
    # Ejemplo de acceso a viajes (primeros 3)
    print(f"\nPrimeros 3 viajes:")
    for viaje in datos.viajes[:3]:
        print(f"  Viaje {viaje.id_viaje}: [{viaje.tiempo_inicio}, {viaje.tiempo_fin}]")
    
    # Ejemplo de consulta de costos
    print(f"\nEjemplos de costos:")
    for i in range(min(3, datos.numero_viajes)):
        for j in range(min(3, datos.numero_viajes)):
            if i != j:
                costo = datos.obtener_costo(i, j)
                factible = "Factible" if datos.es_factible(i, j) else "Infactible"
                print(f"  Costo [{i}→{j}]: {costo:>10.0f} ({factible})")


def ejecutar_prueba_rendimiento(cargador: MDVSPDataLoader, 
                               instancias_prueba: list) -> None:
    """
    Ejecuta pruebas de rendimiento del cargador.
    
    Args:
        cargador: Instancia del cargador
        instancias_prueba: Lista de nombres de instancias para probar
    """
    tiempos_carga = []
    
    for nombre_instancia in instancias_prueba:
        print(f"Cargando {nombre_instancia}...", end=" ")
        
        inicio = time.perf_counter()
        try:
            datos = cargador.cargar_instancia(nombre_instancia)
            tiempo = time.perf_counter() - inicio
            tiempos_carga.append(tiempo)
            
            # Calcula métricas de la instancia
            estadisticas = datos.calcular_estadisticas_factibilidad()
            
            print(f"{tiempo:.4f}s "
                  f"({datos.numero_viajes} viajes, "
                  f"{estadisticas['porcentaje_infactibles']:.1f}% infactible)")
            
        except Exception as e:
            print(f"Error: {str(e)}")
    
    if tiempos_carga:
        tiempo_promedio = sum(tiempos_carga) / len(tiempos_carga)
        tiempo_min = min(tiempos_carga)
        tiempo_max = max(tiempos_carga)
        
        print(f"\nResumen de rendimiento:")
        print(f"  Tiempo promedio: {tiempo_promedio:.4f}s")
        print(f"  Tiempo mínimo: {tiempo_min:.4f}s")
        print(f"  Tiempo máximo: {tiempo_max:.4f}s")


if __name__ == "__main__":
    main()