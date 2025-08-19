"""
Script principal para ejecutar experimentos con el algoritmo Concurrent Schedule.
Ejecuta el algoritmo en todas las instancias disponibles y genera reportes detallados.
"""

import sys
import argparse
from pathlib import Path
from memory_profiler import profile

# Agrega directorios al path para importar módulos
sys.path.append(str(Path(__file__).parent))

from algorithms.experiment_runner import ExperimentRunner
from algorithms.concurrent_schedule import ConcurrentScheduleAlgorithm
from data.mdvsp_data_loader import MDVSPDataLoader


@profile
def main():
    """Función principal del script de experimentación."""
    
    # Configuración de argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Ejecutor de experimentos MDVSP')
    parser.add_argument('--instancias', '-i', type=str, default='fischetti',
                       help='Directorio con instancias MDVSP (default: fischetti)')
    parser.add_argument('--resultados', '-r', type=str, default='resultados',
                       help='Directorio para guardar resultados (default: resultados)')
    parser.add_argument('--limite', '-l', type=int, default=None,
                       help='Número máximo de instancias a procesar (default: todas)')
    parser.add_argument('--muestra', '-m', action='store_true',
                       help='Ejecuta solo una muestra pequeña (primeras 5 instancias)')
    parser.add_argument('--detalle', '-d', action='store_true',
                       help='Muestra detalles de las mejores soluciones')
    
    args = parser.parse_args()
    
    # Ajusta límite si se solicita muestra
    if args.muestra:
        args.limite = 5
        print("Modo muestra: procesando solo las primeras 5 instancias")
    
    try:
        # Inicializa el ejecutor de experimentos
        experimento = ExperimentRunner(
            directorio_instancias=args.instancias,
            directorio_resultados=args.resultados
        )
        
        # Ejecuta el experimento completo
        resumen = experimento.ejecutar_experimento_completo(limite_instancias=args.limite)
        
        # Muestra análisis adicional si se solicita
        if args.detalle and resumen['instancias_exitosas'] > 0:
            mostrar_analisis_detallado(experimento)
        
        # Sugiere siguientes pasos
        print(f"\n=== SIGUIENTES PASOS ===")
        print(f"1. Revisar resultados detallados en: {args.resultados}/resultados_detallados.csv")
        print(f"2. Leer reporte completo en: {args.resultados}/reporte_experimento.txt")
        print(f"3. Analizar patrones por tipo de instancia (número de depósitos)")
        
        if resumen['instancias_fallidas'] > 0:
            print(f"4. Investigar causas de {resumen['instancias_fallidas']} instancias fallidas")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("Asegúrate de que el directorio de instancias existe y contiene archivos .cst y .tim")
        return 1
        
    except Exception as e:
        print(f"Error inesperado: {str(e)}")
        return 1


def mostrar_analisis_detallado(experimento: ExperimentRunner) -> None:
    """
    Muestra análisis detallado de los resultados.
    
    Args:
        experimento: Ejecutor de experimentos con resultados
    """
    print(f"\n=== ANÁLISIS DETALLADO ===")
    
    # Mejores resultados por diferentes criterios
    criterios = [
        ('costo', 'Menor Costo'),
        ('vehiculos', 'Menos Vehículos'),
        ('tiempo', 'Menor Tiempo'),
        ('utilizacion', 'Mayor Utilización')
    ]
    
    for criterio, nombre in criterios:
        print(f"\n{nombre} (Top 5):")
        mejores = experimento.obtener_mejores_resultados(criterio=criterio, top=5)
        
        for i, resultado in enumerate(mejores, 1):
            print(f"  {i}. {resultado['instancia']:12s} - "
                  f"Costo: {resultado['costo_total']:>8.0f}, "
                  f"Vehículos: {resultado['vehiculos_usados']:>3d}, "
                  f"Tiempo: {resultado['tiempo_construccion']:>6.3f}s, "
                  f"Util: {resultado['utilizacion_vehiculos']:>5.1f}%")
    
    # Análisis por número de depósitos
    print(f"\n=== ANÁLISIS POR TIPO DE INSTANCIA ===")
    analizar_por_depositos(experimento)


def analizar_por_depositos(experimento: ExperimentRunner) -> None:
    """
    Analiza resultados agrupados por número de depósitos.
    
    Args:
        experimento: Ejecutor de experimentos con resultados
    """
    resultados_exitosos = [r for r in experimento.resultados_experimento if r['error'] is None]
    
    # Agrupa por número de depósitos
    grupos = {}
    for resultado in resultados_exitosos:
        num_depositos = resultado['numero_depositos']
        if num_depositos not in grupos:
            grupos[num_depositos] = []
        grupos[num_depositos].append(resultado)
    
    # Analiza cada grupo
    for num_depositos in sorted(grupos.keys()):
        grupo = grupos[num_depositos]
        
        costos = [r['costo_total'] for r in grupo]
        vehiculos = [r['vehiculos_usados'] for r in grupo]
        tiempos = [r['tiempo_construccion'] for r in grupo]
        utilizaciones = [r['utilizacion_vehiculos'] for r in grupo]
        
        print(f"\nDepósitos {num_depositos} ({len(grupo)} instancias):")
        print(f"  Costo promedio: {sum(costos)/len(costos):>8.0f} "
              f"(rango: {min(costos):>8.0f} - {max(costos):>8.0f})")
        print(f"  Vehículos promedio: {sum(vehiculos)/len(vehiculos):>5.1f} "
              f"(rango: {min(vehiculos):>3d} - {max(vehiculos):>3d})")
        print(f"  Tiempo promedio: {sum(tiempos)/len(tiempos):>8.4f}s "
              f"(rango: {min(tiempos):>6.4f}s - {max(tiempos):>6.4f}s)")
        print(f"  Utilización promedio: {sum(utilizaciones)/len(utilizaciones):>5.1f}% "
              f"(rango: {min(utilizaciones):>5.1f}% - {max(utilizaciones):>5.1f}%)")


def ejecutar_prueba_rapida() -> None:
    """Ejecuta una prueba rápida con una sola instancia."""
    print("=== PRUEBA RÁPIDA ===")
    
    try:
        # Carga una instancia
        cargador = MDVSPDataLoader("fischetti")
        instancias = cargador.obtener_instancias_disponibles()
        
        if not instancias:
            print("No se encontraron instancias")
            return
        
        instancia = cargador.cargar_instancia(instancias[0])
        print(f"Instancia de prueba: {instancia.nombre_archivo_instancia}")
        print(instancia.obtener_resumen())
        
        # Ejecuta algoritmo
        algoritmo = ConcurrentScheduleAlgorithm(verbose=True)
        solucion = algoritmo.resolver(instancia)
        
        # Muestra resultados
        print(f"\n{solucion.obtener_resumen()}")
        print(f"\n{solucion.obtener_detalle_rutas(5)}")
        
    except Exception as e:
        print(f"Error en prueba rápida: {str(e)}")


if __name__ == "__main__":
    # Verifica si se debe ejecutar prueba rápida
    if len(sys.argv) > 1 and sys.argv[1] == "--prueba":
        ejecutar_prueba_rapida()
    else:
        sys.exit(main())