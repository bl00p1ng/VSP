"""
Script principal para resolver problemas VSP con archivos específicos.
Recibe paths de archivos .tim, .cst y .solucion como parámetros.
"""

import sys
import argparse
from pathlib import Path
from memory_profiler import profile

# Agrega directorios al path para importar módulos
sys.path.append(str(Path(__file__).parent))

from data.vsp_data_loader import VSPDataLoader
from algorithms.vsp_constructive import VSPConstructiveAlgorithm


@profile
def main():
    """Función principal que resuelve una instancia VSP específica."""
    
    # Configuración de argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Resolvedor VSP para archivos específicos')
    parser.add_argument('archivo_tim', type=str,
                       help='Path completo al archivo .tim con tiempos de servicios')
    parser.add_argument('archivo_cst', type=str, 
                       help='Path completo al archivo .cst con matriz de costos')
    parser.add_argument('archivo_solucion', type=str,
                       help='Path completo al archivo .solucion de salida')
    parser.add_argument('--estrategia', '-e', type=str, default='tiempo_inicio',
                       choices=['tiempo_inicio', 'tiempo_fin', 'duracion', 'mixta'],
                       help='Estrategia del algoritmo constructivo (default: tiempo_inicio)')
    
    args = parser.parse_args()
    
    # Valida que los archivos de entrada existan
    archivo_tim = Path(args.archivo_tim)
    archivo_cst = Path(args.archivo_cst)
    archivo_solucion = Path(args.archivo_solucion)
    
    if not archivo_tim.exists():
        print(f"Error: Archivo .tim no encontrado: {archivo_tim}")
        return 1
    
    if not archivo_cst.exists():
        print(f"Error: Archivo .cst no encontrado: {archivo_cst}")
        return 1
    
    # Valida extensiones
    if archivo_tim.suffix.lower() != '.tim':
        print(f"Error: El primer archivo debe tener extensión .tim")
        return 1
    
    if archivo_cst.suffix.lower() != '.cst':
        print(f"Error: El segundo archivo debe tener extensión .cst")
        return 1
    
    if archivo_solucion.suffix.lower() != '.solucion':
        print(f"Error: El archivo de salida debe tener extensión .solucion")
        return 1
    
    try:
        print("=== RESOLVEDOR VSP ===")
        print(f"Archivo .tim: {archivo_tim}")
        print(f"Archivo .cst: {archivo_cst}")
        print(f"Archivo .solucion: {archivo_solucion}")
        print(f"Estrategia: {args.estrategia}")
        
        # Carga la instancia desde archivos específicos
        cargador = VSPDataLoader()
        instancia = cargador.cargar_instancia_desde_archivos(
            str(archivo_cst), 
            str(archivo_tim)
        )
        
        print(f"\n=== INSTANCIA CARGADA ===")
        print(instancia.obtener_resumen())
        
        # Resuelve la instancia
        algoritmo = VSPConstructiveAlgorithm()
        solucion = algoritmo.resolver(instancia, estrategia=args.estrategia)
        
        print(f"\n=== SOLUCIÓN ENCONTRADA ===")
        print(f"Factible: {'Sí' if solucion.es_factible else 'No'}")
        print(f"Vehículos usados: {solucion.numero_vehiculos_usados}")
        print(f"Servicios asignados: {len(solucion.servicios_asignados)}")
        print(f"Costo total: {solucion.costo_total:.0f}")
        print(f"Tiempo construcción: {solucion.tiempo_construccion:.4f}s")
        
        # Genera archivo .solucion
        generar_archivo_solucion(solucion, archivo_solucion)
        
        print(f"\n=== ARCHIVO GENERADO ===")
        print(f"Solución guardada en: {archivo_solucion}")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error de archivo: {str(e)}")
        return 1
        
    except Exception as e:
        print(f"Error durante la resolución: {str(e)}")
        return 1


def generar_archivo_solucion(solucion, archivo_salida):
    """
    Genera el archivo .solucion con el formato especificado.
    Cada fila = vehículo, cada columna = servicios que recorre (índices desde 0).
    
    Args:
        solucion: Solución VSP obtenida
        archivo_salida: Path donde guardar el archivo .solucion
    """
    try:
        with open(archivo_salida, 'w', encoding='utf-8') as archivo:
            # Procesa solo rutas activas (con servicios asignados)
            rutas_activas = solucion.obtener_rutas_activas()
            
            for ruta in rutas_activas:
                if not ruta.es_vacia():
                    # Convierte servicios a string separados por espacio
                    # Los índices ya empiezan en 0 por diseño
                    servicios_str = ' '.join(map(str, ruta.servicios))
                    archivo.write(servicios_str + '\n')
        
        print(f"  ✓ Archivo .solucion generado con {len(rutas_activas)} rutas")
        
        # Muestra resumen del archivo generado
        print(f"  - Formato: cada fila = vehículo, cada columna = servicio")
        print(f"  - Índices de servicios: desde 0")
        print(f"  - Total de vehículos usados: {len(rutas_activas)}")
        
    except IOError as e:
        raise IOError(f"Error escribiendo archivo .solucion: {str(e)}")


if __name__ == "__main__":
    sys.exit(main())