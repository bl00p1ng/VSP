"""
Ejecutor de experimentos para evaluar el algoritmo Concurrent Schedule en todas las instancias.
Maneja la carga de datos, ejecución del algoritmo y reporte de resultados.
"""

import time
from pathlib import Path
from typing import List, Dict, Optional
import csv
from memory_profiler import profile

from data.mdvsp_data_loader import MDVSPDataLoader
from data.mdvsp_data_model import MDVSPData
from .concurrent_schedule import ConcurrentScheduleAlgorithm
from .solution_model import SolucionMDVSP


class ExperimentRunner:
    """
    Ejecutor de experimentos que evalúa el algoritmo en múltiples instancias.
    Genera reportes detallados de rendimiento y calidad de soluciones.
    """
    
    def __init__(self, directorio_instancias: str = "fischetti", 
                 directorio_resultados: str = "resultados") -> None:
        """
        Inicializa el ejecutor de experimentos.
        
        Args:
            directorio_instancias: Directorio con instancias MDVSP
            directorio_resultados: Directorio para guardar resultados
        """
        self.cargador = MDVSPDataLoader(directorio_instancias)
        self.algoritmo = ConcurrentScheduleAlgorithm(verbose=False)
        self.directorio_resultados = Path(directorio_resultados)
        self.directorio_resultados.mkdir(exist_ok=True)
        
        self.resultados_experimento = []
        self.instancias_cargadas = []
        self.tiempo_total_experimento = 0.0
    
    @profile
    def ejecutar_experimento_completo(self, limite_instancias: Optional[int] = None) -> Dict:
        """
        Ejecuta el experimento completo en todas las instancias disponibles.
        
        Args:
            limite_instancias: Número máximo de instancias a procesar (None = todas)
            
        Returns:
            Diccionario con resumen del experimento
        """
        print("=== INICIO DEL EXPERIMENTO MDVSP ===")
        inicio_experimento = time.perf_counter()
        
        # Carga todas las instancias
        print("\n1. Cargando instancias...")
        self._cargar_instancias(limite_instancias)
        
        # Ejecuta algoritmo en cada instancia
        print(f"\n2. Ejecutando algoritmo en {len(self.instancias_cargadas)} instancias...")
        self._ejecutar_algoritmo_todas_instancias()
        
        # Genera reportes
        print("\n3. Generando reportes...")
        resumen = self._generar_reportes()
        
        self.tiempo_total_experimento = time.perf_counter() - inicio_experimento
        
        print(f"\n=== EXPERIMENTO COMPLETADO EN {self.tiempo_total_experimento:.2f}s ===")
        
        return resumen
    
    def _cargar_instancias(self, limite: Optional[int]) -> None:
        """
        Carga las instancias disponibles.
        
        Args:
            limite: Número máximo de instancias a cargar
        """
        instancias_disponibles = self.cargador.obtener_instancias_disponibles()
        
        if limite is not None:
            instancias_disponibles = instancias_disponibles[:limite]
        
        print(f"  Instancias encontradas: {len(instancias_disponibles)}")
        
        for i, nombre_instancia in enumerate(instancias_disponibles):
            try:
                instancia = self.cargador.cargar_instancia(nombre_instancia)
                self.instancias_cargadas.append(instancia)
                
                if (i + 1) % 20 == 0:
                    print(f"  Cargadas: {i + 1}/{len(instancias_disponibles)}")
                    
            except Exception as e:
                print(f"  ✗ Error cargando {nombre_instancia}: {str(e)}")
        
        print(f"  ✓ {len(self.instancias_cargadas)} instancias cargadas exitosamente")
    
    def _ejecutar_algoritmo_todas_instancias(self) -> None:
        """Ejecuta el algoritmo en todas las instancias cargadas."""
        
        for i, instancia in enumerate(self.instancias_cargadas):
            print(f"  [{i+1:3d}/{len(self.instancias_cargadas)}] {instancia.nombre_archivo_instancia}")
            
            try:
                # Ejecuta algoritmo
                solucion = self.algoritmo.resolver(instancia)
                
                # Guarda resultado
                resultado = self._crear_registro_resultado(instancia, solucion)
                self.resultados_experimento.append(resultado)
                
                # Muestra progreso
                print(f"      Costo: {solucion.costo_total:>8.0f}, "
                      f"Vehículos: {solucion.numero_vehiculos_usados:>3d}, "
                      f"Tiempo: {solucion.tiempo_construccion:>6.3f}s")
                
            except Exception as e:
                print(f"      ✗ Error: {str(e)}")
                # Registra el error
                resultado = self._crear_registro_error(instancia, str(e))
                self.resultados_experimento.append(resultado)
    
    def _crear_registro_resultado(self, instancia: MDVSPData, solucion: SolucionMDVSP) -> Dict:
        """
        Crea un registro de resultado para una instancia resuelta.
        
        Args:
            instancia: Instancia original
            solucion: Solución obtenida
            
        Returns:
            Diccionario con datos del resultado
        """
        estadisticas = solucion.obtener_estadisticas()
        
        return {
            'instancia': instancia.nombre_archivo_instancia,
            'numero_depositos': instancia.numero_depositos,
            'numero_viajes': instancia.numero_viajes,
            'vehiculos_disponibles': instancia.numero_total_vehiculos,
            'costo_total': solucion.costo_total,
            'vehiculos_usados': solucion.numero_vehiculos_usados,
            'viajes_asignados': len(solucion.viajes_asignados),
            'tiempo_construccion': solucion.tiempo_construccion,
            'es_factible': solucion.es_factible,
            'rutas_activas': estadisticas['numero_rutas_activas'],
            'viajes_por_ruta_promedio': estadisticas['viajes_por_ruta_promedio'],
            'viajes_por_ruta_max': estadisticas['viajes_por_ruta_max'],
            'viajes_por_ruta_min': estadisticas['viajes_por_ruta_min'],
            'costo_por_vehiculo': estadisticas['costo_por_vehiculo_promedio'],
            'utilizacion_vehiculos': estadisticas['utilizacion_vehiculos'],
            'error': None
        }
    
    def _crear_registro_error(self, instancia: MDVSPData, mensaje_error: str) -> Dict:
        """
        Crea un registro de error para una instancia no resuelta.
        
        Args:
            instancia: Instancia que causó error
            mensaje_error: Descripción del error
            
        Returns:
            Diccionario con datos del error
        """
        return {
            'instancia': instancia.nombre_archivo_instancia,
            'numero_depositos': instancia.numero_depositos,
            'numero_viajes': instancia.numero_viajes,
            'vehiculos_disponibles': instancia.numero_total_vehiculos,
            'costo_total': None,
            'vehiculos_usados': None,
            'viajes_asignados': None,
            'tiempo_construccion': None,
            'es_factible': False,
            'rutas_activas': None,
            'viajes_por_ruta_promedio': None,
            'viajes_por_ruta_max': None,
            'viajes_por_ruta_min': None,
            'costo_por_vehiculo': None,
            'utilizacion_vehiculos': None,
            'error': mensaje_error
        }
    
    def _generar_reportes(self) -> Dict:
        """
        Genera reportes detallados del experimento.
        
        Returns:
            Diccionario con resumen del experimento
        """
        # Filtra resultados exitosos
        resultados_exitosos = [r for r in self.resultados_experimento if r['error'] is None]
        resultados_fallidos = [r for r in self.resultados_experimento if r['error'] is not None]
        
        # Genera archivo CSV con resultados detallados
        self._guardar_resultados_csv()
        
        # Calcula estadísticas agregadas
        resumen = self._calcular_estadisticas_agregadas(resultados_exitosos, resultados_fallidos)
        
        # Genera reporte textual
        self._generar_reporte_textual(resumen)
        
        # Muestra resumen en consola
        self._mostrar_resumen_consola(resumen)
        
        return resumen
    
    def _guardar_resultados_csv(self) -> None:
        """Guarda los resultados detallados en archivo CSV."""
        archivo_csv = self.directorio_resultados / "resultados_detallados.csv"
        
        if not self.resultados_experimento:
            return
        
        # Obtiene todas las columnas
        columnas = list(self.resultados_experimento[0].keys())
        
        with open(archivo_csv, 'w', newline='', encoding='utf-8') as archivo:
            escritor = csv.DictWriter(archivo, fieldnames=columnas)
            escritor.writeheader()
            escritor.writerows(self.resultados_experimento)
        
        print(f"  ✓ Resultados detallados guardados en: {archivo_csv}")
    
    def _calcular_estadisticas_agregadas(self, resultados_exitosos: List[Dict], 
                                       resultados_fallidos: List[Dict]) -> Dict:
        """
        Calcula estadísticas agregadas del experimento.
        
        Args:
            resultados_exitosos: Lista de resultados sin errores
            resultados_fallidos: Lista de resultados con errores
            
        Returns:
            Diccionario con estadísticas agregadas
        """
        if not resultados_exitosos:
            return {
                'total_instancias': len(self.resultados_experimento),
                'instancias_exitosas': 0,
                'instancias_fallidas': len(resultados_fallidos),
                'tasa_exito': 0.0,
                'tiempo_total': self.tiempo_total_experimento
            }
        
        # Estadísticas de costo
        costos = [r['costo_total'] for r in resultados_exitosos]
        
        # Estadísticas de vehículos
        vehiculos_usados = [r['vehiculos_usados'] for r in resultados_exitosos]
        vehiculos_disponibles = [r['vehiculos_disponibles'] for r in resultados_exitosos]
        
        # Estadísticas de tiempo
        tiempos = [r['tiempo_construccion'] for r in resultados_exitosos]
        
        # Estadísticas de utilización
        utilizaciones = [r['utilizacion_vehiculos'] for r in resultados_exitosos]
        
        return {
            'total_instancias': len(self.resultados_experimento),
            'instancias_exitosas': len(resultados_exitosos),
            'instancias_fallidas': len(resultados_fallidos),
            'tasa_exito': (len(resultados_exitosos) / len(self.resultados_experimento)) * 100,
            'tiempo_total': self.tiempo_total_experimento,
            'tiempo_promedio_instancia': sum(tiempos) / len(tiempos),
            'tiempo_min': min(tiempos),
            'tiempo_max': max(tiempos),
            'costo_promedio': sum(costos) / len(costos),
            'costo_min': min(costos),
            'costo_max': max(costos),
            'vehiculos_promedio': sum(vehiculos_usados) / len(vehiculos_usados),
            'vehiculos_disponibles_promedio': sum(vehiculos_disponibles) / len(vehiculos_disponibles),
            'utilizacion_promedio': sum(utilizaciones) / len(utilizaciones),
            'todas_factibles': all(r['es_factible'] for r in resultados_exitosos)
        }
    
    def _generar_reporte_textual(self, resumen: Dict) -> None:
        """
        Genera un reporte textual detallado.
        
        Args:
            resumen: Estadísticas agregadas del experimento
        """
        archivo_reporte = self.directorio_resultados / "reporte_experimento.txt"
        
        with open(archivo_reporte, 'w', encoding='utf-8') as archivo:
            archivo.write("REPORTE DEL EXPERIMENTO - ALGORITMO CONCURRENT SCHEDULE\n")
            archivo.write("=" * 60 + "\n\n")
            
            archivo.write(f"Fecha y hora: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            archivo.write(f"Algoritmo: Concurrent Schedule (Constructivo)\n")
            archivo.write(f"Total instancias procesadas: {resumen['total_instancias']}\n\n")
            
            archivo.write("RESULTADOS GENERALES:\n")
            archivo.write("-" * 30 + "\n")
            archivo.write(f"Instancias exitosas: {resumen['instancias_exitosas']}\n")
            archivo.write(f"Instancias fallidas: {resumen['instancias_fallidas']}\n")
            archivo.write(f"Tasa de éxito: {resumen['tasa_exito']:.1f}%\n")
            archivo.write(f"Tiempo total: {resumen['tiempo_total']:.2f}s\n\n")
            
            if resumen['instancias_exitosas'] > 0:
                archivo.write("ESTADÍSTICAS DE RENDIMIENTO:\n")
                archivo.write("-" * 30 + "\n")
                archivo.write(f"Tiempo promedio por instancia: {resumen['tiempo_promedio_instancia']:.4f}s\n")
                archivo.write(f"Tiempo mínimo: {resumen['tiempo_min']:.4f}s\n")
                archivo.write(f"Tiempo máximo: {resumen['tiempo_max']:.4f}s\n\n")
                
                archivo.write("ESTADÍSTICAS DE CALIDAD:\n")
                archivo.write("-" * 30 + "\n")
                archivo.write(f"Costo promedio: {resumen['costo_promedio']:.0f}\n")
                archivo.write(f"Costo mínimo: {resumen['costo_min']:.0f}\n")
                archivo.write(f"Costo máximo: {resumen['costo_max']:.0f}\n")
                archivo.write(f"Vehículos usados (promedio): {resumen['vehiculos_promedio']:.1f}\n")
                archivo.write(f"Vehículos disponibles (promedio): {resumen['vehiculos_disponibles_promedio']:.1f}\n")
                archivo.write(f"Utilización promedio: {resumen['utilizacion_promedio']:.1f}%\n")
                archivo.write(f"Todas las soluciones factibles: {'Sí' if resumen['todas_factibles'] else 'No'}\n")
        
        print(f"  ✓ Reporte textual guardado en: {archivo_reporte}")
    
    def _mostrar_resumen_consola(self, resumen: Dict) -> None:
        """
        Muestra un resumen del experimento en la consola.
        
        Args:
            resumen: Estadísticas agregadas del experimento
        """
        print(f"\n=== RESUMEN DEL EXPERIMENTO ===")
        print(f"Instancias procesadas: {resumen['total_instancias']}")
        print(f"Tasa de éxito: {resumen['tasa_exito']:.1f}%")
        print(f"Tiempo total: {resumen['tiempo_total']:.2f}s")
        
        if resumen['instancias_exitosas'] > 0:
            print(f"\nRendimiento:")
            print(f"  - Tiempo promedio: {resumen['tiempo_promedio_instancia']:.4f}s")
            print(f"  - Costo promedio: {resumen['costo_promedio']:.0f}")
            print(f"  - Vehículos promedio: {resumen['vehiculos_promedio']:.1f}")
            print(f"  - Utilización promedio: {resumen['utilizacion_promedio']:.1f}%")
    
    def obtener_mejores_resultados(self, criterio: str = 'costo', top: int = 10) -> List[Dict]:
        """
        Obtiene los mejores resultados según un criterio específico.
        
        Args:
            criterio: Criterio de ordenamiento ('costo', 'vehiculos', 'tiempo', 'utilizacion')
            top: Número de resultados a retornar
            
        Returns:
            Lista de los mejores resultados
        """
        resultados_exitosos = [r for r in self.resultados_experimento if r['error'] is None]
        
        if not resultados_exitosos:
            return []
        
        # Define función de ordenamiento según criterio
        if criterio == 'costo':
            key_func = lambda x: x['costo_total']
            reverso = False
        elif criterio == 'vehiculos':
            key_func = lambda x: x['vehiculos_usados']
            reverso = False
        elif criterio == 'tiempo':
            key_func = lambda x: x['tiempo_construccion']
            reverso = False
        elif criterio == 'utilizacion':
            key_func = lambda x: x['utilizacion_vehiculos']
            reverso = True
        else:
            raise ValueError(f"Criterio no válido: {criterio}")
        
        resultados_ordenados = sorted(resultados_exitosos, key=key_func, reverse=reverso)
        return resultados_ordenados[:top]