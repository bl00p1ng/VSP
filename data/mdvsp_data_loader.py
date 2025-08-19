"""
Cargador de datos optimizado para instancias del problema MDVSP.
Implementa la lectura eficiente de archivos .cst y .tim con validación robusta.
"""

import os
import time
from pathlib import Path
from typing import List, Tuple
import numpy as np
from memory_profiler import profile

from .mdvsp_data_model import MDVSPData, Deposito, Viaje


class MDVSPDataLoader:
    """
    Cargador optimizado para instancias MDVSP con capacidades de monitoreo de rendimiento.
    Implementa el patrón Single Responsibility para la carga de datos.
    """
    
    def __init__(self, directorio_instancias: str = "fischetti") -> None:
        """
        Inicializa el cargador con el directorio de instancias.
        
        Args:
            directorio_instancias: Ruta al directorio que contiene los archivos de instancias
        """
        self.directorio_instancias = Path(directorio_instancias)
        self._validar_directorio()
    
    def _validar_directorio(self) -> None:
        """Valida que el directorio de instancias exista y sea accesible."""
        if not self.directorio_instancias.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {self.directorio_instancias}")
        
        if not self.directorio_instancias.is_dir():
            raise NotADirectoryError(f"La ruta no es un directorio: {self.directorio_instancias}")
    
    def obtener_instancias_disponibles(self) -> List[str]:
        """
        Obtiene la lista de instancias disponibles en el directorio.
        
        Returns:
            Lista de nombres de instancias sin extensión
        """
        archivos_cst = set()
        archivos_tim = set()
        
        for archivo in self.directorio_instancias.iterdir():
            if archivo.suffix == ".cst":
                archivos_cst.add(archivo.stem)
            elif archivo.suffix == ".tim":
                archivos_tim.add(archivo.stem)
        
        # Retorna solo las instancias que tienen ambos archivos
        instancias_completas = sorted(archivos_cst.intersection(archivos_tim))
        return instancias_completas
    
    @profile
    def cargar_instancia(self, nombre_instancia: str) -> MDVSPData:
        """
        Carga una instancia completa del problema MDVSP.
        
        Args:
            nombre_instancia: Nombre de la instancia sin extensión
            
        Returns:
            Objeto MDVSPData con todos los datos cargados
            
        Raises:
            FileNotFoundError: Si los archivos de la instancia no existen
            ValueError: Si hay errores en el formato de los datos
        """
        inicio_tiempo = time.perf_counter()
        
        try:
            # Carga archivos por separado para mejor control de errores
            matriz_costos, depositos, numero_viajes = self._cargar_archivo_cst(nombre_instancia)
            viajes = self._cargar_archivo_tim(nombre_instancia, numero_viajes)
            
            # Calcula el total de vehículos
            numero_total_vehiculos = sum(deposito.numero_vehiculos for deposito in depositos)
            
            # Crea la instancia completa
            instancia = MDVSPData(
                nombre_archivo_instancia=nombre_instancia,
                numero_depositos=len(depositos),
                numero_viajes=numero_viajes,
                numero_total_vehiculos=numero_total_vehiculos,
                depositos=depositos,
                viajes=viajes,
                matriz_viajes=matriz_costos
            )
            
            tiempo_total = time.perf_counter() - inicio_tiempo
            print(f"Instancia {nombre_instancia} cargada en {tiempo_total:.4f} segundos")
            
            return instancia
            
        except Exception as e:
            raise ValueError(f"Error cargando instancia {nombre_instancia}: {str(e)}") from e
    
    def _cargar_archivo_cst(self, nombre_instancia: str) -> Tuple[np.ndarray, List[Deposito], int]:
        """
        Carga el archivo .cst con matriz de costos e información de depósitos.
        
        Args:
            nombre_instancia: Nombre de la instancia
            
        Returns:
            Tuple con matriz de costos, lista de depósitos y número de viajes
        """
        archivo_cst = self.directorio_instancias / f"{nombre_instancia}.cst"
        
        if not archivo_cst.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {archivo_cst}")
        
        with open(archivo_cst, 'r', encoding='utf-8') as archivo:
            try:
                # Lee la primera línea que contiene: num_depositos num_viajes num_veh_dep1 num_veh_dep2 ...
                primera_linea = archivo.readline().strip().split()
                
                if len(primera_linea) < 2:
                    raise ValueError("Primera línea debe contener al menos número de depósitos y viajes")
                
                numero_depositos = int(primera_linea[0])
                numero_viajes = int(primera_linea[1])
                
                if numero_depositos <= 0:
                    raise ValueError("Número de depósitos debe ser positivo")
                if numero_viajes <= 0:
                    raise ValueError("Número de viajes debe ser positivo")
                
                # Extrae número de vehículos por depósito desde la primera línea
                if len(primera_linea) != (2 + numero_depositos):
                    raise ValueError(f"Primera línea debe contener {2 + numero_depositos} valores")
                
                depositos = []
                for i in range(numero_depositos):
                    numero_vehiculos = int(primera_linea[2 + i])
                    deposito = Deposito(id_deposito=i, numero_vehiculos=numero_vehiculos)
                    depositos.append(deposito)
                
                # Lee matriz de costos
                matriz_costos = self._leer_matriz_costos(archivo, numero_viajes + numero_depositos)
                
                return matriz_costos, depositos, numero_viajes
                
            except (ValueError, IndexError) as e:
                raise ValueError(f"Error en formato del archivo {archivo_cst}: {str(e)}") from e
    
    def _leer_matriz_costos(self, archivo, dimension: int) -> np.ndarray:
        """
        Lee la matriz de costos de forma optimizada.
        
        Args:
            archivo: Handle del archivo abierto
            dimension: Dimensión de la matriz cuadrada
            
        Returns:
            Matriz de costos como numpy array
        """
        # Pre-alloca la matriz para mejor rendimiento
        matriz = np.empty((dimension, dimension), dtype=np.float64)
        
        # Lee todos los valores restantes del archivo
        contenido_restante = archivo.read()
        valores = contenido_restante.split()
        
        if len(valores) != dimension * dimension:
            raise ValueError(f"Número de valores en matriz ({len(valores)}) "
                           f"no coincide con dimensión esperada ({dimension * dimension})")
        
        # Convierte y asigna valores de forma vectorizada
        try:
            valores_numericos = np.array([float(valor) for valor in valores], dtype=np.float64)
            matriz = valores_numericos.reshape((dimension, dimension))
        except ValueError as e:
            raise ValueError(f"Error convirtiendo valores de matriz: {str(e)}") from e
        
        return matriz
    
    def _cargar_archivo_tim(self, nombre_instancia: str, numero_viajes: int) -> List[Viaje]:
        """
        Carga el archivo .tim con tiempos de inicio y fin de viajes.
        
        Args:
            nombre_instancia: Nombre de la instancia
            numero_viajes: Número esperado de viajes
            
        Returns:
            Lista de objetos Viaje
        """
        archivo_tim = self.directorio_instancias / f"{nombre_instancia}.tim"
        
        if not archivo_tim.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {archivo_tim}")
        
        with open(archivo_tim, 'r', encoding='utf-8') as archivo:
            try:
                contenido = archivo.read()
                valores = contenido.split()
                
                if len(valores) != 2 * numero_viajes:
                    raise ValueError(f"Número de tiempos ({len(valores)}) "
                                   f"no coincide con viajes esperados ({2 * numero_viajes})")
                
                # Separa tiempos de inicio y fin
                tiempos_inicio = [int(valores[i]) for i in range(numero_viajes)]
                tiempos_fin = [int(valores[i + numero_viajes]) for i in range(numero_viajes)]
                
                # Crea objetos Viaje
                viajes = []
                for i in range(numero_viajes):
                    viaje = Viaje(
                        id_viaje=i,
                        tiempo_inicio=tiempos_inicio[i],
                        tiempo_fin=tiempos_fin[i]
                    )
                    viajes.append(viaje)
                
                return viajes
                
            except (ValueError, IndexError) as e:
                raise ValueError(f"Error en formato del archivo {archivo_tim}: {str(e)}") from e
    
    def cargar_todas_las_instancias(self) -> List[MDVSPData]:
        """
        Carga todas las instancias disponibles en el directorio.
        
        Returns:
            Lista de objetos MDVSPData con todas las instancias
        """
        instancias_disponibles = self.obtener_instancias_disponibles()
        instancias_cargadas = []
        
        print(f"Cargando {len(instancias_disponibles)} instancias...")
        
        for nombre_instancia in instancias_disponibles:
            try:
                instancia = self.cargar_instancia(nombre_instancia)
                instancias_cargadas.append(instancia)
                print(f"✓ {nombre_instancia} cargada exitosamente")
            except Exception as e:
                print(f"✗ Error cargando {nombre_instancia}: {str(e)}")
        
        return instancias_cargadas
    
    def validar_integridad_instancia(self, instancia: MDVSPData) -> bool:
        """
        Valida la integridad de una instancia cargada.
        
        Args:
            instancia: Instancia a validar
            
        Returns:
            True si la instancia es válida, False en caso contrario
        """
        try:
            # Valida consistencia temporal de viajes
            for viaje in instancia.viajes:
                if viaje.tiempo_inicio >= viaje.tiempo_fin:
                    print(f"Error: Viaje {viaje.id_viaje} tiene tiempos inconsistentes")
                    return False
            
            # Valida matriz de costos
            if instancia.matriz_viajes.size == 0:
                print("Error: Matriz de costos vacía")
                return False
            
            # Valida que existan aristas factibles
            estadisticas = instancia.calcular_estadisticas_factibilidad()
            if estadisticas['aristas_factibles'] == 0:
                print("Error: No existen aristas factibles")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error validando instancia: {str(e)}")
            return False