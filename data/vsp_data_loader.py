"""
Cargador de datos optimizado para instancias del problema VSP.
Implementa la lectura eficiente de archivos con construcción dinámica de restricciones.
"""

import os
import time
from pathlib import Path
from typing import List, Tuple
import numpy as np
from memory_profiler import profile

from data.vsp_data_model import VSPData, DepositoVSP, Servicio


class VSPDataLoader:
    """
    Cargador optimizado para instancias VSP con aplicación dinámica de restricciones.
    Implementa restricciones de factibilidad temporal y de conexión.
    """
    
    def __init__(self, directorio_instancias: str = "instancias_vsp") -> None:
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
    def cargar_instancia_desde_archivos(self, archivo_cst: str, archivo_tim: str) -> VSPData:
        """
        Carga una instancia VSP desde archivos específicos.
        
        Args:
            archivo_cst: Path completo al archivo .cst
            archivo_tim: Path completo al archivo .tim
            
        Returns:
            Objeto VSPData con todos los datos cargados
            
        Raises:
            FileNotFoundError: Si algún archivo no existe
            ValueError: Si hay errores en el formato de los datos
        """
        inicio_tiempo = time.perf_counter()
        
        archivo_cst_path = Path(archivo_cst)
        archivo_tim_path = Path(archivo_tim)
        
        # Valida que los archivos existan
        if not archivo_cst_path.exists():
            raise FileNotFoundError(f"Archivo .cst no encontrado: {archivo_cst_path}")
        
        if not archivo_tim_path.exists():
            raise FileNotFoundError(f"Archivo .tim no encontrado: {archivo_tim_path}")
        
        try:
            # Carga datos básicos
            matriz_costos_base, deposito, numero_servicios = self._cargar_archivo_cst_individual(archivo_cst_path)
            servicios = self._cargar_archivo_tim_individual(archivo_tim_path, numero_servicios)
            
            # Construye matriz de costos con restricciones VSP
            matriz_costos_final = self._construir_matriz_vsp(
                matriz_costos_base, deposito, servicios, numero_servicios
            )
            
            # Nombre de instancia basado en archivos
            nombre_instancia = archivo_cst_path.stem
            
            # Crea la instancia VSP completa
            instancia = VSPData(
                nombre_instancia=nombre_instancia,
                numero_servicios=numero_servicios,
                deposito=deposito,
                servicios=servicios,
                matriz_costos=matriz_costos_final
            )
            
            tiempo_total = time.perf_counter() - inicio_tiempo
            print(f"Instancia VSP '{nombre_instancia}' cargada en {tiempo_total:.4f} segundos")
            
            return instancia
            
        except Exception as e:
            raise ValueError(f"Error cargando instancia VSP desde archivos: {str(e)}") from e

    @profile
    def cargar_instancia(self, nombre_instancia: str) -> VSPData:
        """
        Carga una instancia completa del problema VSP.
        
        Args:
            nombre_instancia: Nombre de la instancia sin extensión
            
        Returns:
            Objeto VSPData con todos los datos cargados
            
        Raises:
            FileNotFoundError: Si los archivos de la instancia no existen
            ValueError: Si hay errores en el formato de los datos
        """
        inicio_tiempo = time.perf_counter()
        
        try:
            # Carga datos básicos
            matriz_costos_base, deposito, numero_servicios = self._cargar_archivo_cst(nombre_instancia)
            servicios = self._cargar_archivo_tim(nombre_instancia, numero_servicios)
            
            # Construye matriz de costos con restricciones VSP
            matriz_costos_final = self._construir_matriz_vsp(
                matriz_costos_base, deposito, servicios, numero_servicios
            )
            
            # Crea la instancia VSP completa
            instancia = VSPData(
                nombre_instancia=nombre_instancia,
                numero_servicios=numero_servicios,
                deposito=deposito,
                servicios=servicios,
                matriz_costos=matriz_costos_final
            )
            
            tiempo_total = time.perf_counter() - inicio_tiempo
            print(f"Instancia VSP {nombre_instancia} cargada en {tiempo_total:.4f} segundos")
            
            return instancia
            
        except Exception as e:
            raise ValueError(f"Error cargando instancia VSP {nombre_instancia}: {str(e)}") from e
    
    def _cargar_archivo_cst(self, nombre_instancia: str) -> Tuple[np.ndarray, DepositoVSP, int]:
        """
        Carga el archivo .cst con matriz de costos básica e información del depósito.
        
        Args:
            nombre_instancia: Nombre de la instancia
            
        Returns:
            Tuple con matriz de costos básica, depósito y número de servicios
        """
        archivo_cst = self.directorio_instancias / f"{nombre_instancia}.cst"
        
        if not archivo_cst.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {archivo_cst}")
        
        with open(archivo_cst, 'r', encoding='utf-8') as archivo:
            try:
                # Lee la primera línea: num_servicios num_vehiculos_deposito
                primera_linea = archivo.readline().strip().split()
                
                if len(primera_linea) < 2:
                    raise ValueError("Primera línea debe contener número de servicios y vehículos")
                
                numero_servicios = int(primera_linea[0])
                numero_vehiculos = int(primera_linea[1])
                
                if numero_servicios <= 0:
                    raise ValueError("Número de servicios debe ser positivo")
                if numero_vehiculos <= 0:
                    raise ValueError("Número de vehículos debe ser positivo")
                
                # Crea el depósito único
                deposito = DepositoVSP(
                    id_deposito=0,
                    numero_vehiculos=numero_vehiculos,
                    nombre_deposito=f"Deposito_{nombre_instancia}"
                )
                
                # Lee matriz de costos (servicios + 1 para depósito)
                matriz_costos_base = self._leer_matriz_costos(archivo, numero_servicios + 1)
                
                return matriz_costos_base, deposito, numero_servicios
                
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
    
    def _cargar_archivo_tim(self, nombre_instancia: str, numero_servicios: int) -> List[Servicio]:
        """
        Carga el archivo .tim con tiempos de inicio y fin de servicios.
        
        Args:
            nombre_instancia: Nombre de la instancia
            numero_servicios: Número esperado de servicios
            
        Returns:
            Lista de objetos Servicio
        """
        archivo_tim = self.directorio_instancias / f"{nombre_instancia}.tim"
        
        if not archivo_tim.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {archivo_tim}")
        
        with open(archivo_tim, 'r', encoding='utf-8') as archivo:
            try:
                contenido = archivo.read()
                valores = contenido.split()
                
                if len(valores) != 2 * numero_servicios:
                    raise ValueError(f"Número de tiempos ({len(valores)}) "
                                   f"no coincide con servicios esperados ({2 * numero_servicios})")
                
                # Separa tiempos de inicio y fin
                tiempos_inicio = [int(valores[i]) for i in range(numero_servicios)]
                tiempos_fin = [int(valores[i + numero_servicios]) for i in range(numero_servicios)]
                
                # Crea objetos Servicio
                servicios = []
                for i in range(numero_servicios):
                    servicio = Servicio(
                        id_servicio=i,
                        tiempo_inicio=tiempos_inicio[i],
                        tiempo_fin=tiempos_fin[i],
                        ubicacion_inicio=f"Inicio_{i}",
                        ubicacion_fin=f"Fin_{i}"
                    )
                    servicios.append(servicio)
                
                return servicios
                
            except (ValueError, IndexError) as e:
                raise ValueError(f"Error en formato del archivo {archivo_tim}: {str(e)}") from e
    
    def _construir_matriz_vsp(self, matriz_base: np.ndarray, deposito: DepositoVSP,
                             servicios: List[Servicio], numero_servicios: int) -> np.ndarray:
        """
        Construye la matriz de costos VSP aplicando todas las restricciones específicas.
        
        Args:
            matriz_base: Matriz de costos básica leída del archivo
            deposito: Depósito del VSP
            servicios: Lista de servicios
            numero_servicios: Número total de servicios
            
        Returns:
            Matriz de costos final con todas las restricciones VSP aplicadas
        """
        INFACTIBLE = 100000000.0
        PROHIBIDO = 0.0
        
        dimension_matriz = numero_servicios + 1  # +1 para el depósito
        restricciones_aplicadas = 0
        
        # Copia la matriz base para modificarla
        matriz_final = matriz_base.copy()
        
        print(f"Construyendo matriz VSP: {numero_servicios} servicios + 1 depósito")
        
        # Aplica restricciones específicas del VSP
        for i in range(dimension_matriz):
            for j in range(dimension_matriz):
                
                # Índice del depósito es el último
                indice_deposito = numero_servicios
                
                # RESTRICCIÓN 1: Conexiones prohibidas por matriz (0 o 100000000)
                if (matriz_final[i, j] == PROHIBIDO or matriz_final[i, j] >= INFACTIBLE):
                    matriz_final[i, j] = INFACTIBLE
                    # Aplicar restricción bidireccional
                    matriz_final[j, i] = INFACTIBLE
                    restricciones_aplicadas += 1
                
                # RESTRICCIÓN 2: Depósito a sí mismo (infactible)
                elif i == indice_deposito and j == indice_deposito:
                    matriz_final[i, j] = INFACTIBLE
                    restricciones_aplicadas += 1
                
                # RESTRICCIÓN 3: Servicio a sí mismo (infactible)
                elif i == j and i < numero_servicios:
                    matriz_final[i, j] = INFACTIBLE
                    restricciones_aplicadas += 1
                
                # RESTRICCIÓN 4: Restricciones temporales entre servicios
                elif i < numero_servicios and j < numero_servicios and i != j:
                    servicio_i = servicios[i]
                    servicio_j = servicios[j]
                    
                    # Verifica traslapes temporales
                    if servicio_i.se_traslapa_con(servicio_j):
                        matriz_final[i, j] = INFACTIBLE
                        matriz_final[j, i] = INFACTIBLE  # Bidireccional
                        restricciones_aplicadas += 1
                    
                    # Verifica precedencia temporal con tiempo de desplazamiento
                    elif not self._es_factible_secuencia_temporal(servicio_i, servicio_j, matriz_final[i, j]):
                        matriz_final[i, j] = INFACTIBLE
                        restricciones_aplicadas += 1
        
        print(f"Restricciones VSP aplicadas: {restricciones_aplicadas}")
        
        # Genera archivo de diagnóstico
        self._generar_archivo_diagnostico_vsp(matriz_final, servicios, nombre_instancia=f"{numero_servicios}_servicios_vsp.csv")
        
        return matriz_final
    
    def _es_factible_secuencia_temporal(self, servicio_origen: Servicio, servicio_destino: Servicio,
                                       tiempo_desplazamiento: float) -> bool:
        """
        Verifica si dos servicios pueden conectarse considerando tiempo de desplazamiento.
        
        Args:
            servicio_origen: Servicio de origen
            servicio_destino: Servicio de destino
            tiempo_desplazamiento: Tiempo necesario para desplazarse entre servicios
            
        Returns:
            True si la secuencia es temporalmente factible
        """
        # Si el tiempo de desplazamiento es infactible, no puede conectarse
        if tiempo_desplazamiento >= 100000000.0:
            return False
    
    def _cargar_archivo_cst_individual(self, archivo_cst: Path) -> Tuple[np.ndarray, DepositoVSP, int]:
        """
        Carga el archivo .cst individual con matriz de costos e información del depósito.
        ADAPTADO: Detecta automáticamente si es formato MDVSP y lo convierte a VSP.
        
        Args:
            archivo_cst: Path al archivo .cst
            
        Returns:
            Tuple con matriz de costos básica, depósito y número de servicios
        """
        with open(archivo_cst, 'r', encoding='utf-8') as archivo:
            try:
                # Lee la primera línea
                primera_linea = archivo.readline().strip().split()
                
                if len(primera_linea) < 2:
                    raise ValueError("Primera línea debe contener al menos 2 valores")
                
                # DETECCIÓN AUTOMÁTICA DEL FORMATO
                if len(primera_linea) == 2:
                    numero_servicios = int(primera_linea[0])
                    numero_vehiculos = int(primera_linea[1])
                    
                    if numero_servicios <= 0 or numero_vehiculos <= 0:
                        raise ValueError("Número de servicios y vehículos debe ser positivo")
                    
                    deposito = DepositoVSP(
                        id_deposito=0,
                        numero_vehiculos=numero_vehiculos,
                        nombre_deposito=f"Deposito_{archivo_cst.stem}"
                    )
                    
                    # Lee matriz VSP: (servicios + 1) x (servicios + 1)
                    matriz_costos_base = self._leer_matriz_costos(archivo, numero_servicios + 1)
                    
                    return matriz_costos_base, deposito, numero_servicios
                
                else:
                    # FORMATO MDVSP: num_depositos num_viajes num_veh_dep1 num_veh_dep2 ...
                    print(f"  Detectado formato MDVSP, convirtiendo a VSP...")
                    
                    numero_depositos = int(primera_linea[0])
                    numero_viajes = int(primera_linea[1])  # Estos serán nuestros "servicios"
                    
                    if numero_depositos <= 0 or numero_viajes <= 0:
                        raise ValueError("Número de depósitos y viajes debe ser positivo")
                    
                    if len(primera_linea) != (2 + numero_depositos):
                        raise ValueError(f"Primera línea debe contener {2 + numero_depositos} valores para formato MDVSP")
                    
                    # Suma todos los vehículos de todos los depósitos
                    numero_total_vehiculos = sum(int(primera_linea[2 + i]) for i in range(numero_depositos))
                    
                    deposito_vsp = DepositoVSP(
                        id_deposito=0,
                        numero_vehiculos=numero_total_vehiculos,
                        nombre_deposito=f"DepositoUnificado_{archivo_cst.stem}"
                    )
                    
                    # Lee matriz MDVSP completa: (depositos + viajes) x (depositos + viajes) 
                    dimension_mdvsp = numero_depositos + numero_viajes
                    matriz_mdvsp = self._leer_matriz_costos(archivo, dimension_mdvsp)
                    
                    # CONVIERTE matriz MDVSP a formato VSP
                    matriz_vsp = self._convertir_matriz_mdvsp_a_vsp(
                        matriz_mdvsp, numero_depositos, numero_viajes
                    )
                    
                    print(f"  Conversión completada: {numero_viajes} viajes -> servicios, {numero_total_vehiculos} vehículos total")
                    
                    return matriz_vsp, deposito_vsp, numero_viajes
                
            except (ValueError, IndexError) as e:
                raise ValueError(f"Error en formato del archivo {archivo_cst}: {str(e)}") from e
    
    def _convertir_matriz_mdvsp_a_vsp(self, matriz_mdvsp: np.ndarray, 
                                     numero_depositos: int, numero_viajes: int) -> np.ndarray:
        """
        Convierte una matriz MDVSP a formato VSP.
        
        MDVSP: [Viajes | Depositos]  ->  VSP: [Servicios | Deposito_Unificado]
               [Viajes | Depositos]           [Servicios | Deposito_Unificado]
        
        Args:
            matriz_mdvsp: Matriz MDVSP original
            numero_depositos: Número de depósitos en MDVSP
            numero_viajes: Número de viajes (que serán servicios en VSP)
            
        Returns:
            Matriz VSP de tamaño (numero_viajes + 1) x (numero_viajes + 1)
        """
        INFACTIBLE = 100000000.0
        
        # Crea matriz VSP: servicios + 1 depósito unificado
        dimension_vsp = numero_viajes + 1
        matriz_vsp = np.full((dimension_vsp, dimension_vsp), INFACTIBLE, dtype=np.float64)
        
        # PASO 1: Copia la submatriz viaje-viaje (servicios entre sí)
        for i in range(numero_viajes):
            for j in range(numero_viajes):
                matriz_vsp[i, j] = matriz_mdvsp[i, j]
        
        # PASO 2: Calcula costos desde depósito unificado a servicios
        # Usa el MÍNIMO costo de todos los depósitos MDVSP
        indice_deposito_vsp = numero_viajes  # Último índice
        
        for servicio in range(numero_viajes):
            # Costo mínimo desde cualquier depósito MDVSP al servicio
            costos_depositos_a_servicio = []
            for dep in range(numero_depositos):
                indice_dep_mdvsp = numero_viajes + dep
                costo = matriz_mdvsp[indice_dep_mdvsp, servicio]
                if costo < INFACTIBLE:
                    costos_depositos_a_servicio.append(costo)
            
            if costos_depositos_a_servicio:
                matriz_vsp[indice_deposito_vsp, servicio] = min(costos_depositos_a_servicio)
            else:
                matriz_vsp[indice_deposito_vsp, servicio] = INFACTIBLE
        
        # PASO 3: Calcula costos desde servicios al depósito unificado  
        for servicio in range(numero_viajes):
            # Costo mínimo desde servicio a cualquier depósito MDVSP
            costos_servicio_a_depositos = []
            for dep in range(numero_depositos):
                indice_dep_mdvsp = numero_viajes + dep
                costo = matriz_mdvsp[servicio, indice_dep_mdvsp]
                if costo < INFACTIBLE:
                    costos_servicio_a_depositos.append(costo)
            
            if costos_servicio_a_depositos:
                matriz_vsp[servicio, indice_deposito_vsp] = min(costos_servicio_a_depositos)
            else:
                matriz_vsp[servicio, indice_deposito_vsp] = INFACTIBLE
        
        # PASO 4: Depósito a sí mismo (infactible)
        matriz_vsp[indice_deposito_vsp, indice_deposito_vsp] = INFACTIBLE
        
        return matriz_vsp
    
    def _cargar_archivo_tim_individual(self, archivo_tim: Path, numero_servicios: int) -> List[Servicio]:
        """
        Carga el archivo .tim individual con tiempos de servicios.
        
        Args:
            archivo_tim: Path al archivo .tim
            numero_servicios: Número esperado de servicios
            
        Returns:
            Lista de objetos Servicio
        """
        with open(archivo_tim, 'r', encoding='utf-8') as archivo:
            try:
                contenido = archivo.read()
                valores = contenido.split()
                
                if len(valores) != 2 * numero_servicios:
                    raise ValueError(f"Número de tiempos ({len(valores)}) "
                                   f"no coincide con servicios esperados ({2 * numero_servicios})")
                
                # Separa tiempos de inicio y fin
                tiempos_inicio = [int(valores[i]) for i in range(numero_servicios)]
                tiempos_fin = [int(valores[i + numero_servicios]) for i in range(numero_servicios)]
                
                # Crea objetos Servicio
                servicios = []
                for i in range(numero_servicios):
                    servicio = Servicio(
                        id_servicio=i,
                        tiempo_inicio=tiempos_inicio[i],
                        tiempo_fin=tiempos_fin[i],
                        ubicacion_inicio=f"Inicio_{i}",
                        ubicacion_fin=f"Fin_{i}"
                    )
                    servicios.append(servicio)
                
                return servicios
                
            except (ValueError, IndexError) as e:
                raise ValueError(f"Error en formato del archivo {archivo_tim}: {str(e)}") from e
        
        # Verifica que el servicio origen termine + tiempo desplazamiento <= inicio servicio destino
        tiempo_llegada = servicio_origen.tiempo_fin + tiempo_desplazamiento
        return tiempo_llegada <= servicio_destino.tiempo_inicio
    
    def _generar_archivo_diagnostico_vsp(self, matriz: np.ndarray, servicios: List[Servicio],
                                        nombre_instancia: str) -> None:
        """
        Genera archivo CSV con la matriz de costos VSP para diagnóstico.
        
        Args:
            matriz: Matriz de costos a exportar
            servicios: Lista de servicios para información adicional
            nombre_instancia: Nombre del archivo de salida
        """
        archivo_salida = Path(nombre_instancia)
        
        try:
            with open(archivo_salida, 'w', encoding='utf-8') as archivo:
                # Encabezado con información de la instancia
                archivo.write(f"# Matriz VSP: {nombre_instancia}\n")
                archivo.write(f"# Servicios: {len(servicios)}\n")
                archivo.write(f"# Dimensión: {matriz.shape[0]}x{matriz.shape[1]}\n")
                archivo.write(f"# Servicios (ID:Inicio-Fin):")
                for servicio in servicios:
                    archivo.write(f" {servicio.id_servicio}:{servicio.tiempo_inicio}-{servicio.tiempo_fin}")
                archivo.write("\n# Última fila/columna = Depósito\n\n")
                
                # Matriz de costos
                for i in range(matriz.shape[0]):
                    fila = ";".join(f"{matriz[i, j]:.0f}" for j in range(matriz.shape[1]))
                    archivo.write(f"{fila};\n")
        
        except IOError as e:
            print(f"Advertencia: No se pudo generar archivo de diagnóstico VSP: {e}")
    
    def cargar_todas_las_instancias(self) -> List[VSPData]:
        """
        Carga todas las instancias VSP disponibles en el directorio.
        
        Returns:
            Lista de objetos VSPData con todas las instancias
        """
        instancias_disponibles = self.obtener_instancias_disponibles()
        instancias_cargadas = []
        
        print(f"Cargando {len(instancias_disponibles)} instancias VSP...")
        
        for nombre_instancia in instancias_disponibles:
            try:
                instancia = self.cargar_instancia(nombre_instancia)
                instancias_cargadas.append(instancia)
                print(f"✓ {nombre_instancia} VSP cargada exitosamente")
            except Exception as e:
                print(f"✗ Error cargando VSP {nombre_instancia}: {str(e)}")
        
        return instancias_cargadas
    
    def validar_integridad_instancia(self, instancia: VSPData) -> bool:
        """
        Valida la integridad de una instancia VSP cargada.
        
        Args:
            instancia: Instancia VSP a validar
            
        Returns:
            True si la instancia es válida
        """
        try:
            # Valida dimensiones básicas
            if instancia.numero_servicios <= 0:
                print("Error: Número de servicios debe ser positivo")
                return False
            
            # Valida consistencia de datos
            if len(instancia.servicios) != instancia.numero_servicios:
                print("Error: Inconsistencia en número de servicios")
                return False
            
            # Valida matriz de costos
            dimension_esperada = instancia.numero_servicios + 1
            if instancia.matriz_costos.shape != (dimension_esperada, dimension_esperada):
                print(f"Error: Matriz debe ser {dimension_esperada}x{dimension_esperada}")
                return False
            
            # Valida que no haya traslapes temporales entre servicios
            for i in range(len(instancia.servicios)):
                for j in range(i + 1, len(instancia.servicios)):
                    if instancia.servicios[i].se_traslapa_con(instancia.servicios[j]):
                        print(f"Error: Servicios {i} y {j} tienen traslapes temporales")
                        return False
            
            # Valida que haya al menos algunas conexiones factibles
            stats = instancia.obtener_estadisticas()
            if stats['conexiones_factibles'] == 0:
                print("Error: No existen conexiones factibles entre servicios")
                return False
            
            print(f"✓ Instancia VSP válida: {stats['conexiones_factibles']} conexiones factibles")
            return True
            
        except Exception as e:
            print(f"Error validando instancia VSP: {str(e)}")
            return False