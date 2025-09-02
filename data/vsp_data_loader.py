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
            Tupla con (matriz_costos, deposito, numero_servicios)
        """
        archivo_cst = self.directorio_instancias / f"{nombre_instancia}.cst"
        return self._cargar_archivo_cst_individual(archivo_cst)
    
    def _cargar_archivo_cst_individual(self, archivo_cst: Path) -> Tuple[np.ndarray, DepositoVSP, int]:
        """
        Carga el archivo .cst individual con matriz de costos e información del depósito.
        ADAPTADO: Detecta automáticamente si es formato MDVSP y lo convierte a VSP.
        
        Args:
            archivo_cst: Path al archivo .cst
            
        Returns:
            Tupla con (matriz_costos, deposito, numero_servicios)
        """
        if not archivo_cst.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {archivo_cst}")
        
        with open(archivo_cst, 'r', encoding='utf-8') as archivo:
            try:
                contenido = archivo.read()
                lineas = contenido.strip().split('\n')
                
                # Carga línea de cabecera
                cabecera = lineas[0].split()
                if len(cabecera) < 3:
                    raise ValueError("Cabecera del archivo debe tener al menos 3 valores")
                
                numero_servicios = int(cabecera[0])
                numero_depositos = int(cabecera[1])
                total_vehiculos = int(cabecera[2])
                
                # Detecta formato MDVSP y convierte a VSP
                if numero_depositos > 1:
                    print(f"  Detectado formato MDVSP, convirtiendo a VSP...")
                    vehiculos_por_deposito = [int(x) for x in cabecera[3:3 + numero_depositos]]
                    total_vehiculos_mdvsp = sum(vehiculos_por_deposito)
                    print(f"  Conversión completada: {numero_servicios} viajes -> servicios, {total_vehiculos_mdvsp} vehículos total")
                    # Para VSP, usa el total de vehículos disponibles
                    total_vehiculos = total_vehiculos_mdvsp
                
                # Carga matriz de costos
                dimension_matriz = numero_servicios + numero_depositos
                matriz = np.zeros((dimension_matriz, dimension_matriz), dtype=float)
                
                for i in range(1, len(lineas)):
                    if lineas[i].strip():
                        valores = lineas[i].strip().split()
                        if len(valores) >= dimension_matriz:
                            fila_idx = i - 1
                            for j in range(dimension_matriz):
                                try:
                                    matriz[fila_idx, j] = float(valores[j])
                                except (ValueError, IndexError):
                                    matriz[fila_idx, j] = 100000000.0  # Costo infactible por defecto
                
                # Para MDVSP, convierte a formato VSP (matriz de servicios + 1 depósito único)
                if numero_depositos > 1:
                    matriz_vsp = self._convertir_mdvsp_a_vsp(matriz, numero_servicios, numero_depositos)
                else:
                    matriz_vsp = matriz
                
                # Crea depósito VSP único
                deposito = DepositoVSP(
                    id_deposito=0,
                    numero_vehiculos=total_vehiculos,
                    nombre_deposito=f"Deposito_VSP_{archivo_cst.stem}",
                    ubicacion="Centro"
                )
                
                return matriz_vsp, deposito, numero_servicios
                
            except (ValueError, IndexError) as e:
                raise ValueError(f"Error en formato del archivo {archivo_cst}: {str(e)}") from e
    
    def _convertir_mdvsp_a_vsp(self, matriz_mdvsp: np.ndarray, numero_servicios: int, numero_depositos: int) -> np.ndarray:
        """
        Convierte una matriz MDVSP (Multiple Depot) a formato VSP (Single Depot).
        Combina múltiples depósitos en uno solo tomando el mínimo costo.
        
        Args:
            matriz_mdvsp: Matriz original con múltiples depósitos
            numero_servicios: Número de servicios en la instancia
            numero_depositos: Número de depósitos en MDVSP
            
        Returns:
            Matriz VSP con un solo depósito
        """
        # Nueva dimensión: servicios + 1 depósito único
        nueva_dimension = numero_servicios + 1
        matriz_vsp = np.zeros((nueva_dimension, nueva_dimension), dtype=float)
        
        # Copia costos entre servicios (permanecen iguales)
        for i in range(numero_servicios):
            for j in range(numero_servicios):
                matriz_vsp[i, j] = matriz_mdvsp[i, j]
        
        # Para conexiones depósito -> servicios, toma el mínimo entre todos los depósitos
        indice_deposito_vsp = numero_servicios
        for j in range(numero_servicios):
            costos_desde_depositos = []
            for d in range(numero_depositos):
                indice_deposito_mdvsp = numero_servicios + d
                if indice_deposito_mdvsp < matriz_mdvsp.shape[0]:
                    costos_desde_depositos.append(matriz_mdvsp[indice_deposito_mdvsp, j])
            
            if costos_desde_depositos:
                matriz_vsp[indice_deposito_vsp, j] = min(costos_desde_depositos)
            else:
                matriz_vsp[indice_deposito_vsp, j] = 100000000.0
        
        # Para conexiones servicios -> depósito, toma el mínimo hacia cualquier depósito
        for i in range(numero_servicios):
            costos_hacia_depositos = []
            for d in range(numero_depositos):
                indice_deposito_mdvsp = numero_servicios + d
                if indice_deposito_mdvsp < matriz_mdvsp.shape[1]:
                    costos_hacia_depositos.append(matriz_mdvsp[i, indice_deposito_mdvsp])
            
            if costos_hacia_depositos:
                matriz_vsp[i, indice_deposito_vsp] = min(costos_hacia_depositos)
            else:
                matriz_vsp[i, indice_deposito_vsp] = 100000000.0
        
        # Depósito a sí mismo: infactible
        matriz_vsp[indice_deposito_vsp, indice_deposito_vsp] = 100000000.0
        
        return matriz_vsp
    
    def _cargar_archivo_tim(self, nombre_instancia: str, numero_servicios: int) -> List[Servicio]:
        """
        Carga el archivo .tim con tiempos de servicios.
        
        Args:
            nombre_instancia: Nombre de la instancia
            numero_servicios: Número esperado de servicios
            
        Returns:
            Lista de objetos Servicio
        """
        archivo_tim = self.directorio_instancias / f"{nombre_instancia}.tim"
        return self._cargar_archivo_tim_individual(archivo_tim, numero_servicios)
    
    def _cargar_archivo_tim_individual(self, archivo_tim: Path, numero_servicios: int) -> List[Servicio]:
        """
        Carga el archivo .tim individual con tiempos de servicios.
        
        Args:
            archivo_tim: Path al archivo .tim
            numero_servicios: Número esperado de servicios
            
        Returns:
            Lista de objetos Servicio
        """
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
                    
                    # Verifica traslapes temporales - los marca como infactibles
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
        
        # El servicio destino debe iniciar después de que termine el origen + tiempo de desplazamiento
        tiempo_llegada_mas_temprana = servicio_origen.tiempo_fin + int(tiempo_desplazamiento)
        
        return tiempo_llegada_mas_temprana <= servicio_destino.tiempo_inicio
    
    def _generar_archivo_diagnostico_vsp(self, matriz: np.ndarray, servicios: List[Servicio], nombre_instancia: str) -> None:
        """
        Genera un archivo de diagnóstico con la matriz VSP construida.
        
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
        Reporta traslapes temporales como información pero no falla por ellos.
        
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
            
            # Reporta traslapes temporales como información (no como error)
            traslapes_detectados = 0
            for i in range(len(instancia.servicios)):
                for j in range(i + 1, len(instancia.servicios)):
                    if instancia.servicios[i].se_traslapa_con(instancia.servicios[j]):
                        traslapes_detectados += 1
            
            if traslapes_detectados > 0:
                print(f"Información: Detectados {traslapes_detectados} pares de servicios con traslapes temporales "
                      f"(conexiones marcadas como infactibles)")
            
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