# =============================================================================
# PROYECTO TFM: SISTEMA DE PROCESAMIENTO DISTRIBUIDO DE DATOS CENSALES
# =============================================================================
#                    Universidad de Málaga
# Máster en Ingeniería del Software e Inteligencia Artificial
#
# Título: Sistema de procesamiento distribuido de datos censales
#
# Proyecto: misia-procesamiento-distribuido-sociodemografico
# Versión: 1.0.0
# Fecha: 2025
#
# Autores: - Ramiro Ricardo Merchán Mora
#
# Director: Antonio Jesús Nebro Urbaneja
#
# =============================================================================
# ARCHIVO: etl/bronze/parametric_ingestion.py
# =============================================================================
#
# Propósito: Sistema de ingesta parametrizada para fuentes de datos heterogéneas
#
# Funcionalidades principales:
# - Ingesta configurable basada en archivos de parámetros CSV
# - Procesamiento distribuido optimizado para datasets censales masivos
# - Validación automática de integridad y calidad de datos
# - Generación de reportes detallados de ingesta para auditoría académica
#
# Dependencias:
# - pyspark.sql
# - pandas (para configuración de parámetros)
# - core.logger
# - config.data_config
#
# Uso:
# from etl.bronze.parametric_ingestion import ParametricIngestionEngine
# engine = ParametricIngestionEngine(spark_session, base_path)
# results = engine.execute_parametric_ingestion()
#
# =============================================================================

import os
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, count, sum as spark_sum, avg

from core.logger import get_logger, TFMLogger
from config.data_config import get_data_config


@dataclass
class IngestionConfiguration:
    """
    Configuración para una operación de ingesta específica.
    
    Esta clase encapsula todos los parámetros necesarios para ejecutar
    la ingesta de un archivo específico, incluyendo rutas, delimitadores,
    opciones de particionamiento y validaciones.
    """
    id_ingesta: str
    nombre_archivo: str
    nombre_tabla: str
    path_origen: str
    path_destino: str
    delimitador: str = ","
    columna_particion: Optional[str] = None
    grupo_ingesta: str = "default"
    orden_grupo: int = 1
    estado: int = 1  # 1=activo, 0=inactivo
    validaciones_requeridas: List[str] = None
    
    def __post_init__(self):
        """Inicialización posterior para validar configuración."""
        if self.validaciones_requeridas is None:
            self.validaciones_requeridas = []


@dataclass
class IngestionResult:
    """
    Resultado de una operación de ingesta parametrizada.
    
    Esta clase almacena todos los detalles relevantes del proceso
    de ingesta, facilitando la generación de reportes y análisis
    de rendimiento posterior.
    """
    id_ingesta: str
    success: bool
    records_processed: int
    columns_count: int
    processing_time_seconds: float
    file_size_mb: float
    throughput_records_per_second: float
    partition_applied: Optional[str] = None
    validation_results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class ParametricConfigurationLoader:
    """
    Cargador de configuraciones parametrizadas desde archivos CSV.
    
    Esta clase implementa la lógica de lectura y validación de archivos
    de configuración que definen los parámetros de ingesta para cada
    fuente de datos del sistema.
    """
    
    def __init__(self, base_path: str):
        """
        Inicializa el cargador de configuraciones.
        
        Args:
            base_path (str): Ruta base del Data Lake
        """
        self.base_path = Path(base_path)
        self.logger = get_logger(__name__)
        
        # Ruta estándar del archivo de configuración
        self.config_file_path = self.base_path / "sftp" / "parametria_ingesta.csv"
    
    def load_active_configurations(self) -> List[IngestionConfiguration]:
        """
        Carga todas las configuraciones activas desde el archivo de parámetros.
        
        Returns:
            List[IngestionConfiguration]: Lista de configuraciones activas
            
        Raises:
            FileNotFoundError: Si el archivo de configuración no existe
            ValueError: Si el archivo tiene formato inválido
        """
        self.logger.info("Cargando configuraciones de ingesta parametrizada")
        
        # Verificar que el archivo de configuración existe
        if not self.config_file_path.exists():
            error_msg = f"Archivo de configuración no encontrado: {self.config_file_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            # Cargar archivo CSV con configuraciones
            df_config = pd.read_csv(self.config_file_path, encoding='utf-8')
            
            # Validar columnas requeridas
            required_columns = [
                'id_ingesta', 'estado', 'nombre_archivo', 'nombre_tabla',
                'path_origen', 'path_destino'
            ]
            
            missing_columns = [col for col in required_columns if col not in df_config.columns]
            if missing_columns:
                raise ValueError(f"Columnas requeridas faltantes: {missing_columns}")
            
            # Filtrar solo configuraciones activas
            active_configs = df_config[df_config['estado'] == 1].copy()
            
            if active_configs.empty:
                self.logger.warning("No se encontraron configuraciones activas")
                return []
            
            # Ordenar por grupo y orden
            if 'grupo_ingesta' in active_configs.columns and 'orden_grupo' in active_configs.columns:
                active_configs = active_configs.sort_values(['grupo_ingesta', 'orden_grupo'])
            
            # Convertir a objetos de configuración
            configurations = []
            for _, row in active_configs.iterrows():
                config = IngestionConfiguration(
                    id_ingesta=str(row['id_ingesta']),
                    nombre_archivo=str(row['nombre_archivo']),
                    nombre_tabla=str(row['nombre_tabla']),
                    path_origen=str(row['path_origen']),
                    path_destino=str(row['path_destino']),
                    delimitador=str(row.get('delimitador', ',')),
                    columna_particion=str(row['columna_particion']) if pd.notna(row.get('columna_particion')) else None,
                    grupo_ingesta=str(row.get('grupo_ingesta', 'default')),
                    orden_grupo=int(row.get('orden_grupo', 1)),
                    estado=int(row['estado'])
                )
                configurations.append(config)
            
            self.logger.info(f"Configuraciones cargadas: {len(configurations)} activas")
            
            # Log resumen por grupo
            self._log_configuration_summary(configurations)
            
            return configurations
            
        except Exception as e:
            error_msg = f"Error cargando configuraciones: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _log_configuration_summary(self, configurations: List[IngestionConfiguration]) -> None:
        """
        Genera un log resumen de las configuraciones cargadas.
        
        Args:
            configurations (List[IngestionConfiguration]): Configuraciones cargadas
        """
        # Agrupar por grupo de ingesta
        groups = {}
        for config in configurations:
            group = config.grupo_ingesta
            if group not in groups:
                groups[group] = []
            groups[group].append(config)
        
        self.logger.info("Resumen de configuraciones por grupo:")
        for group_name, group_configs in groups.items():
            self.logger.info(f"  Grupo '{group_name}': {len(group_configs)} archivos")
            for config in sorted(group_configs, key=lambda x: x.orden_grupo):
                self.logger.info(
                    f"    {config.orden_grupo}. {config.nombre_archivo} -> {config.nombre_tabla}"
                )


class DistributedIngestionProcessor:
    """
    Procesador distribuido de ingesta para datasets masivos.
    
    Esta clase implementa la lógica de procesamiento distribuido optimizada
    para la ingesta de datasets censales, aprovechando las capacidades de
    paralelización de Apache Spark.
    """
    
    def __init__(self, spark: SparkSession):
        """
        Inicializa el procesador distribuido.
        
        Args:
            spark (SparkSession): Sesión de Spark configurada
        """
        self.spark = spark
        self.logger = get_logger(__name__)
    
    def process_file_ingestion(self, config: IngestionConfiguration) -> IngestionResult:
        """
        Procesa la ingesta de un archivo individual con optimizaciones distribuidas.
        
        Args:
            config (IngestionConfiguration): Configuración de ingesta
            
        Returns:
            IngestionResult: Resultado detallado del procesamiento
        """
        start_time = datetime.now()
        
        self.logger.info(f"Iniciando ingesta distribuida: {config.id_ingesta}")
        self.logger.info(f"Archivo origen: {config.nombre_archivo}")
        self.logger.info(f"Tabla destino: {config.nombre_tabla}")
        
        try:
            # Construir rutas completas
            source_path = os.path.join(config.path_origen, config.nombre_archivo)
            destination_path = os.path.join(config.path_destino, config.nombre_tabla)
            
            # Verificar que el archivo origen existe
            if not os.path.exists(source_path):
                error_msg = f"Archivo origen no encontrado: {source_path}"
                return self._create_error_result(config.id_ingesta, error_msg, start_time)
            
            # Cargar archivo con configuración distribuida optimizada
            df_loaded = self._load_file_distributed(source_path, config.delimitador)
            
            # Obtener métricas básicas
            records_count = df_loaded.count()
            columns_count = len(df_loaded.columns)
            file_size_mb = self._calculate_file_size_mb(source_path)
            
            self.logger.info(f"Archivo cargado: {records_count:,} registros, {columns_count} columnas")
            
            # Aplicar optimizaciones de escritura distribuida
            optimized_df = self._apply_distributed_optimizations(df_loaded)
            
            # Configurar escritura con particionamiento si está especificado
            writer = optimized_df.write.mode("overwrite").option("compression", "snappy")
            
            partition_applied = None
            if config.columna_particion:
                if config.columna_particion in optimized_df.columns:
                    writer = writer.partitionBy(config.columna_particion)
                    partition_applied = config.columna_particion
                    self.logger.info(f"Particionamiento aplicado por: {config.columna_particion}")
                else:
                    self.logger.warning(f"Columna de partición no encontrada: {config.columna_particion}")
            
            # Ejecutar escritura distribuida en formato Parquet
            self.logger.info("Ejecutando escritura distribuida...")
            writer.parquet(destination_path)
            
            # Validar resultado de escritura
            validation_results = self._validate_written_data(destination_path, records_count)
            
            # Calcular métricas de rendimiento
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            throughput = records_count / processing_time if processing_time > 0 else 0
            
            self.logger.info(f"Ingesta completada exitosamente:")
            self.logger.info(f"  Registros procesados: {records_count:,}")
            self.logger.info(f"  Tiempo de procesamiento: {processing_time:.2f} segundos")
            self.logger.info(f"  Throughput: {throughput:.0f} registros/segundo")
            
            return IngestionResult(
                id_ingesta=config.id_ingesta,
                success=True,
                records_processed=records_count,
                columns_count=columns_count,
                processing_time_seconds=processing_time,
                file_size_mb=file_size_mb,
                throughput_records_per_second=throughput,
                partition_applied=partition_applied,
                validation_results=validation_results
            )
            
        except Exception as e:
            error_msg = f"Error en ingesta distribuida: {str(e)}"
            self.logger.error(f"Fallo en ingesta {config.id_ingesta}: {error_msg}")
            return self._create_error_result(config.id_ingesta, error_msg, start_time)
    
    def _load_file_distributed(self, file_path: str, delimiter: str) -> DataFrame:
        """
        Carga un archivo con configuraciones distribuidas optimizadas.
        
        Args:
            file_path (str): Ruta del archivo a cargar
            delimiter (str): Delimitador utilizado en el archivo
            
        Returns:
            DataFrame: DataFrame de Spark cargado y optimizado
        """
        # Configuración optimizada para lectura distribuida
        df = self.spark.read \
            .option("header", "true") \
            .option("delimiter", delimiter) \
            .option("encoding", "UTF-8") \
            .option("multiline", "true") \
            .option("escape", '"') \
            .option("quote", '"') \
            .option("inferSchema", "false") \
            .csv(file_path)
        
        # Log información de carga
        self.logger.debug(f"Archivo cargado con delimitador: '{delimiter}'")
        
        return df
    
    def _apply_distributed_optimizations(self, df: DataFrame) -> DataFrame:
        """
        Aplica optimizaciones distribuidas al DataFrame.
        
        Args:
            df (DataFrame): DataFrame original
            
        Returns:
            DataFrame: DataFrame optimizado para escritura distribuida
        """
        # Optimizar número de particiones para escritura
        # Regla heurística: ~200MB por partición para datasets grandes
        record_count = df.count()
        
        if record_count > 1000000:  # > 1M registros
            optimal_partitions = max(4, min(200, record_count // 50000))
            df_optimized = df.coalesce(optimal_partitions)
            self.logger.info(f"Optimización aplicada: {optimal_partitions} particiones")
        else:
            # Para datasets pequeños, usar menos particiones
            df_optimized = df.coalesce(4)
            self.logger.info("Optimización aplicada: 4 particiones (dataset pequeño)")
        
        return df_optimized
    
    def _validate_written_data(self, destination_path: str, expected_records: int) -> Dict[str, Any]:
        """
        Valida los datos escritos en el destino.
        
        Args:
            destination_path (str): Ruta de destino de los datos
            expected_records (int): Número esperado de registros
            
        Returns:
            Dict[str, Any]: Resultados de validación
        """
        try:
            # Leer datos escritos para validación
            df_validation = self.spark.read.parquet(destination_path)
            actual_records = df_validation.count()
            
            # Validar integridad de registros
            records_match = actual_records == expected_records
            
            validation_results = {
                'records_integrity_check': records_match,
                'expected_records': expected_records,
                'actual_records': actual_records,
                'data_readable': True
            }
            
            if not records_match:
                self.logger.warning(
                    f"Discrepancia en conteo de registros: "
                    f"esperados {expected_records}, encontrados {actual_records}"
                )
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error en validación de datos escritos: {str(e)}")
            return {
                'records_integrity_check': False,
                'data_readable': False,
                'validation_error': str(e)
            }
    
    def _calculate_file_size_mb(self, file_path: str) -> float:
        """
        Calcula el tamaño de un archivo en megabytes.
        
        Args:
            file_path (str): Ruta del archivo
            
        Returns:
            float: Tamaño en megabytes
        """
        try:
            file_size_bytes = os.path.getsize(file_path)
            return file_size_bytes / (1024 * 1024)
        except Exception as e:
            self.logger.warning(f"No se pudo calcular tamaño de {file_path}: {e}")
            return 0.0
    
    def _create_error_result(self, ingestion_id: str, error_message: str, 
                           start_time: datetime) -> IngestionResult:
        """
        Crea un resultado de error para ingesta fallida.
        
        Args:
            ingestion_id (str): ID de la ingesta fallida
            error_message (str): Mensaje de error
            start_time (datetime): Tiempo de inicio del procesamiento
            
        Returns:
            IngestionResult: Resultado de error
        """
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return IngestionResult(
            id_ingesta=ingestion_id,
            success=False,
            records_processed=0,
            columns_count=0,
            processing_time_seconds=processing_time,
            file_size_mb=0.0,
            throughput_records_per_second=0.0,
            error_message=error_message
        )


class ParametricIngestionEngine:
    """
    Motor principal de ingesta parametrizada para datasets censales.
    
    Esta clase orquesta todo el proceso de ingesta parametrizada,
    integrando la carga de configuraciones, el procesamiento distribuido
    y la generación de reportes de auditoría académica.
    """
    
    def __init__(self, spark: SparkSession, base_path: str = "C:/DataLake"):
        """
        Inicializa el motor de ingesta parametrizada.
        
        Args:
            spark (SparkSession): Sesión de Spark configurada
            base_path (str): Ruta base del Data Lake
        """
        self.spark = spark
        self.base_path = base_path
        self.data_config = get_data_config(base_path)
        self.logger = get_logger(__name__)
        
        # Inicializar componentes especializados
        self.config_loader = ParametricConfigurationLoader(base_path)
        self.processor = DistributedIngestionProcessor(spark)
        
        self.logger.info(f"ParametricIngestionEngine inicializado para: {base_path}")
    
    def execute_parametric_ingestion(self) -> Dict[str, Any]:
        """
        Ejecuta el proceso completo de ingesta parametrizada.
        
        Returns:
            Dict[str, Any]: Resultados completos del proceso de ingesta
        """
        TFMLogger.log_pipeline_start("INGESTA_PARAMETRIZADA", self.logger)
        
        start_time = datetime.now()
        
        try:
            # Paso 1: Cargar configuraciones activas
            configurations = self.config_loader.load_active_configurations()
            
            if not configurations:
                self.logger.warning("No se encontraron configuraciones activas para procesar")
                return self._create_empty_result()
            
            # Paso 2: Ejecutar ingesta para cada configuración
            ingestion_results = []
            
            for config in configurations:
                result = self.processor.process_file_ingestion(config)
                ingestion_results.append(result)
                
                # Log progreso
                status = "EXITOSA" if result.success else "FALLIDA"
                self.logger.info(f"Ingesta {config.id_ingesta}: {status}")
            
            # Paso 3: Generar análisis de resultados
            analysis = self._analyze_ingestion_results(ingestion_results)
            
            # Paso 4: Generar reporte de auditoría
            report_path = self._generate_audit_report(configurations, ingestion_results, analysis)
            
            # Calcular métricas finales
            end_time = datetime.now()
            total_duration = (end_time - start_time).total_seconds()
            
            # Log resumen final
            successful_count = len([r for r in ingestion_results if r.success])
            total_records = sum(r.records_processed for r in ingestion_results if r.success)
            
            TFMLogger.log_pipeline_end(
                "INGESTA_PARAMETRIZADA", 
                self.logger, 
                total_duration, 
                successful_count == len(configurations)
            )
            
            return {
                'success': successful_count == len(configurations),
                'total_configurations': len(configurations),
                'successful_ingestions': successful_count,
                'failed_ingestions': len(configurations) - successful_count,
                'total_records_processed': total_records,
                'total_duration_seconds': total_duration,
                'average_throughput': total_records / total_duration if total_duration > 0 else 0,
                'ingestion_results': ingestion_results,
                'analysis': analysis,
                'audit_report_path': report_path
            }
            
        except Exception as e:
            end_time = datetime.now()
            total_duration = (end_time - start_time).total_seconds()
            
            error_msg = f"Error crítico en ingesta parametrizada: {str(e)}"
            self.logger.error(error_msg)
            
            TFMLogger.log_pipeline_end("INGESTA_PARAMETRIZADA", self.logger, total_duration, False)
            
            return {
                'success': False,
                'error': error_msg,
                'total_duration_seconds': total_duration
            }
    
    def _analyze_ingestion_results(self, results: List[IngestionResult]) -> Dict[str, Any]:
        """
        Analiza los resultados de ingesta para generar métricas académicas.
        
        Args:
            results (List[IngestionResult]): Resultados de todas las ingestas
            
        Returns:
            Dict[str, Any]: Análisis detallado de resultados
        """
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        if not successful_results:
            return {
                'success_rate': 0.0,
                'total_throughput': 0.0,
                'performance_metrics': {},
                'error_analysis': self._analyze_errors(failed_results)
            }
        
        # Métricas de rendimiento
        total_records = sum(r.records_processed for r in successful_results)
        total_time = sum(r.processing_time_seconds for r in successful_results)
        
        throughput_values = [r.throughput_records_per_second for r in successful_results]
        avg_throughput = sum(throughput_values) / len(throughput_values)
        max_throughput = max(throughput_values)
        min_throughput = min(throughput_values)
        
        # Análisis de distribución de archivos
        file_sizes = [r.file_size_mb for r in successful_results]
        avg_file_size = sum(file_sizes) / len(file_sizes) if file_sizes else 0
        
        analysis = {
            'success_rate': len(successful_results) / len(results) * 100,
            'total_records_processed': total_records,
            'total_processing_time': total_time,
            'performance_metrics': {
                'average_throughput': avg_throughput,
                'maximum_throughput': max_throughput,
                'minimum_throughput': min_throughput,
                'average_file_size_mb': avg_file_size
            },
            'distribution_analysis': {
                'files_with_partitioning': len([r for r in successful_results if r.partition_applied]),
                'average_columns_per_table': sum(r.columns_count for r in successful_results) / len(successful_results)
            }
        }
        
        if failed_results:
            analysis['error_analysis'] = self._analyze_errors(failed_results)
        
        return analysis
    
    def _analyze_errors(self, failed_results: List[IngestionResult]) -> Dict[str, Any]:
        """
        Analiza los errores ocurridos durante la ingesta.
        
        Args:
            failed_results (List[IngestionResult]): Resultados fallidos
            
        Returns:
            Dict[str, Any]: Análisis de errores
        """
        if not failed_results:
            return {}
        
        # Categorizar errores por tipo
        error_categories = {}
        for result in failed_results:
            error_msg = result.error_message or "Error desconocido"
            
            # Clasificar errores en categorías
            if "no encontrado" in error_msg.lower():
                category = "archivo_no_encontrado"
            elif "permission" in error_msg.lower():
                category = "permisos_archivo"
            elif "format" in error_msg.lower() or "delimiter" in error_msg.lower():
                category = "formato_datos"
            else:
                category = "error_general"
            
            if category not in error_categories:
                error_categories[category] = []
            error_categories[category].append(result.id_ingesta)
        
        return {
            'total_failed': len(failed_results),
            'error_categories': error_categories,
            'failed_ingestion_ids': [r.id_ingesta for r in failed_results]
        }
    
    def _generate_audit_report(self, configurations: List[IngestionConfiguration],
                             results: List[IngestionResult], 
                             analysis: Dict[str, Any]) -> str:
        """
        Genera un reporte de auditoría académica del proceso de ingesta.
        
        Args:
            configurations (List[IngestionConfiguration]): Configuraciones procesadas
            results (List[IngestionResult]): Resultados de ingesta
            analysis (Dict[str, Any]): Análisis de resultados
            
        Returns:
            str: Ruta del archivo de reporte generado
        """
        import json
        
        # Crear reporte completo para auditoría académica
        audit_report = {
            'metadata': {
                'report_timestamp': datetime.now().isoformat(),
                'report_type': 'parametric_ingestion_audit',
                'tfm_project': 'Sistema de procesamiento distribuido de datos censales',
                'author': 'Ramiro Ricardo Merchán Mora',
                'university': 'Universidad de Málaga'
            },
            'process_summary': {
                'total_configurations': len(configurations),
                'successful_ingestions': len([r for r in results if r.success]),
                'failed_ingestions': len([r for r in results if not r.success]),
                'success_rate_percentage': analysis.get('success_rate', 0),
                'total_records_processed': analysis.get('total_records_processed', 0)
            },
            'performance_analysis': analysis.get('performance_metrics', {}),
            'detailed_results': [
                {
                    'id_ingesta': r.id_ingesta,
                    'success': r.success,
                    'records_processed': r.records_processed,
                    'processing_time_seconds': r.processing_time_seconds,
                    'throughput_records_per_second': r.throughput_records_per_second,
                    'file_size_mb': r.file_size_mb,
                    'partition_applied': r.partition_applied,
                    'error_message': r.error_message
                }
                for r in results
            ],
            'configurations_used': [
                {
                    'id_ingesta': c.id_ingesta,
                    'nombre_archivo': c.nombre_archivo,
                    'nombre_tabla': c.nombre_tabla,
                    'grupo_ingesta': c.grupo_ingesta,
                    'orden_grupo': c.orden_grupo
                }
                for c in configurations
            ]
        }
        
        # Guardar reporte en formato JSON para análisis posterior
        reports_path = self.data_config.paths.reports_path
        reports_path.mkdir(parents=True, exist_ok=True)
        
        report_filename = f"parametric_ingestion_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = reports_path / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(audit_report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Reporte de auditoría generado: {report_path}")
        
        return str(report_path)
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """
        Crea un resultado vacío cuando no hay configuraciones activas.
        
        Returns:
            Dict[str, Any]: Resultado vacío
        """
        return {
            'success': True,
            'total_configurations': 0,
            'successful_ingestions': 0,
            'failed_ingestions': 0,
            'total_records_processed': 0,
            'total_duration_seconds': 0.0,
            'message': 'No hay configuraciones activas para procesar'
        }