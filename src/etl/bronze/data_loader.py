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
# ARCHIVO: etl/bronze/data_loader.py
# =============================================================================
#
# Propósito: Cargador especializado de datos para la capa Bronze del Data Lake
#
# Funcionalidades principales:
# - Carga optimizada de tablas Bronze con validación de esquemas
# - Gestión de diccionarios de recodificación del INEC
# - Validación automática de integridad de datos censales
# - Logging académico estructurado para trazabilidad
#
# Dependencias:
# - pyspark.sql
# - core.logger
# - config.data_config
# - utils.validation_utils
#
# Uso:
# from etl.bronze.data_loader import BronzeDataLoader
# loader = BronzeDataLoader(spark_session, base_path)
# tables = loader.load_all_bronze_tables()
#
# =============================================================================

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType

from core.logger import get_logger
from config.data_config import get_data_config, TableType
from utils.validation_utils import DataValidator


@dataclass
class LoadResult:
    """
    Resultado de la operación de carga de una tabla.
    
    Esta clase encapsula toda la información relevante sobre el proceso
    de carga de una tabla específica, facilitando el análisis posterior
    y la generación de reportes de calidad.
    """
    table_name: str
    success: bool
    record_count: int
    column_count: int
    file_size_mb: float
    load_time_seconds: float
    validation_results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class BronzeDataLoader:
    """
    Cargador especializado para la capa Bronze del Data Lake.
    
    Esta clase implementa la lógica de carga optimizada para datasets
    censales y de ENEMDU, incluyendo validación de esquemas, gestión
    de diccionarios de recodificación y métricas de calidad de datos.
    
    El diseño sigue el principio de responsabilidad única, enfocándose
    exclusivamente en la carga eficiente de datos crudos desde archivos
    Parquet hacia DataFrames de Spark validados.
    """
    
    def __init__(self, spark: SparkSession, base_path: str = "C:/DataLake"):
        """
        Inicializa el cargador de datos Bronze.
        
        Args:
            spark (SparkSession): Sesión de Spark configurada
            base_path (str): Ruta base del Data Lake
        """
        self.spark = spark
        self.data_config = get_data_config(base_path)
        self.bronze_path = self.data_config.paths.bronze_path
        self.logger = get_logger(__name__)
        self.validator = DataValidator()
        
        # Configuración de tablas esperadas en Bronze
        self.expected_tables = {
            "poblacion.parquet": TableType.CENSO_POBLACION,
            "hogar.parquet": TableType.CENSO_HOGAR,
            "vivienda.parquet": TableType.CENSO_VIVIENDA,
            "enemdu_personas.parquet": TableType.ENEMDU_PERSONAS,
            "enemdu_vivienda.parquet": TableType.ENEMDU_VIVIENDA
        }
        
        # Configuración de diccionarios de recodificación
        self.dictionary_tables = {
            "ZR_Sftp_DICCIONARIO_PROVINCIA_2022": TableType.DICCIONARIO_PROVINCIA,
            "ZR_Sftp_DICCIONARIO_CANTON_2022": TableType.DICCIONARIO_CANTON,
            "ZR_Sftp_DICCIONARIO_PARROQUIA_2022": TableType.DICCIONARIO_PARROQUIA,
            "ZR_Sftp_DICCIONARIO_CENSO_2022": TableType.DICCIONARIO_CENSO,
            "ZR_Sftp_DICCIONARIO_ENEMDU_2022": TableType.DICCIONARIO_ENEMDU
        }
        
        self.logger.info(f"BronzeDataLoader inicializado con ruta: {self.bronze_path}")
    
    def load_all_bronze_tables(self) -> Dict[str, DataFrame]:
        """
        Carga todas las tablas disponibles en la capa Bronze.
        
        Este método ejecuta la carga completa de datasets censales y
        diccionarios de recodificación, aplicando validaciones de calidad
        y generando métricas detalladas del proceso.
        
        Returns:
            Dict[str, DataFrame]: Diccionario con todas las tablas cargadas
            
        Raises:
            FileNotFoundError: Si el directorio Bronze no existe
            ValueError: Si no se encuentra ninguna tabla válida
        """
        self.logger.info("Iniciando carga completa de tablas Bronze")
        
        # Verificar que el directorio Bronze existe
        if not self.bronze_path.exists():
            error_msg = f"Directorio Bronze no encontrado: {self.bronze_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Cargar tablas principales de datos
        main_tables = self._load_main_tables()
        
        # Cargar diccionarios de recodificación
        dictionary_tables = self._load_dictionary_tables()
        
        # Combinar resultados
        all_tables = {**main_tables, **dictionary_tables}
        
        # Generar reporte de carga
        self._generate_load_report(all_tables)
        
        self.logger.info(f"Carga Bronze completada: {len(all_tables)} tablas cargadas")
        
        return all_tables
    
    def _load_main_tables(self) -> Dict[str, DataFrame]:
        """
        Carga las tablas principales de datos censales y ENEMDU.
        
        Returns:
            Dict[str, DataFrame]: Diccionario con tablas principales cargadas
        """
        self.logger.info("Cargando tablas principales de datos")
        
        loaded_tables = {}
        load_results = []
        
        for file_name, table_type in self.expected_tables.items():
            # Intentar cargar cada tabla
            load_result = self._load_single_table(file_name, table_type)
            load_results.append(load_result)
            
            # Si la carga fue exitosa, agregar al diccionario
            if load_result.success:
                table_key = file_name.replace('.parquet', '')
                loaded_tables[table_key] = load_result.dataframe
                
                self.logger.info(
                    f"Tabla cargada: {table_key} - "
                    f"{load_result.record_count:,} registros, "
                    f"{load_result.column_count} columnas"
                )
            else:
                self.logger.warning(
                    f"Fallo al cargar tabla: {file_name} - {load_result.error_message}"
                )
        
        # Validar que se cargaron tablas mínimas requeridas
        self._validate_minimum_tables_loaded(loaded_tables)
        
        return loaded_tables
    
    def _load_dictionary_tables(self) -> Dict[str, DataFrame]:
        """
        Carga los diccionarios de recodificación del INEC.
        
        Returns:
            Dict[str, DataFrame]: Diccionario con tablas de diccionarios cargadas
        """
        self.logger.info("Cargando diccionarios de recodificación")
        
        loaded_dictionaries = {}
        
        for file_name, table_type in self.dictionary_tables.items():
            # Intentar cargar cada diccionario
            load_result = self._load_single_table(file_name, table_type)
            
            if load_result.success:
                # Generar clave descriptiva para el diccionario
                dict_key = self._generate_dictionary_key(file_name)
                loaded_dictionaries[dict_key] = load_result.dataframe
                
                self.logger.info(
                    f"Diccionario cargado: {dict_key} - "
                    f"{load_result.record_count:,} mapeos"
                )
            else:
                self.logger.warning(
                    f"Diccionario no disponible: {file_name} - {load_result.error_message}"
                )
        
        return loaded_dictionaries
    
    def _load_single_table(self, file_name: str, table_type: TableType) -> LoadResult:
        """
        Carga una tabla individual con validación completa.
        
        Args:
            file_name (str): Nombre del archivo a cargar
            table_type (TableType): Tipo de tabla según configuración
            
        Returns:
            LoadResult: Resultado detallado de la operación de carga
        """
        import time
        
        start_time = time.time()
        table_path = self.bronze_path / file_name
        
        try:
            # Verificar que el archivo existe
            if not table_path.exists():
                return LoadResult(
                    table_name=file_name,
                    success=False,
                    record_count=0,
                    column_count=0,
                    file_size_mb=0.0,
                    load_time_seconds=0.0,
                    error_message=f"Archivo no encontrado: {table_path}"
                )
            
            # Cargar DataFrame desde Parquet
            df = self.spark.read.parquet(str(table_path))
            
            # Obtener métricas básicas
            record_count = df.count()
            column_count = len(df.columns)
            file_size_mb = self._calculate_file_size_mb(table_path)
            load_time = time.time() - start_time
            
            # Validar estructura de tabla si hay esquema definido
            validation_results = None
            try:
                schema = self.data_config.get_schema(table_type)
                validation_results = self.data_config.validate_table_structure(
                    table_type, df.columns
                )
                
                # Log resultados de validación
                if not validation_results['is_valid']:
                    self.logger.warning(
                        f"Problemas de esquema en {file_name}: "
                        f"Columnas faltantes: {validation_results['missing_required_columns']}"
                    )
                
            except ValueError:
                # No hay esquema definido para este tipo de tabla
                self.logger.debug(f"Sin esquema definido para {table_type}")
            
            # Crear resultado exitoso
            result = LoadResult(
                table_name=file_name,
                success=True,
                record_count=record_count,
                column_count=column_count,
                file_size_mb=file_size_mb,
                load_time_seconds=load_time,
                validation_results=validation_results
            )
            
            # Agregar DataFrame al resultado (para uso interno)
            result.dataframe = df
            
            return result
            
        except Exception as e:
            load_time = time.time() - start_time
            
            self.logger.error(f"Error cargando {file_name}: {str(e)}")
            
            return LoadResult(
                table_name=file_name,
                success=False,
                record_count=0,
                column_count=0,
                file_size_mb=0.0,
                load_time_seconds=load_time,
                error_message=str(e)
            )
    
    def _calculate_file_size_mb(self, file_path: Path) -> float:
        """
        Calcula el tamaño total de un archivo o directorio en MB.
        
        Args:
            file_path (Path): Ruta del archivo o directorio
            
        Returns:
            float: Tamaño en megabytes
        """
        total_size = 0
        
        try:
            if file_path.is_file():
                total_size = file_path.stat().st_size
            elif file_path.is_dir():
                # Para directorios Parquet, sumar todos los archivos
                for file_item in file_path.rglob('*'):
                    if file_item.is_file():
                        total_size += file_item.stat().st_size
        except Exception as e:
            self.logger.warning(f"No se pudo calcular tamaño de {file_path}: {e}")
            return 0.0
        
        return total_size / (1024 * 1024)  # Convertir a MB
    
    def _generate_dictionary_key(self, file_name: str) -> str:
        """
        Genera una clave descriptiva para diccionarios basada en el nombre de archivo.
        
        Args:
            file_name (str): Nombre del archivo de diccionario
            
        Returns:
            str: Clave descriptiva para el diccionario
        """
        # Mapeo de nombres de archivo a claves descriptivas
        key_mapping = {
            "ZR_Sftp_DICCIONARIO_PROVINCIA_2022": "dict_provincia",
            "ZR_Sftp_DICCIONARIO_CANTON_2022": "dict_canton",
            "ZR_Sftp_DICCIONARIO_PARROQUIA_2022": "dict_parroquia",
            "ZR_Sftp_DICCIONARIO_CENSO_2022": "dict_censo",
            "ZR_Sftp_DICCIONARIO_ENEMDU_2022": "dict_enemdu"
        }
        
        return key_mapping.get(file_name, f"dict_{file_name.lower()}")
    
    def _validate_minimum_tables_loaded(self, loaded_tables: Dict[str, DataFrame]) -> None:
        """
        Valida que se hayan cargado las tablas mínimas requeridas.
        
        Args:
            loaded_tables (Dict[str, DataFrame]): Tablas cargadas exitosamente
            
        Raises:
            ValueError: Si no se cumplen los requisitos mínimos
        """
        required_tables = {"poblacion", "hogar", "vivienda"}  # Mínimo para Censo
        loaded_table_names = set(loaded_tables.keys())
        
        missing_required = required_tables - loaded_table_names
        
        if missing_required:
            error_msg = (
                f"Faltan tablas requeridas para el análisis censal: {missing_required}. "
                f"Tablas disponibles: {loaded_table_names}"
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.info("Validación de tablas mínimas: EXITOSA")
    
    def _generate_load_report(self, all_tables: Dict[str, DataFrame]) -> None:
        """
        Genera un reporte detallado del proceso de carga.
        
        Args:
            all_tables (Dict[str, DataFrame]): Todas las tablas cargadas
        """
        total_records = sum(df.count() for df in all_tables.values())
        main_tables_count = len([k for k in all_tables.keys() if not k.startswith('dict_')])
        dict_tables_count = len([k for k in all_tables.keys() if k.startswith('dict_')])
        
        self.logger.info("=" * 60)
        self.logger.info("REPORTE DE CARGA BRONZE LAYER")
        self.logger.info("=" * 60)
        self.logger.info(f"Tablas principales cargadas: {main_tables_count}")
        self.logger.info(f"Diccionarios cargados: {dict_tables_count}")
        self.logger.info(f"Total registros procesados: {total_records:,}")
        
        # Detalle por tabla
        for table_name, df in all_tables.items():
            record_count = df.count()
            column_count = len(df.columns)
            table_type = "DICCIONARIO" if table_name.startswith('dict_') else "DATOS"
            
            self.logger.info(
                f"  {table_name:20} | {table_type:12} | "
                f"{record_count:>10,} registros | {column_count:>3} columnas"
            )
        
        self.logger.info("=" * 60)
    
    def get_table_metadata(self, table_name: str, df: DataFrame) -> Dict[str, Any]:
        """
        Extrae metadatos detallados de una tabla específica.
        
        Args:
            table_name (str): Nombre de la tabla
            df (DataFrame): DataFrame de Spark
            
        Returns:
            Dict[str, Any]: Metadatos completos de la tabla
        """
        # Calcular estadísticas básicas
        record_count = df.count()
        column_count = len(df.columns)
        
        # Analizar tipos de columnas
        schema_analysis = self._analyze_schema(df)
        
        # Detectar columnas con valores nulos
        null_analysis = self._analyze_null_values(df)
        
        metadata = {
            'table_name': table_name,
            'record_count': record_count,
            'column_count': column_count,
            'schema_analysis': schema_analysis,
            'null_analysis': null_analysis,
            'load_timestamp': self._get_current_timestamp()
        }
        
        return metadata
    
    def _analyze_schema(self, df: DataFrame) -> Dict[str, Any]:
        """
        Analiza el esquema de un DataFrame.
        
        Args:
            df (DataFrame): DataFrame a analizar
            
        Returns:
            Dict[str, Any]: Análisis del esquema
        """
        schema = df.schema
        
        type_counts = {}
        for field in schema.fields:
            field_type = str(field.dataType)
            type_counts[field_type] = type_counts.get(field_type, 0) + 1
        
        return {
            'column_names': [field.name for field in schema.fields],
            'data_types': [(field.name, str(field.dataType)) for field in schema.fields],
            'type_distribution': type_counts
        }
    
    def _analyze_null_values(self, df: DataFrame) -> Dict[str, Any]:
        """
        Analiza valores nulos en un DataFrame.
        
        Args:
            df (DataFrame): DataFrame a analizar
            
        Returns:
            Dict[str, Any]: Análisis de valores nulos
        """
        from pyspark.sql.functions import col, sum as spark_sum, isnan, isnull
        
        # Calcular nulos por columna
        null_counts = {}
        total_records = df.count()
        
        for column_name in df.columns:
            null_count = df.filter(col(column_name).isNull()).count()
            null_percentage = (null_count / total_records) * 100 if total_records > 0 else 0
            
            null_counts[column_name] = {
                'null_count': null_count,
                'null_percentage': round(null_percentage, 2)
            }
        
        return {
            'total_records': total_records,
            'null_counts_by_column': null_counts,
            'columns_with_nulls': [col for col, info in null_counts.items() if info['null_count'] > 0]
        }
    
    def _get_current_timestamp(self) -> str:
        """
        Obtiene timestamp actual en formato ISO.
        
        Returns:
            str: Timestamp en formato ISO 8601
        """
        from datetime import datetime
        return datetime.now().isoformat()


class BronzeDataValidator:
    """
    Validador especializado para datos de la capa Bronze.
    
    Esta clase complementa al BronzeDataLoader proporcionando
    validaciones específicas para datos censales y de ENEMDU.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def validate_census_data_quality(self, tables: Dict[str, DataFrame]) -> Dict[str, Any]:
        """
        Valida la calidad de datos censales cargados.
        
        Args:
            tables (Dict[str, DataFrame]): Tablas censales cargadas
            
        Returns:
            Dict[str, Any]: Resultados de validación de calidad
        """
        validation_results = {
            'overall_quality_score': 0.0,
            'table_validations': {},
            'critical_issues': [],
            'recommendations': []
        }
        
        # Validar tabla de población
        if 'poblacion' in tables:
            pop_validation = self._validate_population_table(tables['poblacion'])
            validation_results['table_validations']['poblacion'] = pop_validation
        
        # Validar tabla de hogar
        if 'hogar' in tables:
            household_validation = self._validate_household_table(tables['hogar'])
            validation_results['table_validations']['hogar'] = household_validation
        
        # Validar tabla de vivienda
        if 'vivienda' in tables:
            housing_validation = self._validate_housing_table(tables['vivienda'])
            validation_results['table_validations']['vivienda'] = housing_validation
        
        # Calcular score general
        validation_results['overall_quality_score'] = self._calculate_overall_quality_score(
            validation_results['table_validations']
        )
        
        return validation_results
    
    def _validate_population_table(self, df: DataFrame) -> Dict[str, Any]:
        """
        Valida la tabla de población del censo.
        
        Args:
            df (DataFrame): DataFrame de población
            
        Returns:
            Dict[str, Any]: Resultados de validación
        """
        # Validaciones específicas para población
        validation = {
            'record_count_valid': df.count() > 1000,  # Mínimo esperado
            'required_columns_present': all(col in df.columns for col in ['ID_VIV', 'ID_HOG', 'ID_PER']),
            'age_values_reasonable': True,  # Se validará después
            'geographic_coverage_adequate': True  # Se validará después
        }
        
        # Validar rangos de edad si existe columna P03
        if 'P03' in df.columns:
            from pyspark.sql.functions import col, min as spark_min, max as spark_max
            
            age_stats = df.select(
                spark_min(col('P03')).alias('min_age'),
                spark_max(col('P03')).alias('max_age')
            ).collect()[0]
            
            validation['age_values_reasonable'] = (
                age_stats['min_age'] >= 0 and age_stats['max_age'] <= 120
            )
        
        return validation
    
    def _validate_household_table(self, df: DataFrame) -> Dict[str, Any]:
        """
        Valida la tabla de hogares del censo.
        
        Args:
            df (DataFrame): DataFrame de hogares
            
        Returns:
            Dict[str, Any]: Resultados de validación
        """
        validation = {
            'record_count_valid': df.count() > 500,  # Mínimo esperado para hogares
            'required_columns_present': all(col in df.columns for col in ['ID_VIV', 'ID_HOG']),
            'household_size_reasonable': True
        }
        
        # Validar tamaño de hogar si existe columna H1303
        if 'H1303' in df.columns:
            from pyspark.sql.functions import col, max as spark_max
            
            max_household_size = df.select(spark_max(col('H1303'))).collect()[0][0]
            validation['household_size_reasonable'] = max_household_size <= 20  # Límite razonable
        
        return validation
    
    def _validate_housing_table(self, df: DataFrame) -> Dict[str, Any]:
        """
        Valida la tabla de viviendas del censo.
        
        Args:
            df (DataFrame): DataFrame de viviendas
            
        Returns:
            Dict[str, Any]: Resultados de validación
        """
        validation = {
            'record_count_valid': df.count() > 500,  # Mínimo esperado para viviendas
            'required_columns_present': 'ID_VIV' in df.columns,
            'housing_types_valid': True
        }
        
        # Validar tipos de vivienda si existe columna V01
        if 'V01' in df.columns:
            from pyspark.sql.functions import col
            
            distinct_types = df.select('V01').distinct().count()
            validation['housing_types_valid'] = distinct_types > 0 and distinct_types < 50
        
        return validation
    
    def _calculate_overall_quality_score(self, table_validations: Dict[str, Dict[str, Any]]) -> float:
        """
        Calcula un score general de calidad basado en validaciones individuales.
        
        Args:
            table_validations (Dict[str, Dict[str, Any]]): Validaciones por tabla
            
        Returns:
            float: Score de calidad entre 0 y 100
        """
        if not table_validations:
            return 0.0
        
        total_checks = 0
        passed_checks = 0
        
        for table_name, validations in table_validations.items():
            for check_name, check_result in validations.items():
                if isinstance(check_result, bool):
                    total_checks += 1
                    if check_result:
                        passed_checks += 1
        
        if total_checks == 0:
            return 0.0
        
        quality_score = (passed_checks / total_checks) * 100
        return round(quality_score, 1)