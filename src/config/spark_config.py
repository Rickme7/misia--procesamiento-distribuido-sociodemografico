# =============================================================================
# PROYECTO TFM: SISTEMA DE PROCESAMIENTO DISTRIBUIDO DE DATOS CENSALES
# =============================================================================
#                    Universidad de Málaga
# Máster en Ingeniería del Software e Inteligencia Artificial
#
# Título: Sistema inteligente de procesamiento distribuido para análisis 
#         predictivo de patrones sociodemográficos censales
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
# ARCHIVO: config/spark_config.py
# =============================================================================
#
# Propósito: Configuración centralizada de Apache Spark para el proyecto TFM
#
# Funcionalidades principales:
# - Configuraciones optimizadas para diferentes entornos de ejecución
# - Parámetros de memoria y paralelización para datasets censales
# - Configuración de Delta Lake y optimizaciones avanzadas
# - Perfiles predefinidos para desarrollo, testing y producción
#
# Dependencias:
# - pyspark
# - os (Python standard library)
# - dataclasses (Python standard library)
#
# Uso:
# from config.spark_config import get_spark_session, SparkEnvironment
# spark = get_spark_session(SparkEnvironment.ACADEMIC)
#
# =============================================================================

import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from pyspark.sql import SparkSession


class SparkEnvironment(Enum):
    """
    Enumeración de entornos de ejecución soportados.
    """
    DEVELOPMENT = "development"
    ACADEMIC = "academic"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class SparkConfiguration:
    """
    Clase de configuración para parámetros de Spark.
    
    Esta clase encapsula todos los parámetros necesarios para configurar
    una sesión de Spark optimizada para el procesamiento de datos censales.
    """
    # Configuración básica
    app_name: str
    master: str
    
    # Configuración de memoria
    driver_memory: str
    driver_max_result_size: str
    executor_memory: str
    
    # Configuración de paralelización
    sql_shuffle_partitions: int
    default_parallelism: Optional[int] = None
    
    # Configuraciones de optimización
    adaptive_enabled: bool = True
    adaptive_coalesce_partitions: bool = True
    adaptive_skew_join: bool = True
    adaptive_local_shuffle_reader: bool = True
    
    # Configuración de serialización
    serializer: str = "org.apache.spark.serializer.KryoSerializer"
    
    # Configuración de Delta Lake
    enable_delta: bool = True
    delta_package: str = "io.delta:delta-spark_2.12:3.2.0"
    
    # Configuración de Arrow
    arrow_enabled: bool = False
    
    # Configuraciones adicionales
    additional_configs: Optional[Dict[str, str]] = None


class SparkConfigurationFactory:
    """
    Factory para crear configuraciones de Spark según el entorno.
    
    Esta clase implementa el patrón Factory para generar configuraciones
    optimizadas de Spark adaptadas a diferentes contextos de ejecución.
    """
    
    @staticmethod
    def create_configuration(environment: SparkEnvironment) -> SparkConfiguration:
        """
        Crea una configuración de Spark para el entorno especificado.
        
        Args:
            environment (SparkEnvironment): Entorno de ejecución objetivo
            
        Returns:
            SparkConfiguration: Configuración optimizada para el entorno
        """
        if environment == SparkEnvironment.DEVELOPMENT:
            return SparkConfigurationFactory._create_development_config()
        elif environment == SparkEnvironment.ACADEMIC:
            return SparkConfigurationFactory._create_academic_config()
        elif environment == SparkEnvironment.PRODUCTION:
            return SparkConfigurationFactory._create_production_config()
        elif environment == SparkEnvironment.TESTING:
            return SparkConfigurationFactory._create_testing_config()
        else:
            raise ValueError(f"Entorno no soportado: {environment}")
    
    @staticmethod
    def _create_development_config() -> SparkConfiguration:
        """
        Configuración optimizada para desarrollo local.
        
        Esta configuración prioriza la velocidad de inicio y el debugging
        sobre el rendimiento máximo.
        """
        return SparkConfiguration(
            app_name="TFM_Development_Census_Analytics",
            master="local[*]",
            driver_memory="8g",
            driver_max_result_size="2g",
            executor_memory="6g",
            sql_shuffle_partitions=100,
            arrow_enabled=False,
            additional_configs={
                "spark.sql.adaptive.advisoryPartitionSizeInBytes": "64MB",
                "spark.sql.adaptive.nonEmptyPartitionRatioForBroadcastJoin": "0.2"
            }
        )
    
    @staticmethod
    def _create_academic_config() -> SparkConfiguration:
        """
        Configuración optimizada para entorno académico.
        
        Esta configuración balancea rendimiento y recursos para la ejecución
        de experimentos de investigación en hardware académico estándar.
        """
        return SparkConfiguration(
            app_name="TFM_Academic_Census_Predictive_Analytics",
            master="local[*]",
            driver_memory="12g",
            driver_max_result_size="4g",
            executor_memory="8g",
            sql_shuffle_partitions=200,
            arrow_enabled=False,
            additional_configs={
                "spark.sql.adaptive.advisoryPartitionSizeInBytes": "128MB",
                "spark.sql.adaptive.nonEmptyPartitionRatioForBroadcastJoin": "0.3",
                "spark.sql.adaptive.maxShuffledHashJoinLocalMapThreshold": "64MB",
                "spark.sql.join.preferSortMergeJoin": "true"
            }
        )
    
    @staticmethod
    def _create_production_config() -> SparkConfiguration:
        """
        Configuración optimizada para entorno de producción.
        
        Esta configuración maximiza el rendimiento y la estabilidad para
        cargas de trabajo de producción con datasets completos.
        """
        return SparkConfiguration(
            app_name="TFM_Production_Census_Analytics",
            master="local[*]",
            driver_memory="16g",
            driver_max_result_size="8g",
            executor_memory="12g",
            sql_shuffle_partitions=400,
            arrow_enabled=True,
            additional_configs={
                "spark.sql.adaptive.advisoryPartitionSizeInBytes": "256MB",
                "spark.sql.adaptive.nonEmptyPartitionRatioForBroadcastJoin": "0.4",
                "spark.sql.adaptive.maxShuffledHashJoinLocalMapThreshold": "128MB",
                "spark.sql.join.preferSortMergeJoin": "true",
                "spark.sql.sources.parallelPartitionDiscovery.threshold": "32"
            }
        )
    
    @staticmethod
    def _create_testing_config() -> SparkConfiguration:
        """
        Configuración optimizada para tests unitarios.
        
        Esta configuración minimiza el uso de recursos para permitir
        ejecuciones rápidas de tests automatizados.
        """
        return SparkConfiguration(
            app_name="TFM_Testing_Census_Analytics",
            master="local[2]",
            driver_memory="2g",
            driver_max_result_size="512m",
            executor_memory="1g",
            sql_shuffle_partitions=4,
            enable_delta=False,  # Simplificar para tests
            arrow_enabled=False,
            additional_configs={
                "spark.sql.adaptive.enabled": "false",  # Simplificar para tests
                "spark.sql.adaptive.coalescePartitions.enabled": "false"
            }
        )


class SparkSessionManager:
    """
    Gestor de sesiones de Spark para el proyecto.
    
    Esta clase implementa el patrón Singleton para garantizar una única
    sesión de Spark activa y facilitar su gestión centralizada.
    """
    
    _instance = None
    _session = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SparkSessionManager, cls).__new__(cls)
        return cls._instance
    
    def get_or_create_session(self, environment: SparkEnvironment) -> SparkSession:
        """
        Obtiene la sesión actual o crea una nueva con la configuración especificada.
        
        Args:
            environment (SparkEnvironment): Entorno de configuración
            
        Returns:
            SparkSession: Sesión de Spark configurada
        """
        if self._session is None:
            self._session = self._create_session(environment)
        return self._session
    
    def _create_session(self, environment: SparkEnvironment) -> SparkSession:
        """
        Crea una nueva sesión de Spark con la configuración del entorno.
        
        Args:
            environment (SparkEnvironment): Entorno de configuración
            
        Returns:
            SparkSession: Nueva sesión de Spark configurada
        """
        # Configurar variables de entorno necesarias
        self._setup_environment_variables()
        
        # Obtener configuración para el entorno
        config = SparkConfigurationFactory.create_configuration(environment)
        
        # Construir SparkSession
        builder = SparkSession.builder \
            .appName(config.app_name) \
            .master(config.master) \
            .config("spark.driver.memory", config.driver_memory) \
            .config("spark.driver.maxResultSize", config.driver_max_result_size) \
            .config("spark.executor.memory", config.executor_memory) \
            .config("spark.sql.shuffle.partitions", str(config.sql_shuffle_partitions))
        
        # Configuraciones de optimización adaptativa
        if config.adaptive_enabled:
            builder = builder.config("spark.sql.adaptive.enabled", "true")
            
        if config.adaptive_coalesce_partitions:
            builder = builder.config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            
        if config.adaptive_skew_join:
            builder = builder.config("spark.sql.adaptive.skewJoin.enabled", "true")
            
        if config.adaptive_local_shuffle_reader:
            builder = builder.config("spark.sql.adaptive.localShuffleReader.enabled", "true")
        
        # Configuración de serialización
        builder = builder.config("spark.serializer", config.serializer)
        
        # Configuración de Delta Lake
        if config.enable_delta:
            builder = builder \
                .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
                .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
                .config("spark.jars.packages", config.delta_package)
        
        # Configuración de Arrow
        if config.arrow_enabled:
            builder = builder.config("spark.sql.execution.arrow.pyspark.enabled", "true")
        
        # Configuraciones adicionales
        if config.additional_configs:
            for key, value in config.additional_configs.items():
                builder = builder.config(key, value)
        
        # Crear sesión
        session = builder.getOrCreate()
        
        # Configurar nivel de logging
        session.sparkContext.setLogLevel("WARN")
        
        return session
    
    def _setup_environment_variables(self) -> None:
        """
        Configura las variables de entorno necesarias para Spark.
        """
        # Configuración específica para entorno Windows académico
        hadoop_home = r'C:\Users\ramir\anaconda3\envs\LAB_MISIA_TFM\hadoop'
        
        if os.path.exists(hadoop_home):
            os.environ['HADOOP_HOME'] = hadoop_home
            os.environ['hadoop.home.dir'] = hadoop_home
            os.environ['PATH'] += os.pathsep + os.path.join(hadoop_home, 'bin')
    
    def stop_session(self) -> None:
        """
        Detiene la sesión actual de Spark de manera controlada.
        """
        if self._session is not None:
            self._session.stop()
            self._session = None


def get_spark_session(environment: SparkEnvironment = SparkEnvironment.ACADEMIC) -> SparkSession:
    """
    Función de conveniencia para obtener una sesión de Spark configurada.
    
    Args:
        environment (SparkEnvironment): Entorno de configuración deseado
        
    Returns:
        SparkSession: Sesión de Spark configurada
    """
    manager = SparkSessionManager()
    return manager.get_or_create_session(environment)


def stop_spark_session() -> None:
    """
    Función de conveniencia para detener la sesión actual de Spark.
    """
    manager = SparkSessionManager()
    manager.stop_session()


# Configuraciones predefinidas para casos de uso específicos
CENSUS_DATA_CONFIG = {
    "spark.sql.adaptive.advisoryPartitionSizeInBytes": "128MB",
    "spark.sql.adaptive.maxShuffledHashJoinLocalMapThreshold": "64MB",
    "spark.sql.join.preferSortMergeJoin": "true"
}

ML_TRAINING_CONFIG = {
    "spark.sql.adaptive.advisoryPartitionSizeInBytes": "256MB",
    "spark.sql.adaptive.maxShuffledHashJoinLocalMapThreshold": "128MB",
    "spark.sql.sources.parallelPartitionDiscovery.threshold": "32"
}

ETL_PROCESSING_CONFIG = {
    "spark.sql.adaptive.advisoryPartitionSizeInBytes": "64MB",
    "spark.sql.adaptive.nonEmptyPartitionRatioForBroadcastJoin": "0.2",
    "spark.sql.join.preferSortMergeJoin": "false"
}