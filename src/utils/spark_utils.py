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
# ARCHIVO: utils/spark_utils.py
# =============================================================================
#
# Propósito: Utilidades específicas de Apache Spark para optimización distribuida
#
# =============================================================================

from typing import Dict, List, Optional, Any
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, count


def optimize_dataframe_partitions(df: DataFrame, target_partition_size_mb: int = 128) -> DataFrame:
    """Optimiza particiones de DataFrame según tamaño objetivo."""
    current_partitions = df.rdd.getNumPartitions()
    estimated_size_mb = estimate_dataframe_size_mb(df)
    
    if estimated_size_mb == 0:
        return df
    
    optimal_partitions = max(1, int(estimated_size_mb / target_partition_size_mb))
    
    if optimal_partitions != current_partitions:
        if optimal_partitions < current_partitions:
            return df.coalesce(optimal_partitions)
        else:
            return df.repartition(optimal_partitions)
    
    return df


def estimate_dataframe_size_mb(df: DataFrame) -> float:
    """Estima tamaño de DataFrame en MB."""
    try:
        record_count = df.count()
        column_count = len(df.columns)
        
        # Estimación heurística: 50 bytes promedio por celda
        estimated_bytes = record_count * column_count * 50
        estimated_mb = estimated_bytes / (1024 * 1024)
        
        return estimated_mb
    except:
        return 0.0


def get_spark_metrics(spark: SparkSession) -> Dict[str, Any]:
    """Obtiene métricas actuales de Spark."""
    return {
        'app_name': spark.sparkContext.appName,
        'master': spark.sparkContext.master,
        'default_parallelism': spark.sparkContext.defaultParallelism,
        'spark_version': spark.version,
        'active_stages': len(spark.sparkContext.statusTracker().getActiveStageIds()),
        'active_jobs': len(spark.sparkContext.statusTracker().getActiveJobIds())
    }


def broadcast_small_dataframe(df: DataFrame, threshold_mb: float = 200.0) -> DataFrame:
    """Marca DataFrame para broadcast si es pequeño."""
    from pyspark.sql.functions import broadcast
    
    estimated_size = estimate_dataframe_size_mb(df)
    if estimated_size > 0 and estimated_size <= threshold_mb:
        return broadcast(df)
    return df


def analyze_dataframe_skew(df: DataFrame, partition_col: str) -> Dict[str, Any]:
    """Analiza distribución de datos por partición."""
    if partition_col not in df.columns:
        return {'error': f'Columna {partition_col} no encontrada'}
    
    partition_counts = df.groupBy(partition_col).count().collect()
    
    if not partition_counts:
        return {'error': 'No hay datos para analizar'}
    
    counts = [row['count'] for row in partition_counts]
    total_records = sum(counts)
    
    return {
        'total_partitions': len(counts),
        'total_records': total_records,
        'avg_records_per_partition': total_records / len(counts),
        'max_partition_size': max(counts),
        'min_partition_size': min(counts),
        'skew_ratio': max(counts) / min(counts) if min(counts) > 0 else float('inf'),
        'is_skewed': (max(counts) / min(counts)) > 3.0 if min(counts) > 0 else True
    }