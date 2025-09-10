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
# ARCHIVO: main.py
# =============================================================================
#
# Propósito: Orquestador principal del pipeline completo TFM
#
# =============================================================================

import argparse
import sys
from datetime import datetime
from typing import Dict, Any

from core.logger import setup_logging, get_logger, TFMLogger
from config.spark_config import get_spark_session, stop_spark_session, SparkEnvironment
from etl.bronze.data_loader import BronzeDataLoader
from etl.silver.data_transformer import SilverDataTransformer
from etl.gold.ml_pipeline import DistributedMLPipeline


def main():
    """Función principal del sistema de procesamiento distribuido."""
    parser = argparse.ArgumentParser(
        description="Sistema de procesamiento distribuido de datos censales"
    )
    parser.add_argument("--base-path", default="C:/DataLake", help="Ruta base del Data Lake")
    parser.add_argument("--environment", choices=["development", "academic", "production"], 
                       default="academic", help="Entorno de ejecución")
    parser.add_argument("--layer", choices=["bronze", "silver", "gold", "all"], 
                       default="all", help="Capa específica a ejecutar")
    
    args = parser.parse_args()
    
    # Configurar logging
    setup_logging(args.base_path)
    logger = get_logger(__name__)
    
    TFMLogger.log_pipeline_start("TFM_COMPLETE_PIPELINE", logger)
    start_time = datetime.now()
    
    spark = None
    try:
        # Inicializar Spark
        spark_env = SparkEnvironment(args.environment)
        spark = get_spark_session(spark_env)
        
        # Ejecutar pipeline según capa especificada
        if args.layer == "all":
            results = execute_complete_pipeline(spark, args.base_path)
        elif args.layer == "bronze":
            results = execute_bronze_layer(spark, args.base_path)
        elif args.layer == "silver":
            results = execute_silver_layer(spark, args.base_path)
        elif args.layer == "gold":
            results = execute_gold_layer(spark, args.base_path)
        
        # Log resultados finales
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        TFMLogger.log_pipeline_end("TFM_COMPLETE_PIPELINE", logger, duration, results.get('success', False))
        
        if results.get('success'):
            logger.info("Pipeline TFM completado exitosamente")
            logger.info(f"Duración total: {duration:.2f} segundos")
        else:
            logger.error("Pipeline TFM falló")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Error crítico en pipeline principal: {e}")
        sys.exit(1)
    
    finally:
        if spark:
            stop_spark_session()


def execute_complete_pipeline(spark, base_path: str) -> Dict[str, Any]:
    """Ejecuta pipeline completo Bronze → Silver → Gold."""
    logger = get_logger(__name__)
    
    # Bronze Layer
    logger.info("Ejecutando Bronze Layer")
    bronze_loader = BronzeDataLoader(spark, base_path)
    bronze_tables = bronze_loader.load_all_bronze_tables()
    
    # Silver Layer
    logger.info("Ejecutando Silver Layer")
    silver_transformer = SilverDataTransformer(spark, base_path)
    silver_results = silver_transformer.execute_complete_silver_pipeline(bronze_tables)
    
    if not silver_results.get('success'):
        return {'success': False, 'error': 'Silver layer falló'}
    
    # Gold Layer (cargar desde Delta)
    logger.info("Ejecutando Gold Layer")
    # Cargar master tables desde Delta
    silver_tables = {}
    for table_name, path in silver_results.get('delta_paths', {}).items():
        if path:
            silver_tables[table_name] = spark.read.format("delta").load(path)
    
    gold_pipeline = DistributedMLPipeline(spark, base_path)
    gold_results = gold_pipeline.execute_research_ml_pipeline(silver_tables)
    
    return {
        'success': gold_results.get('success', False),
        'bronze_tables': len(bronze_tables),
        'silver_results': silver_results,
        'gold_results': gold_results
    }


def execute_bronze_layer(spark, base_path: str) -> Dict[str, Any]:
    """Ejecuta solo Bronze Layer."""
    loader = BronzeDataLoader(spark, base_path)
    tables = loader.load_all_bronze_tables()
    
    return {
        'success': len(tables) > 0,
        'tables_loaded': len(tables)
    }


def execute_silver_layer(spark, base_path: str) -> Dict[str, Any]:
    """Ejecuta solo Silver Layer."""
    # Cargar datos Bronze
    bronze_loader = BronzeDataLoader(spark, base_path)
    bronze_tables = bronze_loader.load_all_bronze_tables()
    
    # Transformar a Silver
    silver_transformer = SilverDataTransformer(spark, base_path)
    return silver_transformer.execute_complete_silver_pipeline(bronze_tables)


def execute_gold_layer(spark, base_path: str) -> Dict[str, Any]:
    """Ejecuta solo Gold Layer."""
    # Cargar datos Silver desde Delta
    from pathlib import Path
    silver_path = Path(base_path) / "silver"
    
    silver_tables = {}
    for delta_dir in silver_path.glob("*.delta"):
        table_name = delta_dir.stem
        silver_tables[table_name] = spark.read.format("delta").load(str(delta_dir))
    
    # Ejecutar ML pipeline
    gold_pipeline = DistributedMLPipeline(spark, base_path)
    return gold_pipeline.execute_research_ml_pipeline(silver_tables)


if __name__ == "__main__":
    main()