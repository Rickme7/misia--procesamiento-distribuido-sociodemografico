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
# ARCHIVO: etl/silver/data_transformer.py
# =============================================================================
#
# Propósito: Orquestador principal del pipeline de transformación Silver Layer
#
# Funcionalidades principales:
# - Integración completa de recodificación, feature engineering y master building
# - Persistencia optimizada en formato Delta Lake
# - Validación de calidad y generación de reportes académicos
# - Preparación específica para análisis distribuido de ML
#
# Dependencias:
# - pyspark.sql
# - core.logger
# - etl.silver.recoder
# - etl.silver.feature_engineer
# - etl.silver.master_builder
#
# Uso:
# from etl.silver.data_transformer import SilverDataTransformer
# transformer = SilverDataTransformer(spark_session, base_path)
# results = transformer.execute_complete_silver_pipeline(bronze_tables)
#
# =============================================================================

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from pyspark.sql import SparkSession, DataFrame

from core.logger import get_logger, TFMLogger
from config.data_config import get_data_config
from etl.silver.recoder import VariableRecoder
from etl.silver.feature_engineer import AutomatedFeatureEngineer
from etl.silver.master_builder import MasterTableBuilder


class SilverDataPersister:
    """
    Persistidor optimizado para datos Silver en formato Delta Lake.
    
    Esta clase maneja la persistencia eficiente de master tables
    en formato Delta con optimizaciones para análisis distribuido.
    """
    
    def __init__(self, spark: SparkSession, base_path: str):
        self.spark = spark
        self.data_config = get_data_config(base_path)
        self.silver_path = self.data_config.paths.silver_path
        self.logger = get_logger(__name__)
    
    def persist_master_tables_delta(self, master_tables: Dict[str, DataFrame]) -> Dict[str, Any]:
        """
        Persiste master tables en formato Delta Lake con optimizaciones.
        
        Args:
            master_tables (Dict[str, DataFrame]): Master tables a persistir
            
        Returns:
            Dict[str, Any]: Resultados de persistencia por tabla
        """
        self.logger.info("Iniciando persistencia Delta Lake para master tables")
        
        persistence_results = {}
        
        for table_name, df in master_tables.items():
            result = self._persist_single_table_delta(table_name, df)
            persistence_results[table_name] = result
        
        return persistence_results
    
    def _persist_single_table_delta(self, table_name: str, df: DataFrame) -> Dict[str, Any]:
        """Persiste una master table individual en formato Delta."""
        start_time = datetime.now()
        
        try:
            # Determinar ruta de destino
            table_path = self.silver_path / f"{table_name}.delta"
            
            # Optimizar DataFrame para escritura
            df_optimized = self._optimize_for_persistence(df, table_name)
            
            # Determinar particionamiento
            partition_column = self._determine_partition_column(df_optimized, table_name)
            
            # Configurar escritura Delta
            writer = df_optimized.write.format("delta").mode("overwrite")
            writer = writer.option("mergeSchema", "true").option("overwriteSchema", "true")
            
            if partition_column:
                writer = writer.partitionBy(partition_column)
                self.logger.info(f"Particionando {table_name} por: {partition_column}")
            
            # Ejecutar escritura
            self.logger.info(f"Escribiendo {table_name} en formato Delta...")
            writer.save(str(table_path))
            
            # Aplicar optimizaciones Delta
            self._apply_delta_optimizations(table_path, table_name)
            
            # Validar escritura
            validation_result = self._validate_persisted_table(table_path, df_optimized.count())
            
            # Calcular métricas
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return {
                'success': True,
                'table_path': str(table_path),
                'records_written': validation_result['record_count'],
                'columns_written': validation_result['column_count'],
                'duration_seconds': duration,
                'partition_column': partition_column,
                'optimizations_applied': True
            }
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.logger.error(f"Error persistiendo {table_name}: {str(e)}")
            
            return {
                'success': False,
                'error_message': str(e),
                'duration_seconds': duration
            }
    
    def _optimize_for_persistence(self, df: DataFrame, table_name: str) -> DataFrame:
        """Optimiza DataFrame para persistencia Delta."""
        record_count = df.count()
        
        # Determinar número óptimo de particiones
        if record_count > 5000000:  # > 5M registros
            target_partitions = 32
        elif record_count > 1000000:  # > 1M registros
            target_partitions = 16
        else:
            target_partitions = 8
        
        df_optimized = df.coalesce(target_partitions)
        self.logger.info(f"Optimizando {table_name}: {target_partitions} particiones")
        
        return df_optimized
    
    def _determine_partition_column(self, df: DataFrame, table_name: str) -> Optional[str]:
        """Determina columna óptima para particionamiento."""
        # Configuración de particionamiento por tabla
        partition_config = {
            'censo_master': ['I01', 'provinciaCodigo', 'codigo_provincia_2d'],
            'enemdu_master': ['ciudad', 'codigo_provincia_2d']
        }
        
        candidates = partition_config.get(table_name, [])
        
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
        
        return None
    
    def _apply_delta_optimizations(self, table_path: Path, table_name: str) -> None:
        """Aplica optimizaciones específicas de Delta Lake."""
        try:
            table_path_str = str(table_path)
            
            # OPTIMIZE para compactar archivos
            self.spark.sql(f"OPTIMIZE delta.`{table_path_str}`")
            self.logger.info(f"OPTIMIZE aplicado a {table_name}")
            
            # Z-ORDER por columnas frecuentemente consultadas
            zorder_columns = self._get_zorder_columns(table_name)
            if zorder_columns:
                zorder_sql = f"OPTIMIZE delta.`{table_path_str}` ZORDER BY ({', '.join(zorder_columns)})"
                self.spark.sql(zorder_sql)
                self.logger.info(f"Z-ORDER aplicado a {table_name}: {zorder_columns}")
            
        except Exception as e:
            self.logger.warning(f"Error en optimizaciones Delta para {table_name}: {e}")
    
    def _get_zorder_columns(self, table_name: str) -> List[str]:
        """Obtiene columnas para Z-ORDER según tabla."""
        zorder_config = {
            'censo_master': ['provinciaCodigo', 'edad'],
            'enemdu_master': ['ciudad', 'edad']
        }
        
        return zorder_config.get(table_name, [])
    
    def _validate_persisted_table(self, table_path: Path, expected_records: int) -> Dict[str, Any]:
        """Valida tabla persistida."""
        try:
            df_validation = self.spark.read.format("delta").load(str(table_path))
            actual_records = df_validation.count()
            actual_columns = len(df_validation.columns)
            
            return {
                'record_count': actual_records,
                'column_count': actual_columns,
                'integrity_check': actual_records == expected_records
            }
            
        except Exception as e:
            self.logger.error(f"Error validando tabla persistida: {e}")
            return {
                'record_count': 0,
                'column_count': 0,
                'integrity_check': False,
                'validation_error': str(e)
            }


class SilverQualityReporter:
    """
    Generador de reportes de calidad para el Silver Layer.
    
    Esta clase crea reportes académicos detallados del proceso
    de transformación Silver para documentación del TFM.
    """
    
    def __init__(self, base_path: str):
        self.data_config = get_data_config(base_path)
        self.reports_path = self.data_config.paths.reports_path
        self.logger = get_logger(__name__)
    
    def generate_comprehensive_silver_report(self, pipeline_results: Dict[str, Any]) -> str:
        """
        Genera reporte comprehensivo del pipeline Silver.
        
        Args:
            pipeline_results (Dict[str, Any]): Resultados completos del pipeline
            
        Returns:
            str: Ruta del reporte generado
        """
        self.logger.info("Generando reporte comprehensivo Silver Layer")
        
        # Estructura del reporte académico
        report_data = {
            'metadata': {
                'report_type': 'silver_layer_comprehensive_analysis',
                'generation_timestamp': datetime.now().isoformat(),
                'tfm_project': 'Sistema de procesamiento distribuido de datos censales',
                'author': 'Ramiro Ricardo Merchán Mora',
                'university': 'Universidad de Málaga',
                'academic_year': '2024-2025'
            },
            'pipeline_summary': self._extract_pipeline_summary(pipeline_results),
            'recoding_analysis': self._extract_recoding_analysis(pipeline_results),
            'feature_engineering_analysis': self._extract_feature_analysis(pipeline_results),
            'master_tables_analysis': self._extract_master_tables_analysis(pipeline_results),
            'quality_metrics': self._extract_quality_metrics(pipeline_results),
            'research_questions_coverage': self._extract_research_coverage(pipeline_results),
            'technical_performance': self._extract_performance_metrics(pipeline_results)
        }
        
        # Guardar reporte
        report_filename = f"silver_comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = self.reports_path / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Reporte Silver generado: {report_path}")
        
        # Generar también versión resumida en texto
        summary_path = self._generate_text_summary(report_data, report_path.stem)
        
        return str(report_path)
    
    def _extract_pipeline_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae resumen general del pipeline."""
        return {
            'total_duration_seconds': results.get('total_duration_seconds', 0),
            'success_status': results.get('success', False),
            'tables_processed': results.get('tables_processed', 0),
            'master_tables_created': len(results.get('master_tables', {})),
            'total_records_processed': sum(
                df.count() for df in results.get('master_tables', {}).values()
            ) if results.get('master_tables') else 0
        }
    
    def _extract_recoding_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae análisis de recodificación."""
        recoding_stats = results.get('recoding_statistics', {})
        
        return {
            'total_recodifications': recoding_stats.get('overall_statistics', {}).get('total_operations', 0),
            'success_rate': recoding_stats.get('overall_statistics', {}).get('overall_success_rate', 0),
            'geographic_recodifications': len([
                op for op in recoding_stats.get('statistics_by_operation_type', {}).keys()
                if 'geographic' in op
            ]),
            'variable_recodifications': len([
                op for op in recoding_stats.get('statistics_by_operation_type', {}).keys()
                if 'variable' in op
            ])
        }
    
    def _extract_feature_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae análisis de feature engineering."""
        feature_catalog = results.get('feature_catalog', [])
        research_coverage = results.get('research_question_coverage', {})
        
        feature_type_counts = {}
        for feature in feature_catalog:
            feature_type = feature.get('feature_type', 'unknown')
            feature_type_counts[feature_type] = feature_type_counts.get(feature_type, 0) + 1
        
        return {
            'total_features_created': len(feature_catalog),
            'features_by_type': feature_type_counts,
            'research_coverage': research_coverage,
            'automated_features_percentage': (
                feature_type_counts.get('demographic', 0) + 
                feature_type_counts.get('composite_index', 0)
            ) / len(feature_catalog) * 100 if feature_catalog else 0
        }
    
    def _extract_master_tables_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae análisis de master tables."""
        validation_results = results.get('validation_results', {})
        persistence_results = results.get('persistence_results', {})
        
        tables_analysis = {}
        for table_name, metrics in validation_results.items():
            tables_analysis[table_name] = {
                'record_count': getattr(metrics, 'record_count', 0),
                'column_count': getattr(metrics, 'column_count', 0),
                'quality_score': getattr(metrics, 'data_quality_score', 0),
                'feature_count': getattr(metrics, 'feature_count', 0),
                'target_variables': getattr(metrics, 'target_variables', []),
                'geographic_coverage': getattr(metrics, 'geographic_coverage', 0),
                'persistence_success': persistence_results.get(table_name, {}).get('success', False)
            }
        
        return tables_analysis
    
    def _extract_quality_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae métricas de calidad."""
        validation_results = results.get('validation_results', {})
        
        quality_scores = []
        null_percentages = []
        duplicate_percentages = []
        
        for metrics in validation_results.values():
            if hasattr(metrics, 'data_quality_score'):
                quality_scores.append(metrics.data_quality_score)
            if hasattr(metrics, 'null_percentage'):
                null_percentages.append(metrics.null_percentage)
            if hasattr(metrics, 'duplicate_percentage'):
                duplicate_percentages.append(metrics.duplicate_percentage)
        
        return {
            'average_quality_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            'average_null_percentage': sum(null_percentages) / len(null_percentages) if null_percentages else 0,
            'average_duplicate_percentage': sum(duplicate_percentages) / len(duplicate_percentages) if duplicate_percentages else 0,
            'quality_threshold_met': all(score >= 70 for score in quality_scores)
        }
    
    def _extract_research_coverage(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae cobertura de preguntas de investigación."""
        coverage = results.get('research_question_coverage', {})
        
        return {
            'question_1_distributed_processing': {
                'coverage_met': coverage.get('question_1_distributed_processing', {}).get('coverage_adequate', False),
                'features_count': coverage.get('question_1_distributed_processing', {}).get('features_count', 0)
            },
            'question_2_automated_features': {
                'coverage_met': coverage.get('question_2_automated_features', {}).get('coverage_adequate', False),
                'features_count': coverage.get('question_2_automated_features', {}).get('features_count', 0)
            },
            'question_3_ensemble_targets': {
                'coverage_met': coverage.get('question_3_ensemble_targets', {}).get('coverage_adequate', False),
                'features_count': coverage.get('question_3_ensemble_targets', {}).get('features_count', 0)
            }
        }
    
    def _extract_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae métricas de rendimiento técnico."""
        persistence_results = results.get('persistence_results', {})
        
        total_duration = results.get('total_duration_seconds', 0)
        total_records = sum(
            df.count() for df in results.get('master_tables', {}).values()
        ) if results.get('master_tables') else 0
        
        return {
            'total_processing_time_seconds': total_duration,
            'total_records_processed': total_records,
            'average_throughput_records_per_second': total_records / total_duration if total_duration > 0 else 0,
            'delta_optimizations_applied': all(
                result.get('optimizations_applied', False) 
                for result in persistence_results.values()
            ),
            'all_tables_persisted_successfully': all(
                result.get('success', False) 
                for result in persistence_results.values()
            )
        }
    
    def _generate_text_summary(self, report_data: Dict[str, Any], base_filename: str) -> str:
        """Genera resumen en texto plano para lectura rápida."""
        summary_path = self.reports_path / f"{base_filename}_summary.txt"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RESUMEN EJECUTIVO - SILVER LAYER PROCESSING\n")
            f.write("=" * 80 + "\n\n")
            
            pipeline_summary = report_data['pipeline_summary']
            f.write(f"Estado: {'EXITOSO' if pipeline_summary['success_status'] else 'FALLIDO'}\n")
            f.write(f"Duración total: {pipeline_summary['total_duration_seconds']:.2f} segundos\n")
            f.write(f"Tablas procesadas: {pipeline_summary['tables_processed']}\n")
            f.write(f"Master tables creadas: {pipeline_summary['master_tables_created']}\n")
            f.write(f"Registros procesados: {pipeline_summary['total_records_processed']:,}\n\n")
            
            feature_analysis = report_data['feature_engineering_analysis']
            f.write(f"Features creados: {feature_analysis['total_features_created']}\n")
            f.write(f"Porcentaje automatizado: {feature_analysis['automated_features_percentage']:.1f}%\n\n")
            
            quality_metrics = report_data['quality_metrics']
            f.write(f"Score promedio de calidad: {quality_metrics['average_quality_score']:.1f}/100\n")
            f.write(f"Umbral de calidad alcanzado: {'SÍ' if quality_metrics['quality_threshold_met'] else 'NO'}\n\n")
            
            research_coverage = report_data['research_questions_coverage']
            f.write("COBERTURA PREGUNTAS DE INVESTIGACIÓN:\n")
            for question, data in research_coverage.items():
                status = "✓" if data['coverage_met'] else "✗"
                f.write(f"  {status} {question}: {data['features_count']} features\n")
        
        return str(summary_path)


class SilverDataTransformer:
    """
    Orquestador principal del pipeline de transformación Silver Layer.
    
    Esta clase integra todos los componentes del Silver Layer en un
    workflow coherente optimizado para análisis distribuido.
    """
    
    def __init__(self, spark: SparkSession, base_path: str = "C:/DataLake"):
        """
        Inicializa el transformador de datos Silver.
        
        Args:
            spark (SparkSession): Sesión de Spark configurada
            base_path (str): Ruta base del Data Lake
        """
        self.spark = spark
        self.base_path = base_path
        self.data_config = get_data_config(base_path)
        self.logger = get_logger(__name__)
        
        # Inicializar componentes del pipeline
        self.persister = SilverDataPersister(spark, base_path)
        self.reporter = SilverQualityReporter(base_path)
        
        self.logger.info(f"SilverDataTransformer inicializado para: {base_path}")
    
    def execute_complete_silver_pipeline(self, bronze_tables: Dict[str, DataFrame]) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo de transformación Silver Layer.
        
        Args:
            bronze_tables (Dict[str, DataFrame]): Tablas de la capa Bronze
            
        Returns:
            Dict[str, Any]: Resultados completos del pipeline Silver
        """
        TFMLogger.log_pipeline_start("SILVER_LAYER_TRANSFORMATION", self.logger)
        
        start_time = datetime.now()
        
        try:
            # Separar tablas principales de diccionarios
            main_tables, dictionaries = self._separate_tables_and_dictionaries(bronze_tables)
            
            if not dictionaries:
                raise ValueError("No se encontraron diccionarios de recodificación")
            
            # Paso 1: Recodificación de variables
            self.logger.info("PASO 1: Recodificación de variables categóricas")
            recoder = VariableRecoder(self.spark, dictionaries)
            recoded_tables = recoder.recode_all_tables(main_tables)
            recoding_statistics = recoder.get_recoding_statistics()
            
            # Paso 2: Feature Engineering automatizado
            self.logger.info("PASO 2: Feature engineering automatizado")
            feature_engineer = AutomatedFeatureEngineer(self.spark)
            enhanced_tables = feature_engineer.create_comprehensive_features(recoded_tables)
            feature_catalog = feature_engineer.get_feature_catalog()
            research_coverage = feature_engineer.get_research_question_coverage()
            
            # Paso 3: Construcción de Master Tables
            self.logger.info("PASO 3: Construcción de master tables")
            master_builder = MasterTableBuilder(self.spark)
            master_tables = master_builder.build_separated_master_tables(enhanced_tables)
            validation_results = master_builder.validate_all_master_tables(master_tables)
            
            # Paso 4: Persistencia en Delta Lake
            self.logger.info("PASO 4: Persistencia optimizada Delta Lake")
            persistence_results = self.persister.persist_master_tables_delta(master_tables)
            
            # Paso 5: Generación de reportes
            self.logger.info("PASO 5: Generación de reportes de calidad")
            
            # Consolidar resultados para reporte
            pipeline_results = {
                'success': True,
                'total_duration_seconds': 0,  # Se calculará al final
                'tables_processed': len(main_tables),
                'master_tables': master_tables,
                'recoding_statistics': recoding_statistics,
                'feature_catalog': feature_catalog,
                'research_question_coverage': research_coverage,
                'validation_results': validation_results,
                'persistence_results': persistence_results
            }
            
            report_path = self.reporter.generate_comprehensive_silver_report(pipeline_results)
            
            # Calcular métricas finales
            end_time = datetime.now()
            total_duration = (end_time - start_time).total_seconds()
            
            # Actualizar duración en resultados
            pipeline_results['total_duration_seconds'] = total_duration
            
            # Determinar éxito general
            all_tables_persisted = all(
                result.get('success', False) 
                for result in persistence_results.values()
            )
            
            pipeline_success = len(master_tables) > 0 and all_tables_persisted
            
            TFMLogger.log_pipeline_end("SILVER_LAYER_TRANSFORMATION", self.logger, total_duration, pipeline_success)
            
            # Resultado final
            final_results = {
                'success': pipeline_success,
                'total_duration_seconds': total_duration,
                'master_tables_created': len(master_tables),
                'total_records_processed': sum(df.count() for df in master_tables.values()),
                'features_created': len(feature_catalog),
                'average_quality_score': self._calculate_average_quality_score(validation_results),
                'research_questions_covered': self._count_covered_research_questions(research_coverage),
                'delta_paths': {
                    name: result.get('table_path') 
                    for name, result in persistence_results.items() 
                    if result.get('success')
                },
                'report_path': report_path,
                'detailed_results': pipeline_results
            }
            
            return final_results
            
        except Exception as e:
            end_time = datetime.now()
            total_duration = (end_time - start_time).total_seconds()
            
            error_msg = f"Error crítico en pipeline Silver: {str(e)}"
            self.logger.error(error_msg)
            
            TFMLogger.log_pipeline_end("SILVER_LAYER_TRANSFORMATION", self.logger, total_duration, False)
            
            return {
                'success': False,
                'error_message': error_msg,
                'total_duration_seconds': total_duration
            }
    
    def _separate_tables_and_dictionaries(self, bronze_tables: Dict[str, DataFrame]) -> tuple:
        """Separa tablas principales de diccionarios."""
        main_tables = {k: v for k, v in bronze_tables.items() if not k.startswith('dict_')}
        dictionaries = {k: v for k, v in bronze_tables.items() if k.startswith('dict_')}
        
        self.logger.info(f"Tablas principales: {len(main_tables)}, Diccionarios: {len(dictionaries)}")
        
        return main_tables, dictionaries
    
    def _calculate_average_quality_score(self, validation_results: Dict[str, Any]) -> float:
        """Calcula score promedio de calidad."""
        scores = [
            getattr(metrics, 'data_quality_score', 0) 
            for metrics in validation_results.values()
        ]
        
        return sum(scores) / len(scores) if scores else 0
    
    def _count_covered_research_questions(self, research_coverage: Dict[str, Any]) -> int:
        """Cuenta preguntas de investigación cubiertas adecuadamente."""
        covered_count = 0
        
        for question_data in research_coverage.values():
            if question_data.get('coverage_adequate', False):
                covered_count += 1
        
        return covered_count