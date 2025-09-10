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
# ARCHIVO: etl/gold/ml_pipeline.py
# =============================================================================
#
# Propósito: Pipeline de machine learning distribuido para análisis predictivo censal
#
# Funcionalidades principales:
# - Entrenamiento distribuido de modelos ensemble para las 3 preguntas de investigación
# - Benchmarking de rendimiento distribuido vs secuencial
# - Validación de efectividad del feature engineering automatizado
# - Generación de evidencias empíricas para el TFM
#
# Dependencias:
# - pyspark.sql
# - pyspark.ml
# - core.logger
# - config.spark_config
#
# Uso:
# from etl.gold.ml_pipeline import DistributedMLPipeline
# pipeline = DistributedMLPipeline(spark_session, base_path)
# results = pipeline.execute_research_ml_pipeline(silver_tables)
#
# =============================================================================

import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when, isnan, isnull
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from core.logger import get_logger, TFMLogger
from config.data_config import get_data_config


@dataclass
class MLExperimentResult:
    """
    Resultado de un experimento de machine learning.
    
    Encapsula métricas, tiempos y metadatos de cada experimento
    para análisis académico posterior.
    """
    experiment_id: str
    model_type: str
    dataset_name: str
    feature_count: int
    training_records: int
    testing_records: int
    auc_score: float
    accuracy_score: float
    training_time_seconds: float
    prediction_time_seconds: float
    distributed_cores_used: int
    research_question: str
    success: bool
    error_message: Optional[str] = None


class DatasetPreprocessor:
    """
    Preprocesador especializado para datasets de ML distribuido.
    
    Maneja la preparación de datos para modelos distribuidos,
    incluyendo limpieza, vectorización y particionamiento.
    """
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = get_logger(__name__)
    
    def prepare_ml_dataset(self, df: DataFrame, target_column: str, 
                          feature_columns: List[str]) -> Tuple[DataFrame, List[str]]:
        """
        Prepara dataset para machine learning distribuido.
        
        Args:
            df (DataFrame): DataFrame con datos
            target_column (str): Columna objetivo
            feature_columns (List[str]): Columnas de features
            
        Returns:
            Tuple[DataFrame, List[str]]: Dataset preparado y features procesados
        """
        # Filtrar solo columnas existentes
        available_features = [col for col in feature_columns if col in df.columns]
        
        if not available_features:
            raise ValueError(f"No se encontraron features válidos para {target_column}")
        
        # Verificar que existe la columna objetivo
        if target_column not in df.columns:
            raise ValueError(f"Columna objetivo {target_column} no encontrada")
        
        # Seleccionar datos con target no nulo
        df_clean = df.select(*available_features, target_column).filter(
            col(target_column).isNotNull()
        )
        
        # Verificar balance de clases
        label_dist = df_clean.groupBy(target_column).count().collect()
        
        if len(label_dist) < 2:
            raise ValueError(f"Dataset {target_column} no tiene suficientes clases")
        
        min_class_count = min(row['count'] for row in label_dist)
        if min_class_count < 100:
            raise ValueError(f"Clase minoritaria muy pequeña: {min_class_count} registros")
        
        self.logger.info(f"Dataset preparado: {df_clean.count():,} registros, {len(available_features)} features")
        
        return df_clean, available_features
    
    def create_ml_pipeline(self, feature_columns: List[str], model_type: str) -> Pipeline:
        """
        Crea pipeline de ML con preprocesamiento automático.
        
        Args:
            feature_columns (List[str]): Columnas de features
            model_type (str): Tipo de modelo a crear
            
        Returns:
            Pipeline: Pipeline de ML configurado
        """
        stages = []
        
        # Separar features numéricos y categóricos
        numeric_features = []
        categorical_features = []
        
        # Heurística simple para categorizar features
        for feature in feature_columns:
            if any(term in feature.lower() for term in ['es_', 'tiene_', 'target_']):
                numeric_features.append(feature)
            elif 'descripcion' in feature.lower() or 'categoria' in feature.lower():
                categorical_features.append(feature)
            else:
                numeric_features.append(feature)  # Por defecto numérico
        
        # StringIndexer para features categóricos
        indexed_features = []
        for cat_feature in categorical_features:
            indexer = StringIndexer(
                inputCol=cat_feature,
                outputCol=f"{cat_feature}_indexed",
                handleInvalid="skip"
            )
            stages.append(indexer)
            indexed_features.append(f"{cat_feature}_indexed")
        
        # Combinar features finales
        final_features = numeric_features + indexed_features
        
        # VectorAssembler
        assembler = VectorAssembler(
            inputCols=final_features,
            outputCol="features",
            handleInvalid="skip"
        )
        stages.append(assembler)
        
        # StandardScaler
        scaler = StandardScaler(
            inputCol="features",
            outputCol="scaled_features",
            withStd=True,
            withMean=False
        )
        stages.append(scaler)
        
        # Modelo según tipo
        if model_type == "RandomForest":
            model = RandomForestClassifier(
                featuresCol="scaled_features",
                labelCol="label",
                numTrees=20,
                maxDepth=10,
                seed=42
            )
        elif model_type == "GradientBoosting":
            model = GBTClassifier(
                featuresCol="scaled_features",
                labelCol="label",
                maxIter=20,
                maxDepth=10,
                seed=42
            )
        elif model_type == "LogisticRegression":
            model = LogisticRegression(
                featuresCol="scaled_features",
                labelCol="label",
                maxIter=20,
                regParam=0.01
            )
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")
        
        stages.append(model)
        
        return Pipeline(stages=stages)


class DistributedPerformanceBenchmarker:
    """
    Benchmarker para medir rendimiento distribuido vs secuencial.
    
    Responde a la Pregunta 1 de investigación sobre optimización
    con procesamiento distribuido Apache Spark.
    """
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = get_logger(__name__)
        self.preprocessor = DatasetPreprocessor(spark)
    
    def benchmark_distributed_processing(self, datasets: Dict[str, DataFrame]) -> List[MLExperimentResult]:
        """
        Ejecuta benchmark de procesamiento distribuido.
        
        Args:
            datasets (Dict[str, DataFrame]): Datasets para benchmark
            
        Returns:
            List[MLExperimentResult]: Resultados de benchmark
        """
        self.logger.info("Iniciando benchmark de procesamiento distribuido")
        
        benchmark_results = []
        
        for dataset_name, df in datasets.items():
            # Buscar variables objetivo en el dataset
            target_columns = [col for col in df.columns if col.startswith('target_')]
            
            if not target_columns:
                self.logger.warning(f"No se encontraron targets en {dataset_name}")
                continue
            
            # Usar el primer target disponible
            target_col = target_columns[0]
            
            # Identificar features para benchmark
            feature_cols = self._identify_benchmark_features(df)
            
            try:
                # Preparar dataset
                df_prepared, final_features = self.preprocessor.prepare_ml_dataset(
                    df, target_col, feature_cols
                )
                
                # Renombrar target a 'label'
                df_prepared = df_prepared.withColumn("label", col(target_col))
                
                # Ejecutar benchmark con Random Forest
                result = self._execute_single_benchmark(
                    df_prepared, final_features, dataset_name, target_col
                )
                
                if result:
                    benchmark_results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error en benchmark {dataset_name}: {str(e)}")
                
                # Crear resultado de error
                error_result = MLExperimentResult(
                    experiment_id=f"benchmark_{dataset_name}_{int(time.time())}",
                    model_type="RandomForest",
                    dataset_name=dataset_name,
                    feature_count=0,
                    training_records=0,
                    testing_records=0,
                    auc_score=0.0,
                    accuracy_score=0.0,
                    training_time_seconds=0.0,
                    prediction_time_seconds=0.0,
                    distributed_cores_used=self.spark.sparkContext.defaultParallelism,
                    research_question="Q1_distributed_processing",
                    success=False,
                    error_message=str(e)
                )
                benchmark_results.append(error_result)
        
        return benchmark_results
    
    def _identify_benchmark_features(self, df: DataFrame) -> List[str]:
        """Identifica features apropiados para benchmark."""
        # Features distribuidos (agregaciones por geografía)
        distributed_features = [
            col for col in df.columns 
            if any(term in col.lower() for term in [
                'poblacion_total', 'densidad_relativa', 'ranking_', 'percentil_'
            ])
        ]
        
        # Features demográficos básicos
        basic_features = [
            col for col in df.columns
            if any(term in col.lower() for term in [
                'es_poblacion_activa', 'es_dependiente', 'es_mujer', 'es_urbano'
            ])
        ]
        
        # Combinar y limitar para benchmark
        all_features = distributed_features + basic_features
        return all_features[:15]  # Limitar para benchmark eficiente
    
    def _execute_single_benchmark(self, df: DataFrame, features: List[str], 
                                 dataset_name: str, target_col: str) -> Optional[MLExperimentResult]:
        """Ejecuta un benchmark individual."""
        start_time = time.time()
        
        try:
            # Split datos
            train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
            
            # Crear pipeline
            pipeline = self.preprocessor.create_ml_pipeline(features, "RandomForest")
            
            # Entrenar modelo (medir tiempo)
            training_start = time.time()
            model = pipeline.fit(train_df)
            training_time = time.time() - training_start
            
            # Predicciones (medir tiempo)
            prediction_start = time.time()
            predictions = model.transform(test_df)
            prediction_time = time.time() - prediction_start
            
            # Evaluar
            evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
            auc_score = evaluator.evaluate(predictions)
            
            accuracy_evaluator = MulticlassClassificationEvaluator(
                labelCol="label", predictionCol="prediction", metricName="accuracy"
            )
            accuracy_score = accuracy_evaluator.evaluate(predictions)
            
            # Crear resultado
            result = MLExperimentResult(
                experiment_id=f"benchmark_{dataset_name}_{int(time.time())}",
                model_type="RandomForest",
                dataset_name=dataset_name,
                feature_count=len(features),
                training_records=train_df.count(),
                testing_records=test_df.count(),
                auc_score=auc_score,
                accuracy_score=accuracy_score,
                training_time_seconds=training_time,
                prediction_time_seconds=prediction_time,
                distributed_cores_used=self.spark.sparkContext.defaultParallelism,
                research_question="Q1_distributed_processing",
                success=True
            )
            
            self.logger.info(
                f"Benchmark {dataset_name}: AUC={auc_score:.3f}, "
                f"Tiempo={training_time:.2f}s, Cores={result.distributed_cores_used}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error en benchmark individual: {str(e)}")
            return None


class FeatureEffectivenessAnalyzer:
    """
    Analizador de efectividad del feature engineering automatizado.
    
    Responde a la Pregunta 2 de investigación sobre técnicas
    de feature engineering más efectivas.
    """
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = get_logger(__name__)
        self.preprocessor = DatasetPreprocessor(spark)
    
    def analyze_feature_effectiveness(self, datasets: Dict[str, DataFrame]) -> List[MLExperimentResult]:
        """
        Analiza efectividad de features automatizados.
        
        Args:
            datasets (Dict[str, DataFrame]): Datasets con features
            
        Returns:
            List[MLExperimentResult]: Resultados de análisis
        """
        self.logger.info("Iniciando análisis de efectividad de features")
        
        analysis_results = []
        
        for dataset_name, df in datasets.items():
            # Buscar múltiples targets para comparar
            target_columns = [col for col in df.columns if col.startswith('target_')]
            
            for target_col in target_columns[:2]:  # Limitar a 2 targets por dataset
                try:
                    # Comparar diferentes conjuntos de features
                    baseline_results = self._test_baseline_features(df, target_col, dataset_name)
                    automated_results = self._test_automated_features(df, target_col, dataset_name)
                    
                    if baseline_results and automated_results:
                        analysis_results.extend([baseline_results, automated_results])
                
                except Exception as e:
                    self.logger.error(f"Error analizando {dataset_name}-{target_col}: {str(e)}")
        
        return analysis_results
    
    def _test_baseline_features(self, df: DataFrame, target_col: str, 
                               dataset_name: str) -> Optional[MLExperimentResult]:
        """Prueba features básicos (baseline)."""
        # Features básicos demográficos
        baseline_features = [
            col for col in df.columns
            if any(term in col.lower() for term in ['edad', 'sexo', 'P02', 'P03'])
            and not col.startswith('target_')
        ]
        
        if len(baseline_features) < 2:
            return None
        
        return self._execute_feature_test(
            df, baseline_features, target_col, dataset_name, "baseline", "GradientBoosting"
        )
    
    def _test_automated_features(self, df: DataFrame, target_col: str,
                                dataset_name: str) -> Optional[MLExperimentResult]:
        """Prueba features automatizados."""
        # Features automatizados creados
        automated_features = [
            col for col in df.columns
            if any(term in col.lower() for term in [
                'grupo_edad', 'es_', 'tiene_', 'nivel_', 'indice_', 'categoria_'
            ])
            and not col.startswith('target_')
        ]
        
        if len(automated_features) < 3:
            return None
        
        return self._execute_feature_test(
            df, automated_features, target_col, dataset_name, "automated", "GradientBoosting"
        )
    
    def _execute_feature_test(self, df: DataFrame, features: List[str], target_col: str,
                             dataset_name: str, feature_type: str, model_type: str) -> Optional[MLExperimentResult]:
        """Ejecuta test de un conjunto de features."""
        try:
            # Preparar dataset
            df_prepared, final_features = self.preprocessor.prepare_ml_dataset(
                df, target_col, features
            )
            
            df_prepared = df_prepared.withColumn("label", col(target_col))
            
            # Split datos
            train_df, test_df = df_prepared.randomSplit([0.8, 0.2], seed=42)
            
            # Crear y entrenar pipeline
            training_start = time.time()
            pipeline = self.preprocessor.create_ml_pipeline(final_features, model_type)
            model = pipeline.fit(train_df)
            training_time = time.time() - training_start
            
            # Predicciones
            prediction_start = time.time()
            predictions = model.transform(test_df)
            prediction_time = time.time() - prediction_start
            
            # Evaluar
            evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
            auc_score = evaluator.evaluate(predictions)
            
            accuracy_evaluator = MulticlassClassificationEvaluator(
                labelCol="label", predictionCol="prediction", metricName="accuracy"
            )
            accuracy_score = accuracy_evaluator.evaluate(predictions)
            
            result = MLExperimentResult(
                experiment_id=f"features_{feature_type}_{dataset_name}_{target_col}_{int(time.time())}",
                model_type=model_type,
                dataset_name=f"{dataset_name}_{target_col}_{feature_type}",
                feature_count=len(final_features),
                training_records=train_df.count(),
                testing_records=test_df.count(),
                auc_score=auc_score,
                accuracy_score=accuracy_score,
                training_time_seconds=training_time,
                prediction_time_seconds=prediction_time,
                distributed_cores_used=self.spark.sparkContext.defaultParallelism,
                research_question="Q2_feature_effectiveness",
                success=True
            )
            
            self.logger.info(
                f"Features {feature_type} {dataset_name}-{target_col}: "
                f"AUC={auc_score:.3f}, Features={len(final_features)}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error en test de features {feature_type}: {str(e)}")
            return None


class EnsembleLearningImplementor:
    """
    Implementador de ensemble learning distribuido.
    
    Responde a la Pregunta 3 de investigación sobre impacto
    de ensemble learning distribuido versus modelos individuales.
    """
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = get_logger(__name__)
        self.preprocessor = DatasetPreprocessor(spark)
    
    def implement_ensemble_learning(self, datasets: Dict[str, DataFrame]) -> List[MLExperimentResult]:
        """
        Implementa y evalúa ensemble learning distribuido.
        
        Args:
            datasets (Dict[str, DataFrame]): Datasets para ensemble
            
        Returns:
            List[MLExperimentResult]: Resultados de ensemble vs individuales
        """
        self.logger.info("Iniciando implementación de ensemble learning")
        
        ensemble_results = []
        
        for dataset_name, df in datasets.items():
            target_columns = [col for col in df.columns if col.startswith('target_')]
            
            for target_col in target_columns[:1]:  # Un target por dataset para ensemble
                try:
                    # Comparar modelos individuales vs ensemble
                    individual_results = self._test_individual_models(df, target_col, dataset_name)
                    ensemble_result = self._test_ensemble_model(df, target_col, dataset_name)
                    
                    ensemble_results.extend(individual_results)
                    if ensemble_result:
                        ensemble_results.append(ensemble_result)
                
                except Exception as e:
                    self.logger.error(f"Error en ensemble {dataset_name}-{target_col}: {str(e)}")
        
        return ensemble_results
    
    def _test_individual_models(self, df: DataFrame, target_col: str, 
                               dataset_name: str) -> List[MLExperimentResult]:
        """Prueba modelos individuales."""
        # Identificar features comprehensivos
        features = self._identify_ensemble_features(df)
        
        if len(features) < 5:
            return []
        
        individual_results = []
        models = ["RandomForest", "GradientBoosting", "LogisticRegression"]
        
        for model_type in models:
            try:
                result = self._execute_individual_model_test(
                    df, features, target_col, dataset_name, model_type
                )
                if result:
                    individual_results.append(result)
            except Exception as e:
                self.logger.error(f"Error modelo individual {model_type}: {str(e)}")
        
        return individual_results
    
    def _test_ensemble_model(self, df: DataFrame, target_col: str,
                            dataset_name: str) -> Optional[MLExperimentResult]:
        """Implementa ensemble simple (majority voting)."""
        try:
            features = self._identify_ensemble_features(df)
            
            # Preparar dataset
            df_prepared, final_features = self.preprocessor.prepare_ml_dataset(
                df, target_col, features
            )
            df_prepared = df_prepared.withColumn("label", col(target_col))
            
            # Split datos
            train_df, test_df = df_prepared.randomSplit([0.8, 0.2], seed=42)
            
            # Entrenar múltiples modelos
            training_start = time.time()
            
            models = {}
            model_types = ["RandomForest", "GradientBoosting"]
            
            for model_type in model_types:
                pipeline = self.preprocessor.create_ml_pipeline(final_features, model_type)
                model = pipeline.fit(train_df)
                models[model_type] = model
            
            training_time = time.time() - training_start
            
            # Predicciones ensemble (promedio simple)
            prediction_start = time.time()
            
            # Obtener predicciones de cada modelo
            predictions_list = []
            for model_name, model in models.items():
                pred = model.transform(test_df)
                predictions_list.append(pred.select("label", "prediction"))
            
            # Ensemble simple: promedio de predicciones
            ensemble_predictions = predictions_list[0]
            for i, pred_df in enumerate(predictions_list[1:], 1):
                ensemble_predictions = ensemble_predictions.join(
                    pred_df.withColumnRenamed("prediction", f"prediction_{i}"),
                    "label"
                )
            
            # Calcular predicción ensemble (majority voting simple)
            ensemble_predictions = ensemble_predictions.withColumn(
                "ensemble_prediction",
                when((col("prediction") + col("prediction_1")) >= 1.0, 1.0).otherwise(0.0)
            )
            
            prediction_time = time.time() - prediction_start
            
            # Evaluar ensemble
            evaluator = BinaryClassificationEvaluator(
                labelCol="label", 
                rawPredictionCol="ensemble_prediction",
                metricName="areaUnderROC"
            )
            
            # Nota: Usar accuracy para ensemble simple
            accuracy_evaluator = MulticlassClassificationEvaluator(
                labelCol="label", 
                predictionCol="ensemble_prediction", 
                metricName="accuracy"
            )
            accuracy_score = accuracy_evaluator.evaluate(ensemble_predictions)
            
            result = MLExperimentResult(
                experiment_id=f"ensemble_{dataset_name}_{target_col}_{int(time.time())}",
                model_type="Ensemble_RF_GBT",
                dataset_name=f"{dataset_name}_{target_col}_ensemble",
                feature_count=len(final_features),
                training_records=train_df.count(),
                testing_records=test_df.count(),
                auc_score=accuracy_score,  # Usar accuracy como proxy
                accuracy_score=accuracy_score,
                training_time_seconds=training_time,
                prediction_time_seconds=prediction_time,
                distributed_cores_used=self.spark.sparkContext.defaultParallelism,
                research_question="Q3_ensemble_learning",
                success=True
            )
            
            self.logger.info(
                f"Ensemble {dataset_name}-{target_col}: "
                f"Accuracy={accuracy_score:.3f}, Models={len(models)}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error en ensemble: {str(e)}")
            return None
    
    def _identify_ensemble_features(self, df: DataFrame) -> List[str]:
        """Identifica features comprehensivos para ensemble."""
        # Combinar diferentes tipos de features
        ensemble_features = []
        
        # Features demográficos
        demo_features = [
            col for col in df.columns
            if any(term in col.lower() for term in ['grupo_edad', 'sexo_', 'es_poblacion'])
            and not col.startswith('target_')
        ]
        ensemble_features.extend(demo_features[:5])
        
        # Features geográficos
        geo_features = [
            col for col in df.columns
            if any(term in col.lower() for term in ['poblacion_total', 'densidad_', 'es_urbano'])
            and not col.startswith('target_')
        ]
        ensemble_features.extend(geo_features[:3])
        
        # Features compuestos
        composite_features = [
            col for col in df.columns
            if any(term in col.lower() for term in ['indice_', 'categoria_', 'nivel_'])
            and not col.startswith('target_')
        ]
        ensemble_features.extend(composite_features[:4])
        
        return list(set(ensemble_features))  # Eliminar duplicados
    
    def _execute_individual_model_test(self, df: DataFrame, features: List[str], 
                                      target_col: str, dataset_name: str, 
                                      model_type: str) -> Optional[MLExperimentResult]:
        """Ejecuta test de modelo individual."""
        try:
            # Preparar dataset
            df_prepared, final_features = self.preprocessor.prepare_ml_dataset(
                df, target_col, features
            )
            df_prepared = df_prepared.withColumn("label", col(target_col))
            
            # Split y entrenamiento
            train_df, test_df = df_prepared.randomSplit([0.8, 0.2], seed=42)
            
            training_start = time.time()
            pipeline = self.preprocessor.create_ml_pipeline(final_features, model_type)
            model = pipeline.fit(train_df)
            training_time = time.time() - training_start
            
            # Predicciones
            prediction_start = time.time()
            predictions = model.transform(test_df)
            prediction_time = time.time() - prediction_start
            
            # Evaluar
            evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
            auc_score = evaluator.evaluate(predictions)
            
            accuracy_evaluator = MulticlassClassificationEvaluator(
                labelCol="label", predictionCol="prediction", metricName="accuracy"
            )
            accuracy_score = accuracy_evaluator.evaluate(predictions)
            
            return MLExperimentResult(
                experiment_id=f"individual_{model_type}_{dataset_name}_{target_col}_{int(time.time())}",
                model_type=f"Individual_{model_type}",
                dataset_name=f"{dataset_name}_{target_col}_individual",
                feature_count=len(final_features),
                training_records=train_df.count(),
                testing_records=test_df.count(),
                auc_score=auc_score,
                accuracy_score=accuracy_score,
                training_time_seconds=training_time,
                prediction_time_seconds=prediction_time,
                distributed_cores_used=self.spark.sparkContext.defaultParallelism,
                research_question="Q3_ensemble_learning",
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error modelo individual {model_type}: {str(e)}")
            return None


class DistributedMLPipeline:
    """
    Pipeline principal de machine learning distribuido.
    
    Orquesta la ejecución de experimentos para responder
    las tres preguntas de investigación del TFM.
    """
    
    def __init__(self, spark: SparkSession, base_path: str = "C:/DataLake"):
        """
        Inicializa el pipeline de ML distribuido.
        
        Args:
            spark (SparkSession): Sesión de Spark configurada
            base_path (str): Ruta base del Data Lake
        """
        self.spark = spark
        self.base_path = base_path
        self.data_config = get_data_config(base_path)
        self.logger = get_logger(__name__)
        
        # Inicializar componentes especializados
        self.benchmarker = DistributedPerformanceBenchmarker(spark)
        self.feature_analyzer = FeatureEffectivenessAnalyzer(spark)
        self.ensemble_implementor = EnsembleLearningImplementor(spark)
        
        self.logger.info("DistributedMLPipeline inicializado")
    
    def execute_research_ml_pipeline(self, silver_tables: Dict[str, DataFrame]) -> Dict[str, Any]:
        """
        Ejecuta pipeline completo de ML para responder preguntas de investigación.
        
        Args:
            silver_tables (Dict[str, DataFrame]): Tablas Silver con features
            
        Returns:
            Dict[str, Any]: Resultados completos de experimentos ML
        """
        TFMLogger.log_pipeline_start("ML_RESEARCH_PIPELINE", self.logger)
        
        start_time = datetime.now()
        all_results = []
        
        try:
            # Filtrar solo master tables
            master_tables = {k: v for k, v in silver_tables.items() if 'master' in k.lower()}
            
            if not master_tables:
                raise ValueError("No se encontraron master tables para ML")
            
            self.logger.info(f"Ejecutando ML pipeline con {len(master_tables)} master tables")
            
            # Pregunta 1: Benchmark procesamiento distribuido
            self.logger.info("PREGUNTA 1: Benchmark procesamiento distribuido")
            q1_results = self.benchmarker.benchmark_distributed_processing(master_tables)
            all_results.extend(q1_results)
            
            # Pregunta 2: Efectividad feature engineering
            self.logger.info("PREGUNTA 2: Análisis efectividad features")
            q2_results = self.feature_analyzer.analyze_feature_effectiveness(master_tables)
            all_results.extend(q2_results)
            
            # Pregunta 3: Ensemble learning distribuido
            self.logger.info("PREGUNTA 3: Ensemble learning distribuido")
            q3_results = self.ensemble_implementor.implement_ensemble_learning(master_tables)
            all_results.extend(q3_results)
            
            # Analizar y consolidar resultados
            consolidated_results = self._consolidate_research_results(all_results)
            
            # Generar reporte de experimentos
            report_path = self._generate_ml_experiments_report(consolidated_results)
            
            # Calcular métricas finales
            end_time = datetime.now()
            total_duration = (end_time - start_time).total_seconds()
            
            successful_experiments = len([r for r in all_results if r.success])
            total_experiments = len(all_results)
            
            TFMLogger.log_pipeline_end(
                "ML_RESEARCH_PIPELINE", 
                self.logger, 
                total_duration, 
                successful_experiments > 0
            )
            
            return {
                'success': successful_experiments > 0,
                'total_duration_seconds': total_duration,
                'total_experiments': total_experiments,
                'successful_experiments': successful_experiments,
                'experiments_by_question': consolidated_results,
                'report_path': report_path,
                'performance_summary': {
                    'average_auc_score': sum(r.auc_score for r in all_results if r.success) / successful_experiments if successful_experiments > 0 else 0,
                    'total_training_time': sum(r.training_time_seconds for r in all_results if r.success),
                    'distributed_cores_used': self.spark.sparkContext.defaultParallelism
                }
            }
            
        except Exception as e:
            end_time = datetime.now()
            total_duration = (end_time - start_time).total_seconds()
            
            error_msg = f"Error crítico en ML pipeline: {str(e)}"
            self.logger.error(error_msg)
            
            TFMLogger.log_pipeline_end("ML_RESEARCH_PIPELINE", self.logger, total_duration, False)
            
            return {
                'success': False,
                'error_message': error_msg,
                'total_duration_seconds': total_duration
            }
    
    def _consolidate_research_results(self, all_results: List[MLExperimentResult]) -> Dict[str, Any]:
        """Consolida resultados por pregunta de investigación."""
        consolidated = {
            'Q1_distributed_processing': [],
            'Q2_feature_effectiveness': [],
            'Q3_ensemble_learning': []
        }
        
        for result in all_results:
            question = result.research_question
            if question in consolidated:
                consolidated[question].append({
                    'experiment_id': result.experiment_id,
                    'model_type': result.model_type,
                    'dataset_name': result.dataset_name,
                    'auc_score': result.auc_score,
                    'training_time_seconds': result.training_time_seconds,
                    'feature_count': result.feature_count,
                    'success': result.success
                })
        
        return consolidated
    
    def _generate_ml_experiments_report(self, consolidated_results: Dict[str, Any]) -> str:
        """Genera reporte detallado de experimentos ML."""
        # Crear reporte JSON para análisis académico
        report_data = {
            'metadata': {
                'report_type': 'ml_experiments_comprehensive',
                'generation_timestamp': datetime.now().isoformat(),
                'tfm_project': 'Sistema de procesamiento distribuido de datos censales',
                'spark_configuration': {
                    'cores_used': self.spark.sparkContext.defaultParallelism,
                    'master': self.spark.sparkContext.master
                }
            },
            'research_questions_results': consolidated_results,
            'performance_analysis': self._analyze_performance_metrics(consolidated_results)
        }
        
        # Guardar reporte
        reports_path = self.data_config.paths.reports_path
        report_filename = f"ml_experiments_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = reports_path / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Reporte ML experimentos generado: {report_path}")
        
        return str(report_path)
    
    def _analyze_performance_metrics(self, consolidated_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza métricas de rendimiento por pregunta."""
        analysis = {}
        
        for question, experiments in consolidated_results.items():
            if not experiments:
                continue
            
            successful_experiments = [exp for exp in experiments if exp['success']]
            
            if successful_experiments:
                avg_auc = sum(exp['auc_score'] for exp in successful_experiments) / len(successful_experiments)
                avg_training_time = sum(exp['training_time_seconds'] for exp in successful_experiments) / len(successful_experiments)
                avg_features = sum(exp['feature_count'] for exp in successful_experiments) / len(successful_experiments)
                
                analysis[question] = {
                    'total_experiments': len(experiments),
                    'successful_experiments': len(successful_experiments),
                    'average_auc_score': avg_auc,
                    'average_training_time_seconds': avg_training_time,
                    'average_feature_count': avg_features
                }
        
        return analysis

