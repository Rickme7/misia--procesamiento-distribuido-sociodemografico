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
# ARCHIVO: etl/gold/model_trainer.py
# =============================================================================
#
# Propósito: Entrenador especializado de modelos ML distribuidos para análisis censal
#
# Funcionalidades principales:
# - Entrenamiento optimizado de modelos ensemble distribuidos
# - Validación cruzada y hyperparameter tuning automático
# - Persistencia de modelos entrenados en formato MLlib
# - Métricas detalladas de rendimiento para evaluación académica
#
# Dependencias:
# - pyspark.ml
# - pyspark.sql
# - core.logger
#
# Uso:
# from etl.gold.model_trainer import DistributedModelTrainer
# trainer = DistributedModelTrainer(spark_session, base_path)
# models = trainer.train_research_models(prepared_datasets)
#
# =============================================================================

import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit

from core.logger import get_logger
from config.data_config import get_data_config


@dataclass
class ModelTrainingResult:
    """
    Resultado completo del entrenamiento de un modelo.
    
    Encapsula métricas, modelo entrenado y metadatos
    para análisis académico posterior.
    """
    model_id: str
    model_type: str
    dataset_name: str
    target_variable: str
    features_used: List[str]
    training_records: int
    validation_records: int
    best_params: Dict[str, Any]
    cv_auc_score: float
    cv_accuracy_score: float
    validation_auc_score: float
    validation_accuracy_score: float
    training_time_seconds: float
    model_path: Optional[str]
    feature_importance: Optional[Dict[str, float]]
    success: bool
    error_message: Optional[str] = None


class OptimizedModelBuilder:
    """
    Constructor optimizado de modelos ML para datasets censales.
    
    Crea pipelines de ML optimizados específicamente para
    el análisis de datos socioeconómicos distribuidos.
    """
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = get_logger(__name__)
    
    def create_optimized_pipeline(self, feature_columns: List[str], 
                                 model_type: str, target_col: str = "label") -> Pipeline:
        """
        Crea pipeline optimizado para datos censales.
        
        Args:
            feature_columns (List[str]): Columnas de features
            model_type (str): Tipo de modelo a crear
            target_col (str): Columna objetivo
            
        Returns:
            Pipeline: Pipeline optimizado configurado
        """
        stages = []
        
        # Preprocesamiento inteligente de features
        processed_features = self._preprocess_census_features(feature_columns, stages)
        
        # VectorAssembler
        assembler = VectorAssembler(
            inputCols=processed_features,
            outputCol="features",
            handleInvalid="skip"
        )
        stages.append(assembler)
        
        # StandardScaler para normalización
        scaler = StandardScaler(
            inputCol="features",
            outputCol="scaled_features",
            withStd=True,
            withMean=False
        )
        stages.append(scaler)
        
        # Modelo optimizado según tipo
        model = self._create_optimized_classifier(model_type, target_col)
        stages.append(model)
        
        return Pipeline(stages=stages)
    
    def _preprocess_census_features(self, feature_columns: List[str], 
                                   stages: List) -> List[str]:
        """Preprocesa features específicos para datos censales."""
        numeric_features = []
        categorical_features = []
        processed_features = []
        
        # Clasificar features por tipo
        for feature in feature_columns:
            if self._is_categorical_feature(feature):
                categorical_features.append(feature)
            else:
                numeric_features.append(feature)
        
        # Procesar features categóricos
        for cat_feature in categorical_features:
            indexed_name = f"{cat_feature}_indexed"
            
            indexer = StringIndexer(
                inputCol=cat_feature,
                outputCol=indexed_name,
                handleInvalid="skip"
            )
            stages.append(indexer)
            processed_features.append(indexed_name)
        
        # Agregar features numéricos directamente
        processed_features.extend(numeric_features)
        
        return processed_features
    
    def _is_categorical_feature(self, feature_name: str) -> bool:
        """Determina si un feature es categórico basado en su nombre."""
        categorical_indicators = [
            'descripcion', 'categoria', 'tipo', 'nombre', 'grupo_edad',
            'region_', 'sexo_descripcion'
        ]
        
        return any(indicator in feature_name.lower() for indicator in categorical_indicators)
    
    def _create_optimized_classifier(self, model_type: str, target_col: str):
        """Crea clasificador optimizado para datos censales."""
        if model_type == "RandomForest":
            return RandomForestClassifier(
                featuresCol="scaled_features",
                labelCol=target_col,
                numTrees=50,
                maxDepth=15,
                maxBins=32,
                minInstancesPerNode=10,
                seed=42
            )
        elif model_type == "GradientBoosting":
            return GBTClassifier(
                featuresCol="scaled_features",
                labelCol=target_col,
                maxIter=50,
                maxDepth=10,
                stepSize=0.1,
                seed=42
            )
        elif model_type == "LogisticRegression":
            return LogisticRegression(
                featuresCol="scaled_features",
                labelCol=target_col,
                maxIter=100,
                regParam=0.01,
                elasticNetParam=0.1
            )
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")


class HyperparameterOptimizer:
    """
    Optimizador de hiperparámetros para modelos distribuidos.
    
    Implementa búsqueda automática de hiperparámetros optimizada
    para datasets censales usando validación cruzada distribuida.
    """
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = get_logger(__name__)
    
    def optimize_model_hyperparameters(self, pipeline: Pipeline, train_df: DataFrame,
                                     model_type: str) -> Tuple[PipelineModel, Dict[str, Any], float]:
        """
        Optimiza hiperparámetros usando validación cruzada.
        
        Args:
            pipeline (Pipeline): Pipeline a optimizar
            train_df (DataFrame): Datos de entrenamiento
            model_type (str): Tipo de modelo
            
        Returns:
            Tuple[PipelineModel, Dict[str, Any], float]: Mejor modelo, parámetros y score
        """
        # Crear grid de parámetros según tipo de modelo
        param_grid = self._create_parameter_grid(pipeline, model_type)
        
        # Configurar evaluador
        evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            metricName="areaUnderROC"
        )
        
        # Configurar validación cruzada optimizada
        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=3,  # Reducido para eficiencia en datasets grandes
            seed=42,
            parallelism=2  # Paralelización de CV
        )
        
        self.logger.info(f"Iniciando optimización de hiperparámetros para {model_type}")
        
        # Ejecutar validación cruzada
        cv_model = cv.fit(train_df)
        
        # Extraer mejores parámetros
        best_model = cv_model.bestModel
        best_params = self._extract_best_parameters(cv_model, model_type)
        best_score = max(cv_model.avgMetrics)
        
        self.logger.info(f"Mejor score CV para {model_type}: {best_score:.4f}")
        
        return best_model, best_params, best_score
    
    def _create_parameter_grid(self, pipeline: Pipeline, model_type: str) -> List[Dict]:
        """Crea grid de parámetros específico por tipo de modelo."""
        # Obtener la etapa del modelo en el pipeline
        model_stage = None
        for stage in pipeline.getStages():
            if hasattr(stage, 'numTrees') or hasattr(stage, 'maxIter'):
                model_stage = stage
                break
        
        if model_stage is None:
            return [{}]
        
        param_builder = ParamGridBuilder()
        
        if model_type == "RandomForest":
            param_builder.addGrid(model_stage.numTrees, [20, 50]) \
                         .addGrid(model_stage.maxDepth, [10, 15]) \
                         .addGrid(model_stage.minInstancesPerNode, [5, 10])
        
        elif model_type == "GradientBoosting":
            param_builder.addGrid(model_stage.maxIter, [20, 50]) \
                         .addGrid(model_stage.maxDepth, [5, 10]) \
                         .addGrid(model_stage.stepSize, [0.05, 0.1])
        
        elif model_type == "LogisticRegression":
            param_builder.addGrid(model_stage.regParam, [0.001, 0.01, 0.1]) \
                         .addGrid(model_stage.elasticNetParam, [0.0, 0.1, 0.5])
        
        return param_builder.build()
    
    def _extract_best_parameters(self, cv_model, model_type: str) -> Dict[str, Any]:
        """Extrae los mejores parámetros del modelo de CV."""
        best_params = {}
        
        try:
            # Obtener el mejor modelo
            best_pipeline = cv_model.bestModel
            
            # Extraer parámetros según tipo de modelo
            for stage in best_pipeline.stages:
                if model_type == "RandomForest" and hasattr(stage, 'numTrees'):
                    best_params.update({
                        'numTrees': stage.getNumTrees(),
                        'maxDepth': stage.getMaxDepth(),
                        'minInstancesPerNode': stage.getMinInstancesPerNode()
                    })
                elif model_type == "GradientBoosting" and hasattr(stage, 'maxIter'):
                    best_params.update({
                        'maxIter': stage.getMaxIter(),
                        'maxDepth': stage.getMaxDepth(),
                        'stepSize': stage.getStepSize()
                    })
                elif model_type == "LogisticRegression" and hasattr(stage, 'regParam'):
                    best_params.update({
                        'regParam': stage.getRegParam(),
                        'elasticNetParam': stage.getElasticNetParam()
                    })
        
        except Exception as e:
            self.logger.warning(f"No se pudieron extraer parámetros: {e}")
        
        return best_params


class ModelPersistenceManager:
    """
    Gestor de persistencia para modelos entrenados.
    
    Maneja el guardado y carga de modelos distribuidos
    en formato MLlib optimizado.
    """
    
    def __init__(self, base_path: str):
        self.data_config = get_data_config(base_path)
        self.models_path = self.data_config.paths.models_path
        self.logger = get_logger(__name__)
        
        # Asegurar que el directorio de modelos existe
        self.models_path.mkdir(parents=True, exist_ok=True)
    
    def save_trained_model(self, model: PipelineModel, model_id: str, 
                          metadata: Dict[str, Any]) -> str:
        """
        Guarda modelo entrenado con metadatos.
        
        Args:
            model (PipelineModel): Modelo entrenado
            model_id (str): ID único del modelo
            metadata (Dict[str, Any]): Metadatos del modelo
            
        Returns:
            str: Ruta donde se guardó el modelo
        """
        try:
            # Crear directorio específico para el modelo
            model_dir = self.models_path / model_id
            model_dir.mkdir(exist_ok=True)
            
            # Guardar modelo MLlib
            model_path = model_dir / "pipeline_model"
            model.write().overwrite().save(str(model_path))
            
            # Guardar metadatos
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Modelo guardado: {model_id} en {model_dir}")
            
            return str(model_path)
            
        except Exception as e:
            self.logger.error(f"Error guardando modelo {model_id}: {e}")
            return ""
    
    def load_trained_model(self, model_id: str) -> Tuple[Optional[PipelineModel], Dict[str, Any]]:
        """
        Carga modelo entrenado con metadatos.
        
        Args:
            model_id (str): ID del modelo a cargar
            
        Returns:
            Tuple[Optional[PipelineModel], Dict[str, Any]]: Modelo y metadatos
        """
        try:
            model_dir = self.models_path / model_id
            
            if not model_dir.exists():
                self.logger.error(f"Modelo {model_id} no encontrado")
                return None, {}
            
            # Cargar modelo
            model_path = model_dir / "pipeline_model"
            model = PipelineModel.load(str(model_path))
            
            # Cargar metadatos
            metadata_path = model_dir / "metadata.json"
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    import json
                    metadata = json.load(f)
            
            self.logger.info(f"Modelo cargado: {model_id}")
            
            return model, metadata
            
        except Exception as e:
            self.logger.error(f"Error cargando modelo {model_id}: {e}")
            return None, {}


class DistributedModelTrainer:
    """
    Entrenador principal de modelos ML distribuidos.
    
    Orquesta el entrenamiento de modelos optimizados para
    responder las preguntas de investigación del TFM.
    """
    
    def __init__(self, spark: SparkSession, base_path: str = "C:/DataLake"):
        """
        Inicializa el entrenador de modelos distribuidos.
        
        Args:
            spark (SparkSession): Sesión de Spark configurada
            base_path (str): Ruta base del Data Lake
        """
        self.spark = spark
        self.base_path = base_path
        self.logger = get_logger(__name__)
        
        # Inicializar componentes especializados
        self.model_builder = OptimizedModelBuilder(spark)
        self.hyperparameter_optimizer = HyperparameterOptimizer(spark)
        self.persistence_manager = ModelPersistenceManager(base_path)
        
        self.logger.info("DistributedModelTrainer inicializado")
    
    def train_research_models(self, datasets: Dict[str, DataFrame]) -> Dict[str, List[ModelTrainingResult]]:
        """
        Entrena modelos para responder preguntas de investigación.
        
        Args:
            datasets (Dict[str, DataFrame]): Datasets preparados con targets
            
        Returns:
            Dict[str, List[ModelTrainingResult]]: Resultados de entrenamiento por dataset
        """
        self.logger.info("Iniciando entrenamiento de modelos de investigación")
        
        training_results = {}
        
        for dataset_name, df in datasets.items():
            self.logger.info(f"Entrenando modelos para dataset: {dataset_name}")
            
            # Identificar variables objetivo
            target_columns = [col for col in df.columns if col.startswith('target_')]
            
            dataset_results = []
            
            for target_col in target_columns[:2]:  # Limitar a 2 targets por dataset
                # Identificar features relevantes
                feature_columns = self._identify_relevant_features(df, target_col)
                
                if len(feature_columns) < 3:
                    self.logger.warning(f"Insuficientes features para {target_col}")
                    continue
                
                # Entrenar múltiples tipos de modelos
                model_types = ["RandomForest", "GradientBoosting", "LogisticRegression"]
                
                for model_type in model_types:
                    result = self._train_single_model(
                        df, target_col, feature_columns, model_type, dataset_name
                    )
                    
                    if result:
                        dataset_results.append(result)
            
            training_results[dataset_name] = dataset_results
        
        # Generar resumen de entrenamiento
        self._log_training_summary(training_results)
        
        return training_results
    
    def _identify_relevant_features(self, df: DataFrame, target_col: str) -> List[str]:
        """Identifica features relevantes para un target específico."""
        # Excluir columnas que no son features
        excluded_patterns = [
            'target_', 'id_', 'ID_', 'codigo_', '_codigo', 'timestamp'
        ]
        
        all_columns = df.columns
        feature_candidates = []
        
        for col in all_columns:
            # Excluir columnas no relevantes
            if any(pattern in col for pattern in excluded_patterns):
                continue
            
            # Excluir el target actual
            if col == target_col:
                continue
            
            feature_candidates.append(col)
        
        # Priorizar features más relevantes
        priority_features = []
        secondary_features = []
        
        for feature in feature_candidates:
            if any(term in feature.lower() for term in [
                'grupo_', 'es_', 'tiene_', 'indice_', 'categoria_', 'nivel_'
            ]):
                priority_features.append(feature)
            else:
                secondary_features.append(feature)
        
        # Combinar y limitar
        selected_features = priority_features[:10] + secondary_features[:5]
        
        return selected_features
    
    def _train_single_model(self, df: DataFrame, target_col: str, feature_columns: List[str],
                           model_type: str, dataset_name: str) -> Optional[ModelTrainingResult]:
        """Entrena un modelo individual."""
        start_time = datetime.now()
        model_id = f"{dataset_name}_{target_col}_{model_type}_{int(start_time.timestamp())}"
        
        try:
            # Preparar datos
            df_prepared = self._prepare_training_data(df, target_col, feature_columns)
            
            if df_prepared.count() < 1000:
                self.logger.warning(f"Dataset muy pequeño para {model_id}")
                return None
            
            # Split entrenamiento/validación
            train_df, validation_df = df_prepared.randomSplit([0.8, 0.2], seed=42)
            
            # Crear pipeline
            pipeline = self.model_builder.create_optimized_pipeline(
                feature_columns, model_type, "label"
            )
            
            # Optimizar hiperparámetros
            self.logger.info(f"Optimizando hiperparámetros para {model_id}")
            best_model, best_params, cv_score = self.hyperparameter_optimizer.optimize_model_hyperparameters(
                pipeline, train_df, model_type
            )
            
            # Evaluar en conjunto de validación
            validation_predictions = best_model.transform(validation_df)
            
            # Calcular métricas
            auc_evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
            validation_auc = auc_evaluator.evaluate(validation_predictions)
            
            accuracy_evaluator = MulticlassClassificationEvaluator(
                labelCol="label", predictionCol="prediction", metricName="accuracy"
            )
            validation_accuracy = accuracy_evaluator.evaluate(validation_predictions)
            
            # Extraer importancia de features si es posible
            feature_importance = self._extract_feature_importance(best_model, feature_columns, model_type)
            
            # Persistir modelo
            model_metadata = {
                'model_id': model_id,
                'model_type': model_type,
                'dataset_name': dataset_name,
                'target_variable': target_col,
                'features_used': feature_columns,
                'best_params': best_params,
                'cv_auc_score': cv_score,
                'validation_auc_score': validation_auc,
                'training_timestamp': start_time.isoformat()
            }
            
            model_path = self.persistence_manager.save_trained_model(
                best_model, model_id, model_metadata
            )
            
            # Calcular tiempo total
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            result = ModelTrainingResult(
                model_id=model_id,
                model_type=model_type,
                dataset_name=dataset_name,
                target_variable=target_col,
                features_used=feature_columns,
                training_records=train_df.count(),
                validation_records=validation_df.count(),
                best_params=best_params,
                cv_auc_score=cv_score,
                cv_accuracy_score=0.0,  # No calculado en CV
                validation_auc_score=validation_auc,
                validation_accuracy_score=validation_accuracy,
                training_time_seconds=training_time,
                model_path=model_path,
                feature_importance=feature_importance,
                success=True
            )
            
            self.logger.info(f"Modelo entrenado exitosamente: {model_id} (AUC: {validation_auc:.3f})")
            
            return result
            
        except Exception as e:
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            self.logger.error(f"Error entrenando modelo {model_id}: {str(e)}")
            
            return ModelTrainingResult(
                model_id=model_id,
                model_type=model_type,
                dataset_name=dataset_name,
                target_variable=target_col,
                features_used=feature_columns,
                training_records=0,
                validation_records=0,
                best_params={},
                cv_auc_score=0.0,
                cv_accuracy_score=0.0,
                validation_auc_score=0.0,
                validation_accuracy_score=0.0,
                training_time_seconds=training_time,
                model_path=None,
                feature_importance=None,
                success=False,
                error_message=str(e)
            )

    def _prepare_training_data(self, df: DataFrame, target_col: str, 
                              feature_columns: List[str]) -> DataFrame:
        """Prepara datos para entrenamiento."""
        # Seleccionar solo columnas necesarias
        selected_columns = feature_columns + [target_col]
        df_selected = df.select(*selected_columns)
        
        # Filtrar registros con target válido
        df_clean = df_selected.filter(col(target_col).isNotNull())
        
        # Renombrar target a 'label'
        df_final = df_clean.withColumn("label", col(target_col))
        
        return df_final

    def _extract_feature_importance(self, model: PipelineModel, feature_columns: List[str],
                                   model_type: str) -> Optional[Dict[str, float]]:
        """Extrae importancia de features si el modelo lo soporta."""
        try:
            # Buscar el modelo en las etapas del pipeline
            for stage in model.stages:
                if hasattr(stage, 'featureImportances') and model_type in ["RandomForest", "GradientBoosting"]:
                    importances = stage.featureImportances.toArray()
                    
                    # Mapear importancias a nombres de features
                    importance_dict = {}
                    for i, importance in enumerate(importances):
                        if i < len(feature_columns):
                            importance_dict[feature_columns[i]] = float(importance)
                    
                    return importance_dict
        
        except Exception as e:
            self.logger.warning(f"No se pudo extraer importancia de features: {e}")
        
        return None

    def _log_training_summary(self, training_results: Dict[str, List[ModelTrainingResult]]) -> None:
        """Genera resumen de entrenamiento."""
        total_models = sum(len(results) for results in training_results.values())
        successful_models = sum(len([r for r in results if r.success]) for results in training_results.values())
        
        self.logger.info("=" * 60)
        self.logger.info("RESUMEN ENTRENAMIENTO DE MODELOS")
        self.logger.info("=" * 60)
        self.logger.info(f"Total modelos entrenados: {successful_models}/{total_models}")
        
        for dataset_name, results in training_results.items():
            successful_dataset = len([r for r in results if r.success])
            avg_auc = sum(r.validation_auc_score for r in results if r.success) / successful_dataset if successful_dataset > 0 else 0
            
            self.logger.info(f"  {dataset_name}: {successful_dataset} modelos, AUC promedio: {avg_auc:.3f}")
        
        self.logger.info("=" * 60)