# Referencia de API

## Descripción

Esta documentación describe la API completa del sistema de procesamiento distribuido de datos censales.

## Módulos Principales

### ETL Pipeline

#### Bronze Layer
- `BronzeDataLoader`: Cargador de datos crudos
- `ParametricIngestion`: Sistema de ingesta parametrizada

#### Silver Layer  
- `DataTransformer`: Transformador de datos
- `FeatureEngineer`: Motor de feature engineering
- `Recoder`: Sistema de recodificación

#### Gold Layer
- `MLPipeline`: Pipeline de Machine Learning
- `ModelTrainer`: Entrenador de modelos

### Configuración

#### SparkConfig
- `get_spark_session()`: Obtener sesión configurada de Spark
- `SparkEnvironment`: Enumeración de entornos

#### DataConfig
- `get_data_config()`: Obtener configuración de datos
- `TableSchema`: Definición de esquemas de tablas

## Ejemplos de Uso

Ver notebooks en el directorio `notebooks/` para ejemplos detallados.
