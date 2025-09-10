# Ejemplos de Uso

## Ejecución del Pipeline Completo

### Pipeline Básico
```python
from src.main import run_full_pipeline

# Ejecutar pipeline completo
results = run_full_pipeline(
    environment="academic",
    base_path="C:/DataLake"
)
```

### Pipeline por Capas

#### Bronze Layer
```python
from src.etl.bronze import BronzeDataLoader

loader = BronzeDataLoader(spark, base_path)
tables = loader.load_all_tables()
```

#### Silver Layer
```python
from src.etl.silver import DataTransformer, FeatureEngineer

transformer = DataTransformer(spark, base_path)
feature_engineer = FeatureEngineer(spark)

# Transformar datos
cleaned_data = transformer.transform_tables(tables)

# Crear features
enhanced_data = feature_engineer.create_features(cleaned_data)
```

#### Gold Layer
```python
from src.etl.gold import MLPipeline

ml_pipeline = MLPipeline(spark, base_path)
models = ml_pipeline.train_models(enhanced_data)
```

## Configuración Personalizada

### Spark
```python
from src.config.spark_config import get_spark_session, SparkEnvironment

# Entorno de desarrollo
spark = get_spark_session(SparkEnvironment.DEVELOPMENT)

# Entorno académico
spark = get_spark_session(SparkEnvironment.ACADEMIC)
```

### Datos
```python
from src.config.data_config import get_data_config

config = get_data_config("/custom/path")
schema = config.get_schema(TableType.CENSO_POBLACION)
```

## Análisis Exploratorio

Ver notebooks:
- `notebooks/01_exploratory_analysis.ipynb`
- `notebooks/02_data_quality_assessment.ipynb`
- `notebooks/03_results_visualization.ipynb`

## Scripts de Utilidad

### Validar Resultados
```bash
python scripts/validate_results.py --layer silver
```

### Configurar Entorno
```bash
bash scripts/setup_environment.sh
```
