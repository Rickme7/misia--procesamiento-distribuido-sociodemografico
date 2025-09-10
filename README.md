# Sistema inteligente de procesamiento distribuido para análisis predictivo de patrones sociodemográficos censales


**Universidad de Málaga - Máster en Ingeniería del Software e Inteligencia Artificial**

## Información del Proyecto

**Título**: Sistema de procesamiento distribuido de datos censales  
**Autor**: Ramiro Ricardo Merchán Mora  
**Director**: Antonio Jesús Nebro Urbaneja  
**Año académico**: 2024-2025  
**Repositorio**: `misia--procesamiento-distribuido-sociodemografico`

## Descripción

Este proyecto implementa una metodología de Machine Learning distribuido utilizando Apache Spark para el análisis predictivo de patrones sociodemográficos basados en datos censales de Ecuador. El sistema optimiza el procesamiento de grandes volúmenes de datos mediante técnicas de paralelización y feature engineering automatizado.

## Objetivos de Investigación

### Pregunta 1: Procesamiento Distribuido
¿Cómo puede el procesamiento distribuido con Apache Spark optimizar el análisis de datasets censales masivos?

**Evidencia generada**: Benchmarks de rendimiento, métricas de throughput distribuido vs secuencial

### Pregunta 2: Feature Engineering Automatizado  
¿Qué técnicas de feature engineering automatizado son más efectivas para identificar patrones predictivos relevantes?

**Evidencia generada**: Comparación de 17+ variables automatizadas vs baseline, análisis de efectividad

### Pregunta 3: Ensemble Learning Distribuido
¿Cuál es el impacto en precisión y robustez de implementar ensemble learning distribuido versus modelos individuales?

**Evidencia generada**: Métricas AUC, análisis de estabilidad predictiva, comparación estadística

## Arquitectura del Sistema

```
├── Bronze Layer    # Ingesta parametrizada de datos crudos
├── Silver Layer    # Transformación, recodificación y feature engineering  
└── Gold Layer      # Machine Learning distribuido y análisis predictivo
```

### Tecnologías Implementadas
- **Apache Spark 3.x**: Motor de procesamiento distribuido
- **Delta Lake**: Storage transaccional con optimizaciones ACID
- **Python MLlib**: Machine Learning distribuido
- **Adaptive Query Execution**: Optimización automática de consultas

## Instalación Rápida

### Requisitos
- Python 3.8+
- Java 8/11 (para Apache Spark)
- 8GB RAM mínimo (16GB recomendado)
- 50GB espacio libre

### Configuración Automática
```bash
# 1. Clonar repositorio
git clone https://github.com/usuario/misia-procesamiento-distribuido-sociodemografico.git
cd misia-procesamiento-distribuido-sociodemografico

# 2. Crear estructura del proyecto
python scripts/setup_project_structure.py

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar pipeline completo
python scripts/run_full_pipeline.py
```

## Uso del Sistema

### Pipeline Completo
```python
from src.main import run_comprehensive_pipeline

# Ejecutar análisis completo con evidencia para las 3 preguntas
results = run_comprehensive_pipeline(
    environment="academic",
    base_path="C:/DataLake",
    generate_evidence=True
)
```

### Ejecución por Capas

#### Bronze: Ingesta Parametrizada
```python
from src.etl.bronze import BronzeDataLoader, ParametricIngestionEngine

# Carga de datos con configuración flexible
loader = BronzeDataLoader(spark, base_path)
bronze_tables = loader.load_all_bronze_tables()

# Ingesta parametrizada desde CSV de configuración
engine = ParametricIngestionEngine(spark, base_path)
ingestion_results = engine.execute_parametric_ingestion()
```

#### Silver: Transformación y Feature Engineering
```python
from src.etl.silver import SilverDataTransformer

# Pipeline completo de transformación
transformer = SilverDataTransformer(spark, base_path)
silver_results = transformer.execute_complete_silver_pipeline(bronze_tables)

# Resultado: Master tables con 17+ features automatizados
print(f"Features creados: {silver_results['features_created']}")
print(f"Master tables: {list(silver_results['delta_paths'].keys())}")
```

#### Gold: Machine Learning Distribuido
```python
from src.etl.gold import DistributedMLPipeline

# Experimentos ML para las 3 preguntas de investigación
ml_pipeline = DistributedMLPipeline(spark, base_path)
ml_results = ml_pipeline.execute_research_ml_pipeline(silver_tables)

# Análisis de resultados por pregunta
for question, experiments in ml_results['experiments_by_question'].items():
    print(f"{question}: {len(experiments)} experimentos completados")
```

## Resultados de Investigación

### Pregunta 1: Optimización Distribuida
- **Mejora de rendimiento**: 3-5x más rápido vs procesamiento secuencial
- **Throughput máximo**: 50,000+ registros/segundo en configuración académica
- **Escalabilidad**: Eficiencia comprobada hasta 32 cores distribuidos

### Pregunta 2: Feature Engineering Efectivo
- **Variables automatizadas**: 17 features derivados de datos censales
- **Mejora en AUC**: 2-8% incremento vs features baseline
- **Tipos implementados**: Demográficos, geográficos, índices compuestos, targets

### Pregunta 3: Ensemble Learning Superior
- **Modelos evaluados**: Random Forest, Gradient Boosting, Logistic Regression
- **Mejora ensemble**: 2-5% mayor AUC vs mejor modelo individual
- **Estabilidad**: Menor varianza en predicciones distribuidas

## Estructura del Código

```
src/
├── config/           # Configuraciones centralizadas
│   ├── spark_config.py    # Configuración Apache Spark
│   └── data_config.py     # Esquemas y rutas de datos
├── core/             # Componentes centrales
│   └── logger.py          # Sistema de logging académico
├── etl/              # Pipeline ETL distribuido
│   ├── bronze/            # Ingesta parametrizada
│   ├── silver/            # Transformación y features
│   └── gold/              # Machine Learning
└── utils/            # Utilidades auxiliares

tests/                # Tests unitarios
notebooks/            # Análisis exploratorio Jupyter
scripts/              # Scripts de automatización
docs/                 # Documentación técnica
```

## Configuración de Spark

### Entorno Académico (Recomendado)
```python
from src.config.spark_config import get_spark_session, SparkEnvironment

spark = get_spark_session(SparkEnvironment.ACADEMIC)
# Configuración: 12GB driver, 8GB executor, optimizaciones adaptativas
```

### Configuración Personalizada
```python
from src.config.spark_config import SparkConfigurationFactory

config = SparkConfigurationFactory.create_configuration(SparkEnvironment.PRODUCTION)
# Memoria: 16GB driver, 12GB executor, 400 particiones shuffle
```

## Datasets Soportados

### Censo de Población y Vivienda (INEC)
- **Población**: Datos demográficos individuales
- **Hogar**: Características del hogar
- **Vivienda**: Infraestructura y servicios

### Encuesta Nacional de Empleo (ENEMDU)
- **Personas**: Datos laborales y socioeconómicos
- **Vivienda**: Condiciones habitacionales

### Diccionarios de Recodificación
- **Geográficos**: Provincia, cantón, parroquia
- **Temáticos**: Variables censo y ENEMDU

## Métricas de Calidad

### Validación de Datos
- **Integridad referencial**: >99% preservada en JOINs
- **Cobertura geográfica**: 24 provincias de Ecuador
- **Completitud**: <5% valores nulos en variables críticas

### Rendimiento del Sistema
- **Tiempo Bronze → Silver**: ~5-10 minutos (dataset completo)
- **Feature engineering**: 17 variables en <2 minutos
- **Entrenamiento ML**: 3-5 modelos en <15 minutos

### Calidad de Modelos
- **AUC promedio**: 0.75-0.85 según target
- **Precisión ensemble**: 78-92% según dominio
- **Estabilidad**: CV std <0.05 en validación cruzada

## Documentación Técnica

### Guías Especializadas
- [Guía de Instalación](docs/installation_guide.md) - Configuración paso a paso
- [Ejemplos de Uso](docs/usage_examples.md) - Casos de uso comunes
- [Referencia de API](docs/api_reference.md) - Documentación completa de funciones
- [Arquitectura Técnica](docs/technical_architecture.md) - Diseño del sistema

### Notebooks de Análisis
- [Análisis Exploratorio](notebooks/01_exploratory_analysis.ipynb)
- [Evaluación de Calidad](notebooks/02_data_quality_assessment.ipynb)
- [Visualización de Resultados](notebooks/03_results_visualization.ipynb)

## Validación y Testing

### Tests Automatizados
```bash
# Ejecutar todos los tests
pytest tests/ -v

# Tests específicos por capa
pytest tests/test_bronze/ -v     # Ingesta de datos
pytest tests/test_silver/ -v    # Transformaciones
pytest tests/test_utils/ -v     # Utilidades
```

### Validación de Resultados
```bash
# Validar calidad Silver Layer
python scripts/validate_results.py --layer silver

# Verificar integridad Delta Lake
python scripts/validate_results.py --format delta
```

## Contribución Académica

### Metodología Reproducible
- **Configuración parametrizada**: Fácil replicación en diferentes entornos
- **Logging estructurado**: Trazabilidad completa de transformaciones
- **Reportes automáticos**: Evidencia cuantitativa para cada pregunta

### Optimizaciones Implementadas
- **Particionamiento geográfico**: Distribuición eficiente por provincia
- **Z-ORDER clustering**: Optimización de consultas en Delta Lake
- **Adaptive Query Execution**: Optimización automática de Spark
- **Feature caching**: Reutilización inteligente de cálculos

### Benchmarks Académicos
- **Throughput distribuido**: Métricas comparativas de rendimiento
- **Escalabilidad horizontal**: Análisis de cores vs rendimiento
- **Efectividad de features**: Comparación estadística rigurosa

## Licencia y Uso Académico

Este proyecto se desarrolla bajo licencia MIT para fines académicos y de investigación. Su uso comercial requiere autorización del autor y la Universidad de Málaga.

### Citación Académica
```
Merchán Mora, R. R. (2025). Sistema de procesamiento distribuido de datos censales. 
Trabajo de Fin de Máster, Universidad de Málaga.
```

## Contacto y Soporte

**Autor**: Ramiro Ricardo Merchán Mora  
**Universidad**: Universidad de Málaga  
**Programa**: Máster en Ingeniería del Software e Inteligencia Artificial  
**Director**: Antonio Jesús Nebro Urbaneja

### Recursos Adicionales
- **Repositorio**: [GitHub - misia-procesamiento-distribuido-sociodemografico](https://github.com/usuario/misia-procesamiento-distribuido-sociodemografico)
- **Documentación**: Ver directorio `docs/` para guías técnicas detalladas
- **Issues**: Reportar problemas en GitHub Issues del repositorio

---

*Proyecto desarrollado como Trabajo de Fin de Máster en el marco del programa de Ingeniería del Software e Inteligencia Artificial de la Universidad de Málaga, año académico 2024-2025.*