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
# ARCHIVO: etl/silver/feature_engineer.py
# =============================================================================
#
# Propósito: Motor de feature engineering automatizado para análisis predictivo censal
#
# Funcionalidades principales:
# - Creación automática de variables demográficas y socioeconómicas
# - Features distribuidos optimizados para Apache Spark
# - Variables objetivo para modelos de machine learning
# - Índices compuestos de vulnerabilidad y desarrollo
#
# Dependencias:
# - pyspark.sql
# - core.logger
# - config.data_config
#
# Uso:
# from etl.silver.feature_engineer import AutomatedFeatureEngineer
# engineer = AutomatedFeatureEngineer(spark_session)
# enhanced_tables = engineer.create_comprehensive_features(recoded_tables)
#
# =============================================================================

from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import (
    col, when, count, avg, sum as spark_sum, max as spark_max, min as spark_min,
    stddev, row_number, rank, dense_rank, ntile, lit, coalesce, cast,
    regexp_replace, trim, upper, lower, desc, asc
)

from core.logger import get_logger
from config.data_config import FeatureEngineeringConfig


class FeatureType(Enum):
    """Enumeración de tipos de features para categorización."""
    DEMOGRAPHIC = "demographic"
    GEOGRAPHIC = "geographic"
    SOCIOECONOMIC = "socioeconomic"
    COMPOSITE_INDEX = "composite_index"
    TARGET_VARIABLE = "target_variable"
    DISTRIBUTED_AGGREGATE = "distributed_aggregate"


@dataclass
class FeatureMetadata:
    """
    Metadatos de un feature creado.
    
    Esta clase mantiene información detallada sobre cada feature
    generado para facilitar la trazabilidad y documentación académica.
    """
    feature_name: str
    feature_type: FeatureType
    source_columns: List[str]
    description: str
    creation_method: str
    data_type: str
    validation_rules: Optional[Dict[str, Any]] = None
    creation_timestamp: str = None
    
    def __post_init__(self):
        if self.creation_timestamp is None:
            self.creation_timestamp = datetime.now().isoformat()


class DemographicFeatureCreator:
    """
    Creador especializado de features demográficos.
    
    Esta clase implementa la lógica de creación de variables demográficas
    basadas en edad, sexo, estado civil y otras características de población.
    """
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = get_logger(__name__)
        self.features_created: List[FeatureMetadata] = []
    
    def create_age_based_features(self, df: DataFrame) -> DataFrame:
        """
        Crea features basados en edad utilizando la variable P03 (edad real).
        
        Args:
            df (DataFrame): DataFrame con datos de población
            
        Returns:
            DataFrame: DataFrame con features de edad agregados
        """
        if 'P03' not in df.columns:
            self.logger.warning("Columna P03 (edad) no disponible para features demográficos")
            return df
        
        df_enhanced = df
        
        # Grupos etarios estandarizados
        df_enhanced = df_enhanced.withColumn("grupo_edad_detallado",
            when(col("P03").cast("int") < 5, "PRIMERA_INFANCIA")
            .when(col("P03").cast("int") < 12, "INFANCIA")
            .when(col("P03").cast("int") < 18, "ADOLESCENCIA")
            .when(col("P03").cast("int") < 27, "JUVENTUD")
            .when(col("P03").cast("int") < 60, "ADULTEZ")
            .otherwise("ADULTO_MAYOR")
        )
        
        # Grupos etarios para análisis económico
        df_enhanced = df_enhanced.withColumn("grupo_edad_economico",
            when(col("P03").cast("int") < 15, "DEPENDIENTE_MENOR")
            .when(col("P03").cast("int") < 65, "POBLACION_ACTIVA")
            .otherwise("DEPENDIENTE_MAYOR")
        )
        
        # Indicadores binarios de dependencia demográfica
        df_enhanced = df_enhanced.withColumn("es_dependiente_demografico",
            when((col("P03").cast("int") < 15) | (col("P03").cast("int") >= 65), 1)
            .otherwise(0)
        )
        
        df_enhanced = df_enhanced.withColumn("es_poblacion_activa",
            when((col("P03").cast("int") >= 15) & (col("P03").cast("int") < 65), 1)
            .otherwise(0)
        )
        
        # Registrar features creados
        self._register_features([
            FeatureMetadata("grupo_edad_detallado", FeatureType.DEMOGRAPHIC, ["P03"],
                          "Clasificación detallada por grupos etarios", "age_classification", "string"),
            FeatureMetadata("grupo_edad_economico", FeatureType.DEMOGRAPHIC, ["P03"],
                          "Clasificación económica por edad", "economic_age_classification", "string"),
            FeatureMetadata("es_dependiente_demografico", FeatureType.DEMOGRAPHIC, ["P03"],
                          "Indicador de dependencia demográfica", "age_threshold", "integer"),
            FeatureMetadata("es_poblacion_activa", FeatureType.DEMOGRAPHIC, ["P03"],
                          "Indicador de población económicamente activa", "age_threshold", "integer")
        ])
        
        return df_enhanced
    
    def create_gender_features(self, df: DataFrame) -> DataFrame:
        """
        Crea features basados en género utilizando la variable P02.
        
        Args:
            df (DataFrame): DataFrame con datos de población
            
        Returns:
            DataFrame: DataFrame con features de género agregados
        """
        if 'P02' not in df.columns:
            self.logger.warning("Columna P02 (sexo) no disponible para features de género")
            return df
        
        df_enhanced = df
        
        # Descripción textual del sexo
        df_enhanced = df_enhanced.withColumn("sexo_descripcion",
            when(col("P02") == "1", "HOMBRE")
            .when(col("P02") == "2", "MUJER")
            .otherwise("NO_ESPECIFICADO")
        )
        
        # Indicadores binarios
        df_enhanced = df_enhanced.withColumn("es_mujer",
            when(col("P02") == "2", 1).otherwise(0)
        )
        
        df_enhanced = df_enhanced.withColumn("es_hombre",
            when(col("P02") == "1", 1).otherwise(0)
        )
        
        # Registrar features creados
        self._register_features([
            FeatureMetadata("sexo_descripcion", FeatureType.DEMOGRAPHIC, ["P02"],
                          "Descripción textual del sexo", "categorical_mapping", "string"),
            FeatureMetadata("es_mujer", FeatureType.DEMOGRAPHIC, ["P02"],
                          "Indicador binario de género femenino", "binary_encoding", "integer"),
            FeatureMetadata("es_hombre", FeatureType.DEMOGRAPHIC, ["P02"],
                          "Indicador binario de género masculino", "binary_encoding", "integer")
        ])
        
        return df_enhanced
    
    def create_education_features(self, df: DataFrame) -> DataFrame:
        """
        Crea features educativos basados en columnas recodificadas disponibles.
        
        Args:
            df (DataFrame): DataFrame con datos de población
            
        Returns:
            DataFrame: DataFrame con features educativos agregados
        """
        # Buscar columnas de educación recodificadas
        education_columns = [col for col in df.columns 
                           if any(term in col.lower() for term in ['educacion', 'instruccion', 'escolar'])]
        
        if not education_columns:
            self.logger.warning("No se encontraron columnas de educación para feature engineering")
            return df
        
        df_enhanced = df
        
        # Usar la primera columna de educación encontrada
        education_col = education_columns[0]
        
        # Indicador de educación superior
        df_enhanced = df_enhanced.withColumn("tiene_educacion_superior",
            when(col(education_col).rlike("(?i)(superior|universitario|post-grado)"), 1)
            .otherwise(0)
        )
        
        # Indicador de educación básica completa
        df_enhanced = df_enhanced.withColumn("completo_educacion_basica",
            when(col(education_col).rlike("(?i)(secundaria|bachillerato)"), 1)
            .otherwise(0)
        )
        
        # Nivel educativo numérico (orden aproximado)
        df_enhanced = df_enhanced.withColumn("nivel_educativo_numerico",
            when(col(education_col).rlike("(?i)(analfabet|ninguno)"), 0)
            .when(col(education_col).rlike("(?i)(primaria)"), 1)
            .when(col(education_col).rlike("(?i)(secundaria|bachillerato)"), 2)
            .when(col(education_col).rlike("(?i)(superior|universitario)"), 3)
            .when(col(education_col).rlike("(?i)(post-grado|maestr|doctor)"), 4)
            .otherwise(1)
        )
        
        # Registrar features creados
        self._register_features([
            FeatureMetadata("tiene_educacion_superior", FeatureType.DEMOGRAPHIC, [education_col],
                          "Indicador de educación superior", "regex_classification", "integer"),
            FeatureMetadata("completo_educacion_basica", FeatureType.DEMOGRAPHIC, [education_col],
                          "Indicador de educación básica completa", "regex_classification", "integer"),
            FeatureMetadata("nivel_educativo_numerico", FeatureType.DEMOGRAPHIC, [education_col],
                          "Nivel educativo en escala numérica", "ordinal_encoding", "integer")
        ])
        
        return df_enhanced
    
    def _register_features(self, features: List[FeatureMetadata]) -> None:
        """Registra features creados para seguimiento."""
        self.features_created.extend(features)
        feature_names = [f.feature_name for f in features]
        self.logger.info(f"Features demográficos creados: {feature_names}")


class DistributedAggregateCreator:
    """
    Creador de features distribuidos usando agregaciones de Apache Spark.
    
    Esta clase implementa features que requieren agregaciones distribuidas
    para responder a la pregunta de investigación sobre procesamiento distribuido.
    """
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = get_logger(__name__)
        self.features_created: List[FeatureMetadata] = []
    
    def create_geographic_aggregates(self, df: DataFrame, dataset_type: str) -> DataFrame:
        """
        Crea agregaciones distribuidas por geografía.
        
        Args:
            df (DataFrame): DataFrame con información geográfica
            dataset_type (str): Tipo de dataset (censo/enemdu)
            
        Returns:
            DataFrame: DataFrame con agregaciones geográficas
        """
        # Identificar columna de provincia
        provincia_col = self._identify_provincia_column(df)
        if not provincia_col:
            self.logger.warning("No se encontró columna de provincia para agregaciones")
            return df
        
        # Window para agregaciones por provincia
        window_provincia = Window.partitionBy(provincia_col)
        
        # Población total por provincia
        df_enhanced = df.withColumn(f"poblacion_total_provincia_{dataset_type}",
                                   count("*").over(window_provincia))
        
        # Densidad relativa (normalizada por 100,000)
        df_enhanced = df_enhanced.withColumn(f"densidad_relativa_provincia_{dataset_type}",
                                           col(f"poblacion_total_provincia_{dataset_type}") / lit(100000))
        
        # Ranking de provincias por población
        window_ranking = Window.orderBy(desc(f"poblacion_total_provincia_{dataset_type}"))
        df_enhanced = df_enhanced.withColumn(f"ranking_provincial_{dataset_type}",
                                           dense_rank().over(window_ranking))
        
        # Percentil poblacional por provincia
        df_enhanced = df_enhanced.withColumn(f"percentil_poblacional_{dataset_type}",
                                           ntile(100).over(window_ranking))
        
        # Registrar features creados
        self._register_features([
            FeatureMetadata(f"poblacion_total_provincia_{dataset_type}", FeatureType.DISTRIBUTED_AGGREGATE,
                          [provincia_col], "Población total por provincia", "distributed_count", "long"),
            FeatureMetadata(f"densidad_relativa_provincia_{dataset_type}", FeatureType.DISTRIBUTED_AGGREGATE,
                          [provincia_col], "Densidad poblacional relativa", "distributed_normalization", "double"),
            FeatureMetadata(f"ranking_provincial_{dataset_type}", FeatureType.DISTRIBUTED_AGGREGATE,
                          [provincia_col], "Ranking de provincia por población", "distributed_ranking", "integer"),
            FeatureMetadata(f"percentil_poblacional_{dataset_type}", FeatureType.DISTRIBUTED_AGGREGATE,
                          [provincia_col], "Percentil poblacional provincial", "distributed_percentile", "integer")
        ])
        
        return df_enhanced
    
    def create_household_aggregates(self, df: DataFrame) -> DataFrame:
        """
        Crea agregaciones distribuidas a nivel de hogar.
        
        Args:
            df (DataFrame): DataFrame con información de hogares
            
        Returns:
            DataFrame: DataFrame con agregaciones de hogar
        """
        # Verificar columnas necesarias para hogares
        household_cols = ['ID_VIV', 'ID_HOG']
        if not all(col in df.columns for col in household_cols):
            self.logger.warning("Columnas de hogar no disponibles para agregaciones")
            return df
        
        # Window para agregaciones por hogar
        window_hogar = Window.partitionBy(['ID_VIV', 'ID_HOG'])
        
        # Tamaño del hogar
        df_enhanced = df.withColumn("tamaño_hogar_calculado",
                                   count("*").over(window_hogar))
        
        # Agregaciones de edad por hogar si está disponible
        if 'P03' in df.columns:
            df_enhanced = df_enhanced.withColumn("edad_promedio_hogar",
                                               avg(col("P03").cast("double")).over(window_hogar))
            
            df_enhanced = df_enhanced.withColumn("edad_maxima_hogar",
                                               spark_max(col("P03").cast("double")).over(window_hogar))
            
            df_enhanced = df_enhanced.withColumn("edad_minima_hogar",
                                               spark_min(col("P03").cast("double")).over(window_hogar))
            
            # Ratio de dependencia del hogar
            df_enhanced = df_enhanced.withColumn("dependientes_hogar",
                spark_sum(
                    when((col("P03").cast("int") < 15) | (col("P03").cast("int") >= 65), 1)
                    .otherwise(0)
                ).over(window_hogar))
            
            df_enhanced = df_enhanced.withColumn("ratio_dependencia_hogar",
                col("dependientes_hogar") / col("tamaño_hogar_calculado"))
        
        # Registrar features creados
        feature_list = [
            FeatureMetadata("tamaño_hogar_calculado", FeatureType.DISTRIBUTED_AGGREGATE,
                          household_cols, "Tamaño calculado del hogar", "distributed_count", "long")
        ]
        
        if 'P03' in df.columns:
            feature_list.extend([
                FeatureMetadata("edad_promedio_hogar", FeatureType.DISTRIBUTED_AGGREGATE,
                              household_cols + ["P03"], "Edad promedio del hogar", "distributed_average", "double"),
                FeatureMetadata("ratio_dependencia_hogar", FeatureType.DISTRIBUTED_AGGREGATE,
                              household_cols + ["P03"], "Ratio de dependencia del hogar", "distributed_ratio", "double")
            ])
        
        self._register_features(feature_list)
        
        return df_enhanced
    
    def _identify_provincia_column(self, df: DataFrame) -> Optional[str]:
        """Identifica la columna de provincia disponible en el DataFrame."""
        possible_columns = [
            'codigo_provincia_2d', 'codigo_provincia_2d_homo',
            'I01', 'provincia_codigo'
        ]
        
        for col_name in possible_columns:
            if col_name in df.columns:
                return col_name
        
        return None
    
    def _register_features(self, features: List[FeatureMetadata]) -> None:
        """Registra features creados para seguimiento."""
        self.features_created.extend(features)
        feature_names = [f.feature_name for f in features]
        self.logger.info(f"Features distribuidos creados: {feature_names}")


class CompositeIndexCreator:
    """
    Creador de índices compuestos para análisis socioeconómico.
    
    Esta clase implementa la creación de índices multidimensionales
    que combinan múltiples variables para análisis predictivo.
    """
    
    def __init__(self, spark: SparkSession, config: FeatureEngineeringConfig):
        self.spark = spark
        self.config = config
        self.logger = get_logger(__name__)
        self.features_created: List[FeatureMetadata] = []
    
    def create_vulnerability_index(self, df: DataFrame) -> DataFrame:
        """
        Crea índice de vulnerabilidad socioeconómica.
        
        Args:
            df (DataFrame): DataFrame con variables socioeconómicas
            
        Returns:
            DataFrame: DataFrame con índice de vulnerabilidad
        """
        # Componentes del índice de vulnerabilidad
        vulnerability_components = []
        component_columns = []
        
        # Componente de dependencia demográfica
        if 'es_dependiente_demografico' in df.columns:
            vulnerability_components.append(col('es_dependiente_demografico'))
            component_columns.append('es_dependiente_demografico')
        
        # Componente educativo
        if 'tiene_educacion_superior' in df.columns:
            vulnerability_components.append(lit(1) - col('tiene_educacion_superior'))
            component_columns.append('tiene_educacion_superior')
        elif 'nivel_educativo_numerico' in df.columns:
            # Invertir escala educativa para que menor educación = mayor vulnerabilidad
            max_education = 4  # Valor máximo esperado
            vulnerability_components.append(
                (lit(max_education) - col('nivel_educativo_numerico')) / lit(max_education)
            )
            component_columns.append('nivel_educativo_numerico')
        
        # Componente de empleo si está disponible
        employment_columns = [col for col in df.columns 
                            if any(term in col.lower() for term in ['empleo', 'ocupacion', 'trabajo'])]
        
        if employment_columns:
            employment_col = employment_columns[0]
            # Asumir que desempleo/inactividad incrementa vulnerabilidad
            vulnerability_components.append(
                when(col(employment_col).rlike("(?i)(desempleo|inactiv|no ocupa)"), 1).otherwise(0)
            )
            component_columns.append(employment_col)
        
        if not vulnerability_components:
            self.logger.warning("No se encontraron componentes para índice de vulnerabilidad")
            return df
        
        # Calcular índice compuesto
        vulnerability_expr = vulnerability_components[0]
        for component in vulnerability_components[1:]:
            vulnerability_expr = vulnerability_expr + component
        
        # Normalizar por número de componentes
        vulnerability_expr = vulnerability_expr / lit(len(vulnerability_components))
        
        df_enhanced = df.withColumn("indice_vulnerabilidad_compuesto", vulnerability_expr)
        
        # Crear categorías de vulnerabilidad
        df_enhanced = df_enhanced.withColumn("categoria_vulnerabilidad",
            when(col("indice_vulnerabilidad_compuesto") >= 0.7, "ALTA")
            .when(col("indice_vulnerabilidad_compuesto") >= 0.4, "MEDIA")
            .otherwise("BAJA")
        )
        
        # Registrar features creados
        self._register_features([
            FeatureMetadata("indice_vulnerabilidad_compuesto", FeatureType.COMPOSITE_INDEX,
                          component_columns, "Índice compuesto de vulnerabilidad socioeconómica",
                          "weighted_aggregation", "double"),
            FeatureMetadata("categoria_vulnerabilidad", FeatureType.COMPOSITE_INDEX,
                          ["indice_vulnerabilidad_compuesto"], "Categorización de vulnerabilidad",
                          "threshold_classification", "string")
        ])
        
        return df_enhanced
    
    def create_development_index(self, df: DataFrame) -> DataFrame:
        """
        Crea índice de desarrollo socioeconómico.
        
        Args:
            df (DataFrame): DataFrame con variables socioeconómicas
            
        Returns:
            DataFrame: DataFrame con índice de desarrollo
        """
        development_components = []
        component_columns = []
        
        # Componente educativo
        if 'nivel_educativo_numerico' in df.columns:
            # Normalizar educación (0-4 a 0-1)
            development_components.append(col('nivel_educativo_numerico') / lit(4))
            component_columns.append('nivel_educativo_numerico')
        
        # Componente de edad productiva
        if 'es_poblacion_activa' in df.columns:
            development_components.append(col('es_poblacion_activa'))
            component_columns.append('es_poblacion_activa')
        
        # Componente urbano/rural si está disponible
        if 'es_urbano' in df.columns:
            development_components.append(col('es_urbano'))
            component_columns.append('es_urbano')
        
        if not development_components:
            self.logger.warning("No se encontraron componentes para índice de desarrollo")
            return df
        
        # Calcular índice compuesto
        development_expr = development_components[0]
        for component in development_components[1:]:
            development_expr = development_expr + component
        
        # Normalizar por número de componentes
        development_expr = development_expr / lit(len(development_components))
        
        df_enhanced = df.withColumn("indice_desarrollo_compuesto", development_expr)
        
        # Crear categorías de desarrollo
        df_enhanced = df_enhanced.withColumn("categoria_desarrollo",
            when(col("indice_desarrollo_compuesto") >= 0.7, "ALTO")
            .when(col("indice_desarrollo_compuesto") >= 0.4, "MEDIO")
            .otherwise("BAJO")
        )
        
        # Registrar features creados
        self._register_features([
            FeatureMetadata("indice_desarrollo_compuesto", FeatureType.COMPOSITE_INDEX,
                          component_columns, "Índice compuesto de desarrollo socioeconómico",
                          "weighted_aggregation", "double"),
            FeatureMetadata("categoria_desarrollo", FeatureType.COMPOSITE_INDEX,
                          ["indice_desarrollo_compuesto"], "Categorización de desarrollo",
                          "threshold_classification", "string")
        ])
        
        return df_enhanced
    
    def _register_features(self, features: List[FeatureMetadata]) -> None:
        """Registra features creados para seguimiento."""
        self.features_created.extend(features)
        feature_names = [f.feature_name for f in features]
        self.logger.info(f"Índices compuestos creados: {feature_names}")


class TargetVariableCreator:
    """
    Creador de variables objetivo para modelos de machine learning.
    
    Esta clase implementa la creación de variables dependientes
    apropiadas para diferentes tareas de clasificación predictiva.
    """
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = get_logger(__name__)
        self.features_created: List[FeatureMetadata] = []
    
    def create_census_targets(self, df: DataFrame) -> DataFrame:
        """
        Crea variables objetivo para datos del censo.
        
        Args:
            df (DataFrame): DataFrame del censo
            
        Returns:
            DataFrame: DataFrame con variables objetivo
        """
        df_enhanced = df
        
        # Target: Vulnerabilidad alta
        if 'categoria_vulnerabilidad' in df.columns:
            df_enhanced = df_enhanced.withColumn("target_vulnerabilidad_alta",
                when(col("categoria_vulnerabilidad") == "ALTA", 1.0).otherwise(0.0)
            )
            
            self._register_target("target_vulnerabilidad_alta", ["categoria_vulnerabilidad"],
                                "Predicción de vulnerabilidad socioeconómica alta")
        
        # Target: Educación superior
        if 'tiene_educacion_superior' in df.columns:
            df_enhanced = df_enhanced.withColumn("target_educacion_superior",
                col("tiene_educacion_superior").cast("double")
            )
            
            self._register_target("target_educacion_superior", ["tiene_educacion_superior"],
                                "Predicción de acceso a educación superior")
        
        # Target: Jefe de hogar (si hay información de parentesco)
        parentesco_columns = [col for col in df.columns if 'parentesco' in col.lower()]
        if parentesco_columns:
            parentesco_col = parentesco_columns[0]
            df_enhanced = df_enhanced.withColumn("target_jefe_hogar",
                when(col(parentesco_col).rlike("(?i)(jefe|representante)"), 1.0).otherwise(0.0)
            )
            
            self._register_target("target_jefe_hogar", [parentesco_col],
                                "Predicción de condición de jefe de hogar")
        
        # Target: Área urbana
        if 'es_urbano' in df.columns:
            df_enhanced = df_enhanced.withColumn("target_area_urbana",
                col("es_urbano").cast("double")
            )
            
            self._register_target("target_area_urbana", ["es_urbano"],
                                "Predicción de residencia en área urbana")
        
        return df_enhanced
    
    def create_enemdu_targets(self, df: DataFrame) -> DataFrame:
        """
        Crea variables objetivo para datos de ENEMDU.
        
        Args:
            df (DataFrame): DataFrame de ENEMDU
            
        Returns:
            DataFrame: DataFrame con variables objetivo
        """
        df_enhanced = df
        
        # Target: Empleo formal (basado en categoría de ocupación)
        occupation_columns = [col for col in df.columns 
                            if any(term in col.lower() for term in ['ocupacion', 'categoria', 'empleo'])]
        
        if occupation_columns:
            occupation_col = occupation_columns[0]
            df_enhanced = df_enhanced.withColumn("target_empleo_formal",
                when(col(occupation_col).rlike("(?i)(empleado|asalariado)"), 1.0).otherwise(0.0)
            )
            
            self._register_target("target_empleo_formal", [occupation_col],
                                "Predicción de empleo formal vs informal")
        
        # Target: Ingresos suficientes (si hay información de ingresos)
        income_columns = [col for col in df.columns if 'ingr' in col.lower()]
        if income_columns:
            income_col = income_columns[0]
            # Usar umbral de salario básico aproximado (400 USD para Ecuador)
            df_enhanced = df_enhanced.withColumn("target_ingresos_suficientes",
                when(col(income_col).cast("double") >= 400, 1.0).otherwise(0.0)
            )
            
            self._register_target("target_ingresos_suficientes", [income_col],
                                "Predicción de ingresos por encima del umbral básico")
        
        # Target: Área urbana (similar al censo)
        if 'es_urbano' in df.columns:
            df_enhanced = df_enhanced.withColumn("target_area_urbana_enemdu",
                col("es_urbano").cast("double")
            )
            
            self._register_target("target_area_urbana_enemdu", ["es_urbano"],
                                "Predicción de residencia urbana en ENEMDU")
        
        return df_enhanced
    
    def _register_target(self, target_name: str, source_columns: List[str], description: str) -> None:
        """Registra una variable objetivo creada."""
        target_metadata = FeatureMetadata(
            feature_name=target_name,
            feature_type=FeatureType.TARGET_VARIABLE,
            source_columns=source_columns,
            description=description,
            creation_method="binary_classification_target",
            data_type="double"
        )
        
        self.features_created.append(target_metadata)
        self.logger.info(f"Variable objetivo creada: {target_name}")


class AutomatedFeatureEngineer:
    """
    Motor principal de feature engineering automatizado.
    
    Esta clase orquesta la creación de todas las categorías de features
    para análisis predictivo de datos censales, integrando los componentes
    especializados en un workflow coherente.
    """
    
    def __init__(self, spark: SparkSession):
        """
        Inicializa el motor de feature engineering.
        
        Args:
            spark (SparkSession): Sesión de Spark configurada
        """
        self.spark = spark
        self.logger = get_logger(__name__)
        self.config = FeatureEngineeringConfig()
        
        # Inicializar componentes especializados
        self.demographic_creator = DemographicFeatureCreator(spark)
        self.distributed_creator = DistributedAggregateCreator(spark)
        self.composite_creator = CompositeIndexCreator(spark, self.config)
        self.target_creator = TargetVariableCreator(spark)
        
        self.logger.info("AutomatedFeatureEngineer inicializado")
    
    def create_comprehensive_features(self, tables: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
        """
        Crea features comprehensivos para todas las tablas.
        
        Args:
            tables (Dict[str, DataFrame]): Tablas recodificadas
            
        Returns:
            Dict[str, DataFrame]: Tablas con features completos
        """
        self.logger.info("Iniciando creación comprehensiva de features")
        
        enhanced_tables = {}
       
       for table_name, df in tables.items():
           if table_name.startswith('dict_'):
               # Los diccionarios no necesitan feature engineering
               enhanced_tables[table_name] = df
               continue
           
           self.logger.info(f"Creando features para tabla: {table_name}")
           
           df_enhanced = df
           
           # Paso 1: Features demográficos básicos
           df_enhanced = self.demographic_creator.create_age_based_features(df_enhanced)
           df_enhanced = self.demographic_creator.create_gender_features(df_enhanced)
           df_enhanced = self.demographic_creator.create_education_features(df_enhanced)
           
           # Paso 2: Features distribuidos por geografía
           dataset_type = 'censo' if 'censo' in table_name or table_name in ['poblacion', 'hogar', 'vivienda'] else 'enemdu'
           df_enhanced = self.distributed_creator.create_geographic_aggregates(df_enhanced, dataset_type)
           
           # Paso 3: Agregaciones de hogar si es tabla de población
           if table_name == 'poblacion':
               df_enhanced = self.distributed_creator.create_household_aggregates(df_enhanced)
           
           # Paso 4: Índices compuestos
           df_enhanced = self.composite_creator.create_vulnerability_index(df_enhanced)
           df_enhanced = self.composite_creator.create_development_index(df_enhanced)
           
           # Paso 5: Variables objetivo según tipo de datos
           if dataset_type == 'censo':
               df_enhanced = self.target_creator.create_census_targets(df_enhanced)
           else:
               df_enhanced = self.target_creator.create_enemdu_targets(df_enhanced)
           
           enhanced_tables[table_name] = df_enhanced
           
           self.logger.info(f"Feature engineering completado para {table_name}")
       
       # Generar resumen completo
       self._generate_feature_engineering_summary()
       
       return enhanced_tables
   
   def _generate_feature_engineering_summary(self) -> None:
       """Genera resumen completo del proceso de feature engineering."""
       all_features = (
           self.demographic_creator.features_created +
           self.distributed_creator.features_created +
           self.composite_creator.features_created +
           self.target_creator.features_created
       )
       
       # Estadísticas por tipo
       feature_type_counts = {}
       for feature in all_features:
           feature_type = feature.feature_type.value
           feature_type_counts[feature_type] = feature_type_counts.get(feature_type, 0) + 1
       
       self.logger.info("=" * 60)
       self.logger.info("RESUMEN DE FEATURE ENGINEERING AUTOMATIZADO")
       self.logger.info("=" * 60)
       self.logger.info(f"Total features creados: {len(all_features)}")
       
       for feature_type, count in feature_type_counts.items():
           self.logger.info(f"  {feature_type}: {count} features")
       
       self.logger.info("=" * 60)
   
   def get_feature_catalog(self) -> List[Dict[str, Any]]:
       """
       Obtiene catálogo completo de features creados.
       
       Returns:
           List[Dict[str, Any]]: Catálogo de features con metadatos
       """
       all_features = (
           self.demographic_creator.features_created +
           self.distributed_creator.features_created +
           self.composite_creator.features_created +
           self.target_creator.features_created
       )
       
       return [
           {
               'feature_name': f.feature_name,
               'feature_type': f.feature_type.value,
               'source_columns': f.source_columns,
               'description': f.description,
               'creation_method': f.creation_method,
               'data_type': f.data_type,
               'creation_timestamp': f.creation_timestamp
           }
           for f in all_features
       ]
   
   def get_research_question_coverage(self) -> Dict[str, Any]:
       """
       Analiza cobertura de features por pregunta de investigación.
       
       Returns:
           Dict[str, Any]: Análisis de cobertura por pregunta
       """
       all_features = (
           self.demographic_creator.features_created +
           self.distributed_creator.features_created +
           self.composite_creator.features_created +
           self.target_creator.features_created
       )
       
       # Pregunta 1: Features distribuidos
       distributed_features = [f for f in all_features if f.feature_type == FeatureType.DISTRIBUTED_AGGREGATE]
       
       # Pregunta 2: Features automatizados
       automated_features = [f for f in all_features if f.feature_type in [
           FeatureType.DEMOGRAPHIC, FeatureType.COMPOSITE_INDEX
       ]]
       
       # Pregunta 3: Variables objetivo para ensemble
       target_features = [f for f in all_features if f.feature_type == FeatureType.TARGET_VARIABLE]
       
       return {
           'question_1_distributed_processing': {
               'features_count': len(distributed_features),
               'features': [f.feature_name for f in distributed_features],
               'coverage_adequate': len(distributed_features) >= 4
           },
           'question_2_automated_features': {
               'features_count': len(automated_features),
               'features': [f.feature_name for f in automated_features],
               'coverage_adequate': len(automated_features) >= 10
           },
           'question_3_ensemble_targets': {
               'features_count': len(target_features),
               'features': [f.feature_name for f in target_features],
               'coverage_adequate': len(target_features) >= 3
           }
       }