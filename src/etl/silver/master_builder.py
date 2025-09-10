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
# ARCHIVO: etl/silver/master_builder.py
# =============================================================================
#
# Propósito: Constructor de tablas master integradas para análisis distribuido
#
# Funcionalidades principales:
# - Construcción de master tables separadas por dominio (Censo/ENEMDU)
# - JOINs optimizados para preservar integridad referencial
# - Validación automática de calidad post-integración
# - Preparación específica para ensemble learning distribuido
#
# Dependencias:
# - pyspark.sql
# - core.logger
# - config.data_config
#
# Uso:
# from etl.silver.master_builder import MasterTableBuilder
# builder = MasterTableBuilder(spark_session)
# master_tables = builder.build_separated_master_tables(enhanced_tables)
#
# =============================================================================

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, count, avg, sum as spark_sum, max as spark_max, min as spark_min

from core.logger import get_logger


@dataclass
class MasterTableMetrics:
    """
    Métricas de calidad de una master table construida.
    
    Esta clase encapsula las métricas de calidad e integridad
    de las master tables para validación académica.
    """
    table_name: str
    record_count: int
    column_count: int
    primary_tables_joined: List[str]
    join_success_rate: float
    data_quality_score: float
    null_percentage: float
    duplicate_percentage: float
    geographic_coverage: int
    feature_count: int
    target_variables: List[str]


class DataIntegrityValidator:
    """
    Validador de integridad de datos para master tables.
    
    Esta clase implementa validaciones específicas para garantizar
    la calidad de los datos integrados en las master tables.
    """
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = get_logger(__name__)
    
    def validate_master_table_integrity(self, df: DataFrame, table_name: str) -> MasterTableMetrics:
        """
        Valida la integridad completa de una master table.
        
        Args:
            df (DataFrame): Master table a validar
            table_name (str): Nombre de la master table
            
        Returns:
            MasterTableMetrics: Métricas detalladas de integridad
        """
        self.logger.info(f"Validando integridad de master table: {table_name}")
        
        # Métricas básicas
        record_count = df.count()
        column_count = len(df.columns)
        
        # Análisis de valores nulos
        null_percentage = self._calculate_null_percentage(df)
        
        # Análisis de duplicados
        duplicate_percentage = self._calculate_duplicate_percentage(df, table_name)
        
        # Cobertura geográfica
        geographic_coverage = self._calculate_geographic_coverage(df)
        
        # Conteo de features y targets
        feature_count = len([col for col in df.columns if any(
            term in col.lower() for term in ['grupo_', 'es_', 'indice_', 'categoria_', 'ratio_']
        )])
        
        target_variables = [col for col in df.columns if col.startswith('target_')]
        
        # Score de calidad compuesto
        data_quality_score = self._calculate_quality_score(
            null_percentage, duplicate_percentage, geographic_coverage, feature_count
        )
        
        metrics = MasterTableMetrics(
            table_name=table_name,
            record_count=record_count,
            column_count=column_count,
            primary_tables_joined=[],  # Se completará en el builder
            join_success_rate=100.0,  # Se calculará en el builder
            data_quality_score=data_quality_score,
            null_percentage=null_percentage,
            duplicate_percentage=duplicate_percentage,
            geographic_coverage=geographic_coverage,
            feature_count=feature_count,
            target_variables=target_variables
        )
        
        return metrics
    
    def _calculate_null_percentage(self, df: DataFrame) -> float:
        """Calcula porcentaje promedio de valores nulos."""
        if df.count() == 0:
            return 100.0
        
        total_cells = df.count() * len(df.columns)
        null_counts = []
        
        for column in df.columns:
            null_count = df.filter(col(column).isNull()).count()
            null_counts.append(null_count)
        
        total_nulls = sum(null_counts)
        return (total_nulls / total_cells) * 100 if total_cells > 0 else 0
    
    def _calculate_duplicate_percentage(self, df: DataFrame, table_name: str) -> float:
        """Calcula porcentaje de registros duplicados."""
        total_records = df.count()
        
        if total_records == 0:
            return 0.0
        
        # Identificar columnas de ID según tipo de tabla
        if 'censo' in table_name.lower() or table_name in ['poblacion', 'hogar', 'vivienda']:
            id_columns = ['ID_VIV', 'ID_HOG', 'ID_PER']
        else:
            id_columns = ['id_vivienda', 'id_hogar', 'id_persona']
        
        # Usar columnas de ID disponibles
        available_id_columns = [col for col in id_columns if col in df.columns]
        
        if not available_id_columns:
            return 0.0
        
        unique_records = df.select(*available_id_columns).distinct().count()
        duplicates = total_records - unique_records
        
        return (duplicates / total_records) * 100 if total_records > 0 else 0
    
    def _calculate_geographic_coverage(self, df: DataFrame) -> int:
        """Calcula número de provincias cubiertas."""
        provincia_columns = ['I01', 'codigo_provincia_2d', 'codigo_provincia_2d_homo']
        
        for col_name in provincia_columns:
            if col_name in df.columns:
                return df.select(col_name).distinct().count()
        
        return 0
    
    def _calculate_quality_score(self, null_pct: float, dup_pct: float, 
                                geo_coverage: int, feature_count: int) -> float:
        """Calcula score compuesto de calidad de datos."""
        # Penalizar nulos y duplicados, premiar cobertura y features
        null_score = max(0, 100 - null_pct * 2)  # Penalizar nulos fuertemente
        dup_score = max(0, 100 - dup_pct * 5)    # Penalizar duplicados muy fuertemente
        geo_score = min(100, geo_coverage * 4)    # Premiar cobertura geográfica
        feature_score = min(100, feature_count * 2)  # Premiar riqueza de features
        
        # Promedio ponderado
        weighted_score = (null_score * 0.3 + dup_score * 0.4 + geo_score * 0.2 + feature_score * 0.1)
        return round(weighted_score, 1)


class CensusMasterBuilder:
    """
    Constructor especializado para master table del Censo.
    
    Esta clase implementa la lógica específica de integración
    de las tablas del censo (población, hogar, vivienda).
    """
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = get_logger(__name__)
    
    def build_censo_master(self, tables: Dict[str, DataFrame]) -> Optional[DataFrame]:
        """
        Construye la master table integrada del censo.
        
        Args:
            tables (Dict[str, DataFrame]): Tablas del censo con features
            
        Returns:
            Optional[DataFrame]: Master table del censo o None si falla
        """
        # Verificar disponibilidad de tablas requeridas
        required_tables = ['poblacion', 'hogar', 'vivienda']
        available_tables = [table for table in required_tables if table in tables]
        
        if len(available_tables) < 3:
            self.logger.error(f"Tablas censo incompletas: {available_tables}")
            return None
        
        self.logger.info("Construyendo master table del censo")
        
        try:
            # Obtener tablas con alias para evitar ambigüedades
            df_poblacion = tables['poblacion'].alias("pob")
            df_hogar = tables['hogar'].alias("hog")
            df_vivienda = tables['vivienda'].alias("viv")
            
            # Log inicial de registros
            pob_count = df_poblacion.count()
            hog_count = df_hogar.count()
            viv_count = df_vivienda.count()
            
            self.logger.info(f"Registros iniciales - Población: {pob_count:,}, Hogar: {hog_count:,}, Vivienda: {viv_count:,}")
            
            # JOIN 1: Población + Hogar
            df_pob_hogar = self._join_poblacion_hogar(df_poblacion, df_hogar)
            
            if df_pob_hogar is None:
                return None
            
            # JOIN 2: Resultado + Vivienda
            df_censo_master = self._join_with_vivienda(df_pob_hogar, df_vivienda)
            
            if df_censo_master is None:
                return None
            
            # Validar integridad final
            final_count = df_censo_master.count()
            self.logger.info(f"Master table censo completada: {final_count:,} registros")
            
            return df_censo_master
            
        except Exception as e:
            self.logger.error(f"Error construyendo master table censo: {str(e)}")
            return None
    
    def _join_poblacion_hogar(self, df_poblacion: DataFrame, df_hogar: DataFrame) -> Optional[DataFrame]:
        """Ejecuta JOIN entre población y hogar."""
        self.logger.info("Ejecutando JOIN Población + Hogar")
        
        try:
            # JOIN usando claves compuestas
            df_joined = df_poblacion.join(
                df_hogar,
                (col("pob.ID_VIV") == col("hog.ID_VIV")) & 
                (col("pob.ID_HOG") == col("hog.ID_HOG")),
                "inner"
            )
            
            # Seleccionar columnas evitando duplicaciones
            selected_columns = self._select_poblacion_hogar_columns(df_joined)
            df_result = df_joined.select(*selected_columns)
            
            result_count = df_result.count()
            self.logger.info(f"JOIN Población + Hogar completado: {result_count:,} registros")
            
            return df_result
            
        except Exception as e:
            self.logger.error(f"Error en JOIN población-hogar: {str(e)}")
            return None
    
    def _join_with_vivienda(self, df_pob_hogar: DataFrame, df_vivienda: DataFrame) -> Optional[DataFrame]:
        """Ejecuta JOIN con vivienda."""
        self.logger.info("Ejecutando JOIN con Vivienda")
        
        try:
            df_joined = df_pob_hogar.alias("resultado").join(
                df_vivienda,
                col("resultado.idVivienda") == col("viv.ID_VIV"),
                "inner"
            )
            
            # Seleccionar todas las columnas del resultado anterior más vivienda
            resultado_columns = [f"resultado.{col}" for col in df_pob_hogar.columns]
            vivienda_columns = self._select_vivienda_columns(df_vivienda)
            
            all_columns = resultado_columns + vivienda_columns
            df_result = df_joined.select(*all_columns)
            
            result_count = df_result.count()
            self.logger.info(f"JOIN con Vivienda completado: {result_count:,} registros")
            
            return df_result
            
        except Exception as e:
            self.logger.error(f"Error en JOIN con vivienda: {str(e)}")
            return None
    
    def _select_poblacion_hogar_columns(self, df_joined: DataFrame) -> List[str]:
        """Selecciona columnas para JOIN población-hogar evitando duplicados."""
        # Columnas principales de población con alias
        poblacion_columns = [
            "pob.ID_VIV AS idVivienda",
            "pob.ID_HOG AS idHogar", 
            "pob.ID_PER AS idPersona",
            "pob.P03 AS edad",
            "pob.P02 AS sexoCodigo",
            "pob.I01 AS provinciaCodigo",
            "pob.I02 AS ubicacionCodigo",
            "pob.CANTON AS cantonCodigo"
        ]
        
        # Agregar columnas recodificadas y features de población
        poblacion_feature_columns = [
            f"pob.{col}" for col in df_joined.columns 
            if col.startswith('pob.') and any(term in col for term in [
                'sexo_descripcion', 'grupo_edad', 'es_', 'tiene_', 'nivel_', 
                'indice_', 'categoria_', 'target_', 'provinciaNombre', 'cantonNombre'
            ])
        ]
        
        # Columnas de hogar
        hogar_columns = [
            "hog.H01 AS numeroDormitorios",
            "hog.H1303 AS totalPersonasHogar"
        ]
        
        # Agregar features de hogar
        hogar_feature_columns = [
            f"hog.{col}" for col in df_joined.columns
            if col.startswith('hog.') and any(term in col for term in [
                'tamano_', 'edad_', 'ratio_', 'dependientes_'
            ])
        ]
        
        return poblacion_columns + poblacion_feature_columns + hogar_columns + hogar_feature_columns
    
    def _select_vivienda_columns(self, df_vivienda: DataFrame) -> List[str]:
        """Selecciona columnas relevantes de vivienda."""
        base_columns = [
            "viv.V01 AS tipoViviendaCodigo",
            "viv.V0201R AS materialParedesCodigo"
        ]
        
        # Agregar columnas recodificadas de vivienda
        feature_columns = [
            f"viv.{col}" for col in df_vivienda.columns
            if any(term in col.lower() for term in [
                'tipo', 'material', 'condicion', 'agua', 'energia', 'servicio'
            ]) and not col.startswith('V0')  # Evitar códigos originales
        ]
        
        return base_columns + feature_columns


class EnemduMasterBuilder:
    """
    Constructor especializado para master table de ENEMDU.
    
    Esta clase implementa la lógica específica de integración
    de las tablas de ENEMDU (personas, vivienda).
    """
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = get_logger(__name__)
    
    def build_enemdu_master(self, tables: Dict[str, DataFrame]) -> Optional[DataFrame]:
        """
        Construye la master table integrada de ENEMDU.
        
        Args:
            tables (Dict[str, DataFrame]): Tablas de ENEMDU con features
            
        Returns:
            Optional[DataFrame]: Master table de ENEMDU o None si falla
        """
        # Verificar disponibilidad de tablas
        required_tables = ['enemdu_personas', 'enemdu_vivienda']
        available_tables = [table for table in required_tables if table in tables]
        
        if len(available_tables) < 2:
            self.logger.error(f"Tablas ENEMDU incompletas: {available_tables}")
            return None
        
        self.logger.info("Construyendo master table de ENEMDU")
        
        try:
            df_personas = tables['enemdu_personas'].alias("per")
            df_vivienda = tables['enemdu_vivienda'].alias("viv")
            
            # Log inicial
            per_count = df_personas.count()
            viv_count = df_vivienda.count()
            self.logger.info(f"Registros iniciales - Personas: {per_count:,}, Vivienda: {viv_count:,}")
            
            # Ejecutar JOIN con validación
            df_enemdu_master = self._join_enemdu_tables(df_personas, df_vivienda)
            
            if df_enemdu_master is None:
                return None
            
            final_count = df_enemdu_master.count()
            self.logger.info(f"Master table ENEMDU completada: {final_count:,} registros")
            
            return df_enemdu_master
            
        except Exception as e:
            self.logger.error(f"Error construyendo master table ENEMDU: {str(e)}")
            return None
    
    def _join_enemdu_tables(self, df_personas: DataFrame, df_vivienda: DataFrame) -> Optional[DataFrame]:
        """Ejecuta JOIN entre tablas ENEMDU con validación."""
        try:
            # Validar claves antes del JOIN
            self._validate_join_keys(df_personas, df_vivienda)
            
            # JOIN con condiciones múltiples
            df_joined = df_personas.join(
                df_vivienda,
                (col("per.id_vivienda") == col("viv.id_vivienda")) &
                (col("per.id_hogar") == col("viv.id_hogar")) &
                col("per.id_vivienda").isNotNull() &
                col("per.id_hogar").isNotNull() &
                col("viv.id_vivienda").isNotNull() &
                col("viv.id_hogar").isNotNull(),
                "inner"
            )
            
            # Seleccionar columnas relevantes
            selected_columns = self._select_enemdu_columns(df_joined)
            df_result = df_joined.select(*selected_columns)
            
            return df_result
            
        except Exception as e:
            self.logger.error(f"Error en JOIN ENEMDU: {str(e)}")
            return None
    
    def _validate_join_keys(self, df_personas: DataFrame, df_vivienda: DataFrame) -> None:
        """Valida las claves de JOIN antes de ejecutar."""
        # Verificar unicidad en vivienda
        viv_total = df_vivienda.count()
        viv_unique = df_vivienda.select("id_vivienda", "id_hogar").distinct().count()
        
        if viv_total != viv_unique:
            self.logger.warning(f"Duplicados en vivienda ENEMDU: {viv_total} vs {viv_unique}")
        
        # Verificar nulos
        per_nulls = df_personas.filter(
            col("id_vivienda").isNull() | col("id_hogar").isNull()
        ).count()
        
        viv_nulls = df_vivienda.filter(
            col("id_vivienda").isNull() | col("id_hogar").isNull()
        ).count()
        
        if per_nulls > 0:
            self.logger.warning(f"Claves nulas en personas ENEMDU: {per_nulls}")
        if viv_nulls > 0:
            self.logger.warning(f"Claves nulas en vivienda ENEMDU: {viv_nulls}")
    
    def _select_enemdu_columns(self, df_joined: DataFrame) -> List[str]:
        """Selecciona columnas para master table ENEMDU."""
        # Columnas principales de personas
        personas_columns = [
            "per.id_vivienda AS idVivienda",
            "per.id_hogar AS idHogar",
            "per.id_persona AS idPersona",
            "per.p24 AS horasTrabajadasSemana",
            "per.ingrl AS ingresoLaboral",
            "per.p03 AS edad"
        ]
        
        # Features de personas
        personas_features = [
            f"per.{col}" for col in df_joined.columns
            if col.startswith('per.') and any(term in col for term in [
                'sexo_', 'grupo_', 'es_', 'tiene_', 'categoria_', 'target_',
                'provinciaNombre', 'cantonNombre', 'region_'
            ])
        ]
        
        # Columnas de vivienda
        vivienda_columns = [
            "viv.vi01 AS tipoViviendaEnemdu",
            "viv.vi02 AS materialParedesEnemdu"
        ]
        
        # Features de vivienda
        vivienda_features = [
            f"viv.{col}" for col in df_joined.columns
            if col.startswith('viv.') and any(term in col for term in [
                'tipo', 'material', 'agua', 'servicio'
            ])
        ]
        
        return personas_columns + personas_features + vivienda_columns + vivienda_features


class MasterTableBuilder:
    """
    Constructor principal de master tables para análisis distribuido.
    
    Esta clase orquesta la construcción de master tables separadas
    optimizadas para ensemble learning distribuido.
    """
    
    def __init__(self, spark: SparkSession):
        """
        Inicializa el constructor de master tables.
        
        Args:
            spark (SparkSession): Sesión de Spark configurada
        """
        self.spark = spark
        self.logger = get_logger(__name__)
        
        # Inicializar componentes especializados
        self.censo_builder = CensusMasterBuilder(spark)
        self.enemdu_builder = EnemduMasterBuilder(spark)
        self.validator = DataIntegrityValidator(spark)
        
        self.logger.info("MasterTableBuilder inicializado")
    
    def build_separated_master_tables(self, enhanced_tables: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
        """
        Construye master tables separadas para análisis distribuido.
        
        Args:
            enhanced_tables (Dict[str, DataFrame]): Tablas con features completos
            
        Returns:
            Dict[str, DataFrame]: Master tables construidas
        """
        self.logger.info("Iniciando construcción de master tables separadas")
        
        master_tables = {}
        
        # Construir master table del censo
        censo_master = self.censo_builder.build_censo_master(enhanced_tables)
        if censo_master is not None:
            master_tables['censo_master'] = censo_master
            self.logger.info("Master table censo construida exitosamente")
        
        # Construir master table de ENEMDU
        enemdu_master = self.enemdu_builder.build_enemdu_master(enhanced_tables)
        if enemdu_master is not None:
            master_tables['enemdu_master'] = enemdu_master
            self.logger.info("Master table ENEMDU construida exitosamente")
        
        # Validar todas las master tables
        validation_results = self.validate_all_master_tables(master_tables)
        
        # Log resumen final
        self._log_construction_summary(master_tables, validation_results)
        
        return master_tables
    
    def validate_all_master_tables(self, master_tables: Dict[str, DataFrame]) -> Dict[str, MasterTableMetrics]:
        """
        Valida todas las master tables construidas.
        
        Args:
            master_tables (Dict[str, DataFrame]): Master tables a validar
            
        Returns:
            Dict[str, MasterTableMetrics]: Métricas de validación por tabla
        """
        self.logger.info("Validando integridad de master tables")
        
        validation_results = {}
        
        for table_name, df in master_tables.items():
            metrics = self.validator.validate_master_table_integrity(df, table_name)
            validation_results[table_name] = metrics
            
            self.logger.info(f"Validación {table_name}: Score {metrics.data_quality_score}/100")
        
        return validation_results
    
    def _log_construction_summary(self, master_tables: Dict[str, DataFrame], 
                                 validation_results: Dict[str, MasterTableMetrics]) -> None:
        """Log resumen de construcción de master tables."""
        self.logger.info("=" * 60)
        self.logger.info("RESUMEN CONSTRUCCIÓN MASTER TABLES")
        self.logger.info("=" * 60)
        
        for table_name, df in master_tables.items():
            metrics = validation_results.get(table_name)
            
            self.logger.info(f"Master Table: {table_name.upper()}")
            self.logger.info(f"  Registros: {df.count():,}")
            self.logger.info(f"  Columnas: {len(df.columns)}")
            
            if metrics:
                self.logger.info(f"  Calidad: {metrics.data_quality_score}/100")
                self.logger.info(f"  Features: {metrics.feature_count}")
                self.logger.info(f"  Targets: {len(metrics.target_variables)}")
                self.logger.info(f"  Cobertura geográfica: {metrics.geographic_coverage} provincias")
        
        self.logger.info("=" * 60)
        self.logger.info("Master tables listas para análisis distribuido")