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
# ARCHIVO: etl/silver/recoder.py
# =============================================================================
#
# Propósito: Sistema de recodificación automática de variables categóricas censales
#
# Funcionalidades principales:
# - Recodificación de variables usando diccionarios INEC
# - Homologación geográfica distribuida para provincias, cantones y parroquias
# - Derivación automática de features geográficos
# - Trazabilidad completa de transformaciones aplicadas
#
# Dependencias:
# - pyspark.sql
# - core.logger
# - config.data_config
#
# Uso:
# from etl.silver.recoder import VariableRecoder
# recoder = VariableRecoder(spark_session, dictionaries)
# tables_recoded = recoder.recode_all_tables(bronze_tables)
#
# =============================================================================

from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import (
    col, when, broadcast, substring, lpad, length,
    regexp_replace, trim, upper, lower, coalesce, cast
)

from core.logger import get_logger


@dataclass
class RecodificationLog:
    """
    Registro de una operación de recodificación específica.
    
    Esta clase mantiene la trazabilidad completa de las transformaciones
    aplicadas para facilitar la auditoría académica y reproducibilidad.
    """
    table_name: str
    operation_type: str
    source_column: str
    target_column: str
    mappings_applied: int
    timestamp: str
    success: bool
    error_message: Optional[str] = None


class GeographicHomologator:
    """
    Homologador especializado para códigos geográficos del Ecuador.
    
    Esta clase implementa la lógica de homologación de códigos geográficos
    utilizando los estándares del INEC para garantizar consistencia en
    la identificación de provincias, cantones y parroquias.
    """
    
    def __init__(self, spark: SparkSession, dictionaries: Dict[str, DataFrame]):
        """
        Inicializa el homologador geográfico.
        
        Args:
            spark (SparkSession): Sesión de Spark
            dictionaries (Dict[str, DataFrame]): Diccionarios geográficos
        """
        self.spark = spark
        self.dictionaries = dictionaries
        self.logger = get_logger(__name__)
        self._geo_lookup_cache = None
        
        # Configuración de columnas geográficas por tabla
        self.geographic_column_mapping = {
            'poblacion': ['I01', 'CANTON'],
            'hogar': ['I01', 'CANTON'],
            'vivienda': ['I01', 'CANTON'],
            'enemdu_personas': ['ciudad'],
            'enemdu_vivienda': ['ciudad']
        }
    
    def create_unified_geographic_lookup(self) -> Optional[DataFrame]:
        """
        Crea lookup geográfico unificado con homologación de códigos.
        
        Returns:
            Optional[DataFrame]: Lookup geográfico homologado o None si falla
        """
        if self._geo_lookup_cache is not None:
            return self._geo_lookup_cache
        
        self.logger.info("Creando lookup geográfico unificado")
        
        try:
            # Obtener diccionarios geográficos
            dict_provincia = self.dictionaries.get('dict_provincia')
            dict_canton = self.dictionaries.get('dict_canton')
            dict_parroquia = self.dictionaries.get('dict_parroquia')
            
            if not all([dict_provincia, dict_canton, dict_parroquia]):
                self.logger.error("Diccionarios geográficos incompletos")
                return None
            
            # Homologar provincias a 2 dígitos
            df_provincia_std = dict_provincia.select(
                lpad(col("CODIGO").cast("string"), 2, "0").alias("codigo_provincia_2d"),
                col("NOMBRE").alias("provinciaNombre")
            ).distinct()
            
            # Homologar cantones a 4 dígitos y extraer provincia
            df_canton_std = dict_canton.select(
                lpad(col("CODIGO").cast("string"), 4, "0").alias("codigo_canton_4d"),
                col("NOMBRE").alias("cantonNombre")
            ).withColumn("codigo_provincia_2d", 
                        substring(col("codigo_canton_4d"), 1, 2)).distinct()
            
            # Homologar parroquias a 6 dígitos y extraer jerarquía
            df_parroquia_std = dict_parroquia.select(
                lpad(col("CODIGO").cast("string"), 6, "0").alias("codigo_parroquia_6d"),
                col("NOMBRE").alias("parroquiaNombre")
            ).withColumn("codigo_provincia_2d", 
                        substring(col("codigo_parroquia_6d"), 1, 2)) \
             .withColumn("codigo_canton_4d", 
                        substring(col("codigo_parroquia_6d"), 1, 4)).distinct()
            
            # Crear lookup completo mediante JOINs
            geo_lookup = df_parroquia_std \
                .join(df_canton_std, ["codigo_provincia_2d", "codigo_canton_4d"], "left") \
                .join(df_provincia_std, ["codigo_provincia_2d"], "left") \
                .select(
                    "codigo_parroquia_6d", "parroquiaNombre",
                    "codigo_canton_4d", "cantonNombre", 
                    "codigo_provincia_2d", "provinciaNombre"
                ).distinct()
            
            # Validar unicidad y cachear resultado
            lookup_count = geo_lookup.count()
            unique_parroquias = geo_lookup.select("codigo_parroquia_6d").distinct().count()
            
            if lookup_count != unique_parroquias:
                self.logger.warning(f"Lookup tiene duplicados: {lookup_count} vs {unique_parroquias}")
            
            geo_lookup.cache()
            self._geo_lookup_cache = geo_lookup
            
            self.logger.info(f"Lookup geográfico creado: {lookup_count:,} registros")
            return geo_lookup
            
        except Exception as e:
            self.logger.error(f"Error creando lookup geográfico: {str(e)}")
            return None
    
    def homologate_geographic_columns(self, df: DataFrame, table_name: str) -> DataFrame:
        """
        Homologa columnas geográficas de una tabla específica.
        
        Args:
            df (DataFrame): DataFrame original
            table_name (str): Nombre de la tabla
            
        Returns:
            DataFrame: DataFrame con columnas geográficas homologadas
        """
        table_columns = self.geographic_column_mapping.get(table_name.lower(), [])
        available_columns = [col for col in table_columns if col in df.columns]
        
        if not available_columns:
            return df
        
        self.logger.info(f"Homologando columnas geográficas para {table_name}: {available_columns}")
        
        df_homologated = df
        
        # Homologar cada columna geográfica detectada
        for column_name in available_columns:
            if column_name == 'I01':  # Provincia
                df_homologated = df_homologated.withColumn(
                    "codigo_provincia_2d_homo",
                    lpad(col(column_name).cast("string"), 2, "0")
                )
            elif column_name == 'CANTON':  # Cantón
                df_homologated = df_homologated.withColumn(
                    "codigo_canton_4d_homo",
                    lpad(col(column_name).cast("string"), 4, "0")
                )
            elif column_name == 'ciudad':  # Parroquia (ENEMDU)
                df_homologated = df_homologated.withColumn(
                    "codigo_parroquia_6d_homo",
                    lpad(col(column_name).cast("string"), 6, "0")
                )
        
        return df_homologated
    
    def apply_geographic_joins(self, df: DataFrame, table_name: str) -> DataFrame:
        """
        Aplica JOINs geográficos basados en columnas disponibles.
        
        Args:
            df (DataFrame): DataFrame con columnas homologadas
            table_name (str): Nombre de la tabla
            
        Returns:
            DataFrame: DataFrame enriquecido con información geográfica
        """
        geo_lookup = self.create_unified_geographic_lookup()
        if geo_lookup is None:
            self.logger.warning("No se puede aplicar enrichment geográfico")
            return df
        
        initial_count = df.count()
        
        # Aplicar JOIN según el nivel geográfico más específico disponible
        if 'codigo_parroquia_6d_homo' in df.columns:
            self.logger.info(f"Aplicando JOIN por parroquia para {table_name}")
            df_enriched = df.join(
                broadcast(geo_lookup),
                col("codigo_parroquia_6d_homo") == col("codigo_parroquia_6d"),
                "left"
            )
        elif 'codigo_canton_4d_homo' in df.columns:
            self.logger.info(f"Aplicando JOIN por cantón para {table_name}")
            canton_lookup = geo_lookup.select(
                "codigo_canton_4d", "cantonNombre",
                "codigo_provincia_2d", "provinciaNombre"
            ).distinct()
            
            df_enriched = df.join(
                broadcast(canton_lookup),
                col("codigo_canton_4d_homo") == col("codigo_canton_4d"),
                "left"
            )
        elif 'codigo_provincia_2d_homo' in df.columns:
            self.logger.info(f"Aplicando JOIN por provincia para {table_name}")
            provincia_lookup = geo_lookup.select(
                "codigo_provincia_2d", "provinciaNombre"
            ).distinct()
            
            df_enriched = df.join(
                broadcast(provincia_lookup),
                col("codigo_provincia_2d_homo") == col("codigo_provincia_2d"),
                "left"
            )
        else:
            return df
        
        # Validar que no se duplicaron registros
        final_count = df_enriched.count()
        if final_count != initial_count:
            self.logger.warning(
                f"Cambio en conteo de registros durante JOIN geográfico: "
                f"{initial_count:,} -> {final_count:,}"
            )
        
        return df_enriched
    
    def derive_geographic_features(self, df: DataFrame) -> DataFrame:
        """
        Deriva features geográficos adicionales basados en códigos.
        
        Args:
            df (DataFrame): DataFrame con información geográfica
            
        Returns:
            DataFrame: DataFrame con features geográficos derivados
        """
        df_enhanced = df
        
        # Indicador urbano/rural basado en columnas disponibles
        if 'area' in df.columns:  # ENEMDU
            df_enhanced = df_enhanced.withColumn("es_urbano",
                when(col("area") == "1", 1).otherwise(0))
        elif 'AUR' in df.columns:  # CENSO
            df_enhanced = df_enhanced.withColumn("es_urbano",
                when(col("AUR") == "1", 1).otherwise(0))
        
        # Región geográfica del Ecuador
        provincia_col = self._get_provincia_column(df_enhanced)
        if provincia_col:
            df_enhanced = df_enhanced.withColumn("region_geografica",
                when(col(provincia_col).isin(["01", "02", "03", "04", "05", "06", "18"]), "SIERRA")
                .when(col(provincia_col).isin(["07", "08", "09", "12", "13", "21", "23", "24"]), "COSTA")
                .when(col(provincia_col).isin(["14", "15", "16", "19", "22"]), "AMAZONIA")
                .when(col(provincia_col) == "20", "GALAPAGOS")
                .otherwise("OTRAS")
            )
        
        # Región específica basada en nombres de provincia
        provincia_nombre_col = self._get_provincia_nombre_column(df_enhanced)
        if provincia_nombre_col:
            df_enhanced = df_enhanced.withColumn("region_especifica",
                when(col(provincia_nombre_col).rlike("(?i)(quito|pichincha)"), "CAPITAL")
                .when(col(provincia_nombre_col).rlike("(?i)(guayas|guayaquil)"), "COMERCIAL")
                .when(col(provincia_nombre_col).rlike("(?i)(azuay|cuenca)"), "PATRIMONIAL")
                .otherwise("OTRAS_PROVINCIAS")
            )
        
        return df_enhanced
    
    def _get_provincia_column(self, df: DataFrame) -> Optional[str]:
        """Identifica la columna de código de provincia disponible."""
        possible_columns = [
            'codigo_provincia_2d', 'codigo_provincia_2d_homo',
            'codigo_provincia_derivado', 'I01'
        ]
        for col_name in possible_columns:
            if col_name in df.columns:
                return col_name
        return None
    
    def _get_provincia_nombre_column(self, df: DataFrame) -> Optional[str]:
        """Identifica la columna de nombre de provincia disponible."""
        possible_columns = ['provinciaNombre', 'provincia_nombre']
        for col_name in possible_columns:
            if col_name in df.columns:
                return col_name
        return None


class VariableRecoder:
    """
    Recodificador principal de variables categóricas usando diccionarios INEC.
    
    Esta clase implementa la lógica completa de recodificación de variables
    categóricas utilizando los diccionarios oficiales del INEC, incluyendo
    tanto recodificación de variables específicas como homologación geográfica.
    """
    
    def __init__(self, spark: SparkSession, dictionaries: Dict[str, DataFrame]):
        """
        Inicializa el recodificador de variables.
        
        Args:
            spark (SparkSession): Sesión de Spark
            dictionaries (Dict[str, DataFrame]): Diccionarios de recodificación
        """
        self.spark = spark
        self.dictionaries = dictionaries
        self.logger = get_logger(__name__)
        self.recodification_log: List[RecodificationLog] = []
        
        # Inicializar componente geográfico
        self.geo_homologator = GeographicHomologator(spark, dictionaries)
        
        # Configuración de recodificación por tabla
        self.table_dictionary_mapping = {
            'poblacion': 'dict_censo',
            'hogar': 'dict_censo',
            'vivienda': 'dict_censo',
            'enemdu_personas': 'dict_enemdu',
            'enemdu_vivienda': 'dict_enemdu'
        }
    
    def recode_all_tables(self, tables: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
        """
        Aplica recodificación completa a todas las tablas.
        
        Args:
            tables (Dict[str, DataFrame]): Tablas originales
            
        Returns:
            Dict[str, DataFrame]: Tablas recodificadas
        """
        self.logger.info("Iniciando recodificación completa de tablas")
        
        recoded_tables = {}
        
        for table_name, df in tables.items():
            if table_name.startswith('dict_'):
                # Los diccionarios no se recodifican
                recoded_tables[table_name] = df
                continue
            
            self.logger.info(f"Recodificando tabla: {table_name}")
            
            # Paso 1: Recodificación de variables específicas
            df_variable_recoded = self._apply_variable_recoding(df, table_name)
            
            # Paso 2: Homologación y enrichment geográfico
            df_fully_recoded = self._apply_geographic_recoding(df_variable_recoded, table_name)
            
            recoded_tables[table_name] = df_fully_recoded
            
            self.logger.info(f"Recodificación completada para {table_name}")
        
        # Generar resumen de recodificación
        self._log_recoding_summary()
        
        return recoded_tables
    
    def _apply_variable_recoding(self, df: DataFrame, table_name: str) -> DataFrame:
        """
        Aplica recodificación de variables específicas usando diccionarios.
        
        Args:
            df (DataFrame): DataFrame original
            table_name (str): Nombre de la tabla
            
        Returns:
            DataFrame: DataFrame con variables recodificadas
        """
        dict_name = self.table_dictionary_mapping.get(table_name)
        if not dict_name or dict_name not in self.dictionaries:
            self.logger.info(f"No hay diccionario específico para {table_name}")
            return df
        
        target_dict = self.dictionaries[dict_name]
        
        try:
            # Filtrar variables que requieren recodificación
            vars_to_recode = target_dict.filter(
                (col('requiere_recodificacion').isNotNull()) &
                (upper(col('requiere_recodificacion')) == 'SI') &
                (col('variable_codigo').isNotNull()) &
                (col('codigo').isNotNull()) &
                (col('etiqueta').isNotNull())
            )
            
            if vars_to_recode.count() == 0:
                self.logger.info(f"No hay variables para recodificar en {table_name}")
                return df
            
            # Obtener variables únicas disponibles en el DataFrame
            unique_variables = vars_to_recode.select('variable_codigo').distinct().collect()
            available_variables = [
                row['variable_codigo'] for row in unique_variables
                if row['variable_codigo'] in df.columns
            ]
            
            if not available_variables:
                self.logger.info(f"Variables de recodificación no encontradas en {table_name}")
                return df
            
            df_recoded = df
            variables_recoded = 0
            
            # Procesar cada variable individualmente
            for variable_codigo in available_variables:
                df_recoded, success = self._recode_single_variable(
                    df_recoded, variable_codigo, vars_to_recode, table_name
                )
                if success:
                    variables_recoded += 1
            
            self.logger.info(f"Variables recodificadas en {table_name}: {variables_recoded}")
            return df_recoded
            
        except Exception as e:
            self.logger.error(f"Error en recodificación de variables para {table_name}: {str(e)}")
            return df
    
    def _recode_single_variable(self, df: DataFrame, variable_codigo: str,
                              vars_dict: DataFrame, table_name: str) -> Tuple[DataFrame, bool]:
        """
        Recodifica una variable específica.
        
        Args:
            df (DataFrame): DataFrame a modificar
            variable_codigo (str): Código de la variable a recodificar
            vars_dict (DataFrame): Diccionario de recodificación
            table_name (str): Nombre de la tabla
            
        Returns:
            Tuple[DataFrame, bool]: DataFrame modificado y éxito de la operación
        """
        try:
            # Obtener mapeos para esta variable
            var_mappings = vars_dict.filter(col('variable_codigo') == variable_codigo)
            mappings_data = var_mappings.select('codigo', 'etiqueta', 'campo').collect()
            
            if not mappings_data:
                return df, False
            
            # Construir expresión de recodificación
            recode_expr = col(variable_codigo)
            for mapping in mappings_data:
                codigo_orig = str(mapping['codigo'])
                etiqueta_nueva = str(mapping['etiqueta'])
                recode_expr = when(col(variable_codigo) == codigo_orig, etiqueta_nueva) \
                            .otherwise(recode_expr)
            
            # Determinar nombre de nueva columna
            new_column_name = mappings_data[0]['campo'] if mappings_data[0]['campo'] else f"{variable_codigo}_recoded"
            
            # Aplicar recodificación
            df_modified = df.withColumn(new_column_name, recode_expr)
            
            # Registrar operación
            self.recodification_log.append(RecodificationLog(
                table_name=table_name,
                operation_type='variable_recoding',
                source_column=variable_codigo,
                target_column=new_column_name,
                mappings_applied=len(mappings_data),
                timestamp=datetime.now().isoformat(),
                success=True
            ))
            
            return df_modified, True
            
        except Exception as e:
            # Registrar error
            self.recodification_log.append(RecodificationLog(
                table_name=table_name,
                operation_type='variable_recoding',
                source_column=variable_codigo,
                target_column='',
                mappings_applied=0,
                timestamp=datetime.now().isoformat(),
                success=False,
                error_message=str(e)
            ))
            
            self.logger.error(f"Error recodificando variable {variable_codigo}: {str(e)}")
            return df, False
    
    def _apply_geographic_recoding(self, df: DataFrame, table_name: str) -> DataFrame:
        """
        Aplica recodificación geográfica completa.
        
        Args:
            df (DataFrame): DataFrame con variables recodificadas
            table_name (str): Nombre de la tabla
            
        Returns:
            DataFrame: DataFrame con recodificación geográfica aplicada
        """
        try:
            # Paso 1: Homologar columnas geográficas
            df_homologated = self.geo_homologator.homologate_geographic_columns(df, table_name)
            
            # Paso 2: Aplicar JOINs geográficos
            df_joined = self.geo_homologator.apply_geographic_joins(df_homologated, table_name)
            
            # Paso 3: Derivar features geográficos
            df_enhanced = self.geo_homologator.derive_geographic_features(df_joined)
            
            # Registrar operación geográfica exitosa
            geo_fields_added = [
                col for col in df_enhanced.columns 
                if col in ['provinciaNombre', 'cantonNombre', 'parroquiaNombre', 'region_geografica', 'es_urbano']
                and col not in df.columns
            ]
            
            self.recodification_log.append(RecodificationLog(
                table_name=table_name,
                operation_type='geographic_recoding',
                source_column='geographic_columns',
                target_column=','.join(geo_fields_added),
                mappings_applied=len(geo_fields_added),
                timestamp=datetime.now().isoformat(),
                success=True
            ))
            
            return df_enhanced
            
        except Exception as e:
            self.logger.error(f"Error en recodificación geográfica para {table_name}: {str(e)}")
            
            # Registrar error geográfico
            self.recodification_log.append(RecodificationLog(
                table_name=table_name,
                operation_type='geographic_recoding',
                source_column='geographic_columns',
                target_column='',
                mappings_applied=0,
                timestamp=datetime.now().isoformat(),
                success=False,
                error_message=str(e)
            ))
            
            return df
    
    def _log_recoding_summary(self) -> None:
        """Genera resumen de operaciones de recodificación."""
        total_operations = len(self.recodification_log)
        successful_operations = len([log for log in self.recodification_log if log.success])
        
        # Agrupar por tabla
        tables_processed = set(log.table_name for log in self.recodification_log)
        
        # Agrupar por tipo de operación
        variable_recodings = len([log for log in self.recodification_log if log.operation_type == 'variable_recoding'])
        geographic_recodings = len([log for log in self.recodification_log if log.operation_type == 'geographic_recoding'])
        
        self.logger.info("=" * 60)
        self.logger.info("RESUMEN DE RECODIFICACIÓN")
        self.logger.info("=" * 60)
        self.logger.info(f"Tablas procesadas: {len(tables_processed)}")
        self.logger.info(f"Operaciones totales: {total_operations}")
        self.logger.info(f"Operaciones exitosas: {successful_operations}")
        self.logger.info(f"Recodificaciones de variables: {variable_recodings}")
        self.logger.info(f"Recodificaciones geográficas: {geographic_recodings}")
        
        # Detalle por tabla
        for table_name in tables_processed:
            table_logs = [log for log in self.recodification_log if log.table_name == table_name]
            successful_table_logs = [log for log in table_logs if log.success]
            self.logger.info(f"  {table_name}: {len(successful_table_logs)}/{len(table_logs)} operaciones exitosas")
        
        self.logger.info("=" * 60)
    
    def get_recoding_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas detalladas de recodificación.
        
        Returns:
            Dict[str, Any]: Estadísticas completas de recodificación
        """
        total_logs = len(self.recodification_log)
        successful_logs = [log for log in self.recodification_log if log.success]
        failed_logs = [log for log in self.recodification_log if not log.success]
        
        # Estadísticas por tabla
        table_stats = {}
        for table_name in set(log.table_name for log in self.recodification_log):
            table_logs = [log for log in self.recodification_log if log.table_name == table_name]
            table_successful = [log for log in table_logs if log.success]
            
            table_stats[table_name] = {
                'total_operations': len(table_logs),
                'successful_operations': len(table_successful),
                'success_rate': len(table_successful) / len(table_logs) * 100 if table_logs else 0
            }
        
        # Estadísticas por tipo de operación
        operation_stats = {}
        for operation_type in set(log.operation_type for log in self.recodification_log):
            type_logs = [log for log in self.recodification_log if log.operation_type == operation_type]
            type_successful = [log for log in type_logs if log.success]
            
            operation_stats[operation_type] = {
                'total_operations': len(type_logs),
                'successful_operations': len(type_successful),
                'success_rate': len(type_successful) / len(type_logs) * 100 if type_logs else 0
            }
        
        return {
            'overall_statistics': {
                'total_operations': total_logs,
                'successful_operations': len(successful_logs),
                'failed_operations': len(failed_logs),
                'overall_success_rate': len(successful_logs) / total_logs * 100 if total_logs else 0
            },
            'statistics_by_table': table_stats,
            'statistics_by_operation_type': operation_stats,
            'failed_operations': [
                {
                    'table_name': log.table_name,
                    'operation_type': log.operation_type,
                    'source_column': log.source_column,
                    'error_message': log.error_message
                }
                for log in failed_logs
            ]
        }