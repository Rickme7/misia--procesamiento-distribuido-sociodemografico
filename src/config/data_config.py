# =============================================================================
# PROYECTO TFM: SISTEMA DE PROCESAMIENTO DISTRIBUIDO DE DATOS CENSALES
# =============================================================================
#                    Universidad de Málaga
# Máster en Ingeniería del Software e Inteligencia Artificial
#
# Título: Sistema inteligente de procesamiento distribuido para análisis 
#         predictivo de patrones sociodemográficos censales
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
# ARCHIVO: config/data_config.py
# =============================================================================
#
# Propósito: Configuración centralizada de rutas, esquemas y metadatos de datos
#
# Funcionalidades principales:
# - Definición de estructura de directorios del Data Lake
# - Configuración de esquemas de datos censales y ENEMDU
# - Parámetros de calidad y validación de datos
# - Metadatos para el pipeline de feature engineering
#
# Dependencias:
# - pathlib (Python standard library)
# - dataclasses (Python standard library)
# - enum (Python standard library)
#
# Uso:
# from config.data_config import DataPaths, get_table_schema
# paths = DataPaths()
# schema = get_table_schema(TableType.CENSO_POBLACION)
#
# =============================================================================

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any


class DataLayer(Enum):
    """Enumeración de las capas del Data Lake."""
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


class TableType(Enum):
    """Enumeración de tipos de tablas en el sistema."""
    CENSO_POBLACION = "censo_poblacion"
    CENSO_HOGAR = "censo_hogar"
    CENSO_VIVIENDA = "censo_vivienda"
    ENEMDU_PERSONAS = "enemdu_personas"
    ENEMDU_VIVIENDA = "enemdu_vivienda"
    DICCIONARIO_PROVINCIA = "dict_provincia"
    DICCIONARIO_CANTON = "dict_canton"
    DICCIONARIO_PARROQUIA = "dict_parroquia"
    DICCIONARIO_CENSO = "dict_censo"
    DICCIONARIO_ENEMDU = "dict_enemdu"


@dataclass
class DataPaths:
    """
    Configuración centralizada de rutas del proyecto.
    
    Esta clase define todas las rutas utilizadas en el pipeline de datos,
    facilitando la gestión y modificación de la estructura de directorios.
    """
    base_path: str = "C:/DataLake"
    
    def __post_init__(self):
        """Inicializa las rutas derivadas después de la creación del objeto."""
        self._base = Path(self.base_path)
        
        # Crear directorios si no existen
        self._ensure_directories_exist()
    
    def _ensure_directories_exist(self) -> None:
        """Crea la estructura de directorios del Data Lake."""
        directories = [
            self.bronze_path,
            self.silver_path,
            self.gold_path,
            self.logs_path,
            self.metadata_path,
            self.reports_path,
            self.models_path
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def bronze_path(self) -> Path:
        """Ruta de la capa Bronze."""
        return self._base / "bronze"
    
    @property
    def silver_path(self) -> Path:
        """Ruta de la capa Silver."""
        return self._base / "silver"
    
    @property
    def gold_path(self) -> Path:
        """Ruta de la capa Gold."""
        return self._base / "gold"
    
    @property
    def logs_path(self) -> Path:
        """Ruta de los logs."""
        return self._base / "logs"
    
    @property
    def metadata_path(self) -> Path:
        """Ruta de metadatos."""
        return self._base / "metadata"
    
    @property
    def reports_path(self) -> Path:
        """Ruta de reportes generados."""
        return self._base / "reports"
    
    @property
    def models_path(self) -> Path:
        """Ruta de modelos ML."""
        return self._base / "models"
    
    @property
    def sftp_path(self) -> Path:
        """Ruta de datos de entrada SFTP."""
        return self._base / "sftp"
    
    def get_table_path(self, layer: DataLayer, table_name: str, 
                      file_format: str = "parquet") -> Path:
        """
        Obtiene la ruta completa de una tabla específica.
        
        Args:
            layer (DataLayer): Capa del Data Lake
            table_name (str): Nombre de la tabla
            file_format (str): Formato del archivo
            
        Returns:
            Path: Ruta completa de la tabla
        """
        if layer == DataLayer.BRONZE:
            base_path = self.bronze_path
        elif layer == DataLayer.SILVER:
            base_path = self.silver_path
        elif layer == DataLayer.GOLD:
            base_path = self.gold_path
        else:
            raise ValueError(f"Capa no válida: {layer}")
        
        if file_format == "delta":
            return base_path / f"{table_name}.delta"
        else:
            return base_path / f"{table_name}.{file_format}"


@dataclass
class TableSchema:
    """
    Definición de esquema para una tabla específica.
    
    Esta clase encapsula la información del esquema de una tabla,
    incluyendo columnas, tipos de datos y metadatos adicionales.
    """
    table_name: str
    primary_keys: List[str]
    required_columns: List[str]
    optional_columns: List[str] = field(default_factory=list)
    partition_columns: List[str] = field(default_factory=list)
    geographic_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    numeric_columns: List[str] = field(default_factory=list)
    date_columns: List[str] = field(default_factory=list)
    
    @property
    def all_columns(self) -> List[str]:
        """Retorna todas las columnas de la tabla."""
        return self.required_columns + self.optional_columns


@dataclass
class DataQualityRules:
    """
    Reglas de calidad de datos para validación.
    
    Esta clase define las reglas de calidad que deben cumplir
    los datos en cada etapa del pipeline.
    """
    max_null_percentage: float = 0.05  # 5% máximo de valores nulos
    min_record_count: int = 1000  # Mínimo número de registros
    max_duplicate_percentage: float = 0.01  # 1% máximo de duplicados
    required_geographic_coverage: List[str] = field(
        default_factory=lambda: ["01", "02", "03", "04", "05", "06", "07", "08", "09"]
    )  # Códigos de provincias principales de Ecuador
    
    def validate_quality_thresholds(self, metrics: Dict[str, Any]) -> Dict[str, bool]:
        """
        Valida métricas contra las reglas de calidad definidas.
        
        Args:
            metrics (Dict[str, Any]): Métricas calculadas de los datos
            
        Returns:
            Dict[str, bool]: Resultado de validaciones por regla
        """
        validations = {}
        
        # Validar porcentaje de nulos
        null_percentage = metrics.get('null_percentage', 0)
        validations['null_percentage_ok'] = null_percentage <= self.max_null_percentage
        
        # Validar número mínimo de registros
        record_count = metrics.get('record_count', 0)
        validations['record_count_ok'] = record_count >= self.min_record_count
        
        # Validar porcentaje de duplicados
        duplicate_percentage = metrics.get('duplicate_percentage', 0)
        validations['duplicate_percentage_ok'] = duplicate_percentage <= self.max_duplicate_percentage
        
        return validations


@dataclass
class FeatureEngineeringConfig:
    """
    Configuración para el proceso de feature engineering.
    
    Esta clase define los parámetros utilizados en la creación
    automática de features para el análisis predictivo.
    """
    # Configuración de variables demográficas
    age_groups: Dict[str, tuple] = field(default_factory=lambda: {
        "MENOR_15": (0, 15),
        "JOVEN_15_24": (15, 25),
        "ADULTO_25_34": (25, 35),
        "ADULTO_35_49": (35, 50),
        "ADULTO_50_64": (50, 65),
        "ADULTO_MAYOR_65": (65, 120)
    })
    
    # Rangos de población económicamente activa
    pea_age_range: tuple = (15, 64)
    
    # Configuración de índices compuestos
    vulnerability_components: List[str] = field(default_factory=lambda: [
        "es_dependiente_edad_real",
        "carece_educacion_superior",
        "esta_desempleado"
    ])
    
    # Umbrales para variables objetivo
    income_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "bajo": 400.0,
        "medio": 800.0,
        "alto": 1500.0
    })
    
    # Configuración de features geográficos
    geographic_feature_types: List[str] = field(default_factory=lambda: [
        "region_geografica",
        "region_especifica",
        "es_urbano",
        "densidad_poblacional"
    ])


class DataConfigurationManager:
    """
    Gestor centralizado de configuraciones de datos.
    
    Esta clase actúa como punto único de acceso para todas las
    configuraciones relacionadas con datos y esquemas.
    """
    
    def __init__(self, base_path: str = "C:/DataLake"):
        self.paths = DataPaths(base_path)
        self.quality_rules = DataQualityRules()
        self.feature_config = FeatureEngineeringConfig()
        self._schemas = self._initialize_schemas()
    
    def _initialize_schemas(self) -> Dict[TableType, TableSchema]:
        """Inicializa los esquemas de todas las tablas del sistema."""
        schemas = {}
        
        # Esquema para tabla de población del censo
        schemas[TableType.CENSO_POBLACION] = TableSchema(
            table_name="poblacion",
            primary_keys=["ID_VIV", "ID_HOG", "ID_PER"],
            required_columns=["I01", "I02", "P02", "P03"],
            optional_columns=["ESCOLA", "P27", "P28"],
            partition_columns=["I01"],
            geographic_columns=["I01", "CANTON"],
            categorical_columns=["P02"],
            numeric_columns=["P03", "ESCOLA"],
        )
        
        # Esquema para tabla de hogar del censo
        schemas[TableType.CENSO_HOGAR] = TableSchema(
            table_name="hogar",
            primary_keys=["ID_VIV", "ID_HOG"],
            required_columns=["H01", "H1303"],
            optional_columns=["H03", "H04", "H09"],
            partition_columns=["I01"],
            geographic_columns=["I01", "CANTON"],
            numeric_columns=["H01", "H1303"],
        )
        
        # Esquema para tabla de vivienda del censo
        schemas[TableType.CENSO_VIVIENDA] = TableSchema(
            table_name="vivienda",
            primary_keys=["ID_VIV"],
            required_columns=["V01", "V0201R"],
            optional_columns=["V05", "V03", "V07"],
            partition_columns=["I01"],
            geographic_columns=["I01", "CANTON"],
            categorical_columns=["V01", "V0201R"],
        )
        
        # Esquema para tabla de personas ENEMDU
        schemas[TableType.ENEMDU_PERSONAS] = TableSchema(
            table_name="enemdu_personas",
            primary_keys=["id_vivienda", "id_hogar", "id_persona"],
            required_columns=["p24", "p40", "p41", "ingrl"],
            optional_columns=["p71b", "p03"],
            partition_columns=["ciudad"],
            geographic_columns=["ciudad"],
            numeric_columns=["p24", "ingrl", "p71b", "p03"],
        )
        
        # Esquema para tabla de vivienda ENEMDU
        schemas[TableType.ENEMDU_VIVIENDA] = TableSchema(
            table_name="enemdu_vivienda",
            primary_keys=["id_vivienda", "id_hogar"],
            required_columns=["vi01", "vi02"],
            optional_columns=["vi05a", "vi03a"],
            partition_columns=["ciudad"],
            geographic_columns=["ciudad"],
            categorical_columns=["vi01", "vi02"],
        )
        
        return schemas
    
    def get_schema(self, table_type: TableType) -> TableSchema:
        """
        Obtiene el esquema de una tabla específica.
        
        Args:
            table_type (TableType): Tipo de tabla
            
        Returns:
            TableSchema: Esquema de la tabla
        """
        if table_type not in self._schemas:
            raise ValueError(f"Esquema no encontrado para tabla: {table_type}")
        
        return self._schemas[table_type]
    
    def get_all_schemas(self) -> Dict[TableType, TableSchema]:
        """Retorna todos los esquemas definidos."""
        return self._schemas.copy()
    
    def validate_table_structure(self, table_type: TableType, 
                                columns: List[str]) -> Dict[str, Any]:
        """
        Valida la estructura de una tabla contra su esquema definido.
        
        Args:
            table_type (TableType): Tipo de tabla a validar
            columns (List[str]): Columnas presentes en la tabla
            
        Returns:
            Dict[str, Any]: Resultado de la validación
        """
        schema = self.get_schema(table_type)
        
        # Verificar columnas requeridas
        missing_required = set(schema.required_columns) - set(columns)
        
        # Verificar columnas extra
        expected_columns = set(schema.all_columns)
        extra_columns = set(columns) - expected_columns
        
        # Verificar claves primarias
        missing_primary_keys = set(schema.primary_keys) - set(columns)
        
        return {
            'is_valid': len(missing_required) == 0 and len(missing_primary_keys) == 0,
            'missing_required_columns': list(missing_required),
            'missing_primary_keys': list(missing_primary_keys),
            'extra_columns': list(extra_columns),
            'schema_compliance_score': self._calculate_compliance_score(
                schema, columns, missing_required, missing_primary_keys
            )
        }
    
    def _calculate_compliance_score(self, schema: TableSchema, columns: List[str],
                                  missing_required: set, missing_primary_keys: set) -> float:
        """
        Calcula un score de cumplimiento del esquema (0-100).
        
        Args:
            schema (TableSchema): Esquema de referencia
            columns (List[str]): Columnas presentes
            missing_required (set): Columnas requeridas faltantes
            missing_primary_keys (set): Claves primarias faltantes
            
        Returns:
            float: Score de cumplimiento (0-100)
        """
        total_expected = len(schema.required_columns) + len(schema.primary_keys)
        total_missing = len(missing_required) + len(missing_primary_keys)
        
        if total_expected == 0:
            return 100.0
        
        compliance = ((total_expected - total_missing) / total_expected) * 100
        return max(0.0, compliance)


# Instancia global del gestor de configuración
def get_data_config(base_path: str = "C:/DataLake") -> DataConfigurationManager:
    """
    Función de conveniencia para obtener el gestor de configuración de datos.
    
    Args:
        base_path (str): Ruta base del Data Lake
        
    Returns:
        DataConfigurationManager: Gestor configurado
    """
    return DataConfigurationManager(base_path)


# Configuraciones predefinidas para diferentes entornos
DEVELOPMENT_CONFIG = {
    'base_path': "C:/DataLake_Dev",
    'max_null_percentage': 0.10,  # Más permisivo en desarrollo
    'min_record_count': 100
}

PRODUCTION_CONFIG = {
    'base_path': "C:/DataLake_Prod",
    'max_null_percentage': 0.02,  # Más estricto en producción
    'min_record_count': 10000
}

TESTING_CONFIG = {
    'base_path': "C:/DataLake_Test",
    'max_null_percentage': 0.20,  # Muy permisivo para tests
    'min_record_count': 10
}