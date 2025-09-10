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
# ARCHIVO: utils/validation_utils.py
# =============================================================================
#
# Propósito: Utilidades de validación de datos para el pipeline TFM
#
# Funcionalidades principales:
# - Validación de calidad de datos censales
# - Verificación de integridad referencial
# - Cálculo de métricas de completitud
# - Validación de esquemas de datos
#
# =============================================================================

from typing import Dict, List, Optional, Any, Tuple
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, sum as spark_sum, isnan, isnull


class DataValidator:
    """Validador centralizado de calidad de datos."""
    
    def __init__(self):
        self.logger = None
    
    def validate_data_quality(self, df: DataFrame, table_name: str) -> Dict[str, Any]:
        """Valida calidad general de datos."""
        total_records = df.count()
        total_columns = len(df.columns)
        
        if total_records == 0:
            return {
                'table_name': table_name,
                'is_valid': False,
                'total_records': 0,
                'quality_score': 0.0,
                'issues': ['No hay registros en el dataset']
            }
        
        # Análisis de nulos
        null_analysis = self._analyze_null_values(df)
        
        # Análisis de duplicados (si hay columnas ID)
        duplicate_analysis = self._analyze_duplicates(df, table_name)
        
        # Calcular score de calidad
        quality_score = self._calculate_quality_score(
            null_analysis['avg_null_percentage'],
            duplicate_analysis['duplicate_percentage']
        )
        
        return {
            'table_name': table_name,
            'is_valid': quality_score >= 70.0,
            'total_records': total_records,
            'total_columns': total_columns,
            'quality_score': quality_score,
            'null_analysis': null_analysis,
            'duplicate_analysis': duplicate_analysis
        }
    
    def _analyze_null_values(self, df: DataFrame) -> Dict[str, Any]:
        """Analiza valores nulos por columna."""
        total_records = df.count()
        null_counts = {}
        
        for column in df.columns:
            null_count = df.filter(col(column).isNull()).count()
            null_percentage = (null_count / total_records) * 100 if total_records > 0 else 0
            null_counts[column] = {
                'null_count': null_count,
                'null_percentage': round(null_percentage, 2)
            }
        
        avg_null_percentage = sum(info['null_percentage'] for info in null_counts.values()) / len(null_counts)
        
        return {
            'null_counts_by_column': null_counts,
            'avg_null_percentage': round(avg_null_percentage, 2),
            'columns_with_high_nulls': [
                col for col, info in null_counts.items() 
                if info['null_percentage'] > 20
            ]
        }
    
    def _analyze_duplicates(self, df: DataFrame, table_name: str) -> Dict[str, Any]:
        """Analiza duplicados basado en columnas ID."""
        total_records = df.count()
        
        # Identificar columnas ID
        id_columns = self._identify_id_columns(df, table_name)
        
        if not id_columns:
            return {
                'duplicate_percentage': 0.0,
                'duplicate_count': 0,
                'id_columns_used': []
            }
        
        # Contar registros únicos
        unique_records = df.select(*id_columns).distinct().count()
        duplicate_count = total_records - unique_records
        duplicate_percentage = (duplicate_count / total_records) * 100 if total_records > 0 else 0
        
        return {
            'duplicate_percentage': round(duplicate_percentage, 2),
            'duplicate_count': duplicate_count,
            'id_columns_used': id_columns
        }
    
    def _identify_id_columns(self, df: DataFrame, table_name: str) -> List[str]:
        """Identifica columnas ID según el tipo de tabla."""
        id_mappings = {
            'poblacion': ['ID_VIV', 'ID_HOG', 'ID_PER'],
            'hogar': ['ID_VIV', 'ID_HOG'],
            'vivienda': ['ID_VIV'],
            'enemdu_personas': ['id_vivienda', 'id_hogar', 'id_persona'],
            'enemdu_vivienda': ['id_vivienda', 'id_hogar']
        }
        
        # Buscar en configuración directa
        if table_name in id_mappings:
            potential_ids = id_mappings[table_name]
            return [col for col in potential_ids if col in df.columns]
        
        # Buscar por patrones en nombres de columnas
        id_patterns = ['ID_', 'id_', '_id', 'codigo_']
        potential_ids = []
        
        for column in df.columns:
            if any(pattern in column for pattern in id_patterns):
                potential_ids.append(column)
        
        return potential_ids[:3]  # Limitar a 3 columnas ID
    
    def _calculate_quality_score(self, avg_null_percentage: float, 
                                duplicate_percentage: float) -> float:
        """Calcula score compuesto de calidad."""
        null_score = max(0, 100 - (avg_null_percentage * 2))
        duplicate_score = max(0, 100 - (duplicate_percentage * 5))
        
        # Promedio ponderado
        quality_score = (null_score * 0.6) + (duplicate_score * 0.4)
        return round(quality_score, 1)


def validate_schema_compliance(df: DataFrame, expected_columns: List[str]) -> Dict[str, Any]:
    """Valida cumplimiento de esquema."""
    actual_columns = set(df.columns)
    expected_columns_set = set(expected_columns)
    
    missing_columns = expected_columns_set - actual_columns
    extra_columns = actual_columns - expected_columns_set
    
    compliance_percentage = (
        len(actual_columns & expected_columns_set) / len(expected_columns_set) * 100
        if expected_columns_set else 100
    )
    
    return {
        'is_compliant': len(missing_columns) == 0,
        'compliance_percentage': round(compliance_percentage, 1),
        'missing_columns': list(missing_columns),
        'extra_columns': list(extra_columns),
        'matching_columns': len(actual_columns & expected_columns_set)
    }


def validate_data_types(df: DataFrame, expected_types: Dict[str, str]) -> Dict[str, Any]:
    """Valida tipos de datos."""
    actual_types = dict(df.dtypes)
    type_mismatches = {}
    
    for column, expected_type in expected_types.items():
        if column in actual_types:
            actual_type = actual_types[column]
            if actual_type != expected_type:
                type_mismatches[column] = {
                    'expected': expected_type,
                    'actual': actual_type
                }
    
    return {
        'all_types_correct': len(type_mismatches) == 0,
        'type_mismatches': type_mismatches,
        'checked_columns': len(expected_types),
        'mismatched_columns': len(type_mismatches)
    }


def validate_geographic_coverage(df: DataFrame, provincia_column: str) -> Dict[str, Any]:
    """Valida cobertura geográfica."""
    if provincia_column not in df.columns:
        return {
            'coverage_adequate': False,
            'error': f'Columna {provincia_column} no encontrada'
        }
    
    # Provincias principales de Ecuador (códigos)
    expected_provinces = [
        '01', '02', '03', '04', '05', '06', '07', '08', '09',
        '10', '11', '12', '13', '14', '15', '16', '17', '18',
        '19', '20', '21', '22', '23', '24'
    ]
    
    actual_provinces = [
        str(row[provincia_column]) 
        for row in df.select(provincia_column).distinct().collect()
        if row[provincia_column] is not None
    ]
    
    coverage_count = len(set(actual_provinces) & set(expected_provinces))
    coverage_percentage = (coverage_count / len(expected_provinces)) * 100
    
    return {
        'coverage_adequate': coverage_count >= 20,  # Al menos 20 provincias
        'provinces_covered': coverage_count,
        'total_expected_provinces': len(expected_provinces),
        'coverage_percentage': round(coverage_percentage, 1),
        'missing_provinces': list(set(expected_provinces) - set(actual_provinces))
    }