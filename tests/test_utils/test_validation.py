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
# ARCHIVO: tests/test_utils/test_validation.py
# =============================================================================
#
# Propósito: Tests para utilidades de validación
#
# =============================================================================

import pytest
from unittest.mock import Mock, patch
from pyspark.sql import DataFrame

from utils.validation_utils import (
    DataValidator, validate_schema_compliance, 
    validate_data_types, validate_geographic_coverage
)


class TestDataValidator:
    """Tests para DataValidator."""
    
    def test_validate_data_quality_empty_dataset(self, spark_session):
        """Test validación con dataset vacío."""
        validator = DataValidator()
        
        mock_df = Mock(spec=DataFrame)
        mock_df.count.return_value = 0
        
        result = validator.validate_data_quality(mock_df, "test_table")
        
        assert result['is_valid'] is False
        assert result['total_records'] == 0
        assert result['quality_score'] == 0.0
        assert 'No hay registros' in result['issues'][0]
    
    def test_validate_data_quality_good_data(self, spark_session):
        """Test validación con datos de buena calidad."""
        validator = DataValidator()
        
        mock_df = Mock(spec=DataFrame)
        mock_df.count.return_value = 1000
        mock_df.columns = ['ID_VIV', 'P02', 'P03']
        
        # Mock análisis de nulos
        validator._analyze_null_values = Mock(return_value={
            'avg_null_percentage': 2.0,
            'null_counts_by_column': {},
            'columns_with_high_nulls': []
        })
        
        # Mock análisis de duplicados
        validator._analyze_duplicates = Mock(return_value={
            'duplicate_percentage': 1.0,
            'duplicate_count': 10,
            'id_columns_used': ['ID_VIV']
        })
        
        result = validator.validate_data_quality(mock_df, "poblacion")
        
        assert result['is_valid'] is True
        assert result['total_records'] == 1000
        assert result['quality_score'] > 70.0
    
    def test_identify_id_columns_censo(self, spark_session):
        """Test identificación de columnas ID para censo."""
        validator = DataValidator()
        
        mock_df = Mock(spec=DataFrame)
        mock_df.columns = ['ID_VIV', 'ID_HOG', 'ID_PER', 'P02', 'P03']
        
        id_columns = validator._identify_id_columns(mock_df, 'poblacion')
        
        assert id_columns == ['ID_VIV', 'ID_HOG', 'ID_PER']
    
    def test_identify_id_columns_enemdu(self, spark_session):
        """Test identificación de columnas ID para ENEMDU."""
        validator = DataValidator()
        
        mock_df = Mock(spec=DataFrame)
        mock_df.columns = ['id_vivienda', 'id_hogar', 'id_persona', 'p24']
        
        id_columns = validator._identify_id_columns(mock_df, 'enemdu_personas')
        
        assert id_columns == ['id_vivienda', 'id_hogar', 'id_persona']
    
    def test_calculate_quality_score(self, spark_session):
        """Test cálculo de score de calidad."""
        validator = DataValidator()
        
        # Datos perfectos
        score = validator._calculate_quality_score(0.0, 0.0)
        assert score == 100.0
        
        # Datos con problemas
        score = validator._calculate_quality_score(10.0, 5.0)
        assert score < 100.0 and score > 0.0
        
        # Datos muy malos
        score = validator._calculate_quality_score(50.0, 20.0)
        assert score == 0.0


class TestValidationFunctions:
    """Tests para funciones de validación."""
    
    def test_validate_schema_compliance_perfect_match(self):
        """Test cumplimiento perfecto de esquema."""
        mock_df = Mock(spec=DataFrame)
        mock_df.columns = ['ID_VIV', 'P02', 'P03']
        
        expected_columns = ['ID_VIV', 'P02', 'P03']
        
        result = validate_schema_compliance(mock_df, expected_columns)
        
        assert result['is_compliant'] is True
        assert result['compliance_percentage'] == 100.0
        assert len(result['missing_columns']) == 0
        assert len(result['extra_columns']) == 0
    
    def test_validate_schema_compliance_missing_columns(self):
        """Test con columnas faltantes."""
        mock_df = Mock(spec=DataFrame)
        mock_df.columns = ['ID_VIV', 'P02']
        
        expected_columns = ['ID_VIV', 'P02', 'P03']
        
        result = validate_schema_compliance(mock_df, expected_columns)
        
        assert result['is_compliant'] is False
        assert result['compliance_percentage'] < 100.0
        assert 'P03' in result['missing_columns']
    
    def test_validate_data_types_correct(self):
        """Test validación de tipos correctos."""
        mock_df = Mock(spec=DataFrame)
        mock_df.dtypes = [('ID_VIV', 'string'), ('P03', 'int')]
        
        expected_types = {'ID_VIV': 'string', 'P03': 'int'}
        
        result = validate_data_types(mock_df, expected_types)
        
        assert result['all_types_correct'] is True
        assert len(result['type_mismatches']) == 0
    
    def test_validate_data_types_incorrect(self):
        """Test validación de tipos incorrectos."""
        mock_df = Mock(spec=DataFrame)
        mock_df.dtypes = [('ID_VIV', 'int'), ('P03', 'string')]
        
        expected_types = {'ID_VIV': 'string', 'P03': 'int'}
        
        result = validate_data_types(mock_df, expected_types)
        
        assert result['all_types_correct'] is False
        assert len(result['type_mismatches']) == 2
        assert 'ID_VIV' in result['type_mismatches']
        assert 'P03' in result['type_mismatches']
    
    def test_validate_geographic_coverage_good(self):
        """Test cobertura geográfica adecuada."""
        mock_df = Mock(spec=DataFrame)
        
        # Mock provincias de Ecuador
        mock_rows = [Mock(**{provincia_col: f'{i:02d}'}) for i in range(1, 22)]
        mock_collect = Mock(return_value=mock_rows)
        mock_df.select.return_value.distinct.return_value.collect = mock_collect
        
        provincia_col = 'I01'
        result = validate_geographic_coverage(mock_df, provincia_col)
        
        assert result['coverage_adequate'] is True
        assert result['provinces_covered'] >= 20
        assert result['coverage_percentage'] > 80.0
    
    def test_validate_geographic_coverage_poor(self):
        """Test cobertura geográfica insuficiente."""
        mock_df = Mock(spec=DataFrame)
        
        # Solo 3 provincias
        mock_rows = [Mock(**{'I01': '01'}), Mock(**{'I01': '02'}), Mock(**{'I01': '03'})]
        mock_collect = Mock(return_value=mock_rows)
        mock_df.select.return_value.distinct.return_value.collect = mock_collect
        
        result = validate_geographic_coverage(mock_df, 'I01')
        
        assert result['coverage_adequate'] is False
        assert result['provinces_covered'] < 20
        assert result['coverage_percentage'] < 20.0
    
    def test_validate_geographic_coverage_missing_column(self):
        """Test con columna geográfica faltante."""
        mock_df = Mock(spec=DataFrame)
        mock_df.columns = ['P02', 'P03']
        
        result = validate_geographic_coverage(mock_df, 'I01')
        
        assert result['coverage_adequate'] is False
        assert 'error' in result
        assert 'no encontrada' in result['error']


# Fixtures para tests de validación
@pytest.fixture
def sample_quality_dataframe(spark_session):
    """DataFrame de muestra para tests de calidad."""
    data = [
        {'ID_VIV': 1, 'ID_HOG': 1, 'ID_PER': 1, 'P02': '1', 'P03': 25, 'I01': '01'},
        {'ID_VIV': 1, 'ID_HOG': 1, 'ID_PER': 2, 'P02': '2', 'P03': 30, 'I01': '01'},
        {'ID_VIV': 2, 'ID_HOG': 1, 'ID_PER': 1, 'P02': '1', 'P03': None, 'I01': '02'},  # Valor nulo
        {'ID_VIV': 3, 'ID_HOG': 1, 'ID_PER': 1, 'P02': '2', 'P03': 45, 'I01': '03'}
    ]
    
    return spark_session.createDataFrame(data)


@pytest.fixture
def sample_poor_quality_dataframe(spark_session):
    """DataFrame con problemas de calidad para tests."""
    data = [
        {'ID_VIV': 1, 'P02': None, 'P03': None},  # Muchos nulos
        {'ID_VIV': 1, 'P02': '1', 'P03': 25},     # Duplicado de ID_VIV
        {'ID_VIV': 2, 'P02': None, 'P03': None}   # Más nulos
    ]
    
    return spark_session.createDataFrame(data)