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
# ARCHIVO: tests/test_bronze/test_data_loader.py
# =============================================================================
#
# Propósito: Tests unitarios para el cargador de datos Bronze
#
# =============================================================================

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from pyspark.sql import DataFrame

from etl.bronze.data_loader import BronzeDataLoader, BronzeDataValidator


class TestBronzeDataLoader:
    """Tests para BronzeDataLoader."""
    
    def test_init(self, spark_session, temp_data_path):
        """Test inicialización del loader."""
        loader = BronzeDataLoader(spark_session, temp_data_path)
        
        assert loader.spark == spark_session
        assert str(loader.bronze_path).endswith("bronze")
        assert len(loader.expected_tables) == 5
        assert len(loader.dictionary_tables) == 5
    
    @patch('os.path.exists')
    def test_load_all_bronze_tables_no_directory(self, mock_exists, spark_session, temp_data_path):
        """Test cuando no existe directorio Bronze."""
        mock_exists.return_value = False
        
        loader = BronzeDataLoader(spark_session, temp_data_path)
        
        with pytest.raises(FileNotFoundError):
            loader.load_all_bronze_tables()
    
    def test_calculate_file_size_mb(self, spark_session, temp_data_path):
        """Test cálculo de tamaño de archivo."""
        loader = BronzeDataLoader(spark_session, temp_data_path)
        
        # Test con archivo inexistente
        size = loader._calculate_file_size_mb(temp_data_path / "nonexistent.parquet")
        assert size == 0.0
    
    def test_generate_dictionary_key(self, spark_session, temp_data_path):
        """Test generación de claves de diccionario."""
        loader = BronzeDataLoader(spark_session, temp_data_path)
        
        key = loader._generate_dictionary_key("ZR_Sftp_DICCIONARIO_PROVINCIA_2022")
        assert key == "dict_provincia"
        
        key = loader._generate_dictionary_key("unknown_file")
        assert key == "dict_unknown_file"
    
    def test_validate_minimum_tables_loaded_success(self, spark_session, temp_data_path, sample_census_data):
        """Test validación exitosa de tablas mínimas."""
        loader = BronzeDataLoader(spark_session, temp_data_path)
        
        mock_tables = {
            'poblacion': sample_census_data['poblacion'],
            'hogar': sample_census_data['hogar'],
            'vivienda': Mock()
        }
        
        # No debe lanzar excepción
        loader._validate_minimum_tables_loaded(mock_tables)
    
    def test_validate_minimum_tables_loaded_failure(self, spark_session, temp_data_path):
        """Test validación fallida de tablas mínimas."""
        loader = BronzeDataLoader(spark_session, temp_data_path)
        
        mock_tables = {'poblacion': Mock()}  # Faltan hogar y vivienda
        
        with pytest.raises(ValueError, match="Faltan tablas requeridas"):
            loader._validate_minimum_tables_loaded(mock_tables)


class TestBronzeDataValidator:
    """Tests para BronzeDataValidator."""
    
    def test_validate_census_data_quality_empty_tables(self, spark_session):
        """Test validación con tablas vacías."""
        validator = BronzeDataValidator()
        empty_tables = {}
        
        result = validator.validate_census_data_quality(empty_tables)
        
        assert result['overall_quality_score'] == 0.0
        assert result['table_validations'] == {}
    
    def test_validate_population_table_basic(self, spark_session):
        """Test validación básica de tabla población."""
        validator = BronzeDataValidator()
        
        # Mock DataFrame de población
        mock_df = Mock(spec=DataFrame)
        mock_df.count.return_value = 5000
        mock_df.columns = ['ID_VIV', 'ID_HOG', 'ID_PER', 'P03']
        
        result = validator._validate_population_table(mock_df)
        
        assert result['record_count_valid'] is True
        assert result['required_columns_present'] is True
    
    def test_validate_household_table(self, spark_session):
        """Test validación de tabla hogar."""
        validator = BronzeDataValidator()
        
        mock_df = Mock(spec=DataFrame)
        mock_df.count.return_value = 2000
        mock_df.columns = ['ID_VIV', 'ID_HOG', 'H1303']
        
        result = validator._validate_household_table(mock_df)
        
        assert result['record_count_valid'] is True
        assert result['required_columns_present'] is True
    
    def test_calculate_overall_quality_score(self, spark_session):
        """Test cálculo de score general de calidad."""
        validator = BronzeDataValidator()
        
        # Mock validaciones exitosas
        validations = {
            'table1': {'check1': True, 'check2': True},
            'table2': {'check1': True, 'check2': False}
        }
        
        score = validator._calculate_overall_quality_score(validations)
        
        assert score == 75.0  # 3 de 4 checks pasaron
    
    def test_calculate_overall_quality_score_empty(self, spark_session):
        """Test score con validaciones vacías."""
        validator = BronzeDataValidator()
        
        score = validator._calculate_overall_quality_score({})
        assert score == 0.0


class TestIntegrationBronzeLoader:
    """Tests de integración para el cargador Bronze."""
    
    def test_get_table_metadata(self, spark_session, temp_data_path, sample_census_data):
        """Test extracción de metadatos de tabla."""
        loader = BronzeDataLoader(spark_session, temp_data_path)
        
        # Crear DataFrame de Spark desde pandas
        df_spark = spark_session.createDataFrame(sample_census_data['poblacion'])
        
        metadata = loader.get_table_metadata('test_table', df_spark)
        
        assert metadata['table_name'] == 'test_table'
        assert metadata['record_count'] > 0
        assert metadata['column_count'] > 0
        assert 'schema_analysis' in metadata
        assert 'null_analysis' in metadata
        assert 'load_timestamp' in metadata


# Fixtures específicas para tests Bronze
@pytest.fixture
def mock_bronze_table():
    """Mock de tabla Bronze para tests."""
    mock_df = Mock(spec=DataFrame)
    mock_df.count.return_value = 1000
    mock_df.columns = ['ID_VIV', 'ID_HOG', 'ID_PER', 'P02', 'P03']
    return mock_df


@pytest.fixture
def mock_dictionary_table():
    """Mock de tabla diccionario para tests."""
    mock_df = Mock(spec=DataFrame)
    mock_df.count.return_value = 100
    mock_df.columns = ['CODIGO', 'NOMBRE', 'DESCRIPCION']
    return mock_df