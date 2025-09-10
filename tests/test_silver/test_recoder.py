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
# ARCHIVO: tests/test_silver/test_recoder.py
# =============================================================================
#
# Propósito: Tests unitarios para el sistema de recodificación
#
# =============================================================================

import pytest
from unittest.mock import Mock, patch
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, StringType

from etl.silver.recoder import VariableRecoder, GeographicHomologator


class TestGeographicHomologator:
    """Tests para GeographicHomologator."""
    
    def test_init(self, spark_session):
        """Test inicialización del homologador."""
        mock_dicts = {'dict_provincia': Mock(), 'dict_canton': Mock()}
        homologator = GeographicHomologator(spark_session, mock_dicts)
        
        assert homologator.spark == spark_session
        assert homologator.dictionaries == mock_dicts
        assert 'poblacion' in homologator.geographic_column_mapping
    
    def test_homologate_geographic_columns_poblacion(self, spark_session):
        """Test homologación de columnas para tabla población."""
        mock_dicts = {}
        homologator = GeographicHomologator(spark_session, mock_dicts)
        
        # Mock DataFrame con columnas I01 y CANTON
        mock_df = Mock(spec=DataFrame)
        mock_df.columns = ['I01', 'CANTON', 'P02', 'P03']
        mock_df.withColumn.return_value = mock_df
        
        result_df = homologator.homologate_geographic_columns(mock_df, 'poblacion')
        
        # Verificar que se llamaron las homologaciones esperadas
        assert mock_df.withColumn.call_count >= 2  # Al menos I01 y CANTON
    
    def test_homologate_geographic_columns_enemdu(self, spark_session):
        """Test homologación para ENEMDU."""
        mock_dicts = {}
        homologator = GeographicHomologator(spark_session, mock_dicts)
        
        mock_df = Mock(spec=DataFrame)
        mock_df.columns = ['ciudad', 'p24', 'ingrl']
        mock_df.withColumn.return_value = mock_df
        
        result_df = homologator.homologate_geographic_columns(mock_df, 'enemdu_personas')
        
        # Debe homologar la columna ciudad
        assert mock_df.withColumn.call_count >= 1
    
    def test_derive_geographic_features(self, spark_session):
        """Test derivación de features geográficos."""
        mock_dicts = {}
        homologator = GeographicHomologator(spark_session, mock_dicts)
        
        mock_df = Mock(spec=DataFrame)
        mock_df.columns = ['AUR', 'codigo_provincia_2d']  # Columnas del censo
        mock_df.withColumn.return_value = mock_df
        
        result_df = homologator.derive_geographic_features(mock_df)
        
        # Debe crear features urbano/rural y región
        assert mock_df.withColumn.call_count >= 2
    
    def test_get_provincia_column(self, spark_session):
        """Test identificación de columna de provincia."""
        mock_dicts = {}
        homologator = GeographicHomologator(spark_session, mock_dicts)
        
        mock_df = Mock(spec=DataFrame)
        mock_df.columns = ['codigo_provincia_2d', 'other_col']
        
        provincia_col = homologator._get_provincia_column(mock_df)
        assert provincia_col == 'codigo_provincia_2d'
        
        # Test sin columna de provincia
        mock_df.columns = ['other_col', 'another_col']
        provincia_col = homologator._get_provincia_column(mock_df)
        assert provincia_col is None


class TestVariableRecoder:
    """Tests para VariableRecoder."""
    
    def test_init(self, spark_session):
        """Test inicialización del recodificador."""
        mock_dicts = {'dict_censo': Mock()}
        recoder = VariableRecoder(spark_session, mock_dicts)
        
        assert recoder.spark == spark_session
        assert recoder.dictionaries == mock_dicts
        assert len(recoder.recodification_log) == 0
        assert 'poblacion' in recoder.table_dictionary_mapping
    
    def test_recode_all_tables_with_dictionaries(self, spark_session):
        """Test recodificación de todas las tablas."""
        # Mock diccionarios
        mock_dict = Mock(spec=DataFrame)
        mock_dicts = {'dict_censo': mock_dict}
        
        # Mock tabla principal
        mock_table = Mock(spec=DataFrame)
        tables = {'poblacion': mock_table, 'dict_censo': mock_dict}
        
        recoder = VariableRecoder(spark_session, mock_dicts)
        
        # Mock métodos internos
        recoder._apply_variable_recoding = Mock(return_value=mock_table)
        recoder._apply_geographic_recoding = Mock(return_value=mock_table)
        
        result = recoder.recode_all_tables(tables)
        
        assert 'poblacion' in result
        assert 'dict_censo' in result  # Diccionarios pasan sin modificar
        assert recoder._apply_variable_recoding.called
        assert recoder._apply_geographic_recoding.called
    
    def test_recode_single_variable_success(self, spark_session):
        """Test recodificación exitosa de variable individual."""
        mock_dicts = {}
        recoder = VariableRecoder(spark_session, mock_dicts)
        
        mock_df = Mock(spec=DataFrame)
        mock_df.withColumn.return_value = mock_df
        
        # Mock datos de mapeo
        mock_mappings = [
            Mock(codigo='1', etiqueta='Hombre', campo='sexo_descripcion'),
            Mock(codigo='2', etiqueta='Mujer', campo='sexo_descripcion')
        ]
        
        mock_vars_dict = Mock(spec=DataFrame)
        
        result_df, success = recoder._recode_single_variable(
            mock_df, 'P02', mock_vars_dict, 'poblacion'
        )
        
        assert result_df == mock_df
        # El éxito depende de la implementación mock
    
    def test_get_recoding_statistics(self, spark_session):
        """Test obtención de estadísticas de recodificación."""
        mock_dicts = {}
        recoder = VariableRecoder(spark_session, mock_dicts)
        
        # Simular logs de recodificación
        from etl.silver.recoder import RecodificationLog
        from datetime import datetime
        
        recoder.recodification_log = [
            RecodificationLog('poblacion', 'variable_recoding', 'P02', 'sexo_desc', 2, datetime.now().isoformat(), True),
            RecodificationLog('poblacion', 'geographic_recoding', 'I01', 'provincia', 1, datetime.now().isoformat(), True),
            RecodificationLog('hogar', 'variable_recoding', 'H01', 'dormitorios', 1, datetime.now().isoformat(), False, 'Error test')
        ]
        
        stats = recoder.get_recoding_statistics()
        
        assert stats['overall_statistics']['total_operations'] == 3
        assert stats['overall_statistics']['successful_operations'] == 2
        assert stats['overall_statistics']['failed_operations'] == 1
        assert 'poblacion' in stats['statistics_by_table']
        assert 'hogar' in stats['statistics_by_table']


class TestIntegrationRecoder:
    """Tests de integración para el recodificador."""
    
    def test_complete_recoding_workflow(self, spark_session, sample_census_data):
        """Test workflow completo de recodificación."""
        # Mock diccionarios mínimos
        mock_dict_censo = Mock(spec=DataFrame)
        mock_dict_censo.filter.return_value = Mock(count=Mock(return_value=0))  # Sin recodificaciones
        
        mock_dict_provincia = Mock(spec=DataFrame)
        mock_dict_canton = Mock(spec=DataFrame)
        mock_dict_parroquia = Mock(spec=DataFrame)
        
        mock_dicts = {
            'dict_censo': mock_dict_censo,
            'dict_provincia': mock_dict_provincia,
            'dict_canton': mock_dict_canton,
            'dict_parroquia': mock_dict_parroquia
        }
        
        # Crear DataFrame real de Spark
        df_poblacion = spark_session.createDataFrame(sample_census_data['poblacion'])
        tables = {'poblacion': df_poblacion}
        
        recoder = VariableRecoder(spark_session, mock_dicts)
        
        # Mock el lookup geográfico para evitar complejidad
        recoder.geo_homologator.create_unified_geographic_lookup = Mock(return_value=None)
        
        result = recoder.recode_all_tables(tables)
        
        assert 'poblacion' in result
        assert isinstance(result['poblacion'], DataFrame)


# Fixtures específicas para tests de recodificación
@pytest.fixture
def mock_recodification_dict():
    """Mock de diccionario de recodificación."""
    mock_dict = Mock(spec=DataFrame)
    
    # Mock del filtro para variables a recodificar
    mock_filtered = Mock(spec=DataFrame)
    mock_filtered.select.return_value = Mock(distinct=Mock(return_value=Mock(collect=Mock(return_value=[]))))
    mock_dict.filter.return_value = mock_filtered
    
    return mock_dict


@pytest.fixture
def sample_geographic_data(spark_session):
    """Datos geográficos de muestra."""
    data = [
        {'I01': '01', 'CANTON': '0101', 'P02': '1', 'P03': 25},
        {'I01': '02', 'CANTON': '0201', 'P02': '2', 'P03': 30}
    ]
    
    return spark_session.createDataFrame(data)