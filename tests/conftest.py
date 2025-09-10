# =============================================================================
# PROYECTO TFM: SISTEMA DE PROCESAMIENTO DISTRIBUIDO DE DATOS CENSALES
# =============================================================================
#                    Universidad de Málaga
# Máster en Ingeniería del Software e Inteligencia Artificial
#
# Título: Sistema inteligente de procesamiento distribuido para análisis 
#           predictivo de patrones sociodemográficos censales
#
# Proyecto: misia--procesamiento-distribuido-sociodemografico
# Versión: 1.0.0
# Fecha: 2025
#
# Autores: - Ramiro Ricardo Merchán Mora
#
# Director: Antonio Jesús Nebro Urbaneja
#
# =============================================================================
# ARCHIVO: tests/conftest.py
# =============================================================================
#
# Propósito: Configuración global para tests con pytest
#
# =============================================================================


import pytest
from unittest.mock import Mock
from pathlib import Path
import tempfile
import shutil

from src.config.spark_config import get_spark_session, SparkEnvironment
from src.config.data_config import get_data_config


@pytest.fixture(scope="session")
def spark_session():
    """
    Fixture que proporciona una sesión de Spark para testing.
    
    Utiliza configuración optimizada para tests con recursos mínimos.
    """
    spark = get_spark_session(SparkEnvironment.TESTING)
    yield spark
    spark.stop()


@pytest.fixture(scope="function")
def temp_data_path():
    """
    Fixture que proporciona un directorio temporal para datos de test.
    
    Se limpia automáticamente después de cada test.
    """
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="function")
def data_config(temp_data_path):
    """
    Fixture que proporciona configuración de datos para testing.
    
    Utiliza directorio temporal para evitar interferencias.
    """
    return get_data_config(temp_data_path)


@pytest.fixture(scope="function")
def sample_census_data():
    """
    Fixture que proporciona datos de muestra del censo para tests.
    
    Returns:
        dict: Diccionario con DataFrames de muestra
    """
    import pandas as pd
    
    # Datos de población de muestra
    poblacion_data = {
        'ID_VIV': [1, 1, 2, 2],
        'ID_HOG': [1, 1, 1, 1], 
        'ID_PER': [1, 2, 1, 2],
        'I01': ['01', '01', '02', '02'],
        'P02': ['1', '2', '1', '2'],
        'P03': [25, 30, 45, 40]
    }
    
    # Datos de hogar de muestra
    hogar_data = {
        'ID_VIV': [1, 2],
        'ID_HOG': [1, 1],
        'I01': ['01', '02'],
        'H01': [3, 2],
        'H1303': [4, 3]
    }
    
    return {
        'poblacion': pd.DataFrame(poblacion_data),
        'hogar': pd.DataFrame(hogar_data)
    }


@pytest.fixture(scope="function") 
def sample_enemdu_data():
    """
    Fixture que proporciona datos de muestra de ENEMDU para tests.
    
    Returns:
        dict: Diccionario con DataFrames de muestra ENEMDU
    """
    import pandas as pd
    
    # Datos de personas ENEMDU de muestra
    personas_data = {
        'id_vivienda': [1, 1, 2, 2],
        'id_hogar': [1, 1, 1, 1],
        'id_persona': [1, 2, 1, 2],
        'ciudad': ['170150', '170150', '090150', '090150'],
        'p24': [40, 35, 45, 0],
        'ingrl': [800, 600, 1200, 0]
    }
    
    return {
        'enemdu_personas': pd.DataFrame(personas_data)
    }


# Configuración global de pytest
def pytest_configure(config):
    """
    Configuración global ejecutada antes de todos los tests.
    """
    # Configurar logging para tests
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    
    # Suprimir warnings de Spark en tests
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


def pytest_collection_modifyitems(config, items):
    """
    Modifica la colección de tests para agregar marcadores automáticos.
    """
    for item in items:
        # Marcar tests que requieren Spark
        if "spark" in item.name or "pyspark" in str(item.fspath):
            item.add_marker(pytest.mark.spark)
        
        # Marcar tests de integración
        if "integration" in item.name or "test_integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Marcar tests lentos
        if "slow" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.slow)
