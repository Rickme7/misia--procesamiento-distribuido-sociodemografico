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
# ARCHIVO: core/logger.py
# =============================================================================
#
# Propósito: Sistema centralizado de logging para todo el proyecto TFM
#
# Funcionalidades principales:
# - Configuración estandarizada de logging académico
# - Separación de logs por módulos y niveles
# - Formateo consistente para análisis posterior
# - Integración con pipeline de monitoreo
#
# Dependencias:
# - logging (Python standard library)
# - pathlib (Python standard library)
# - datetime (Python standard library)
#
# Uso:
# from core.logger import get_logger
# logger = get_logger(__name__)
# logger.info("Mensaje informativo")
#
# =============================================================================

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class TFMLogger:
    """
    Sistema de logging estandarizado para el proyecto TFM.
    
    Esta clase implementa un sistema de logging consistente que facilita
    el seguimiento de la ejecución y la identificación de problemas en
    el pipeline de procesamiento distribuido.
    """
    
    _loggers = {}
    _base_path = None
    
    @classmethod
    def setup_logging_environment(cls, base_path: str = "C:/DataLake") -> None:
        """
        Configura el entorno base de logging para todo el proyecto.
        
        Args:
            base_path (str): Directorio base donde se almacenarán los logs
        """
        cls._base_path = Path(base_path)
        
        # Crear directorio de logs si no existe
        logs_dir = cls._base_path / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurar formato base
        cls._setup_root_logger()
    
    @classmethod
    def _setup_root_logger(cls) -> None:
        """
        Configura el logger raíz con formato académico estándar.
        """
        # Formato académico para logs
        academic_formatter = logging.Formatter(
            fmt='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Handler para consola con nivel INFO
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(academic_formatter)
        
        # Configurar logger raíz
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)
    
    @classmethod
    def get_logger(cls, name: str, log_file: Optional[str] = None) -> logging.Logger:
        """
        Obtiene un logger configurado para un módulo específico.
        
        Args:
            name (str): Nombre del módulo (generalmente __name__)
            log_file (str, optional): Nombre específico del archivo de log
            
        Returns:
            logging.Logger: Logger configurado para el módulo
        """
        if name not in cls._loggers:
            cls._loggers[name] = cls._create_module_logger(name, log_file)
        
        return cls._loggers[name]
    
    @classmethod
    def _create_module_logger(cls, name: str, log_file: Optional[str]) -> logging.Logger:
        """
        Crea un logger específico para un módulo con sus propios handlers.
        
        Args:
            name (str): Nombre del módulo
            log_file (str, optional): Archivo de log específico
            
        Returns:
            logging.Logger: Logger configurado
        """
        logger = logging.getLogger(name)
        
        # Si ya tiene handlers, no agregar más
        if logger.handlers:
            return logger
        
        # Formato detallado para archivos
        file_formatter = logging.Formatter(
            fmt='%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Crear archivo de log específico
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            module_name = name.split('.')[-1]  # Último componente del nombre
            log_file = f"{module_name}_{timestamp}.log"
        
        if cls._base_path:
            log_path = cls._base_path / "logs" / log_file
            
            # Handler para archivo con nivel DEBUG
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(file_formatter)
            
            logger.addHandler(file_handler)
        
        # Heredar nivel del logger raíz
        logger.setLevel(logging.INFO)
        
        return logger
    
    @classmethod
    def log_pipeline_start(cls, pipeline_name: str, logger: logging.Logger) -> None:
        """
        Registra el inicio de un pipeline con formato estándar.
        
        Args:
            pipeline_name (str): Nombre del pipeline que inicia
            logger (logging.Logger): Logger a utilizar
        """
        separator = "=" * 80
        logger.info(f"\n{separator}")
        logger.info(f"INICIANDO PIPELINE: {pipeline_name}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info(f"{separator}")
    
    @classmethod
    def log_pipeline_end(cls, pipeline_name: str, logger: logging.Logger, 
                        duration_seconds: float, success: bool = True) -> None:
        """
        Registra la finalización de un pipeline con métricas.
        
        Args:
            pipeline_name (str): Nombre del pipeline finalizado
            logger (logging.Logger): Logger a utilizar
            duration_seconds (float): Duración en segundos
            success (bool): Si el pipeline fue exitoso
        """
        separator = "=" * 80
        status = "COMPLETADO EXITOSAMENTE" if success else "FINALIZADO CON ERRORES"
        
        logger.info(f"\n{separator}")
        logger.info(f"PIPELINE {pipeline_name}: {status}")
        logger.info(f"Duración total: {duration_seconds:.2f} segundos")
        logger.info(f"Timestamp final: {datetime.now().isoformat()}")
        logger.info(f"{separator}")
    
    @classmethod
    def log_step_metrics(cls, step_name: str, metrics: dict, logger: logging.Logger) -> None:
        """
        Registra métricas de un paso específico del pipeline.
        
        Args:
            step_name (str): Nombre del paso
            metrics (dict): Diccionario con métricas del paso
            logger (logging.Logger): Logger a utilizar
        """
        logger.info(f"PASO COMPLETADO: {step_name}")
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                if isinstance(metric_value, float):
                    logger.info(f"  {metric_name}: {metric_value:.2f}")
                else:
                    logger.info(f"  {metric_name}: {metric_value:,}")
            else:
                logger.info(f"  {metric_name}: {metric_value}")


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Función de conveniencia para obtener un logger configurado.
    
    Args:
        name (str): Nombre del módulo (usar __name__)
        log_file (str, optional): Archivo de log específico
        
    Returns:
        logging.Logger: Logger configurado para el módulo
    """
    return TFMLogger.get_logger(name, log_file)


def setup_logging(base_path: str = "C:/DataLake") -> None:
    """
    Función de conveniencia para configurar el sistema de logging.
    
    Args:
        base_path (str): Directorio base para logs
    """
    TFMLogger.setup_logging_environment(base_path)


# Configuraciones predefinidas para diferentes entornos
ACADEMIC_LOG_CONFIG = {
    'level': logging.INFO,
    'format': '%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
    'datefmt': '%Y-%m-%d %H:%M:%S'
}

DEVELOPMENT_LOG_CONFIG = {
    'level': logging.DEBUG,
    'format': '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
    'datefmt': '%Y-%m-%d %H:%M:%S'
}

PRODUCTION_LOG_CONFIG = {
    'level': logging.WARNING,
    'format': '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    'datefmt': '%Y-%m-%d %H:%M:%S'
}
