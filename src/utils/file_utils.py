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
# ARCHIVO: utils/file_utils.py
# =============================================================================
#
# Propósito: Utilidades para manejo de archivos y directorios del proyecto
#
# =============================================================================

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any


def ensure_directory_exists(directory_path: str) -> bool:
    """Crea directorio si no existe."""
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False


def get_file_size_mb(file_path: str) -> float:
    """Obtiene tamaño de archivo en MB."""
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except:
        return 0.0


def get_directory_size_mb(directory_path: str) -> float:
    """Calcula tamaño total de directorio en MB."""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
    except:
        pass
    return total_size / (1024 * 1024)


def list_files_by_extension(directory_path: str, extension: str) -> List[str]:
    """Lista archivos por extensión en directorio."""
    try:
        directory = Path(directory_path)
        if not directory.exists():
            return []
        
        pattern = f"*.{extension.lstrip('.')}"
        return [str(f) for f in directory.glob(pattern)]
    except:
        return []


def save_json_report(data: Dict[str, Any], file_path: str) -> bool:
    """Guarda reporte en formato JSON."""
    try:
        ensure_directory_exists(os.path.dirname(file_path))
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        return True
    except:
        return False


def load_json_config(file_path: str) -> Optional[Dict[str, Any]]:
    """Carga configuración desde JSON."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None


def cleanup_temp_files(directory_path: str, pattern: str = "*temp*") -> int:
    """Limpia archivos temporales."""
    try:
        directory = Path(directory_path)
        temp_files = list(directory.glob(pattern))
        
        removed_count = 0
        for temp_file in temp_files:
            try:
                temp_file.unlink()
                removed_count += 1
            except:
                continue
        
        return removed_count
    except:
        return 0