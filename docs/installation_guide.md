# Guía de Instalación

## Requisitos del Sistema

### Hardware Mínimo
- RAM: 8 GB (recomendado 16 GB)
- Almacenamiento: 50 GB libres
- Procesador: 4 cores (recomendado 8+ cores)

### Software Requerido
- Python 3.8+
- Java 8 o 11 (para Apache Spark)
- Git

## Instalación

### 1. Clonar el Repositorio
```bash
git clone https://github.com/usuario/misia-procesamiento-distribuido-sociodemografico.git
cd misia-procesamiento-distribuido-sociodemografico
```

### 2. Crear Entorno Virtual
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar Variables de Entorno
```bash
export HADOOP_HOME=/path/to/hadoop
export SPARK_HOME=/path/to/spark
```

### 5. Verificar Instalación
```bash
python scripts/validate_installation.py
```

## Configuración de Datos

1. Crear estructura de directorios:
```bash
python scripts/setup_project_structure.py
```

2. Configurar rutas de datos en `src/config/data_config.py`

## Problemas Comunes

### Error de Java
Si encuentras errores relacionados con Java, verifica que JAVA_HOME esté configurado correctamente.

### Error de Hadoop en Windows
En Windows, asegúrate de tener winutils.exe en el directorio HADOOP_HOME/bin.
