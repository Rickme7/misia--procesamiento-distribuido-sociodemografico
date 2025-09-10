"""
Setup script para el proyecto TFM de procesamiento distribuido de datos censales.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="misia-procesamiento-distribuido-sociodemografico",
    version="1.0.0",
    author="Ramiro Ricardo Merchán Mora",
    author_email="correo@uma.es",
    description="Sistema de procesamiento distribuido de datos censales para análisis predictivo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/usuario/misia-procesamiento-distribuido-sociodemografico",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0", 
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.1"
        ],
        "docs": [
            "sphinx>=7.2.6",
            "sphinx-rtd-theme>=1.3.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "misia-pipeline=src.main:main",
            "misia-setup=scripts.setup_project_structure:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/usuario/misia-procesamiento-distribuido-sociodemografico/issues",
        "Source": "https://github.com/usuario/misia-procesamiento-distribuido-sociodemografico",
        "Documentation": "https://github.com/usuario/misia-procesamiento-distribuido-sociodemografico/blob/main/docs/",
    },
)
