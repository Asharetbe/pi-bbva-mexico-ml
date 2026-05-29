"""
Setup para instalar como paquete (opcional).
Uso: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="pi-bbva-mexico-ml",
    version="0.1.0",
    description="Modelos de ML para estimación de Probabilidad de Incumplimiento crediticia - BBVA México",
    author="P1, P2, P3, P4",
    author_email="asharet.b@gmail.com",
    url="https://github.com/Asharetbe/pi-bbva-mexico-ml.git",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        "tensorflow>=2.11.0",
        "keras>=2.11.0",
        "optuna>=3.0.0",
        "scorecardpy>=0.1.9",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "plotly>=5.0.0",
        "jupyter>=1.0.0",
        "openpyxl>=3.8.0",
        "python-dotenv>=0.19.0",
        "tqdm>=4.62.0",
        "joblib>=1.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "pylint>=2.12.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
