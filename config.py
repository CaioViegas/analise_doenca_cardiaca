from pathlib import Path

# Diretório base
BASE_DIR = Path(__file__).resolve().parent

# Data
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"

# Models
MODELS_DIR = BASE_DIR / "models"

# Reports
REPORTS_DIR = BASE_DIR / "reports"

# Notebooks
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
FUNCTIONS_DIR = NOTEBOOKS_DIR / "functions"

# Tests
TESTS_DIR = BASE_DIR / "tests"

# Source
SRC_DIR = BASE_DIR / "src"
SCRIPTS_DIR = SRC_DIR / "scripts"