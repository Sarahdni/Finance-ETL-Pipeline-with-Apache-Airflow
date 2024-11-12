import os
import sys
from pathlib import Path

# Add project root to Python path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

# Set test environment
os.environ["AIRFLOW_HOME"] = str(root_dir)
os.environ["AIRFLOW__CORE__UNIT_TEST_MODE"] = "True"