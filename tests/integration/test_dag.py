import pytest
from datetime import datetime, timedelta
from airflow import DAG
from airflow.models import DagBag
import os
import pendulum

def get_test_dag():
    """Helper function to get the DAG without database connection."""
    dag_bag = DagBag(os.path.join(os.path.dirname(__file__), "../../dags"), include_examples=False)
    return dag_bag.dags['finance_etl_pipeline']

class TestFinanceETLDAG:
    """Test suite for Finance ETL DAG."""

    def test_dag_loaded(self):
        """Test if DAG is loaded correctly."""
        dag = get_test_dag()
        assert dag is not None
        assert dag.dag_id == "finance_etl_pipeline"

    def test_task_structure(self):
        """Test DAG task structure."""
        dag = get_test_dag()
        tasks = dag.tasks
        task_ids = [task.task_id for task in tasks]
        
        # Verify all required tasks exist
        assert "extract_stock_data" in task_ids
        assert "process_data" in task_ids

    def test_task_dependencies(self):
        """Test task dependencies."""
        dag = get_test_dag()
        
        extract_task = dag.get_task("extract_stock_data")
        process_task = dag.get_task("process_data")
        
        # Verify task dependencies
        downstream_task_ids = [task.task_id for task in extract_task.downstream_list]
        assert "process_data" in downstream_task_ids

    def test_default_args(self):
        """Test DAG default arguments."""
        dag = get_test_dag()
        assert dag.default_args["owner"] == "airflow"
        assert dag.default_args["retries"] == 1
        assert isinstance(dag.default_args["retry_delay"], timedelta)

    def test_dag_configuration(self):
        """Test DAG basic configuration."""
        dag = get_test_dag()
        assert not dag.catchup
        expected_date = pendulum.datetime(2024, 1, 1, tz="UTC")
        assert dag.start_date == expected_date

    def test_target_stocks_configuration(self):
        """Test TARGET_STOCKS configuration."""
        from dags.finance_etl import TARGET_STOCKS
        
        assert isinstance(TARGET_STOCKS, list)
        assert len(TARGET_STOCKS) > 0
        assert all(isinstance(symbol, str) for symbol in TARGET_STOCKS)
        assert "AAPL" in TARGET_STOCKS