"""Airflow DAG to orchestrate the Penguin TFX pipeline.

### EXPLAINER
This DAG triggers the TFX pipeline using the KubeflowDagRunner-like pattern,
except we are wiring through LocalDagRunner-style execution inside an Airflow
PythonOperator for simplicity. For production, use the official TFX Airflow
DagRunner. This example keeps the assignment self-contained.
"""

from __future__ import annotations

import datetime as dt

from airflow import DAG
from airflow.operators.python import PythonOperator

from pipeline.pipeline import create_pipeline
from tfx import v1 as tfx
from pipeline import config as cfg


def _run_pipeline():
    cfg.ensure_dirs()
    runner = tfx.orchestration.LocalDagRunner()
    pipeline = create_pipeline()
    runner.run(pipeline)


def _default_args():
    return {
        'owner': 'airflow',
        'depends_on_past': False,
        'start_date': dt.datetime(2024, 1, 1),
        'retries': 0,
    }


def _make_dag() -> DAG:
    with DAG(
        dag_id='penguin_tfx_pipeline',
        default_args=_default_args(),
        schedule_interval=None,
        catchup=False,
        description='Run Penguin TFX pipeline',
        tags=['tfx', 'penguin'],
    ) as dag:
        run = PythonOperator(
            task_id='run_pipeline',
            python_callable=_run_pipeline,
        )
    return dag


globals()['penguin_tfx_pipeline'] = _make_dag()
