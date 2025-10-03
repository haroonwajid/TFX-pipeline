# Penguin TFX Pipeline - Short Report

## Overview
This project implements an end-to-end TFX pipeline for the Palmer Penguins dataset. It includes schema management with TFDV, preprocessing with tf.Transform, model training (baseline + tuned), evaluation with TFMA sliced by `island` and enforced thresholds, and conditional model pushing only when the Evaluator blesses the model. MLMD uses SQLite.

## Data & Schema
- Input: CSV files `data/penguins_train.csv` and `data/penguins_eval.csv`.
- Schema inferred with TFDV (numeric features: bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g; categorical: island, sex; label: species).
- Anomalies are validated each run. Optional auto-update relaxes minor deltas; otherwise the pipeline blocks on anomalies/drift.

## Preprocessing
- Numeric: mean imputation + z-score via tf.Transform.
- Categorical: fill missing with 'unknown' + vocabulary to integer IDs.
- Label: string → integer id with OOV mapped to -1 and filtered.
- The Transform graph is exported and used in serving signatures for consistency.

## Modeling
- Architecture: MLP with two hidden layers and dropout; softmax over 3 species.
- Baseline hyperparameters from config; then KerasTuner RandomSearch refines HPs.
- Only the better model (by primary metric) is exported.

## Evaluation & Thresholds
- TFMA metrics: SparseCategoricalAccuracy (as `binary_accuracy` name in Keras), AUC.
- Sliced by `island`. Thresholds enforced: accuracy ≥ configured bound, AUC ≥ bound.
- Change thresholds encourage improvement over baseline when available.

## Orchestration
- Local: `python orchestration/local_runner.py` runs the pipeline via LocalDagRunner.
- Airflow: DAG `airflow/dags/penguin_pipeline_dag.py` triggers the same run inside a `PythonOperator`.
- CI: GitHub Actions installs dependencies, does lightweight lint/import checks, and validates pipeline creation.

## Reproducibility
- Config centralizes paths, seeds, thresholds. MLMD is at `outputs/mlmd/metadata.db`.
- Determinism aided by fixed seeds and caching; be aware of nondeterminism in GPU ops.

## Next Steps
- Add model explainability (TFX What-If Tool or SHAP integration).
- Implement data drift dashboards and alerts.
- Promote to a production-grade Airflow/Kubeflow runner and remote artifact store.
