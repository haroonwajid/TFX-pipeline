## Penguin TFX Pipeline (Palmer Penguins)

End-to-end TFX pipeline for multi-class classification on the Palmer Penguins dataset. The pipeline covers data ingestion, schema/anomaly detection, preprocessing with tf.Transform, model training (baseline + tuned), evaluation with TFMA sliced by `island` with thresholds, conditional model push, and CI setup. ML Metadata (MLMD) uses SQLite. Serving exports a SavedModel with the tf.Transform graph for consistent preprocessing.

### Repo structure
```
.
├─ pipeline/
│  ├─ config.py                # Config: paths, feature keys, thresholds, MLMD
│  └─ pipeline.py              # TFX pipeline assembly
├─ modules/
│  ├─ preprocessing.py         # tf.Transform preprocessing_fn and feature specs
│  ├─ schema.py                # TFDV schema inference, anomaly & drift handling
│  ├─ model.py                 # Keras model factory consuming transformed features
│  ├─ tuner.py                 # KerasTuner RandomSearch hypermodel/tuner
│  └─ trainer.py               # TFX Trainer run_fn: baseline + tuned, export best
├─ orchestration/
│  └─ local_runner.py          # Run pipeline locally with LocalDagRunner
├─ serving/
│  └─ serving_fn.py            # Serving signature wrapping tf.Transform graph
├─ airflow/
│  └─ dags/penguin_pipeline_dag.py   # Airflow DAG to trigger pipeline
├─ .github/workflows/ci.yml    # CI: deps install, lint, import checks, build pipeline
├─ data/
│  ├─ penguins_train.csv       # Prepared train split
│  └─ penguins_eval.csv        # Prepared eval split
├─ reports/short_report.md     # Design summary and decisions
├─ scripts/setup_and_run.sh    # One-shot env setup (conda-forge) + run pipeline
└─ requirements.txt            # Pip requirements (baseline reference)
```

### Features implemented
- Consistent tf.Transform preprocessing for training and serving
  - Numeric: z-score; Categorical: vocabulary with OOV; Label: string→id with OOV filtered
- TFMA Evaluator slicing by `island` with thresholds
  - Accuracy and AUC thresholds via `pipeline/config.py:EVAL_THRESHOLDS`
  - Change thresholds to prefer models that improve over baseline
- Schema with TFDV + drift
  - First run infers schema; later runs validate, with optional auto-update
  - PSI-based drift detection with threshold in config
  - `AUTO_UPDATE_SCHEMA` and `BLOCK_ON_ANOMALY` toggles
- Trainer trains baseline and tuned models; exports only if tuned ≥ baseline by primary metric; includes Transform-aware serving signature
- MLMD via SQLite at `outputs/mlmd/metadata.db`
- CI workflow to import-check, lint, and build pipeline object
- Airflow DAG included

### Setup and running (macOS ARM/Apple Silicon)
Because TF/TFX wheels are version-sensitive on Apple Silicon, use one of these:

Option A) Docker (if available)
```
docker run --rm -i \
  -v "$PWD":"$PWD" -w "$PWD" tensorflow/tfx:1.15.0 \
  bash -lc "python -m pip install --upgrade pip && pip install keras-tuner==1.4.6 && python orchestration/local_runner.py"
```

Option B) One-shot conda-forge environment (single env) – no Docker
```
scripts/setup_and_run.sh
```
If the terminal can’t find `conda` on first run, open a new terminal and run again.

Outputs
- Pipeline root: `outputs/tfx_pipeline`
- Pushed (blessed) models: `outputs/pushed_models`
- MLMD (SQLite): `outputs/mlmd/metadata.db`

### Configuration knobs
Edit `pipeline/config.py` or set environment variables:
- Paths: `DATA_ROOT`, `OUTPUT_BASE`, `PIPELINE_ROOT`, `SERVING_MODEL_DIR`, `PUSHED_MODEL_DIR`, `METADATA_PATH`
- Files: `TRAIN_DATA_FILENAME`, `EVAL_DATA_FILENAME`
- Features: `NUMERIC_FEATURE_KEYS`, `CATEGORICAL_FEATURE_KEYS`, `LABEL_KEY`
- Training: `BASELINE_TRAINING_PARAMS` (epochs, batch size, lr, dropout, hidden sizes), `TUNING_MAX_TRIALS`
- Evaluation thresholds: `EVAL_THRESHOLDS` and `EVAL_PRIMARY_METRIC`
- Schema & drift: `AUTO_UPDATE_SCHEMA`, `BLOCK_ON_ANOMALY`, `DRIFT_PSI_THRESHOLD`

### Serving
The exported SavedModel includes a `serving_default` signature that accepts raw features and internally applies the Transform graph before inference. Match dtypes and shapes per `serving/serving_fn.py`.

### CI/CD
- GitHub Actions (`.github/workflows/ci.yml`): installs deps, lints (light), imports modules, builds pipeline object
- Extend with tests and environment-specific runners (Airflow/Kubeflow) for production

### Airflow DAG
`airflow/dags/penguin_pipeline_dag.py` triggers the same LocalDagRunner inside a PythonOperator. For production, switch to `AirflowDagRunner`.

### Data
Prepared CSVs are included. To recreate: download Palmer Penguins, keep columns `species,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex`, drop missing rows, shuffle, and split 80/20.

### Troubleshooting
- Docker extraction errors: free disk space and increase Docker Desktop disk image size
- Pip wheels on Apple Silicon: prefer Docker or the conda-forge script
- zsh extras quoting: `pip install 'apache-beam[gcp]==2.48.0'`

### License
MIT
