"""Configuration constants and thresholds for the Penguin TFX pipeline.

### EXPLAINER
This module centralizes all configurable aspects of the pipeline:
- Directory layout for artifacts, metadata (MLMD), and outputs.
- Data and schema file names.
- Model training hyperparameters and feature/label definitions.
- Evaluator thresholds and slicing specs for TFMA.
- Drift/anomaly configuration for TFDV.
- Orchestration toggles (e.g., auto update schema vs block on anomalies).

It is imported by other modules to ensure consistent settings across
preprocessing, training, evaluation, and serving.

The values are chosen to be sensible defaults for the Penguin dataset.
They can be overridden per-environment by reading environment variables.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional


# ------------------------------------------------------------------------------
# Project name and paths
# ------------------------------------------------------------------------------

PROJECT_NAME: str = os.environ.get("PROJECT_NAME", "penguin_tfx_pipeline")

# Repository root (assumed as current working directory by default).
REPO_ROOT: str = os.environ.get("REPO_ROOT", os.getcwd())

# Data directory; expected to contain CSVs for the Penguin dataset.
DATA_ROOT: str = os.environ.get(
    "DATA_ROOT",
    os.path.join(REPO_ROOT, "data"),
)

# Base pipeline output directories
OUTPUT_BASE: str = os.environ.get("OUTPUT_BASE", os.path.join(REPO_ROOT, "outputs"))
PIPELINE_ROOT: str = os.environ.get(
    "PIPELINE_ROOT", os.path.join(OUTPUT_BASE, "tfx_pipeline")
)
SERVING_MODEL_DIR: str = os.environ.get(
    "SERVING_MODEL_DIR", os.path.join(OUTPUT_BASE, "serving_model")
)
PUSHED_MODEL_DIR: str = os.environ.get(
    "PUSHED_MODEL_DIR", os.path.join(OUTPUT_BASE, "pushed_models")
)

# MLMD (Metadata) - using SQLite per assignment requirement.
METADATA_PATH: str = os.environ.get(
    "METADATA_PATH", os.path.join(OUTPUT_BASE, "mlmd", "metadata.db")
)

# ------------------------------------------------------------------------------
# Data files and schema artifacts
# ------------------------------------------------------------------------------

TRAIN_DATA_FILENAME: str = os.environ.get(
    "TRAIN_DATA_FILENAME", "penguins_train.csv"
)
EVAL_DATA_FILENAME: str = os.environ.get("EVAL_DATA_FILENAME", "penguins_eval.csv")
SCHEMA_FILE: str = os.environ.get("SCHEMA_FILE", os.path.join(OUTPUT_BASE, "schema.pbtxt"))

# Whether to automatically update schema when minor changes are detected.
AUTO_UPDATE_SCHEMA: bool = os.environ.get("AUTO_UPDATE_SCHEMA", "false").lower() in {
    "1",
    "true",
    "yes",
}

# ------------------------------------------------------------------------------
# Features and label specification
# ------------------------------------------------------------------------------

# Raw feature names in the CSV dataset.
NUMERIC_FEATURE_KEYS: List[str] = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]

# Categorical/context features
CATEGORICAL_FEATURE_KEYS: List[str] = [
    "island",  # used for slicing metrics
    "sex",
]

# Label: species classification: Adelie, Chinstrap, Gentoo
LABEL_KEY: str = "species"

# When exporting for serving, we keep a subset of features expected in requests.
SERVING_FEATURE_KEYS: List[str] = NUMERIC_FEATURE_KEYS + ["island", "sex"]

# ------------------------------------------------------------------------------
# Training configuration
# ------------------------------------------------------------------------------

RANDOM_SEED: int = int(os.environ.get("RANDOM_SEED", "42"))

# Base training hyperparameters for the baseline model.
BASELINE_TRAINING_PARAMS: Dict[str, float] = {
    "epochs": int(os.environ.get("BASELINE_EPOCHS", "5")),
    "batch_size": int(os.environ.get("BASELINE_BATCH_SIZE", "32")),
    "learning_rate": float(os.environ.get("BASELINE_LR", "0.001")),
    "dropout_rate": float(os.environ.get("BASELINE_DROPOUT", "0.1")),
    "hidden_units_1": int(os.environ.get("BASELINE_H1", "64")),
    "hidden_units_2": int(os.environ.get("BASELINE_H2", "32")),
}

# Hyperparameter tuning space (KerasTuner)
# These bounds are moderate to keep tuning time reasonable.
TUNING_MAX_TRIALS: int = int(os.environ.get("TUNING_MAX_TRIALS", "10"))
TUNING_EXECUTIONS_PER_TRIAL: int = int(
    os.environ.get("TUNING_EXECUTIONS_PER_TRIAL", "1")
)

# ------------------------------------------------------------------------------
# Evaluation thresholds and slicing
# ------------------------------------------------------------------------------

# The Evaluator will compute metrics (accuracy, AUC) with slicing by "island".
# Thresholds are enforced; a model is only "blessed" if it meets these.
# Additionally, we require improvement over the baseline to push.
EVAL_THRESHOLDS: Dict[str, float] = {
    "binary_accuracy": float(os.environ.get("THRESH_ACC", "0.80")),
    "auc": float(os.environ.get("THRESH_AUC", "0.90")),
}

# Metric keys as used by TFMA for classification; adjust if necessary.
EVAL_PRIMARY_METRIC: str = os.environ.get("PRIMARY_METRIC", "binary_accuracy")

# ------------------------------------------------------------------------------
# Data validation (TFDV) configuration for drift and anomalies
# ------------------------------------------------------------------------------

# Drift detection thresholds; per-feature population stability index (PSI) bounds.
# If PSI exceeds this threshold, drift is flagged.
DRIFT_PSI_THRESHOLD: float = float(os.environ.get("DRIFT_PSI_THRESHOLD", "0.2"))

# Missingness rate threshold for anomaly (example; TFDV has richer configs).
MISSINGNESS_THRESHOLD: float = float(os.environ.get("MISSINGNESS_THRESHOLD", "0.1"))

# Whether to fail the pipeline on anomalies/drift when AUTO_UPDATE_SCHEMA is False.
BLOCK_ON_ANOMALY: bool = os.environ.get("BLOCK_ON_ANOMALY", "true").lower() in {
    "1",
    "true",
    "yes",
}

# ------------------------------------------------------------------------------
# Trainer/export settings
# ------------------------------------------------------------------------------

# Export a SavedModel for TF Serving with preprocessing graph from tf.Transform.
EXPORT_BEST_ONLY: bool = True  # Only push the tuned model if it beats baseline.

# ------------------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------------------

def ensure_dirs() -> None:
    """Ensure that all required directories exist.

    This is useful when orchestrating the pipeline locally or in CI/CD.
    """
    for path in [
        OUTPUT_BASE,
        PIPELINE_ROOT,
        SERVING_MODEL_DIR,
        PUSHED_MODEL_DIR,
        os.path.dirname(METADATA_PATH),
        os.path.dirname(SCHEMA_FILE),
        DATA_ROOT,
    ]:
        os.makedirs(path, exist_ok=True)


def get_data_paths() -> Dict[str, str]:
    """Return absolute paths for train and eval CSV files."""
    return {
        "train": os.path.join(DATA_ROOT, TRAIN_DATA_FILENAME),
        "eval": os.path.join(DATA_ROOT, EVAL_DATA_FILENAME),
    }


def get_metadata_connection_config() -> Dict[str, str]:
    """Return MLMD SQLite configuration.

    Consumers can use this to connect to MLMD in local runners.
    """
    return {"sqlite": METADATA_PATH}


# When this module is run directly, create directories for convenience.
if __name__ == "__main__":
    ensure_dirs()
    print("Initialized configuration and ensured directories exist.")
