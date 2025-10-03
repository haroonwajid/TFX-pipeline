"""Schema management and data validation utilities using TFDV.

### EXPLAINER
This module encapsulates schema creation, anomaly detection, and drift checks
for the Penguin dataset. It supports two modes controlled by config:
- Auto update schema: when enabled, non-breaking changes update and persist
  the schema file automatically.
- Block on anomaly: when disabled, anomalies raise an exception to stop the
  pipeline; when enabled with auto-update off, we still block on severe issues.

It writes/reads the schema at `pipeline/config.py:SCHEMA_FILE`.
"""

from __future__ import annotations

import os
from typing import Optional

import tensorflow_data_validation as tfdv
from tensorflow_metadata.proto.v0 import schema_pb2

from pipeline import config as cfg


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def infer_schema_from_statistics(train_stats: tfdv.Statistics) -> schema_pb2.Schema:
    """Infer a schema from training statistics with reasonable defaults."""
    schema = tfdv.infer_schema(train_stats)

    # Configure domains and constraints for known features.
    # Numeric features should be FLOAT and not required to be present in all examples.
    for feature_name in cfg.NUMERIC_FEATURE_KEYS:
        feature = tfdv.get_feature(schema, feature_name)
        if feature:
            feature.presence.min_fraction = 0.0
            feature.type = schema_pb2.FeatureType.FLOAT

    # Categorical features are STRING with vocabulary inferred later by Transform.
    for feature_name in cfg.CATEGORICAL_FEATURE_KEYS + [cfg.LABEL_KEY]:
        feature = tfdv.get_feature(schema, feature_name)
        if feature:
            feature.presence.min_fraction = 0.0
            feature.type = schema_pb2.FeatureType.STRING

    return schema


def write_schema(schema: schema_pb2.Schema, path: str) -> None:
    _ensure_dir(path)
    tfdv.write_schema_text(schema, path)


def read_schema(path: str) -> Optional[schema_pb2.Schema]:
    if not os.path.exists(path):
        return None
    return tfdv.load_schema_text(path)


def validate_and_update_schema(
    train_stats: tfdv.Statistics,
    eval_stats: Optional[tfdv.Statistics] = None,
) -> schema_pb2.Schema:
    """Validate stats against schema; optionally update or block on anomalies.

    - If no schema exists, infer and persist one from training stats.
    - If anomalies are found:
      * If `cfg.AUTO_UPDATE_SCHEMA` is True, attempt to relax non-breaking
        constraints and write back the updated schema.
      * Else if `cfg.BLOCK_ON_ANOMALY` is True, raise an exception to stop.
      * Otherwise, log but proceed.
    - Also compute drift between train and eval with PSI threshold.
    """
    existing = read_schema(cfg.SCHEMA_FILE)
    if existing is None:
        schema = infer_schema_from_statistics(train_stats)
        write_schema(schema, cfg.SCHEMA_FILE)
        existing = schema

    # Validate training stats
    train_anomalies = tfdv.validate_statistics(train_stats, schema=existing)
    has_train_anomalies = bool(train_anomalies.anomaly_info)

    # Optionally validate eval stats
    has_eval_anomalies = False
    if eval_stats is not None:
        eval_anomalies = tfdv.validate_statistics(eval_stats, schema=existing)
        has_eval_anomalies = bool(eval_anomalies.anomaly_info)

    if has_train_anomalies or has_eval_anomalies:
        if cfg.AUTO_UPDATE_SCHEMA:
            # Attempt to update schema based on training stats
            updated_schema = tfdv.update_schema(train_stats, existing)
            write_schema(updated_schema, cfg.SCHEMA_FILE)
            existing = updated_schema
        elif cfg.BLOCK_ON_ANOMALY:
            details = []
            if has_train_anomalies:
                details.append("train anomalies present")
            if has_eval_anomalies:
                details.append("eval anomalies present")
            raise RuntimeError(
                f"Data anomalies detected and auto-update disabled: {', '.join(details)}"
            )
        # else: proceed but keep existing schema unchanged

    # Drift detection using PSI if both stats provided
    if eval_stats is not None:
        drift_anomalies = tfdv.validate_statistics(
            statistics=eval_stats,
            schema=existing,
            previous_statistics=train_stats,
            serving_statistics=None,
            drift_comparator=tfdv.DriftComparator(
                infinity_norm=tfdv.DriftSkewThreshold(
                    threshold=cfg.DRIFT_PSI_THRESHOLD
                )
            ),
        )
        if drift_anomalies.anomaly_info and cfg.BLOCK_ON_ANOMALY:
            raise RuntimeError(
                "Data drift detected beyond threshold; blocking pipeline run."
            )

    return existing
