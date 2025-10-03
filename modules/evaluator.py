"""TFMA evaluation and blessing configuration.

### EXPLAINER
This module defines the TFMA EvalConfig with slicing by `island` and metric
thresholds (accuracy and AUC). The Evaluator component will use this config
and only bless models that meet the thresholds. Slices are computed for overall
and for each island value.
"""

from __future__ import annotations

import tensorflow_model_analysis as tfma

from pipeline import config as cfg


def build_eval_config(model_specs_label_key: str = None) -> tfma.EvalConfig:
    """Create TFMA EvalConfig with slicing by `island` and thresholds.

    Args:
      model_specs_label_key: optional label key if needed (unused for Keras).

    Returns:
      tfma.EvalConfig
    """
    metrics = [
        tfma.MetricConfig(class_name="SparseCategoricalAccuracy", threshold=tfma.MetricThreshold(
            value_threshold=tfma.GenericValueThreshold(lower_bound={'value': cfg.EVAL_THRESHOLDS['binary_accuracy']}),
            # Keep change threshold to ensure improvement over baseline if available in Evaluator
            change_threshold=tfma.GenericChangeThreshold(direction=tfma.MetricDirection.HIGHER_IS_BETTER)
        )),
        tfma.MetricConfig(class_name="AUC", threshold=tfma.MetricThreshold(
            value_threshold=tfma.GenericValueThreshold(lower_bound={'value': cfg.EVAL_THRESHOLDS['auc']}),
            change_threshold=tfma.GenericChangeThreshold(direction=tfma.MetricDirection.HIGHER_IS_BETTER)
        )),
    ]

    slicing_specs = [
        tfma.SlicingSpec(),  # overall
        tfma.SlicingSpec(feature_keys=["island"])  # by island
    ]

    eval_config = tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(
                label_key=None,  # Keras model outputs probabilities; labels included in data
                prediction_key="predictions",
            )
        ],
        slicing_specs=slicing_specs,
        metrics_specs=[tfma.MetricsSpec(metrics=metrics)],
    )
    return eval_config
