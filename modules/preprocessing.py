"""tf.Transform preprocessing for the Penguin dataset.

### EXPLAINER
This module defines a consistent preprocessing pipeline applied during both
training and serving. We leverage tf.Transform to compute statistics on the
full training dataset and produce a SavedModel that applies the same transforms
at inference time.

Transformations:
- Numeric features: impute missing with mean, then z-score scale.
- Categorical features (island, sex): fill missing with 'unknown' and compute
  vocabulary to integer IDs with out-of-vocab handling.
- Label (species): map string labels to integer class IDs for model training.

Outputs:
- `preprocessing_fn` for the TFX Transform component.
- Utility functions to build feature specs and parse examples.
"""

from __future__ import annotations

import tensorflow as tf
import tensorflow_transform as tft
from typing import Dict, Tuple

# Keep feature keys in sync with pipeline.config
try:
    from pipeline import config as cfg
except Exception:  # pragma: no cover - allows standalone import for docs
    class cfg:  # type: ignore
        NUMERIC_FEATURE_KEYS = [
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
        ]
        CATEGORICAL_FEATURE_KEYS = ["island", "sex"]
        LABEL_KEY = "species"
        SERVING_FEATURE_KEYS = NUMERIC_FEATURE_KEYS + ["island", "sex"]


# Names for transformed features
def _transformed_name(key: str) -> str:
    return f"{key}_xf"


def _vocab_name(key: str) -> str:
    return f"{key}_vocab"


# Label mapping - canonical ordering
_LABELS = ["Adelie", "Chinstrap", "Gentoo"]


def _fill_missing(x: tf.Tensor, default_value: tf.Tensor) -> tf.Tensor:
    """Fill missing values for dense tensors.

    For numeric columns, `default_value` will be 0.0 (before scaling).
    For string columns, `default_value` will be 'unknown'.
    """
    if isinstance(x, tf.SparseTensor):
        x = tf.sparse.to_dense(x, default_value=default_value)
    else:
        x = tf.where(tf.math.is_nan(x), default_value, x)
    return x


def preprocessing_fn(inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """tf.Transform preprocessing function.

    Args:
      inputs: A dict of raw, untransformed feature tensors.

    Returns:
      A dict mapping transformed feature names to transformed tensors.
    """
    outputs: Dict[str, tf.Tensor] = {}

    # Process numeric features: impute and z-score scale
    for key in cfg.NUMERIC_FEATURE_KEYS:
        raw = inputs[key]
        raw = _fill_missing(raw, tf.constant(0.0))
        # Cast to float32 to ensure stable transforms
        raw = tf.cast(raw, tf.float32)
        scaled = tft.scale_to_z_score(raw)
        outputs[_transformed_name(key)] = scaled

    # Process categorical features: string to vocab indices
    for key in cfg.CATEGORICAL_FEATURE_KEYS:
        raw = inputs[key]
        raw = _fill_missing(raw, tf.constant("unknown"))
        # Build vocab and map to integer IDs; reserve OOV bucket
        indices = tft.compute_and_apply_vocabulary(
            raw,
            top_k=None,
            num_oov_buckets=1,
            vocab_filename=_vocab_name(key),
        )
        outputs[_transformed_name(key)] = tf.cast(indices, tf.int64)

    # Map string species label to integer class id
    label_str = _fill_missing(inputs[cfg.LABEL_KEY], tf.constant("unknown"))
    # Ensure label is known; map unknowns to OOV class -1 then filter at trainer
    table = tf.lookup.StaticVocabularyTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(_LABELS),
            values=tf.constant(list(range(len(_LABELS))), dtype=tf.int64),
        ),
        num_oov_buckets=1,
    )
    label_id = table.lookup(label_str)
    # Move OOV bucket to -1 for clarity; known labels 0..num_classes-1
    num_classes = len(_LABELS)
    label_id = tf.where(label_id >= num_classes, tf.constant(-1, tf.int64), label_id)
    outputs[_transformed_name(cfg.LABEL_KEY)] = label_id

    return outputs


def transformed_feature_spec() -> Dict[str, tf.io.FixedLenFeature]:
    """Return feature spec for raw features for tf.Example parsing.

    This spec is used by Transform to parse input examples consistently.
    """
    spec: Dict[str, tf.io.FixedLenFeature] = {}
    for key in cfg.NUMERIC_FEATURE_KEYS:
        spec[key] = tf.io.FixedLenFeature([], tf.float32)
    for key in cfg.CATEGORICAL_FEATURE_KEYS:
        spec[key] = tf.io.FixedLenFeature([], tf.string)
    spec[cfg.LABEL_KEY] = tf.io.FixedLenFeature([], tf.string)
    return spec


def get_transformed_names() -> Tuple[Dict[str, str], str]:
    """Return mapping of feature names to transformed names and transformed label.

    Returns:
      (features_map, transformed_label_key)
    """
    feat_map = {k: _transformed_name(k) for k in (cfg.NUMERIC_FEATURE_KEYS + cfg.CATEGORICAL_FEATURE_KEYS)}
    label = _transformed_name(cfg.LABEL_KEY)
    return feat_map, label
