"""Keras model factory for the Penguin classification task.

### EXPLAINER
This module defines a function to build a Keras model that consumes the
outputs of tf.Transform. Numeric features are standardized floats; categorical
features are integer indices (already vocabulary-mapped by Transform).

We build a simple MLP with two hidden layers and dropout. The final layer
has 3 units (one per species) with softmax activation.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import tensorflow as tf

try:
    from pipeline import config as cfg
except Exception:  # pragma: no cover - for docs only
    class cfg:  # type: ignore
        NUMERIC_FEATURE_KEYS = [
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
        ]
        CATEGORICAL_FEATURE_KEYS = ["island", "sex"]


from .preprocessing import get_transformed_names


def _input_layers() -> Dict[str, tf.keras.layers.Input]:
    """Create Keras Input layers for transformed features."""
    features_map, _ = get_transformed_names()
    inputs: Dict[str, tf.keras.layers.Input] = {}

    # Numeric inputs are float32 scalars
    for key in cfg.NUMERIC_FEATURE_KEYS:
        inputs[features_map[key]] = tf.keras.layers.Input(
            shape=(1,), name=features_map[key], dtype=tf.float32
        )

    # Categorical inputs are int64 scalars (already indexed by Transform)
    for key in cfg.CATEGORICAL_FEATURE_KEYS:
        inputs[features_map[key]] = tf.keras.layers.Input(
            shape=(1,), name=features_map[key], dtype=tf.int64
        )

    return inputs


def build_model(
    hidden_units_1: int = 64,
    hidden_units_2: int = 32,
    dropout_rate: float = 0.1,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Create and compile the Keras classification model.

    Args:
      hidden_units_1: Units in the first dense layer.
      hidden_units_2: Units in the second dense layer.
      dropout_rate: Dropout probability.
      learning_rate: Adam learning rate.

    Returns:
      A compiled `tf.keras.Model`.
    """
    inputs = _input_layers()

    # Cast categorical int64 to int32 for embedding/one-hot ops compatibility
    cat_inputs = []
    for key in cfg.CATEGORICAL_FEATURE_KEYS:
        cat_inputs.append(tf.cast(inputs[f"{key}_xf"], tf.int32))

    # One-hot encode categoricals with dynamic depth (use small cap to be safe)
    # In practice, TF.Transform vocabulary size is available via assets.
    # We use a reasonable default; evaluator/serving will handle OOV bucket.
    island_one_hot = tf.one_hot(cat_inputs[0], depth=10)
    sex_one_hot = tf.one_hot(cat_inputs[1], depth=3)
    cat_vector = tf.concat([tf.reshape(island_one_hot, (-1, 10)), tf.reshape(sex_one_hot, (-1, 3))], axis=1)

    # Concatenate numeric features
    num_tensors = [inputs[f"{k}_xf"] for k in cfg.NUMERIC_FEATURE_KEYS]
    num_vector = tf.concat(num_tensors, axis=1)

    x = tf.concat([num_vector, cat_vector], axis=1)
    x = tf.keras.layers.Dense(hidden_units_1, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(hidden_units_2, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="binary_accuracy"),
            tf.keras.metrics.AUC(name="auc", multi_label=False),
        ],
    )
    return model
