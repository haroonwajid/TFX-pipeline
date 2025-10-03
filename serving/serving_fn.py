"""Serving functions for TF Serving using the Transform graph.

### EXPLAINER
Provides a function to load the Transform graph and the model, exposing a
`serving_default` signature that accepts raw features and applies the same
preprocessing as during training.
"""

from __future__ import annotations

import tensorflow as tf
import tensorflow_transform as tft


def build_signatures(model: tf.keras.Model, transform_output_dir: str):
    tf_transform_output = tft.TFTransformOutput(transform_output_dir)
    raw_feature_spec = tf_transform_output.raw_feature_spec()

    @tf.function(input_signature=[{k: tf.TensorSpec(shape=[None], dtype=v.dtype) for k, v in raw_feature_spec.items()}])
    def serve_raw_features(raw_features):
        transformed = tf_transform_output.transform_raw_features(raw_features)
        preds = model(transformed, training=False)
        return {"predictions": preds}

    return {"serving_default": serve_raw_features}
