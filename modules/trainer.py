"""TFX Trainer run_fn that trains baseline and tuned models, exports best.

### EXPLAINER
- Reads transformed TFRecords produced by the Transform component.
- Builds datasets using the transformed feature spec and label.
- Trains a baseline model with fixed hyperparameters from config.
- Runs KerasTuner RandomSearch to find a tuned model.
- Compares eval metrics (primary metric from config) and exports the better.
- Exports a SavedModel with a serving signature that applies the tf.Transform
  graph so raw features can be sent to the model at inference.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Tuple

import tensorflow as tf
import tensorflow_transform as tft

from pipeline import config as cfg
from .preprocessing import get_transformed_names
from .model import build_model
from .tuner import build_tuner


def _parse_example_fn(
    example_proto: tf.Tensor, feature_spec: Dict[str, tf.io.FixedLenFeature]
) -> Dict[str, tf.Tensor]:
    return tf.io.parse_single_example(example_proto, feature_spec)


def _make_dataset(
    file_pattern: Iterable[str],
    tf_transform_output: tft.TFTransformOutput,
    label_key: str,
    batch_size: int,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Create a tf.data.Dataset from TFRecord files of transformed examples."""
    transformed_feature_spec = tf_transform_output.transformed_feature_spec()

    files = []
    for f in file_pattern if isinstance(file_pattern, (list, tuple)) else [file_pattern]:
        if isinstance(f, bytes):
            f = f.decode()
        files.extend(tf.io.gfile.glob(f))

    dataset = tf.data.TFRecordDataset(files, compression_type="")
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda x: _parse_example_fn(x, transformed_feature_spec),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    def _split_features_labels(features: Dict[str, tf.Tensor]) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
        label = features.pop(label_key)
        # Filter out examples with OOV label mapped to -1 during preprocessing
        return features, label

    dataset = dataset.map(_split_features_labels, num_parallel_calls=tf.data.AUTOTUNE)

    # Remove OOV labels == -1
    dataset = dataset.filter(lambda f, y: tf.greater_equal(y, tf.constant(0, tf.int64)))

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def _build_serving_signature(
    model: tf.keras.Model, tf_transform_output: tft.TFTransformOutput
):
    """Create a serving signature that accepts raw features and applies TFT."""

    raw_feature_spec = tf_transform_output.raw_feature_spec()
    # Remove label from serving spec if present
    raw_feature_spec.pop(cfg.LABEL_KEY, None)

    @tf.function(input_signature=[
        {k: tf.TensorSpec(shape=[None], dtype=v.dtype) for k, v in raw_feature_spec.items()}
    ])
    def serve_tf_examples_fn(raw_features: Dict[str, tf.Tensor]):
        transformed_features = tf_transform_output.transform_raw_features(raw_features)
        outputs = model(transformed_features, training=False)
        return {"predictions": outputs}

    return {"serving_default": serve_tf_examples_fn}


def run_fn(fn_args):
    """TFX Trainer entrypoint.

    Expects:
      - fn_args.train_files, fn_args.eval_files: patterns to transformed TFRecords
      - fn_args.transform_output: path to Transform output
      - fn_args.serving_model_dir: where to export the SavedModel
      - fn_args.model_run_dir: directory for intermediate artifacts (tuning)
    """
    tf.get_logger().setLevel("INFO")

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    features_map, transformed_label_key = get_transformed_names()

    # Datasets
    baseline_batch_size = cfg.BASELINE_TRAINING_PARAMS["batch_size"]
    train_ds = _make_dataset(
        fn_args.train_files,
        tf_transform_output,
        transformed_label_key,
        batch_size=baseline_batch_size,
        shuffle=True,
    )
    eval_ds = _make_dataset(
        fn_args.eval_files,
        tf_transform_output,
        transformed_label_key,
        batch_size=baseline_batch_size,
        shuffle=False,
    )

    # Baseline model
    baseline_model = build_model(
        hidden_units_1=int(cfg.BASELINE_TRAINING_PARAMS["hidden_units_1"]),
        hidden_units_2=int(cfg.BASELINE_TRAINING_PARAMS["hidden_units_2"]),
        dropout_rate=float(cfg.BASELINE_TRAINING_PARAMS["dropout_rate"]),
        learning_rate=float(cfg.BASELINE_TRAINING_PARAMS["learning_rate"]),
    )
    baseline_epochs = int(cfg.BASELINE_TRAINING_PARAMS["epochs"])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_binary_accuracy", patience=3, restore_best_weights=True
        )
    ]

    baseline_model.fit(
        train_ds,
        validation_data=eval_ds,
        epochs=baseline_epochs,
        callbacks=callbacks,
        verbose=2,
    )
    baseline_eval = baseline_model.evaluate(eval_ds, verbose=0, return_dict=True)

    # Tuning
    tuner_dir = os.path.join(fn_args.model_run_dir, "tuner")
    tuner = build_tuner(tuner_dir)
    tuner.search(
        train_ds,
        validation_data=eval_ds,
        epochs=max(baseline_epochs, 8),
        callbacks=callbacks,
        verbose=2,
    )
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    tuned_model = build_model(
        hidden_units_1=best_hp.get("hidden_units_1"),
        hidden_units_2=best_hp.get("hidden_units_2"),
        dropout_rate=best_hp.get("dropout_rate"),
        learning_rate=best_hp.get("learning_rate"),
    )
    tuned_model.fit(
        train_ds,
        validation_data=eval_ds,
        epochs=max(baseline_epochs, 8),
        callbacks=callbacks,
        verbose=2,
    )
    tuned_eval = tuned_model.evaluate(eval_ds, verbose=0, return_dict=True)

    # Select model by primary metric
    metric_key = cfg.EVAL_PRIMARY_METRIC
    baseline_score = float(baseline_eval.get(metric_key, 0.0))
    tuned_score = float(tuned_eval.get(metric_key, 0.0))

    selected = tuned_model if tuned_score >= baseline_score else baseline_model

    # Export SavedModel with TFT serving signature
    signatures = _build_serving_signature(selected, tf_transform_output)
    tf.saved_model.save(selected, fn_args.serving_model_dir, signatures=signatures)

    # Also save Keras model for downstream usage if desired
    try:
        selected.save(os.path.join(fn_args.model_run_dir, "model.keras"))
    except Exception:
        pass
