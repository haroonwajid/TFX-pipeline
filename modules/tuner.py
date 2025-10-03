"""Hyperparameter tuning utilities with KerasTuner for the Penguin model.

### EXPLAINER
Defines a hypermodel function for KerasTuner and a simple tuner factory
that keeps search time reasonable by default. The tuner searches over
hidden units, dropout rate, batch size, and learning rate.
"""

from __future__ import annotations

from typing import Dict

import keras_tuner as kt
import tensorflow as tf

from pipeline import config as cfg
from .model import build_model


def build_hypermodel(hp: kt.HyperParameters) -> tf.keras.Model:
    """Return a compiled model with hyperparameters from `hp`."""
    hidden_units_1 = hp.Int("hidden_units_1", min_value=32, max_value=128, step=32)
    hidden_units_2 = hp.Int("hidden_units_2", min_value=16, max_value=64, step=16)
    dropout_rate = hp.Float("dropout_rate", min_value=0.0, max_value=0.5, step=0.1)
    learning_rate = hp.Choice("learning_rate", values=[1e-3, 5e-4, 1e-4])

    model = build_model(
        hidden_units_1=hidden_units_1,
        hidden_units_2=hidden_units_2,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
    )
    return model


def build_tuner(project_dir: str) -> kt.RandomSearch:
    """Create and return a KerasTuner RandomSearch tuner instance."""
    tuner = kt.RandomSearch(
        build_hypermodel,
        objective=kt.Objective("val_binary_accuracy", direction="max"),
        max_trials=cfg.TUNING_MAX_TRIALS,
        executions_per_trial=cfg.TUNING_EXECUTIONS_PER_TRIAL,
        directory=project_dir,
        project_name="kt_penguin",
        overwrite=True,
        seed=cfg.RANDOM_SEED,
    )
    return tuner
