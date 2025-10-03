"""TFX pipeline assembly for the Penguin dataset.

### EXPLAINER
This module builds a complete TFX pipeline with:
- ExampleGen: reads CSV files for train/eval.
- StatisticsGen: computes data statistics.
- SchemaGen: infers a schema (used with ExampleValidator for anomaly checks).
- ExampleValidator: detects anomalies against schema.
- Transform: applies tf.Transform using `modules/preprocessing.py`.
- Trainer: runs baseline and hyperparameter-tuned training (in-module compare),
  and exports the best model with a tf.Transform-aware serving signature.
- Resolver: fetches the latest blessed model as baseline for Evaluator.
- Evaluator: computes metrics with slicing by `island` and thresholds.
- Pusher: pushes model only if blessed.

MLMD uses SQLite as required. Paths and thresholds come from `pipeline/config.py`.
"""

from __future__ import annotations

import os
from typing import List

from tfx import v1 as tfx

from pipeline import config as cfg


def _get_module_path(relative: str) -> str:
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), relative)


def create_pipeline() -> tfx.dsl.Pipeline:
    """Create the TFX pipeline object."""
    cfg.ensure_dirs()

    # ExampleGen reading specific CSV files and assigning splits
    input_config = tfx.proto.example_gen_pb2.Input(splits=[
        tfx.proto.example_gen_pb2.Input.Split(name='train', pattern=cfg.TRAIN_DATA_FILENAME),
        tfx.proto.example_gen_pb2.Input.Split(name='eval', pattern=cfg.EVAL_DATA_FILENAME),
    ])
    example_gen = tfx.components.CsvExampleGen(
        input_base=cfg.DATA_ROOT,
        input_config=input_config,
    )

    statistics_gen = tfx.components.StatisticsGen(examples=example_gen.outputs['examples'])

    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True
    )

    example_validator = tfx.components.ExampleValidator(
        statistics=statistics_gen.outputs['statistics'], schema=schema_gen.outputs['schema']
    )

    transform = tfx.components.Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=_get_module_path('modules/preprocessing.py'),
    )

    trainer = tfx.components.Trainer(
        module_file=_get_module_path('modules/trainer.py'),
        transformed_examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=tfx.proto.TrainArgs(num_steps=0),
        eval_args=tfx.proto.EvalArgs(num_steps=0),
    )

    # Resolver for latest blessed model to use as baseline in Evaluator
    model_resolver = tfx.dsl.Resolver(
        strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
        model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
        model_blessing=tfx.dsl.Channel(type=tfx.types.standard_artifacts.ModelBlessing),
    ).with_id('latest_blessed_model_resolver')

    # Evaluator with TFMA config
    from modules.evaluator import build_eval_config  # local import to avoid cycles

    eval_config = build_eval_config()
    evaluator = tfx.components.Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config,
    )

    pusher = tfx.components.Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(
                base_directory=cfg.PUSHED_MODEL_DIR
            )
        ),
    )

    pipeline = tfx.dsl.Pipeline(
        pipeline_name=cfg.PROJECT_NAME,
        pipeline_root=cfg.PIPELINE_ROOT,
        components=[
            example_gen,
            statistics_gen,
            schema_gen,
            example_validator,
            transform,
            trainer,
            model_resolver,
            evaluator,
            pusher,
        ],
        enable_cache=True,
        metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(
            cfg.METADATA_PATH
        ),
    )

    return pipeline
