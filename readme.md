# Project: TFX Penguin Pipeline (full assignment)
#
# Purpose:
# - Create a complete TFX pipeline for the Penguin dataset that satisfies Parts A–E
#   (ExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Trainer, Evaluator, Pusher).
# - Store metadata in MLMD (sqlite for local runs).
# - Ensure transforms are consistent at training and serving via TF Transform.
# - Produce evaluation metrics sliced by 'island' feature.
# - Demonstrate anomaly detection, data drift detection, and schema evolution (two modes: auto-update or block).
# - Train two models (baseline & tuned), compare with Evaluator thresholds and only push validated models.
# - Provide a modular repo layout, run script, GitHub Actions CI YAML, and a sample Airflow DAG.
# - Provide short report text for Part E.
#
# Assumptions & dependency notes for Copilot:
# - Use Python 3.9+.
# - Use TensorFlow 2.x, TFX (0.30+ ideally), TF Transform, TF Data Validation, TF Model Analysis, Apache Beam (DirectRunner fine for local).
# - Install libs via requirements.txt (create one).
# - Data: CSV file `data/penguins.csv` containing the Palmer penguins columns:
#   ['species','island','bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g','sex','year']
#   If data path differs, user will edit `PIPELINE_CONFIG` values.
# - We'll use CsvExampleGen (CSV) and LocalDagRunner for local execution.
#
# Repo layout to create (generate these files and directories):
#  - README.md
#  - requirements.txt
#  - pipeline/
#      - __init__.py
#      - pipeline.py                # pipeline factory & programmatic runner
#      - config.py                  # constants e.g., pipeline_root, data_root, metadata_path
#      - runner.py                  # small wrapper to call pipeline.create_pipeline & run
#  - modules/
#      - __init__.py
#      - preprocessing.py           # preprocessing_fn for tf.Transform
#      - trainer_module.py          # run_fn for TFX Trainer, builds and saves Keras model
#      - schema_manager.py          # helpers to update or block schema evolution
#  - scripts/
#      - run_pipeline_local.py      # CLI wrapper to start pipeline (uses pipeline.runner)
#      - inject_anomalies.py        # utility to create corrupted dataset versions
#      - simulate_drift.py          # utility to create data with shifted distributions
#      - compare_models.py          # quick script to show evaluator outputs
#  - infra/
#      - github_actions.yml         # example GitHub Actions workflow
#      - airflow_dag.py             # example Airflow DAG (pseudo/ready-to-adapt)
#  - report/
#      - short_report.md            # Part E markdown (pdf can be exported)
#  - data/
#      - penguins.csv               # (user supplies or script downloads it)
#
# --------------- COPILOT ACTIONS - generate files below ---------------
#
# STEP A: Generate requirements.txt
# - Provide a reasonable requirements file (exact versions optional but suggested).
#
# STEP B: Generate pipeline/config.py
# - Variables:
#    PIPELINE_NAME = 'penguin_pipeline'
#    PROJECT_DIR = os.path.abspath(os.path.join(os.getcwd()))
#    DATA_ROOT = os.path.join(PROJECT_DIR, 'data')
#    PIPELINE_ROOT = os.path.join(PROJECT_DIR, 'pipeline_output')
#    METADATA_PATH = os.path.join(PROJECT_DIR, 'metadata', 'metadata.db')
#    SERVING_MODEL_DIR = os.path.join(PROJECT_DIR, 'serving_model')
#
# STEP C: Generate preprocessing.py (modules/preprocessing.py)
# - Implement `preprocessing_fn(inputs)` for tf.Transform:
#   * Numeric features: bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g -> tft.scale_to_z_score
#   * Categorical features: island, sex -> tft.compute_and_apply_vocabulary
#   * Label: species -> tft.compute_and_apply_vocabulary -> output feature name 'label'
# - Include comments explaining why this ensures consistency at train+serve time.
#
# STEP D: Generate trainer_module.py (modules/trainer_module.py)
# - Implement `run_fn(fn_args)` compatible with TFX Trainer (Keras):
#   * Load tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
#   * Build input_fn that reads transformed TFRecords via tf.data.experimental.make_batched_features_dataset using
#     tf_transform_output.transformed_feature_spec()
#   * Build two model architectures:
#       - build_baseline_model(input_shape/features) - simple dense network
#       - build_tuned_model(...) - more layers, dropout, Adam with different LR and/or L2 regularization
#   * Train both models sequentially (or create two trainer runs) then save models to:
#       - fn_args.serving_model_dir (but include transform graph to ensure raw input compatibility)
#   * Export model with TF Transform graph embedded for serving:
#       - Create a `serving_input_receiver_fn` that expects raw examples, applies the transform graph, then calls Keras model.
#       - Save the resulting SavedModel to fn_args.serving_model_dir.
# - Add logging to produce metrics for training/validation and for TFMA evaluation input.
#
# STEP E: Generate pipeline/pipeline.py
# - Create `create_pipeline(pipeline_root, data_root, metadata_path, serving_model_dir, beam_pipeline_args=[])`
# - Components (ordered):
#    1) example_gen = CsvExampleGen(input_base=data_root)
#    2) statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
#    3) schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True)
#    4) example_validator = ExampleValidator(statistics=statistics_gen.outputs['statistics'], schema=schema_gen.outputs['schema'])
#    5) transform = Transform(examples=example_gen.outputs['examples'], schema=schema_gen.outputs['schema'], module_file=os.path.join(...,'modules/preprocessing.py'))
#    6) trainer = Trainer(module_file=os.path.join(...,'modules/trainer_module.py'), transformed_examples=transform.outputs['transformed_examples'], transform_graph=transform.outputs['transform_graph'], schema=schema_gen.outputs['schema'], train_args=..., eval_args=...)
#    7) evaluator = Evaluator(examples=example_gen.outputs['examples'], model=trainer.outputs['model'], baseline_model=None or reference model, eval_config=<tfma EvalConfig with slice by 'island' and thresholds for accuracy & AUC>)
#    8) pusher = Pusher(model=trainer.outputs['model'], model_blessing=evaluator.outputs['blessing'], push_destination=file system to serving_model_dir)
# - Configure metadata connection via sqlite: metadata.sqlite_metadata_connection_config(metadata_path)
# - Return a TFX Pipeline object
#
# STEP F: Generate pipeline/runner.py and scripts/run_pipeline_local.py
# - `runner.py` should accept CLI args or environment variables (data_root, pipeline_root, metadata_path) and call LocalDagRunner().run(create_pipeline(...))
# - `run_pipeline_local.py` is a thin wrapper CLI to call runner.run() and print status and paths to logs and metadata sqlite.
#
# STEP G: Generate modules/schema_manager.py
# - Implement:
#    - `validate_against_schema(stats_path, schema_path)` -> runs tfdv.validate_statistics & returns anomalies dict
#    - `apply_schema_update(old_schema_path, new_stats_paths, mode='auto'|'block')`:
#         * if mode == 'auto': re-infer schema using combined stats (you can re-run tfdv.infer_schema on stats from train+new) and overwrite schema file (log update).
#         * if mode == 'block': raise Exception or write a "blocked" file and do not update schema (ExampleValidator/Evaluator should stop pipeline)
# - Add a helper `create_enum_domain_for_island(schema_path, valid_values)` to lock allowed island values (so ExampleValidator flags NEW_UNEXPECTED_CATEGORICAL_VALUES).
#
# STEP H: Generate scripts/inject_anomalies.py and simulate_drift.py
# - `inject_anomalies.py`:
#    * Read `data/penguins.csv` with pandas.
#    * Create a mutated copy with:
#       - some missing values in numeric columns (NaN)
#       - wrong types in some columns (put strings in numeric column)
#       - out of range values for flipper_length_mm (e.g., -999, 10000)
#    * Save mutated CSV as `data/penguins_anomalous.csv` for re-run of pipeline and show how ExampleValidator logs anomalies.
# - `simulate_drift.py`:
#    * Read training CSV; create drifted file by shifting `flipper_length_mm` values by +20% or sampling from different distribution; save as `data/penguins_drifted.csv`.
#    * Show how to run pipeline and let Evaluator detect distribution changes (we will show TFMA eval_config slicing by island and a difference check).
#
# STEP I: Generate scripts/compare_models.py
# - Read TFMA evaluation results (evaluator outputs) from pipeline output (or from saved tfma results)
# - Create a table (pandas DataFrame) comparing:
#     * Model name, accuracy (overall), AUC (overall), slices by island accuracy & AUC
# - Print the table and save as CSV `pipeline_output/comparison.csv`
#
# STEP J: Evaluator configuration details
# - EvalConfig must:
#    * Use label_key='label' (produced by preprocessing)
#    * Use slicing_spec = [tfma.SlicingSpec(), tfma.SlicingSpec(feature_keys=['island'])]
#    * Add metrics: Accuracy and AUC; set thresholds:
#         - accuracy lower bound = 0.80 (example) - show how to change
#         - AUC lower bound = 0.85
#    * Add change thresholds such that model must beat baseline on at least one metric OR absolute thresholds to be blessed.
# - Explain inside comments how Evaluator produces a `blessing` artifact and how Pusher uses that artifact to push only validated models.
#
# STEP K: Create infra/github_actions.yml (pseudo)
# - Workflow steps:
#    1) checkout
#    2) set up python
#    3) pip install -r requirements.txt
#    4) run unit tests (if any)
#    5) run scripts/run_pipeline_local.py --pipeline_root=... --metadata_path=... --data_root=...
#    6) Collect artifacts: pipeline_output, metadata/metadata.db, serving_model/
# - Provide optional steps to upload artifacts to Actions artifacts or S3.
#
# STEP L: Create infra/airflow_dag.py (example DAG)
# - Provide an Airflow DAG that calls `python scripts/run_pipeline_local.py` using a BashOperator or PythonOperator.
# - Add tasks: run_pipeline -> check_blessing (script) -> push_model (only if blessed)
#
# STEP M: report/short_report.md (Part E)
# - Include:
#   * Challenges faced (data quality, TF Transform export, schema evolution, local beam runs)
#   * Importance of training-serving skew prevention (tf.Transform, same transform graph)
#   * How to scale this pipeline (MLflow/Kubernetes, distributed TF, Kafka for streaming, feature stores)
#   * Extensions for streaming (use TFX Streaming components or KafkaIO -> use `ExampleGen` with PubSub/Stream, streaming validations, real-time model monitors).
#
# STEP N: Helpful run instructions printed inside scripts and README.md
# - Add commands to:
#    pip install -r requirements.txt
#    python scripts/run_pipeline_local.py --data_root=data --pipeline_root=pipeline_output --metadata_path=metadata/metadata.db
# - Explain where to view metadata (metadata/metadata.db) and where ExampleValidator/TFDV logs appear.
#
# --------------- DEFAULT VALUES & EXAMPLES TO GENERATE ---------------
# - Provide explicit code comments (### EXPLAINERS) inside each generated file that:
#   * tell the user what the file does
#   * indicate where to change thresholds & hyperparameters
#   * show where to check MLMD (metadata sqlite) and tfma output directories
#
# --------------- COPILOT GENERATION STYLE ---------------
# - For each file, produce runnable Python code (no pseudo unless unavoidable).
# - Use abundant comments and docstrings.
# - Add helpful print/logging lines that indicate pipeline progress and artifact URIs.
# - When an API is version sensitive (TFX/TF API), add a short comment with an alternative import/approach.
#
# --------------- After files are created ---------------
# - Provide a final short README.md that lists the deliverables for submission:
#    1) pipeline/ (code)
#    2) scripts/ (how to run & how to inject anomalies)
#    3) report/short_report.md (pdf)
#    4) pipeline_output/ and metadata/ screenshots & sample logs (user will attach)
#
# --------------- COPILOT: now create the files in the repository exactly as described above ---------------
#
# You may start by generating `requirements.txt`, then pipeline/config.py, then modules/preprocessing.py, modules/trainer_module.py, pipeline/pipeline.py, and so on. Make sure each generated file includes robust comments and usage examples.
#
# If you need to split into smaller generations, generate files in the order:
#   1) requirements.txt
#   2) pipeline/config.py
#   3) modules/preprocessing.py
#   4) modules/trainer_module.py
#   5) modules/schema_manager.py
#   6) pipeline/pipeline.py
#   7) pipeline/runner.py
#   8) scripts/*
#   9) infra/*
#  10) report/short_report.md
#
# End of prompt.
