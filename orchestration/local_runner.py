"""Local runner for the Penguin TFX pipeline using LocalDagRunner.

### EXPLAINER
Runs the assembled pipeline locally with MLMD SQLite. Ensure that the data
CSV files exist under `data/` (see `pipeline/config.py`).
"""

from __future__ import annotations

from tfx import v1 as tfx

from pipeline.pipeline import create_pipeline
from pipeline import config as cfg


def main() -> None:
    cfg.ensure_dirs()
    runner = tfx.orchestration.LocalDagRunner()
    pipeline = create_pipeline()
    runner.run(pipeline)


if __name__ == "__main__":
    main()
