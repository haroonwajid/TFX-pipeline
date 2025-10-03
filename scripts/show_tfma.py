"""Show TFMA evaluation metrics overall and sliced by `island`.

### EXPLAINER
Loads the latest TFMA eval_result from the Evaluator outputs and prints metrics
for overall and the `island` slice. Attach this output for assignment deliverables.
"""

from __future__ import annotations

import os
import glob
import tensorflow_model_analysis as tfma


def find_latest_eval_result(base_dir: str) -> str:
    # Heuristic: look for paths ending with eval_result under Evaluator outputs
    candidates = sorted(glob.glob(os.path.join(base_dir, "**", "eval_result"), recursive=True))
    if not candidates:
        raise FileNotFoundError(f"No eval_result found under {base_dir}")
    return candidates[-1]


def main() -> None:
    evaluator_root = os.path.abspath(os.path.join("outputs", "tfx_pipeline"))
    eval_dir = find_latest_eval_result(evaluator_root)

    result = tfma.load_eval_result(output_path=eval_dir)

    print("— Overall metrics —")
    for m in result.metrics_for_slice(tfma.SlicingSpec()):
        print(m)

    print("\n— Sliced by island —")
    for m in result.metrics_for_slice(tfma.SlicingSpec(feature_keys=['island'])):
        print(m)


if __name__ == "__main__":
    main()
