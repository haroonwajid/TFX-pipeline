"""Create anomalous and drifted datasets for Part B demonstrations.

### EXPLAINER
- Writes `data/penguins_anomalous.csv` with missing and wrong-typed values.
- Writes `data/penguins_eval_drift.csv` with shifted flipper length to simulate drift.
Then you can point the pipeline to these files via `pipeline/config.py` and rerun.
"""

from __future__ import annotations

import csv
from pathlib import Path

DATA = Path('data')
TRAIN = DATA / 'penguins_train.csv'
EVAL = DATA / 'penguins_eval.csv'
ANOM = DATA / 'penguins_anomalous.csv'
DRIFT = DATA / 'penguins_eval_drift.csv'


def write_anomalies() -> None:
    rows = list(csv.DictReader(TRAIN.open()))
    for r in rows[:20]:
        r['bill_length_mm'] = ''  # missing
        r['flipper_length_mm'] = 'oops'  # wrong type
    with ANOM.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {ANOM}")


def write_drift() -> None:
    rows = list(csv.DictReader(EVAL.open()))
    for r in rows:
        try:
            r['flipper_length_mm'] = str(float(r['flipper_length_mm']) * 1.2)
        except Exception:
            pass
    with DRIFT.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {DRIFT}")


def main() -> None:
    assert TRAIN.exists() and EVAL.exists(), "Base train/eval CSVs not found."
    write_anomalies()
    write_drift()


if __name__ == '__main__':
    main()
