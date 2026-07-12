# DNA Strand Displacement Prediction

This repository contains the dissertation report, project code, datasets, and
compact artifacts needed to reproduce the reported computational results.

## Contents

- `report/dissertation.pdf` - dissertation report.
- `code/` - project code, cleaned datasets, embeddings, model artifacts, and reproducibility scripts.

## Reproducing Results

Run commands from the `code/` directory.

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python reproduce_all_reported.py
```

For the strongest audit, including retraining the selected model:

```bash
./.venv/bin/python reproduce_all_reported.py --retrain-selected
```

See `code/README.md`, `code/REPRODUCIBILITY.md`, and
`code/config/reproducibility_manifest.json` for the full result-to-command and
figure-to-command mapping.
