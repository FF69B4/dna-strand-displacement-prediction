# Dissertation Code Submission

This folder is a self-contained copy of the code and compact artifacts needed
to inspect and reproduce the study.

## Included

- Reusable framework packages: `api_core`, `experiment_core`, `extract_core`,
  `ml_core`, `runtime`, and `viz_core`
- Dissertation-specific application and experiment code in `dissertation`
- Cleaned input tables in `data/extracted/clean`
- Frozen DNA-BERT and DNABERT-2 embeddings in `bert`
- Compact regression artifacts, predictions, comparisons, and graph artifacts in
  `training/artifacts`
- Human-readable selected experiment settings in `config/selected_experiment.json`

Large exploratory fine-tuned checkpoints are intentionally excluded because
they are not needed to reproduce the selected frozen-embedding regression run.

## Recommended Environment

- Python 3.11
- CPU execution is supported and is the default
- Approximately 8--10 GB of free space is recommended for the Python
  environment, mainly due to the pinned PyTorch dependency stack

## Setup

From this `submission` directory:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

## 1. Verify the Package

```bash
./.venv/bin/python verify_submission.py
```

This checks required files, imports the framework, loads the selected model,
and reports its architecture and metadata.

## 2. Quick Prediction Check

```bash
./.venv/bin/python reproduce_predictions.py
```

This reloads the selected saved artifact, rebuilds Table 3 features, recomputes
the candidate ranking, and compares it with the canonical ranked CSV. Output is
written under `outputs/`.

## 3. Full Selected-Experiment Reproduction

```bash
./.venv/bin/python reproduce_experiment.py
```

This performs the selected training run from Table 1, compares the regenerated
metrics and model tensors with the canonical artifact, regenerates Table 3
predictions, and verifies the external comparison metrics.

To additionally regenerate paper graphs:

```bash
./.venv/bin/python reproduce_experiment.py --render-graphs
```

The full report and regenerated files are written under `outputs/`. Existing
output directories are not overwritten.

## 4. Reproduce Every Reported Result and Figure

```bash
./.venv/bin/python reproduce_all_reported.py
```

This independently reconstructs the reported Table 1 metrics from packaged
model summaries, regenerates the computational graph figures, verifies the
staged graph assets, and recomputes the final Table 3 ranking.

For the strongest audit, including exact retraining of the selected model:

```bash
./.venv/bin/python reproduce_all_reported.py --retrain-selected
```

See `REPRODUCIBILITY.md` and `config/reproducibility_manifest.json` for the
complete result-to-command and figure-to-command mapping.

## Optional API

```bash
./.venv/bin/python -m dissertation.server
```

The JSON API is then available at `http://127.0.0.1:5000`.

Useful endpoints include:

- `/health`
- `/model/final`
- `/runs`
- `/predictions/table3`
- `/predictions/table3/<record_id>`

## Selected Model

The selected model uses:

- Frozen `zhihan1996/DNA_bert_6` mean-pooled embeddings
- Positional and frequency nearest-neighbour dinucleotide features
- Per-position nucleotide availability features
- A `768 -> 256 -> 1` regression head
- Layer normalization, GELU activation, and dropout
- AdamW optimization and early stopping
- `log10(k1)` as the model target

The exact configuration is available in both
`config/selected_experiment.json` and `dissertation/config.py`.
