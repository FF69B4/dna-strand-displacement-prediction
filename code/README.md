# Dissertation Code

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

From this `code` directory:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

## Implementing a New Experiment

The reusable pipeline is configured with dataclasses rather than hard-coded
paths. A new experiment is defined by creating an `ExperimentConfig` that
combines an application configuration with a reproducibility configuration.

The main configuration objects are:

- `DatasetSchema`: names the identifier, sequence, and optional availability
  columns, and supplies the sequence validator and feature builder.
- `ArtifactAppConfig`: defines the project root, artifact locations, final
  prediction files, schema, and optional API endpoint configuration.
- `ReproducibilityConfig`: defines the canonical model artifact, prediction
  inputs, reference outputs, graph renderer, training function, and verifier
  callbacks.
- `ExperimentConfig`: combines the application and reproducibility settings
  under a named experiment.

The graph renderer is a callback that receives an output directory, writes graph
artifacts into that directory, and returns a small verification dictionary.
This keeps plotting code replaceable while allowing the reproducibility runner
to treat graph generation like any other checked artifact.

Minimal shape:

```python
from experiment_core import ArtifactAppConfig, ExperimentConfig, ReproducibilityConfig
from ml_core import DatasetSchema

schema = DatasetSchema(
    id_column="No.",
    sequence_column="Sequence",
    availability_column="Nucleotide Availability",
    validator=sequence_validator,
    feature_set_builder=feature_builder,
)

config = ExperimentConfig(
    name="Example experiment",
    app=ArtifactAppConfig(
        project_root=project_root,
        artifacts_dir=artifacts_dir,
        final_prediction_manifest=manifest_path,
        final_prediction_csv=ranked_predictions_path,
        final_comparison_markdown=None,
        schema=schema,
        api=api_config,
    ),
    reproducibility=ReproducibilityConfig(
        canonical_model_artifact=model_artifact,
        canonical_model_summary=model_summary,
        prediction_table=prediction_table,
        prediction_embeddings=prediction_embeddings,
        canonical_ranked_predictions=canonical_predictions,
        external_summary=external_summary,
        canonical_graphs=canonical_graphs,
        output_artifact_name="model.pt",
        training_args_factory=make_training_args,
        training_runner=train_model,
        prediction_verifier=verify_predictions,
        external_verifier=verify_external_metrics,
        graph_renderer=render_graphs,
    ),
)
```

Graph renderers follow this shape:

```python
def render_graphs(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    # write plots into output_dir
    return {"passed": True, "file_count": len(list(output_dir.iterdir()))}
```

Individual graphs are usually defined as small `plot_*` functions. Each graph
function reads the artifact it needs, creates a Matplotlib figure, applies the
shared theme, and saves the image through `FigureOutput` or the local `save`
helper:

```python
import matplotlib.pyplot as plt
import pandas as pd

from viz_core import FigureOutput, monochrome_serif_theme

theme = monochrome_serif_theme()
figures = FigureOutput(output_dir, theme)

def plot_metric_bar_chart(metrics_path: Path) -> None:
    frame = pd.read_csv(metrics_path)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    bars = ax.bar(frame["model"], frame["r2"])
    theme.apply_bar_patterns(bars)
    ax.set_ylabel("Test R²")
    theme.style_axes(ax)
    figures.save(fig, "metric_bar_chart.png")
```

The renderer then calls whichever `plot_*` functions belong to that experiment
and returns whether the expected files were produced.

The dissertation implementation of this pattern is in `dissertation/config.py`.
Changing the dataclass values redirects the same prediction, retraining,
verification, graphing, and optional API code to a different dataset or model
artifact without rewriting the pipeline itself.

## Supporting Dataclasses

The framework also uses smaller dataclasses for API routing, saved model
handling, graph styling, and PDF extraction.

- `ApiEndpointConfig` defines the local JSON API name and route paths, such as
  health checks, model metadata, run summaries, prediction listings, prediction
  lookup by record ID, and optional prediction recomputation.
- `FeatureRecipe` is loaded from a saved model artifact. It records which
  feature family was used, the nearest-neighbour feature mode, the ordered
  feature names, and the training-set mean and standard deviation needed to
  transform future feature matrices consistently.
- `ModelArtifact` wraps a saved checkpoint path, checkpoint dictionary,
  `FeatureRecipe`, target column, and target transform. It can rebuild the
  regression head, load its trained weights, and invert transformed predictions
  back to the original target scale.
- `ArtifactPredictionResult` is returned after running a model artifact over a
  table. It records the table path, embedding path, artifact path, schema,
  feature recipe, target column, and prediction dataframe.
- `FigureTheme` stores Matplotlib styling choices, including rcParams, hatch
  patterns, line styles, markers, DPI, grid styling, and 3D embedding-view
  angles.
- `FigureOutput` combines an output directory with a `FigureTheme` and provides
  a single `save(fig, filename)` method that applies layout, writes the image,
  and closes the figure.
- `ColumnSpec` defines a cleaned PDF-table column name and cell transform.
- `TableSpec` defines the expected columns, table identifier, output CSV name,
  and header-fragment handling for one extracted PDF table.
- `ExtractionPlan` groups the table specifications and provides the caption
  pattern used to detect which table is currently being parsed.
- `ExtractedTable` stores one extracted table specification and its cleaned
  rows before CSV writing.

## 1. Quick Prediction Check

```bash
./.venv/bin/python reproduce_predictions.py
```

This reloads the selected saved artifact, rebuilds Table 3 features, recomputes
the candidate ranking, and compares it with the canonical ranked CSV. Output is
written under `outputs/`.

## 2. Full Selected-Experiment Reproduction

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

## 3. Reproduce Every Reported Result and Graph

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

The one-command audit runs:

1. `verify_reported_results.py` to reconstruct reported Table 1 metrics.
2. `reproduce_reported_figures.py` to regenerate computational graph artifacts.
3. `reproduce_predictions.py` to recompute the Table 3 candidate ranking.

The `--retrain-selected` option also runs `reproduce_experiment.py` to retrain
the selected frozen-embedding model and compare regenerated metrics, model
tensors, rankings, external comparison metrics, and graph outputs.

## Reproducibility Details

`verify_reported_results.py` reconstructs Table 1 test R2, RMSE, and MAE for:

- embeddings only
- embeddings + nearest-neighbour features
- embeddings + nearest-neighbour + availability features
- selected hybrid model
- unoptimised DNABERT-2
- optimised DNABERT-2

The recomputed values are compared with
`training/artifacts/model_comparison/reported_table1_model_comparison.csv`.
The script also checks the two R2 values stated explicitly in the report:

- selected 6-mer hybrid model test R2: `0.702`
- optimised DNABERT-2 test R2: `0.687`

Computational graphs are regenerated by `dissertation/graphs.py` from packaged
CSV, JSON, prediction, and embedding artifacts. The context-dependent 6-mer PCA
is reconstructed from
`training/artifacts/embedding_context_experiment/same_6mer_token_context_vectors.csv`,
so the original transformer checkpoint is not required to redraw that plot.

Static biology illustrations and concept diagrams in the dissertation PDF are
not reproduced by this code package. The figure reproduction command only
checks graph artifacts generated from packaged data.

Exploratory fine-tuned summaries and per-record prediction tables are retained
as supplementary artifacts, but they are not part of the reported Table 1
comparison. Multi-gigabyte fine-tuned checkpoints are intentionally excluded.

See `config/reproducibility_manifest.json` for the complete result-to-command
and figure-to-command mapping.

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
