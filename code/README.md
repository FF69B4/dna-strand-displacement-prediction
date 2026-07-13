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

### Validators and Feature Builders

`DatasetSchema` delegates representation checking and feature construction to
small pluggable objects. This is what allows the same pipeline to be reused
outside the DNA experiment.

A validator normalises and validates the representation column before features
are prepared. It must expose a `name` attribute and a `validate(value)` method
that returns the normalised representation or raises `ValueError`:

```python
class ExampleSequenceValidator:
    name = "example"

    def validate(self, value: object) -> str:
        sequence = str(value).strip().upper()
        if not sequence:
            raise ValueError("Empty sequence")
        return sequence
```

A feature builder prepares an inference feature matrix from a table, embedding
file, schema, and saved `FeatureRecipe`. It must return a
`PreparedFeatureMatrix` with feature names matching the saved model artifact:

```python
class ExampleFeatureSetBuilder:
    def prepare(
        self,
        *,
        recipe: FeatureRecipe,
        table_path: Path,
        embedding_path: Path,
        schema: DatasetSchema,
    ) -> PreparedFeatureMatrix:
        # Load rows and embeddings, construct features in recipe.feature_names order,
        # then return PreparedFeatureMatrix(features=..., ids=..., feature_names=...).
        ...
```

The DNA implementation uses:

```python
schema = DatasetSchema(
    id_column="No.",
    sequence_column="Sequence",
    availability_column="Nucleotide Availability",
    validator=DnaSequenceValidator(),
    feature_set_builder=DnaFeatureSetBuilder(),
)
```

For non-DNA embedding-only experiments, `NoOpRepresentationValidator` and
`GenericEmbeddingFeatureSetBuilder` can be used instead.

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

- `DnaSequenceValidator` normalises DNA strings and rejects invalid DNA sequence
  content during feature preparation.
- `NoOpRepresentationValidator` passes representation values through as
  strings for non-DNA or already-normalised embedding-only experiments.
- `DnaFeatureSetBuilder` constructs DNA-specific feature matrices, including
  embeddings, nearest-neighbour features, and availability features.
- `GenericEmbeddingFeatureSetBuilder` constructs embedding-only feature
  matrices for non-DNA representations.
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

## Dataclass Templates

The templates below show the expected shape of each configuration object.
Replace the placeholder names with paths, callbacks, validators, or builders
from the experiment being implemented.

### Dataset Schema

```python
schema = DatasetSchema(
    id_column="No.",
    sequence_column="Sequence",
    availability_column="Nucleotide Availability",
    validator=sequence_validator,
    feature_set_builder=feature_builder,
)
```

### API Endpoints

```python
api_config = ApiEndpointConfig(
    name="Example API",
    health_path="/health",
    runtime_path="/runtime",
    model_path="/model/final",
    runs_path="/runs",
    run_summary_path="/runs/<run_name>",
    run_predictions_path="/runs/<run_name>/predictions",
    predictions_path="/predictions",
    prediction_path="/predictions/<record_id>",
    recompute_path="/predictions/recompute",
    recompute_message="Use POST to recompute predictions.",
    recompute_body_example={
        "batch_size": 256,
        "table_path": "optional path",
        "embedding_path": "optional path",
        "artifact_path": "optional path",
    },
)
```

### Application Configuration

```python
app_config = ArtifactAppConfig(
    project_root=project_root,
    artifacts_dir=artifacts_dir,
    final_prediction_manifest=final_prediction_manifest,
    final_prediction_csv=final_prediction_csv,
    final_comparison_markdown=final_comparison_markdown,
    schema=schema,
    api=api_config,
    excluded_run_summaries=frozenset({"artifact_to_hide.json"}),
    prediction_postprocessor=postprocess_predictions,
)
```

### Reproducibility Configuration

```python
reproducibility_config = ReproducibilityConfig(
    canonical_model_artifact=canonical_model_artifact,
    canonical_model_summary=canonical_model_summary,
    prediction_table=prediction_table,
    prediction_embeddings=prediction_embeddings,
    canonical_ranked_predictions=canonical_ranked_predictions,
    external_summary=external_summary,
    canonical_graphs=canonical_graphs,
    output_artifact_name="model.pt",
    training_args_factory=make_training_args,
    training_runner=train_model,
    prediction_verifier=verify_predictions,
    external_verifier=verify_external_metrics,
    graph_renderer=render_graphs,
)
```

### Full Experiment

```python
experiment_config = ExperimentConfig(
    name="Example experiment",
    app=app_config,
    reproducibility=reproducibility_config,
)
```

### Feature Recipe

`FeatureRecipe` is normally loaded from a saved model checkpoint rather than
written by hand. Its shape is:

```python
feature_recipe = FeatureRecipe(
    feature_set="embeddings+nn+availability",
    nn_feature_mode="both",
    feature_names=["feature_1", "feature_2"],
    feature_mean=feature_mean_array,
    feature_std=feature_std_array,
)
```

### Model Artifact

`ModelArtifact` is also normally loaded from disk:

```python
artifact = ModelArtifact.load("training/artifacts/model.pt")
```

Its dataclass shape is:

```python
artifact = ModelArtifact(
    path=artifact_path,
    checkpoint=checkpoint_dict,
    feature_recipe=feature_recipe,
    target_column="k1",
    target_transform="log10",
)
```

### Prediction Result

```python
prediction_result = ArtifactPredictionResult(
    table_path=table_path,
    embedding_path=embedding_path,
    artifact_path=artifact_path,
    schema=schema,
    feature_recipe=feature_recipe,
    target_column="k1",
    frame=prediction_dataframe,
)
```

### Figure Theme and Output

```python
theme = FigureTheme(
    rc_params={"font.family": "serif"},
    bar_hatches=["", "///", "..."],
    line_styles=["-", "--", ":"],
    markers=["o", "s", "^"],
    dpi=220,
)

figures = FigureOutput(
    output_dir=output_dir,
    theme=theme,
)
```

Most experiments can use the provided default instead:

```python
theme = monochrome_serif_theme()
figures = FigureOutput(output_dir, theme)
```

### PDF Extraction

```python
columns = [
    ColumnSpec(name="No."),
    ColumnSpec(name="Sequence", transform=remove_whitespace),
    ColumnSpec(name="k1"),
]

table_spec = TableSpec(
    table_id=1,
    columns=columns,
    output_name="table_1.csv",
    skip_joined_header_fragments=("No. Sequence",),
)

plan = ExtractionPlan(
    tables={1: table_spec},
)

extracted = ExtractedTable(
    spec=table_spec,
    rows=[],
)
```

## Execution Engine

The dataclasses describe the experiment, while the engine classes perform the
training, prediction, and verification work.

- `PreparedFeatureMatrix` is the shared feature container produced by feature
  builders. It stores aligned record IDs, normalised representations, the
  numeric feature matrix, ordered feature names, and embedding metadata.
- `RegressionHead` is the configurable PyTorch feed-forward model. It is built
  from an input dimension, hidden dimensions, activation, normalisation mode,
  and dropout value, then ends in a single regression output.
- `PredictionService` loads a `ModelArtifact`, prepares features using the
  artifact's `FeatureRecipe`, applies the saved scaling statistics, runs the
  regression head on CPU, inverts the target transform, and returns ranked
  predictions.
- `ReproducibilityRunner` executes the configured training runner, compares
  regenerated metrics and tensors with the canonical model artifact, runs the
  prediction and external verifiers, optionally renders graphs, and returns a
  combined pass/fail report.

End-to-end flow:

```text
ExperimentConfig
→ ReproducibilityConfig
→ training_args_factory
→ training_runner
→ ModelArtifact
→ FeatureRecipe
→ feature_set_builder
→ PreparedFeatureMatrix
→ RegressionHead
→ PredictionService
→ prediction_verifier / external_verifier / graph_renderer
→ ReproducibilityRunner report
```

Minimal prediction flow:

```python
artifact = ModelArtifact.load("training/artifacts/model.pt")
service = PredictionService(artifact, schema=schema)

result = service.predict_table(
    table_path="data/example.csv",
    embedding_path="bert/example_embeddings.npz",
    batch_size=256,
)

ranked_predictions = result.frame
```

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
