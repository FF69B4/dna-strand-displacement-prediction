from __future__ import annotations

from experiment_core import ArtifactPredictionPipeline

from dissertation.config import CONFIG


PIPELINE = ArtifactPredictionPipeline(CONFIG.app)
FINAL_PREDICTIONS_JSON = CONFIG.app.final_prediction_manifest
FINAL_PREDICTIONS_CSV = CONFIG.app.final_prediction_csv

final_model_info = PIPELINE.final_model_info
list_table3_predictions = PIPELINE.list_predictions
get_table3_prediction = PIPELINE.get_prediction
list_runs = PIPELINE.list_runs
get_run_summary = PIPELINE.get_run_summary
get_run_predictions = PIPELINE.get_run_predictions
recompute_table_predictions = PIPELINE.recompute_predictions
