from __future__ import annotations

from typing import Any

from dissertation import pipeline
from dissertation.config import CONFIG


class DissertationApiService:
    def health_details(self) -> dict[str, Any]:
        return {
            "pipeline": {
                "final_predictions_available": pipeline.FINAL_PREDICTIONS_CSV.exists(),
                "final_manifest_available": pipeline.FINAL_PREDICTIONS_JSON.exists(),
            },
        }

    def model_info(self) -> dict[str, Any]:
        return pipeline.final_model_info()

    def list_runs(self) -> dict[str, Any]:
        return pipeline.list_runs()

    def get_run_summary(self, run_name: str) -> dict[str, Any] | None:
        return pipeline.get_run_summary(run_name)

    def get_run_predictions(self, run_name: str, limit: int, offset: int) -> dict[str, Any]:
        return pipeline.get_run_predictions(run_name, limit=limit, offset=offset)

    def list_predictions(self, limit: int, offset: int) -> dict[str, Any]:
        return pipeline.list_table3_predictions(limit=limit, offset=offset)

    def recompute_predictions(self, payload: dict[str, Any]) -> dict[str, Any]:
        return pipeline.recompute_table_predictions(
            table_path=payload.get("table_path"),
            embedding_path=payload.get("embedding_path"),
            artifact_path=payload.get("artifact_path"),
            batch_size=int(payload.get("batch_size", 256)),
        )

    def get_prediction(self, record_id: str) -> dict[str, Any] | None:
        return pipeline.get_table3_prediction(record_id)


def dissertation_api_config():
    return CONFIG.app.api
