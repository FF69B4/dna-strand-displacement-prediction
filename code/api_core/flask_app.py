from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from flask import Flask, jsonify, request


RuntimeProvider = Callable[[], Any]


class JsonApiService(Protocol):
    def health_details(self) -> dict[str, Any]:
        ...

    def model_info(self) -> dict[str, Any]:
        ...

    def list_runs(self) -> dict[str, Any]:
        ...

    def get_run_summary(self, run_name: str) -> dict[str, Any] | None:
        ...

    def get_run_predictions(self, run_name: str, limit: int, offset: int) -> dict[str, Any]:
        ...

    def list_predictions(self, limit: int, offset: int) -> dict[str, Any]:
        ...

    def recompute_predictions(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...

    def get_prediction(self, record_id: str) -> dict[str, Any] | None:
        ...


@dataclass(frozen=True)
class ApiEndpointConfig:
    name: str
    health_path: str = "/health"
    runtime_path: str = "/runtime"
    model_path: str = "/model/final"
    runs_path: str = "/runs"
    run_summary_path: str = "/runs/<run_name>"
    run_predictions_path: str = "/runs/<run_name>/predictions"
    predictions_path: str = "/predictions"
    prediction_path: str = "/predictions/<record_id>"
    recompute_path: str = "/predictions/recompute"
    recompute_message: str = "Use POST to recompute predictions."
    recompute_body_example: dict[str, Any] = field(
        default_factory=lambda: {
            "batch_size": 256,
            "table_path": "optional path",
            "embedding_path": "optional path",
            "artifact_path": "optional path",
        }
    )

    def index_endpoints(self) -> dict[str, Any]:
        return {
            "health": self.health_path,
            "runtime": self.runtime_path,
            "model": self.model_path,
            "runs": self.runs_path,
            "run_summary": self.run_summary_path,
            "run_predictions": f"{self.run_predictions_path}?limit=25&offset=0",
            "predictions": f"{self.predictions_path}?limit=25&offset=0",
            "prediction_by_id": self.prediction_path,
            "recompute_predictions": {
                "method": "POST",
                "path": self.recompute_path,
            },
        }


def _error_response(message: str, status_code: int = 400):
    return jsonify({"error": message}), status_code


def _pagination_args() -> tuple[int, int]:
    return (
        request.args.get("limit", default=25, type=int),
        request.args.get("offset", default=0, type=int),
    )


def create_json_api_app(
    service: JsonApiService,
    config: ApiEndpointConfig,
    runtime_provider: RuntimeProvider | None = None,
) -> Flask:
    app = Flask(__name__)

    @app.route("/", methods=["GET"])
    def index():
        return jsonify(
            {
                "name": config.name,
                "status": "running",
                "endpoints": config.index_endpoints(),
            }
        )

    @app.errorhandler(FileNotFoundError)
    def handle_missing_file(error: FileNotFoundError):
        return _error_response(str(error), 404)

    @app.errorhandler(ValueError)
    def handle_value_error(error: ValueError):
        return _error_response(str(error), 400)

    @app.errorhandler(RuntimeError)
    def handle_runtime_error(error: RuntimeError):
        return _error_response(str(error), 500)

    @app.route(config.health_path, methods=["GET"])
    def health():
        payload = {"status": "ok"}
        if runtime_provider is not None:
            runtime = runtime_provider()
            payload["runtime_running"] = runtime.is_running
        payload.update(service.health_details())
        return jsonify(payload)

    if runtime_provider is not None:

        @app.route(config.runtime_path, methods=["GET"])
        def runtime_info():
            runtime = runtime_provider()
            return jsonify(
                {
                    "is_running": runtime.is_running,
                    "state": runtime.get_state(),
                }
            )

    @app.route(config.model_path, methods=["GET"])
    def model_info():
        return jsonify(service.model_info())

    @app.route(config.runs_path, methods=["GET"])
    def runs():
        return jsonify(service.list_runs())

    @app.route(config.run_summary_path, methods=["GET"])
    def run_summary(run_name: str):
        summary = service.get_run_summary(run_name)
        if summary is None:
            return _error_response(f"No run summary found for {run_name!r}", 404)
        return jsonify(summary)

    @app.route(config.run_predictions_path, methods=["GET"])
    def run_predictions(run_name: str):
        limit, offset = _pagination_args()
        return jsonify(service.get_run_predictions(run_name, limit, offset))

    @app.route(config.predictions_path, methods=["GET"])
    def predictions():
        limit, offset = _pagination_args()
        return jsonify(service.list_predictions(limit, offset))

    @app.route(config.recompute_path, methods=["GET"])
    def recompute_predictions_info():
        return jsonify(
            {
                "message": config.recompute_message,
                "method": "POST",
                "path": config.recompute_path,
                "json_body": config.recompute_body_example,
            }
        )

    @app.route(config.recompute_path, methods=["POST"])
    def recompute_predictions():
        payload = request.get_json(silent=True) or {}
        return jsonify(service.recompute_predictions(payload))

    @app.route(config.prediction_path, methods=["GET"])
    def prediction(record_id: str):
        item = service.get_prediction(record_id)
        if item is None:
            return _error_response(f"No prediction found for id {record_id!r}", 404)
        return jsonify(item)

    return app
