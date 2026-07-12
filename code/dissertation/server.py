from __future__ import annotations

from api_core import create_json_api_app
from dissertation.api_service import DissertationApiService, dissertation_api_config
from runtime.runtime import get_runtime


def create_app():
    return create_json_api_app(
        DissertationApiService(),
        dissertation_api_config(),
        runtime_provider=get_runtime,
    )


app = create_app()


def main():
    global runtime
    runtime = get_runtime()
    app.run(debug=True, host="0.0.0.0", port=5000)


if __name__ == "__main__":
    runtime = get_runtime()
    main()
