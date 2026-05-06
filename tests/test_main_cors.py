from fastapi.testclient import TestClient

from app.config.settings import load_settings
from app.main import create_app


def test_cors_preflight_is_allowed_when_enabled() -> None:
    base_settings = load_settings()
    settings = base_settings.model_copy(
        update={
            "server": base_settings.server.model_copy(
                update={"enable_cors": True}
            )
        }
    )
    client = TestClient(create_app(settings))

    response = client.options(
        "/v1/chat/completions",
        headers={
            "Origin": "app://obsidian.md",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,authorization",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "app://obsidian.md"
    assert "POST" in response.headers["access-control-allow-methods"]
