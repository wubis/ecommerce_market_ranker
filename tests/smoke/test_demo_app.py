"""Headless Streamlit smoke coverage for the API-unavailable demo state."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

REPOSITORY_ROOT = Path(__file__).parents[2]
APP = REPOSITORY_ROOT / "src" / "market_rank" / "demo" / "app.py"


def test_demo_renders_controls_and_limitations_when_api_is_unavailable() -> None:
    app = AppTest.from_file(str(APP), default_timeout=10).run()

    assert not app.exception
    assert any(title.value == "MarketRank" for title in app.title)
    assert any("not ready" in warning.value for warning in app.warning)
    assert any("Dataset limitations" in expander.label for expander in app.expander)
    assert any(button.label == "Compare rankings" and button.disabled for button in app.button)
