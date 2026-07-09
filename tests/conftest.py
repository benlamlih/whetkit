import pytest


@pytest.fixture(autouse=True)
def _no_telemetry(monkeypatch):
    """The suite must never send telemetry, even on an opted-in dev machine.
    (tests/test_telemetry.py re-enables it explicitly where needed.)"""
    monkeypatch.setenv("WHETKIT_TELEMETRY", "0")
