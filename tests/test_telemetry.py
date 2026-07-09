"""Opt-in telemetry: disabled by default, exact payload, on/off/status."""

import json
import sys
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from whetkit import telemetry
from whetkit.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch):
    """Point ~ at a temp dir and clear the env override for every test."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("WHETKIT_TELEMETRY", raising=False)
    return tmp_path


def _forbid_http(monkeypatch) -> list:
    calls: list = []

    def _boom(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("no HTTP request may be attempted")

    monkeypatch.setattr(httpx, "post", _boom)
    return calls


class TestDisabledByDefault:
    def test_is_enabled_false_with_no_config(self) -> None:
        assert telemetry.is_enabled() is False

    def test_record_event_is_a_noop(self, monkeypatch) -> None:
        calls = _forbid_http(monkeypatch)
        assert telemetry.record_event("run") is None
        assert calls == []

    def test_cli_command_attempts_no_http(self, tmp_path: Path, monkeypatch) -> None:
        calls = _forbid_http(monkeypatch)
        # a cheap command that goes through the app callback (the hook)
        result = runner.invoke(app, ["diff", str(tmp_path / "a.json"), str(tmp_path / "b.json")])
        assert result.exit_code != 0  # missing files — irrelevant here
        assert calls == []


class TestOnOffStatus:
    def test_roundtrip(self, isolated_home: Path) -> None:
        config_file = isolated_home / ".whetkit" / "telemetry.json"

        result = runner.invoke(app, ["telemetry", "on"])
        assert result.exit_code == 0, result.output
        assert "telemetry enabled" in result.output
        config = json.loads(config_file.read_text())
        assert config["enabled"] is True
        assert config["anonymous_id"]  # created on opt-in
        assert telemetry.is_enabled() is True

        result = runner.invoke(app, ["telemetry", "status"])
        assert result.exit_code == 0
        assert "telemetry is enabled" in result.output

        result = runner.invoke(app, ["telemetry", "off"])
        assert result.exit_code == 0
        assert "telemetry disabled" in result.output
        assert json.loads(config_file.read_text())["enabled"] is False
        assert telemetry.is_enabled() is False

        result = runner.invoke(app, ["telemetry", "status"])
        assert "telemetry is disabled" in result.output

    def test_status_prints_exactly_what_is_collected(self) -> None:
        result = runner.invoke(app, ["telemetry", "status"])
        assert result.exit_code == 0
        for phrase in ("command name", "whetkit version", "Python major.minor", "sys.platform"):
            assert phrase in result.output
        assert "never collected" in result.output
        assert "prompts" in result.output

    def test_bad_action_rejected(self) -> None:
        result = runner.invoke(app, ["telemetry", "sometimes"])
        assert result.exit_code == 2

    def test_env_var_overrides_config(self, monkeypatch) -> None:
        runner.invoke(app, ["telemetry", "on"])
        monkeypatch.setenv("WHETKIT_TELEMETRY", "0")
        assert telemetry.is_enabled() is False
        monkeypatch.setenv("WHETKIT_TELEMETRY", "1")
        assert telemetry.is_enabled() is True


class TestPayload:
    def test_payload_holds_exactly_the_documented_fields(self) -> None:
        from importlib.metadata import version

        payload = telemetry.build_payload("run")
        assert set(payload) == {"api_key", "event", "distinct_id", "properties"}
        assert payload["api_key"] == telemetry.POSTHOG_PROJECT_KEY
        assert payload["event"] == "cli_command"
        assert set(payload["properties"]) == {
            "command",
            "whetkit_version",
            "python_version",
            "platform",
        }
        assert payload["properties"]["command"] == "run"
        assert payload["properties"]["whetkit_version"] == version("whetkit")
        expected_python = f"{sys.version_info.major}.{sys.version_info.minor}"
        assert payload["properties"]["python_version"] == expected_python
        assert payload["properties"]["platform"] == sys.platform

    def test_anonymous_id_is_a_persisted_uuid(self) -> None:
        import uuid

        first = telemetry.anonymous_id()
        uuid.UUID(first)  # must parse as a UUID
        assert telemetry.anonymous_id() == first  # stable across calls
        assert telemetry.build_payload("run")["distinct_id"] == first


class TestSending:
    def test_record_event_posts_when_enabled(self, monkeypatch) -> None:
        monkeypatch.setenv("WHETKIT_TELEMETRY", "1")
        captured: dict = {}

        def fake_post(url, *, json=None, timeout=None):
            captured.update(url=url, payload=json, timeout=timeout)

        monkeypatch.setattr(httpx, "post", fake_post)
        thread = telemetry.record_event("curate")
        assert thread is not None
        thread.join(timeout=5)
        assert captured["url"] == telemetry.POSTHOG_CAPTURE_URL
        assert captured["timeout"] == 2.0
        assert captured["payload"]["properties"]["command"] == "curate"

    def test_failures_are_swallowed(self, monkeypatch) -> None:
        monkeypatch.setenv("WHETKIT_TELEMETRY", "1")

        def fake_post(*args, **kwargs):
            raise httpx.ConnectError("offline")

        monkeypatch.setattr(httpx, "post", fake_post)
        thread = telemetry.record_event("run")
        assert thread is not None
        thread.join(timeout=5)  # must not raise anywhere

    def test_cli_hook_records_the_command_name(self, monkeypatch) -> None:
        recorded: list[str] = []
        monkeypatch.setattr(telemetry, "record_event", lambda cmd: recorded.append(cmd))
        runner.invoke(app, ["doctor", "--server", "typo.json"])  # fails fast, hook already ran
        assert recorded == ["doctor"]

    def test_cli_hook_skips_the_telemetry_command_itself(self, monkeypatch) -> None:
        recorded: list[str] = []
        monkeypatch.setattr(telemetry, "record_event", lambda cmd: recorded.append(cmd))
        runner.invoke(app, ["telemetry", "status"])
        assert recorded == []
