"""Test fixtures for the test_agents package."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_session_state(tmp_path, monkeypatch):
    """Redirect SessionController's state file to a temp directory.

    The SessionController persists state to data/session_state.json and
    restores it on start_session() when the date matches today. This causes
    production state to bleed into tests. Redirecting to a fresh tmp_path
    guarantees each test starts with a clean slate.
    """
    import src.agents.session_controller as sc_module

    monkeypatch.setattr(sc_module, "_STATE_FILE", str(tmp_path / "session_state.json"))
