from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING

import pytest
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from qnt.tests.hl_server import FakeServer

os.environ.setdefault("API_KEY", "default")


def _no_sleep(_seconds: float) -> None:
    return None


class _EveryEventSent(set[str]):
    @override
    def __contains__(self, item: object) -> bool:
        return True


@pytest.fixture(scope="module")
def quiet_tracking() -> Iterator[None]:
    """Disable network usage-tracking events for a whole test module."""
    common = importlib.import_module("qnt.data.common")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(common, "sent_events", _EveryEventSent())
        yield


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Run in an empty directory: no network tracking, local cache, local output files."""
    common = importlib.import_module("qnt.data.common")
    translation = importlib.import_module("qnt.data.id_translation")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OUTPUT_PATH", str(tmp_path / "fractions.nc.gz"))
    monkeypatch.setenv("OUT_STATE_PATH", str(tmp_path / "state.out.pickle.gz"))
    monkeypatch.setenv("IN_STATE_PATH", str(tmp_path / "state.in.pickle.gz"))
    for name in ("SUBMISSION_ID", "MULTI_PASS_MODE_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(common, "sent_events", _EveryEventSent())
    monkeypatch.setattr(common, "CACHE_DIR", str(tmp_path / "data-cache"))
    monkeypatch.setattr(translation, "server_id_to_user_id", None)
    monkeypatch.setattr(translation, "user_id_to_server_id", None)
    yield tmp_path


@pytest.fixture
def server(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> FakeServer:
    """Fake HTTP data server, installed instead of urllib.request.urlopen."""
    request.getfixturevalue("sandbox")
    common = importlib.import_module("qnt.data.common")
    translation = importlib.import_module("qnt.data.id_translation")
    stocks = importlib.import_module("qnt.data.stocks")
    fake: FakeServer = importlib.import_module("qnt.tests.hl_server").FakeServer()
    monkeypatch.setattr("urllib.request.urlopen", fake)
    monkeypatch.setattr(common, "cache_min_mod_time", None)
    monkeypatch.setattr(common.time, "sleep", _no_sleep)
    monkeypatch.setattr(translation, "USE_ID_TRANSLATION", True)
    monkeypatch.setattr(stocks, "FIRST", True)
    return fake
