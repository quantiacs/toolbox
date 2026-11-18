import datetime
import gzip
import io
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

import qnt.data as qndata
import qnt.output as qnout
import qnt.precheck as qnpre
from qnt.tests.hl_server import FakeServer
from qnt.tests.hl_support import make_data, make_weights

pytestmark = pytest.mark.usefixtures("quiet_tracking")

DATES = ["2019-02-01", "2019-03-01", "2019-04-01"]


class FakeProc:
    def __init__(self, code: int) -> None:
        self.stdout = io.BytesIO(b"")
        self.returncode = code
        self._code = code

    def wait(self) -> int:
        return self._code


def patch_popen(monkeypatch: pytest.MonkeyPatch, run: Callable[[str], int]) -> None:
    def fake_popen(command: str, **_kwargs: object) -> FakeProc:
        return FakeProc(run(command))

    monkeypatch.setattr(subprocess, "Popen", fake_popen)


class Workspace:
    def __init__(self, root: Path, data: xr.DataArray) -> None:
        self.root = root
        self.data = data
        self.commands: list[str] = []
        self.weights = qnout.normalize(make_weights(data, seed=14))

    def fake_bash(self, command: str) -> int:
        self.commands.append(command)
        if "init.ipynb" in command:
            return 0
        found = re.search(r"/last/(\d{4}-\d{2}-\d{2})/", command)
        assert found is not None
        until = np.datetime64(found.group(1))
        upto = self.weights.sel(time=slice(None, str(until)))
        Path(qnpre.fractions_fn).write_bytes(gzip.compress(bytes(upto.to_netcdf(compute=True))))
        Path(qnpre.last_data_fn).write_text("last")
        Path(qnpre.html_fn).write_text("<html></html>")
        return 0

    def load(self, data_type: str, **kwargs: object) -> xr.DataArray:
        assert data_type == "futures"
        window = self.data
        assets = kwargs.get("assets")
        if assets is not None:
            window = window.sel(asset=assets)
        start = kwargs.get("min_date")
        stop = kwargs.get("max_date")
        return window.sel(time=slice(str(start) if start else None, str(stop) if stop else None))


@pytest.fixture
def workspace(sandbox: Path, server: FakeServer, monkeypatch: pytest.MonkeyPatch) -> Workspace:
    server.serve_json("referee/submission/forCorrelation", [])
    work = Workspace(sandbox, make_data(n_assets=4, n_days=120, name="futures", seed=31))
    monkeypatch.setattr(qnpre, "fractions_fn", str(sandbox / "fractions.nc.gz"))
    monkeypatch.setattr(qnpre, "last_data_fn", str(sandbox / "last_data.txt"))
    monkeypatch.setattr(qnpre, "html_fn", str(sandbox / "strategy.html"))
    monkeypatch.setattr(qnpre, "result_dir", str(sandbox / "precheck_results"))
    patch_popen(monkeypatch, work.fake_bash)
    monkeypatch.setattr(qndata, "load_data_by_type", work.load)
    return work


def test_evaluate_passes_runs_strategy_per_date_and_archives_results(
    workspace: Workspace, sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stale = sandbox / "precheck_results"
    stale.mkdir()
    (stale / "old.txt").write_text("stale")

    qnpre.evaluate_passes("futures", dates=DATES)

    names = sorted(p.name for p in stale.iterdir())
    assert names == sorted(
        f"{d}.{suffix}"
        for d in DATES
        for suffix in ("fractions.nc.gz", "last_data.txt", "strategy.html")
    )
    assert len(workspace.commands) == 3
    command = workspace.commands[1]
    assert "DATA_BASE_URL=https://data-api.quantiacs.io/last/2019-03-01/" in command
    assert "SUBMISSION_ID=-1" in command
    assert "--ExecutePreprocessor.timeout=600" in command
    assert not Path(qnpre.fractions_fn).exists()
    report = capsys.readouterr()
    assert report.out.count("There are no missed dates.") == 3
    assert "ERROR" not in report.err
    assert "Dates: 2019-02-01 2019-03-01 2019-04-01" in report.out


def test_assemble_output_merges_passes_in_both_modes(workspace: Workspace) -> None:
    qnpre.evaluate_passes("futures", dates=DATES)
    whole = qnpre.assemble_output()
    assert whole is not None
    assert whole.shape == (int(np.busday_count("2019-01-01", "2019-04-02")), 4)
    assert_array_equal(whole.coords["time"].values, workspace.weights.coords["time"].values[: whole.sizes["time"]])
    assert_allclose(
        whole.values, workspace.weights.sel(time=slice(None, "2019-04-01")).values
    )

    single_days = qnpre.assemble_output("one")
    assert single_days is not None
    first_block = int(np.busday_count("2019-01-01", "2019-02-02"))
    assert single_days.sizes["time"] == first_block + 2
    assert [str(t)[:10] for t in single_days.coords["time"].values[-3:]] == DATES
    assert_allclose(
        single_days.values[-2:], workspace.weights.sel(time=DATES[1:]).values
    )
    with pytest.raises(Exception, match="wrong add_mode"):
        qnpre.assemble_output("sideways")


def test_assemble_output_without_results_reports_error(
    workspace: Workspace, sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (sandbox / "precheck_results").mkdir()
    assert qnpre.assemble_output() is None
    assert "ERROR! There are no outputs." in capsys.readouterr().err
    assert len(workspace.commands) == 0


def test_evaluate_passes_picks_dates_from_the_data(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    qnpre.evaluate_passes("futures", passes=4)
    times = workspace.data.coords["time"].values
    expected = sorted(
        {
            str(t)[:10]
            for t in (
                times[-1],
                times[1],
                times[round(len(times) * 1 / 3)],
                times[round(len(times) * 2 / 3)],
            )
        }
    )
    assert f"Dates: {' '.join(expected)}" in capsys.readouterr().out
    assert len(workspace.commands) == len(expected)


def test_init_notebook_is_executed_only_when_present(workspace: Workspace, sandbox: Path) -> None:
    qnpre.run_init()
    assert workspace.commands == []
    (sandbox / "init.ipynb").write_text("{}")
    qnpre.run_init()
    assert len(workspace.commands) == 1
    assert "nbconvert --to html" in workspace.commands[0]


@pytest.mark.usefixtures("workspace")
def test_load_output_adds_missing_time_and_translates_assets(sandbox: Path) -> None:
    (sandbox / "id-translation.csv").write_text("server_id,user_id\nsrv0,NYS:A0\n")
    single = xr.DataArray([0.5, -0.5], dims=["asset"], coords={"asset": ["srv0", "other"]})
    path = sandbox / "single.nc"
    path.write_bytes(bytes(single.to_netcdf(compute=True)))
    loaded = qnpre.load_output(str(path), datetime.date(2019, 3, 5))
    assert loaded.dims == ("time", "asset")
    assert str(loaded.coords["time"].values[0])[:10] == "2019-03-05"
    assert list(loaded.coords["asset"].values) == ["NYS:A0", "other"]
    assert_allclose(loaded.values, [[0.5, -0.5]])


def test_check_output_reports_in_sample_length_and_runs_output_check(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    qnpre.check_output(workspace.weights, "bonds")
    assert "Unsupported data_type bonds" in capsys.readouterr().err

    qnpre.check_output(workspace.weights, "futures")
    report = capsys.readouterr()
    assert "ERROR! In sample period does not contain enough points. 120 < 3528" in report.err
    assert "Load data..." in report.out
    assert "Check liquidity..." in report.out
    assert "Sharpe Ratio =" in report.out
    assert "Ok. This strategy does not correlate with other strategies." in report.out
