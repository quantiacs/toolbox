import datetime
import gzip
import io
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import qnt.data as qndata
import qnt.output as qnout
import qnt.precheck as qnpre
import qnt.stats as qnstats

pytestmark = pytest.mark.usefixtures("quiet_tracking")

ASSETS = ["NYS:A0", "NYS:A1"]
Call = tuple[object, ...]


class Logs:
    def __init__(self) -> None:
        self.info: list[Call] = []
        self.err: list[Call] = []


def row_of(day: str) -> list[float]:
    value = int(day[8:10]) / 100.0
    return [value, -value]


def write_output(path: Path, days: list[str]) -> None:
    values = np.array([row_of(d) for d in days], dtype=float).reshape(len(days), 2)
    arr = xr.DataArray(
        values,
        dims=["time", "asset"],
        coords={"time": pd.DatetimeIndex(days, name="time"), "asset": ASSETS},
    )
    path.write_bytes(gzip.compress(bytes(arr.to_netcdf(compute=True))))


class Bash:
    """Fake shell command runner: records commands and optionally leaves strategy artifacts behind."""

    def __init__(self, days: list[str] | None, code: int = 0) -> None:
        self.days = days
        self.code = code
        self.commands: list[str] = []

    def __call__(self, command: str) -> int:
        self.commands.append(command)
        if self.days is not None:
            write_output(Path(qnpre.fractions_fn), self.days)
            Path(qnpre.last_data_fn).write_text("last")
            Path(qnpre.html_fn).write_text("<html></html>")
        return self.code


@pytest.fixture(autouse=True)
def logs(monkeypatch: pytest.MonkeyPatch) -> Logs:
    spy = Logs()
    monkeypatch.setattr(qnpre, "log_info", lambda *a: spy.info.append(a))
    monkeypatch.setattr(qnpre, "log_err", lambda *a: spy.err.append(a))
    return spy


@pytest.fixture(autouse=True)
def env(sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (sandbox / "id-translation.csv").write_text("server_id,user_id\nsrv0,NYS:A0\n")
    monkeypatch.setattr(qnpre, "fractions_fn", str(sandbox / "fractions.nc.gz"))
    monkeypatch.setattr(qnpre, "last_data_fn", str(sandbox / "last_data.txt"))
    monkeypatch.setattr(qnpre, "html_fn", str(sandbox / "strategy.html"))
    monkeypatch.setattr(qnpre, "result_dir", str(sandbox / "results"))
    return sandbox


class FakeProc:
    def __init__(self, code: int) -> None:
        self.stdout = io.BytesIO(b"")
        self.returncode = code
        self._code = code

    def wait(self) -> int:
        return self._code


def install_bash(monkeypatch: pytest.MonkeyPatch, bash: Bash) -> Bash:
    def fake_popen(command: str, **_kwargs: object) -> FakeProc:
        return FakeProc(bash(command))

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    return bash


# ---------------------------------------------------------------- timeouts and codes

STOCKS = ["stocks", "stocks_long", "stocks_nasdaq100", "stocks_s&p500"]
CRYPTO = [
    "crypto",
    "crypto_futures",
    "cryptofutures",
]


@pytest.mark.parametrize(
    ("data_type", "timeout"),
    [(t, 1800) for t in STOCKS]
    + [("".join(["fut", "ures"]), 600)]
    + [("".join([t[:3], t[3:]]), 300) for t in CRYPTO],
)
def test_timeout_by_data_type(
    monkeypatch: pytest.MonkeyPatch, data_type: str, timeout: int
) -> None:
    bash = install_bash(monkeypatch, Bash(None))
    qnpre.evaluate_passes(data_type, dates=["2019-02-01"])
    assert len(bash.commands) == 1
    assert f"--ExecutePreprocessor.timeout={timeout} --execute strategy.ipynb" in bash.commands[0]


@pytest.mark.parametrize("code", [-9, 1, 3])
def test_nonzero_return_code_is_reported(
    monkeypatch: pytest.MonkeyPatch, logs: Logs, code: int
) -> None:
    install_bash(monkeypatch, Bash(None, code))
    qnpre.evaluate_passes("futures", dates=["2019-02-01"])
    assert ("ERROR! Return code != 0.",) in logs.err
    assert ("return code:", code) in logs.info


def test_zero_return_code_is_not_an_error(monkeypatch: pytest.MonkeyPatch, logs: Logs) -> None:
    install_spies(monkeypatch)
    install_bash(monkeypatch, Bash(["2019-02-01"], 0))
    qnpre.evaluate_passes("futures", dates=["2019-02-01"])
    assert ("ERROR! Return code != 0.",) not in logs.err
    assert ("ERROR! Output is not found.",) not in logs.err


# ---------------------------------------------------------------- result_dir and numbering


def test_previous_results_cleanup_removes_files_only_and_numbers_passes(
    env: Path, monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    results = env / "results"
    (results / "sub").mkdir(parents=True)
    (results / "sub" / "keep.txt").write_text("keep")
    (results / "old.txt").write_text("stale")
    install_bash(monkeypatch, Bash(None))
    qnpre.evaluate_passes("futures", dates=["2019-02-01", "2019-03-01"])
    assert not (results / "old.txt").exists()
    assert (results / "sub" / "keep.txt").read_text() == "keep"
    assert ("Output directory is:", str(results)) in logs.info
    assert ("rm:", str(results) + "/old.txt") in logs.info
    assert not any(c[:1] == ("rm:",) and "sub" in str(c[1]) for c in logs.info)
    assert ("pass:", 1, "/", 2, "max_date:", "2019-02-01") in logs.info
    assert ("pass:", 2, "/", 2, "max_date:", "2019-03-01") in logs.info
    assert ("Dates:", "2019-02-01", "2019-03-01") in logs.info
    assert logs.info[-1] == ("Evaluation complete.",)


# ---------------------------------------------------------------- date selection


def liquid_data(flags: list[float]) -> xr.DataArray:
    times = pd.bdate_range("2019-01-01", periods=len(flags), name="time")
    values = np.ones((len(flags), 2, 2))
    values[:, 1, :] = np.array(flags)[:, None]
    return xr.DataArray(
        values,
        dims=["time", "field", "asset"],
        coords={"time": times, "field": ["close", "is_liquid"], "asset": ASSETS},
    )


def date_strings(data: xr.DataArray, idx: list[int]) -> list[str]:
    return [str(data.coords["time"].values[i])[:10] for i in idx]


def test_default_passes_gives_three_dates(monkeypatch: pytest.MonkeyPatch, logs: Logs) -> None:
    monkeypatch.setenv("SD_FUTURES", "2006-01-01")
    data = liquid_data([1.0] * 9)
    calls: list[tuple[str, dict[str, object]]] = []

    def load(data_type: str, **kwargs: object) -> xr.DataArray:
        calls.append((data_type, kwargs))
        return data

    monkeypatch.setattr(qndata, "load_data_by_type", load)
    bash = install_bash(monkeypatch, Bash(None))
    qnpre.evaluate_passes("futures")
    # picks: the last time, the second time and the middle one (index 4)
    assert ("Dates:", *date_strings(data, [1, 4, 8])) in logs.info
    assert len(bash.commands) == 3
    assert calls == [("futures", {"min_date": pd.Timestamp("2006-01-01").to_pydatetime()})]


def test_dates_are_taken_only_from_liquid_times_by_formula(
    monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    flags = [1.0, 1.0, 0.0, 1.0, 1.0, np.nan, 1.0, 1.0, 1.0, -1.0, 0.0]
    data = liquid_data(flags)
    monkeypatch.setattr(qndata, "load_data_by_type", lambda *_a, **_k: data)
    install_bash(monkeypatch, Bash(None))
    qnpre.evaluate_passes("futures", passes=5)
    # liquid positions 0,1,3,4,6,7,8 (7 of them), picks are -1, 1, round(7/4)=2, round(14/4)=4, round(21/4)=5
    assert ("Dates:", *date_strings(data, [1, 3, 6, 7, 8])) in logs.info


# ---------------------------------------------------------------- checks after a pass


class Spies:
    def __init__(self) -> None:
        self.exposure: list[xr.DataArray] = []
        self.non_liquid: list[tuple[object, object]] = []
        self.missed: list[tuple[object, object]] = []
        self.loads: list[tuple[str, dict[str, object]]] = []


def install_spies(
    monkeypatch: pytest.MonkeyPatch, n_non_liquid: int = 0, missed: list[str] | None = None
) -> Spies:
    spies = Spies()
    marker = xr.DataArray(np.ones((1, 1)), dims=["time", "asset"], coords={"asset": ["x"]})

    def load(data_type: str, **kwargs: object) -> xr.DataArray:
        spies.loads.append((data_type, kwargs))
        return marker

    def calc_non_liquid(data: object, output: object) -> SimpleNamespace:
        spies.non_liquid.append((data, output))
        return SimpleNamespace(time=list(range(n_non_liquid)))

    def find_missed(output: object, data: object) -> list[str]:
        spies.missed.append((output, data))
        return [] if missed is None else missed

    monkeypatch.setattr(qndata, "load_data_by_type", load)
    monkeypatch.setattr(qnstats, "check_exposure", lambda out: spies.exposure.append(out))
    monkeypatch.setattr(qnstats, "calc_non_liquid", calc_non_liquid)
    monkeypatch.setattr(qnstats, "find_missed_dates", find_missed)
    return spies


def test_stocks_pass_checks_exposure_and_loads_data_with_window(
    monkeypatch: pytest.MonkeyPatch, logs: Logs
) -> None:
    spies = install_spies(monkeypatch)
    install_bash(monkeypatch, Bash(["2019-01-31", "2019-02-01"]))
    qnpre.evaluate_passes("stocks", dates=["2019-02-01"])
    assert len(spies.exposure) == 1
    assert [str(t)[:10] for t in spies.exposure[0].coords["time"].values] == ["2019-01-31", "2019-02-01"]
    assert spies.loads == [
        (
            "stocks",
            {
                "assets": ASSETS,
                "min_date": "2019-01-31",
                "max_date": datetime.date(2019, 2, 1),
            },
        )
    ]
    assert ("Check the output...",) in logs.info
    assert ("Load data...",) in logs.info
    assert ("There are no missed dates.",) in logs.info
    assert logs.err == []


def test_futures_pass_skips_exposure_and_liquidity_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    spies = install_spies(monkeypatch)
    install_bash(monkeypatch, Bash(["2019-01-31", "2019-02-01"]))
    qnpre.evaluate_passes("futures", dates=["2019-02-01"])
    assert spies.exposure == []
    assert spies.non_liquid == []
    assert len(spies.missed) == 1
    assert len(spies.loads) == 1


def test_nasdaq_pass_checks_liquidity_but_not_exposure(monkeypatch: pytest.MonkeyPatch) -> None:
    spies = install_spies(monkeypatch)
    install_bash(monkeypatch, Bash(["2019-01-31", "2019-02-01"]))
    qnpre.evaluate_passes("stocks_nasdaq100", dates=["2019-02-01"])
    assert spies.exposure == []
    assert len(spies.non_liquid) == 1


@pytest.mark.parametrize("count", [0, 1, 2])
def test_illiquid_positions_error_only_when_found(
    monkeypatch: pytest.MonkeyPatch, logs: Logs, count: int
) -> None:
    install_spies(monkeypatch, n_non_liquid=count)
    install_bash(monkeypatch, Bash(["2019-01-31", "2019-02-01"]))
    qnpre.evaluate_passes("stocks_nasdaq100", dates=["2019-02-01"])
    assert (("ERROR! The output contains illiquid positions.",) in logs.err) is (count > 0)


@pytest.mark.parametrize("days", [["2019-02-01"], ["2019-01-31", "2019-02-01"]])
def test_nonempty_output_is_checked_for_missed_dates(
    monkeypatch: pytest.MonkeyPatch, logs: Logs, days: list[str]
) -> None:
    spies = install_spies(monkeypatch, missed=["2019-01-31"])
    install_bash(monkeypatch, Bash(days))
    qnpre.evaluate_passes("futures", dates=["2019-02-01"])
    assert len(spies.missed) == 1
    assert ("ERROR: some dates are missed in the output.", ["2019-01-31"]) in logs.err
    assert ("There are no missed dates.",) not in logs.info
    assert ("WARNING! Output is empty.",) not in logs.err


def test_empty_output_characterization(monkeypatch: pytest.MonkeyPatch) -> None:
    # input: strategy output with zero times; expected (by the code's intent): the
    # "WARNING! Output is empty." branch; actual: min_timestamp() of the empty time axis
    # raises ValueError before that branch is reached.
    install_spies(monkeypatch)
    install_bash(monkeypatch, Bash([]))
    with pytest.raises(ValueError, match="zero-size array"):
        qnpre.evaluate_passes("futures", dates=["2019-02-01"])


# ---------------------------------------------------------------- assemble_output


def put(env: Path, date: str, days: list[str]) -> None:
    results = env / "results"
    results.mkdir(exist_ok=True)
    write_output(results / f"{date}.fractions.nc.gz", days)


def fn_of(date: str) -> str:
    return qnpre.result_dir + f"/{date}.fractions.nc.gz"


def make_scenario(env: Path) -> None:
    put(env, "2019-01-29", ["2019-02-01"])  # only times after its date: empty, skipped
    put(env, "2019-02-01", ["2019-01-30"])  # single point
    put(env, "2019-02-04", ["2019-01-30", "2019-02-01", "2019-02-04", "2019-02-05"])
    put(env, "2019-02-05", ["2019-01-30", "2019-02-01"])  # nothing new
    put(env, "2019-02-06", ["2019-02-06"])  # exactly one new point
    (env / "results" / "notes.txt").write_text("ignored")


def check_result(out: xr.DataArray | None, days: list[str]) -> None:
    assert out is not None
    assert [str(t)[:10] for t in out.coords["time"].values] == days
    assert out.values.tolist() == [row_of(d) for d in days]
    assert list(out["asset"].values) == ASSETS


def test_assemble_all_mode_skips_empty_pieces_and_logs_ranges(env: Path, logs: Logs) -> None:
    make_scenario(env)
    mode = "".join(["a", "ll"])
    out = qnpre.assemble_output(mode)
    check_result(out, ["2019-01-30", "2019-02-01", "2019-02-04", "2019-02-06"])
    assert ("Merge outputs...",) in logs.info
    merges = [c for c in logs.info if c[0] in ("init output:", "add output:")]
    assert merges == [
        ("init output:", fn_of("2019-02-01"), "2019-01-30", "2019-01-30"),
        ("add output:", fn_of("2019-02-04"), "2019-02-01", "2019-02-04"),
        ("add output:", fn_of("2019-02-06"), "2019-02-06", "2019-02-06"),
    ]
    assert logs.err == []


def test_assemble_one_mode_takes_only_the_pass_day(env: Path, logs: Logs) -> None:
    make_scenario(env)
    mode = "".join(["o", "ne"])
    out = qnpre.assemble_output(mode)
    check_result(out, ["2019-01-30", "2019-02-04", "2019-02-06"])
    merges = [c for c in logs.info if c[0] in ("init output:", "add output:")]
    assert merges == [
        ("init output:", fn_of("2019-02-01"), "2019-01-30", "2019-01-30"),
        ("add output:", fn_of("2019-02-04"), "2019-02-04", "2019-02-04"),
        ("add output:", fn_of("2019-02-06"), "2019-02-06", "2019-02-06"),
    ]
    assert logs.err == []


@pytest.mark.parametrize("mode", ["abc", "all2", "bogus", "one2", "zzz", "".join(["O", "ne"])])
def test_assemble_rejects_wrong_mode(env: Path, mode: str) -> None:
    put(env, "2019-02-01", ["2019-01-30"])
    put(env, "2019-02-04", ["2019-02-04"])
    with pytest.raises(Exception, match=r"^wrong add_mode$"):
        qnpre.assemble_output(mode)


def test_assemble_without_files_logs_error_and_returns_none(env: Path, logs: Logs) -> None:
    (env / "results").mkdir()
    (env / "results" / "other.txt").write_text("x")
    assert qnpre.assemble_output() is None
    assert ("ERROR! There are no outputs.",) in logs.err


# ---------------------------------------------------------------- check_output


def checked_output(days: list[str]) -> xr.DataArray:
    values = np.array([row_of(d) for d in days])
    return xr.DataArray(
        values,
        dims=["time", "asset"],
        coords={"time": pd.DatetimeIndex(days, name="time"), "asset": ASSETS},
    )


CHECK_DAYS = ["2019-01-08", "2019-01-09", "2019-01-10", "2019-01-11", "2019-01-14", "2019-01-15"]


class CheckEnv:
    def __init__(self) -> None:
        self.loads: list[tuple[str, dict[str, object]]] = []
        self.checks: list[tuple[object, object]] = []
        self.data = object()


@pytest.fixture
def check_env(monkeypatch: pytest.MonkeyPatch) -> CheckEnv:
    state = CheckEnv()
    monkeypatch.setenv("SD_FUTURES", "2019-01-10")

    def load(data_type: str, **kwargs: object) -> object:
        state.loads.append((data_type, kwargs))
        return state.data

    def check(output: object, data: object) -> None:
        state.checks.append((output, data))

    monkeypatch.setattr(qndata, "load_data_by_type", load)
    monkeypatch.setattr(qnout, "check", check)
    return state


@pytest.mark.parametrize(
    ("points", "ok"), [("2", True), ("3", True), ("4", False)], ids=["more", "exact", "fewer"]
)
def test_in_sample_points_boundary_and_message(
    check_env: CheckEnv, monkeypatch: pytest.MonkeyPatch, logs: Logs, points: str, ok: bool
) -> None:
    monkeypatch.setenv("IS_FUTURES", points)
    output = checked_output(CHECK_DAYS)
    qnpre.check_output(output, "futures")
    # points strictly after 2019-01-10 are 01-11, 01-14, 01-15
    if ok:
        assert (f"Ok. In sample period contains enough points.3 >= {points}",) in logs.info
        assert not any("ERROR" in str(c[:1]) for c in logs.err)
    else:
        assert ("ERROR! In sample period does not contain enough points. 3 < 4",) in logs.err
        assert not any(str(c[:1]).startswith("('Ok.") for c in logs.info)
    assert ("Load data...",) in logs.info
    assert check_env.checks == [(output, check_env.data)]


def test_check_output_loads_data_sixty_days_before_start(
    check_env: CheckEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IS_FUTURES", "3")
    qnpre.check_output(checked_output(CHECK_DAYS), "futures")
    assert check_env.loads == [
        ("futures", {"assets": ASSETS, "min_date": pd.Timestamp("2018-11-11").to_pydatetime()})
    ]


def test_unsupported_type_logs_and_skips_loading(check_env: CheckEnv, logs: Logs) -> None:
    qnpre.check_output(checked_output(CHECK_DAYS), "bonds")
    assert logs.err == [("Unsupported data_type", "bonds")]
    assert check_env.loads == []
    assert check_env.checks == []
