import gzip
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

import qnt.output as qnout
from qnt.tests.hl_support import make_data, make_weights, renamed_assets

pytestmark = pytest.mark.usefixtures("quiet_tracking")


@pytest.fixture(scope="module")
def data() -> xr.DataArray:
    return make_data(name="futures", n_days=40)


@pytest.fixture(scope="module")
def weights(data: xr.DataArray) -> xr.DataArray:
    return make_weights(data)


@pytest.fixture(scope="module")
def baseline(data: xr.DataArray, weights: xr.DataArray) -> xr.DataArray:
    return qnout.clean(weights, data, "futures", debug=False)


def with_extra_asset(arr: xr.DataArray) -> xr.DataArray:
    extra = renamed_assets(arr.isel(asset=[0]), ["ZZZ"])
    return xr.concat([arr, extra], "asset")


def with_duplicate_day(arr: xr.DataArray) -> xr.DataArray:
    return xr.concat([arr, arr.isel(time=[0])], "time")


def shuffled(arr: xr.DataArray) -> xr.DataArray:
    order = np.random.default_rng(1).permutation(arr.sizes["time"])
    return arr.isel(time=order, asset=[3, 1, 0, 2, 4, 5, 6, 7])


@pytest.mark.parametrize("damage", [with_extra_asset, with_duplicate_day, shuffled])
def test_clean_repairs_structure_to_the_same_result(
    damage: Callable[[xr.DataArray], xr.DataArray],
    weights: xr.DataArray,
    data: xr.DataArray,
    baseline: xr.DataArray,
) -> None:
    cleaned = qnout.clean(damage(weights), data, "futures", debug=False)
    assert cleaned.dims == ("time", "asset")
    assert_array_equal(cleaned.coords["time"].values, baseline.coords["time"].values)
    assert list(cleaned.coords["asset"].values) == list(baseline.coords["asset"].values)
    assert_allclose(cleaned.values, baseline.values, atol=1e-14)


def test_clean_golden_baseline(baseline: xr.DataArray) -> None:
    values = baseline.values
    assert baseline.shape == (40, 8)
    assert_array_equal(values[:8, 0], 0.0)
    assert_allclose(np.abs(values).sum(axis=1), 1.0, rtol=1e-12)
    assert_allclose(values[8, :4], [0.40603049, -0.11923358, 0.05676762, -0.0063814], atol=1e-8)


def test_clean_neutralizes_infinite_values(weights: xr.DataArray, data: xr.DataArray) -> None:
    broken = weights.copy()
    broken[3, 2] = np.inf
    broken[4, 1] = -np.inf
    zeroed = weights.copy()
    zeroed[3, 2] = 0.0
    zeroed[4, 1] = 0.0
    cleaned = qnout.clean(broken, data, "futures", debug=False)
    expected = qnout.clean(zeroed, data, "futures", debug=False)
    assert np.isfinite(cleaned.values).all()
    assert_allclose(cleaned.values, expected.values, atol=1e-14)


def test_clean_flattens_dates_missing_from_output(
    weights: xr.DataArray, data: xr.DataArray, baseline: xr.DataArray
) -> None:
    kept = [i for i in range(40) if i not in (10, 11)]
    cleaned = qnout.clean(weights.isel(time=kept), data, "futures", debug=False)
    values = cleaned.values
    assert cleaned.shape == (40, 8)
    assert_array_equal(values[10:12], 0.0)
    assert_allclose(
        np.delete(values, [10, 11], axis=0), np.delete(baseline.values, [10, 11], axis=0)
    )


def test_clean_single_day_output_is_placed_on_last_data_date(
    weights: xr.DataArray, data: xr.DataArray
) -> None:
    cleaned = qnout.clean(weights.isel(time=-1, drop=True), data, "futures", debug=False)
    assert cleaned.dims == ("time", "asset")
    assert cleaned.shape == (1, 8)
    assert cleaned.coords["time"].values[0] == data.coords["time"].values[-1]
    assert_allclose(np.abs(cleaned.values).sum(), 1.0)


def test_clean_empty_output_stays_empty(weights: xr.DataArray, data: xr.DataArray) -> None:
    cleaned = qnout.clean(weights.isel(time=slice(0, 0)), data, "futures", debug=False)
    assert cleaned.shape == (0, 8)


def test_clean_without_liquidity_field_keeps_illiquid_positions(
    weights: xr.DataArray, data: xr.DataArray, capsys: pytest.CaptureFixture[str]
) -> None:
    fields = [name for name in data.coords["field"].values if name != "is_liquid"]
    reduced = data.sel(field=fields)
    cleaned = qnout.clean(weights, reduced, "futures", debug=True)
    assert "'is_liquid' field not found" in capsys.readouterr().out
    assert np.abs(cleaned.values[:8, 0]).sum() > 0.1


def test_clean_logs_liquidity_fix_and_kind_mismatch(
    weights: xr.DataArray, data: xr.DataArray, capsys: pytest.CaptureFixture[str]
) -> None:
    qnout.clean(weights, data, "stocks_long", debug=True)
    captured = capsys.readouterr()
    assert (
        "The kind of the data is futures and the kind of the output is stocks_long" in captured.err
    )
    assert "Strategy trades non-liquid assets" in captured.out
    assert "Applying liquidity filter" in captured.out
    assert "Check positive positions" in captured.out


def test_clean_stocks_long_drops_shorts_and_oversized_days() -> None:
    market = make_data(n_assets=40, n_days=40, name="stocks_long")
    cleaned = qnout.clean(make_weights(market), market, "stocks_long", debug=False)
    values = cleaned.values
    assert values.min() == 0.0
    assert 0.0 < values.max() < 0.049
    totals = np.abs(values).sum(axis=1)
    assert totals.min() == 0.0
    assert_allclose(totals.max(), 1.0, rtol=1e-12)


def test_clean_stocks_keeps_longs_and_shorts_under_exposure_limit() -> None:
    market = make_data(n_assets=40, n_days=40, name="stocks")
    cleaned = qnout.clean(make_weights(market), market, "stocks", debug=False)
    values = cleaned.values
    assert values.min() < 0.0 < values.max()
    assert np.abs(values).max() < 0.049
    assert_allclose(np.abs(values).sum(axis=1), 1.0, rtol=1e-12)


def test_clean_crypto_keeps_only_btc() -> None:
    market = make_data(name="crypto", n_days=40)
    renamed = renamed_assets(market, ["BTC", *(str(a) for a in market.coords["asset"].values[1:])])
    cleaned = qnout.clean(make_weights(renamed), renamed, "crypto", debug=False)
    assert list(cleaned.coords["asset"].values) == ["BTC"]
    assert float(np.abs(cleaned.values).max()) <= 1.0


def test_normalize_scales_only_oversized_rows(weights: xr.DataArray) -> None:
    factor = xr.DataArray(np.where(np.arange(40) % 2 == 0, 0.05, 1.0), dims=["time"])
    mixed = weights * factor
    normalized = qnout.normalize(mixed)
    raw = np.nan_to_num(mixed.values)
    values = normalized.values
    raw_total = np.abs(raw).sum(axis=1)
    assert_allclose(values, raw / np.maximum(raw_total, 1.0)[:, None], rtol=1e-12)
    assert (raw_total < 1.0).any()
    assert (raw_total > 1.0).any()
    assert np.isfinite(values).all()


def test_normalize_per_asset_clips_each_position() -> None:
    arr = xr.DataArray(
        [[2.0, -3.0, 0.25, np.nan], [0.5, 0.5, 0.5, 0.5]],
        dims=["time", "asset"],
        coords={"time": [1, 0], "asset": ["d", "c", "b", "a"]},
    )
    result = qnout.normalize(arr, per_asset=True)
    assert list(result.coords["asset"].values) == ["a", "b", "c", "d"]
    assert_allclose(result.values, [[0.5, 0.5, 0.5, 0.5], [0.0, 0.25, -1.0, 1.0]])


def test_normalize_single_day_vector() -> None:
    arr = xr.DataArray([3.0, -1.0, np.inf], dims=["asset"], coords={"asset": ["a", "b", "c"]})
    assert_allclose(qnout.normalize(arr).values, [0.75, -0.25, 0.0])
    small = xr.DataArray([0.2, -0.1], dims=["asset"], coords={"asset": ["a", "b"]})
    assert_allclose(qnout.normalize(small).values, [0.2, -0.1])


def test_align_cuts_window_and_zero_fills(data: xr.DataArray, weights: xr.DataArray) -> None:
    part = weights.isel(time=slice(5, 30))
    aligned = qnout.align(part, data.coords["time"], "2019-01-03", "2019-02-01")
    times = aligned.coords["time"].values
    assert str(times[0])[:10] == "2019-01-03"
    assert str(times[-1])[:10] == "2019-02-01"
    values = aligned.values
    assert_array_equal(values[: 5 - 2], 0.0)
    assert (np.abs(values).sum(axis=1)[3:] > 0).all()


def test_check_flags_each_problem_on_stderr(
    weights: xr.DataArray,
    data: xr.DataArray,
    baseline: xr.DataArray,
    capsys: pytest.CaptureFixture[str],
) -> None:
    qnout.check(weights, data, "futures", check_correlation=False)
    raw_report = capsys.readouterr()
    assert "ERROR! Strategy trades non-liquid assets." in raw_report.err
    assert "ERROR! The Sharpe Ratio is too low." in raw_report.err
    assert "WARNING! There are not enough points in the data" in raw_report.err
    assert "Sharpe Ratio = -1.3118170729618366" in raw_report.out

    qnout.check(baseline, data, "futures", check_correlation=False)
    clean_report = capsys.readouterr()
    assert "non-liquid" not in clean_report.err
    assert clean_report.out.count("Ok.") == 2

    qnout.check(baseline * 0, data, "futures", check_correlation=False)
    assert "ERROR! Output is empty. All positions are zero." in capsys.readouterr().err

    gap = baseline.isel(time=[i for i in range(40) if i != 7])
    qnout.check(gap, data, "futures", check_correlation=False)
    assert "ERROR! Some dates were missed)" in capsys.readouterr().err


def test_check_kind_specific_rules(capsys: pytest.CaptureFixture[str]) -> None:
    index = make_data(name="stocks_nasdaq100", n_days=40)
    qnout.check(make_weights(index), index, "stocks_nasdaq100", check_correlation=False)
    err = capsys.readouterr().err
    assert "ERROR! The max exposure is too high." in err
    assert "Max exposure: 0.5674190696878084 Hard limit: 0.1" in err

    crypto = make_data(name="crypto", n_days=40)
    qnout.check(make_weights(crypto), crypto, "crypto", check_correlation=False)
    assert "ERROR! Output contains not only BTC." in capsys.readouterr().err

    longs = make_data(name="stocks_long", n_days=40)
    qnout.check(make_weights(longs), longs, "stocks_long", check_correlation=False)
    assert "ERROR! Output contains negative positions." in capsys.readouterr().err


def test_check_reports_exceptions_instead_of_raising(
    data: xr.DataArray, capsys: pytest.CaptureFixture[str]
) -> None:
    broken = xr.DataArray(np.zeros((3, 2)), dims=["time", "foo"])
    qnout.check(broken, data, "futures", check_correlation=False)
    assert "cannot reindex or align along dimension 'time'" in capsys.readouterr().err


def test_sharpe_for_check_estimator_depends_on_kind(
    data: xr.DataArray, baseline: xr.DataArray, capsys: pytest.CaptureFixture[str]
) -> None:
    geometric = qnout.calc_sharpe_ratio_for_check(data, baseline, "futures", check_dates=False)
    arithmetic = qnout.calc_sharpe_ratio_for_check(
        data, baseline, "crypto_daily", check_dates=False
    )
    assert_allclose(geometric, -1.3118170729618366, rtol=1e-10)
    assert float(arithmetic) != float(geometric)
    capsys.readouterr()
    qnout.calc_sharpe_ratio_for_check(data, baseline, "futures", check_dates=True)
    err = capsys.readouterr().err
    assert "The first point(2019-01-01) should be earlier than 2006-01-01" in err
    assert "The output series should start from 2006-01-01 or earlier instead of 2019-01-01" in err


def test_write_translates_asset_ids_and_normalizes(sandbox: Path, weights: xr.DataArray) -> None:
    (sandbox / "id-translation.csv").write_text(
        "server_id,user_id\n" + "".join(f"srv{i},NYS:A{i}\n" for i in range(4))
    )
    qnout.write(weights.isel(time=slice(0, 5)))
    with gzip.open(sandbox / "fractions.nc.gz", "rb") as stream:
        stored = xr.open_dataarray(stream.read(), cache=False, decode_times=True).compute()
    assert list(stored.coords["asset"].values) == [
        "NYS:A4", "NYS:A5", "NYS:A6", "NYS:A7", "srv0", "srv1", "srv2", "srv3",
    ]  # fmt: skip
    assert (np.abs(stored.values).sum(axis=1) <= 1.0 + 1e-12).all()
    original = np.nan_to_num(weights.isel(time=slice(0, 5)).values)
    scale = np.maximum(np.abs(original).sum(axis=1), 1.0)
    assert_allclose(stored.values[:, 4:], original[:, :4] / scale[:, None], atol=1e-12)
