import datetime
import gzip
import importlib.util
import os
import pickle
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from email.message import Message
from pathlib import Path
from types import ModuleType, SimpleNamespace, TracebackType

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_array_equal
from typing_extensions import Self, override

import qnt.data.common as qndc
import qnt.log as qlog
from qnt.tests.hl_server import FakeResponse


def moment(text: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(text)


NOW = moment("2024-01-10T12:00:00")
RETENTION = datetime.timedelta(days=7)
SECOND = datetime.timedelta(seconds=1)
DAY = datetime.timedelta(days=1)
SEE_DOCS = "See: https://quantiacs.io/documentation/en/user_guide/local_development.html"


class SpyLog:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def __call__(self, *args: object) -> None:
        self.calls.append(args)


class SizedResponse(FakeResponse):
    def __init__(self, body: bytes, headers: dict[str, str], sizes: list[int]) -> None:
        super().__init__(body, headers)
        self.sizes = sizes

    @override
    def read(self, size: int = -1) -> bytes:
        self.sizes.append(size)
        return super().read(size)


class ProgressRecorder:
    def __init__(self, updates: list[int]) -> None:
        self.updates = updates

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _kind: type[BaseException] | None,
        _error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        return None

    def update(self, value: int) -> None:
        self.updates.append(value)


class Downloads:
    def __init__(self, body: bytes, headers: dict[str, str], failures: int = 0) -> None:
        self.body = body
        self.headers = headers
        self.failures = failures
        self.timeouts: list[float | None] = []
        self.read_sizes: list[int] = []
        self.sleeps: list[float] = []
        self.max_values: list[int | None] = []
        self.updates: list[int] = []

    def urlopen(
        self, request: urllib.request.Request, timeout: float | None = None
    ) -> SizedResponse:
        del request
        self.timeouts.append(timeout)
        if self.failures > 0:
            self.failures -= 1
            raise OSError("down")
        return SizedResponse(self.body, self.headers, self.read_sizes)

    def progress(self, max_value: int | None, poll_interval: int) -> ProgressRecorder:
        del poll_interval
        self.max_values.append(max_value)
        return ProgressRecorder(self.updates)

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


def install_downloads(
    monkeypatch: pytest.MonkeyPatch, body: bytes, headers: dict[str, str], failures: int = 0
) -> Downloads:
    downloads = Downloads(body, headers, failures)
    monkeypatch.setattr("urllib.request.urlopen", downloads.urlopen)
    monkeypatch.setattr(qndc, "time", SimpleNamespace(sleep=downloads.sleep))
    monkeypatch.setattr(qndc, "progressbar", SimpleNamespace(ProgressBar=downloads.progress))
    monkeypatch.setattr(qndc, "cache_min_mod_time", None)
    return downloads


def spy_log_err(monkeypatch: pytest.MonkeyPatch) -> SpyLog:
    spy = SpyLog()
    monkeypatch.setattr(qndc, "log_err", spy)
    return spy


def test_documented_constants() -> None:
    assert qndc.DEFAULT_TAIL == 2190
    assert qndc.TIMEOUT == 60
    assert qndc.RETRY_DELAY == 1


@pytest.mark.usefixtures("sandbox")
def test_download_retries_without_limit_in_submission_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"abc"
    downloads = install_downloads(monkeypatch, payload, {"content-length": "3"}, failures=7)
    spy_log_err(monkeypatch)
    monkeypatch.setenv("SUBMISSION_ID", "run-1")
    assert qndc.request_with_retry("blob-b", None) == payload
    assert downloads.sleeps == [1] * 7


@pytest.mark.parametrize(
    ("length", "with_header", "block"),
    [
        (100, True, 4096),
        (409_700, True, 4097),
        (1_000_000, True, 10_000),
        (10_000, False, 4096),
    ],
)
@pytest.mark.usefixtures("sandbox")
def test_download_reads_in_blocks_sized_from_content_length(
    monkeypatch: pytest.MonkeyPatch,
    length: int,
    with_header: bool,
    block: int,
) -> None:
    payload = bytes(range(256)) * (length // 256) + bytes(range(length % 256))
    headers = {"content-length": str(length)} if with_header else {}
    downloads = install_downloads(monkeypatch, payload, headers)
    assert qndc.request_with_retry(f"blob-c-{length}-{with_header}", None) == payload
    expected_reads = -(-length // block) + 1
    assert downloads.read_sizes == [block] * expected_reads
    assert downloads.max_values == [length if with_header else None]
    assert downloads.updates == [min(block * i, length) for i in range(1, expected_reads)]
    assert downloads.sleeps == []


@pytest.mark.usefixtures("sandbox")
def test_download_decodes_gzip_by_content_encoding_value(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"hello gzip" * 50
    packed = gzip.compress(payload)
    encoding = "".join(("gz", "ip"))
    downloads = install_downloads(
        monkeypatch, packed, {"content-encoding": encoding, "content-length": str(len(packed))}
    )
    assert qndc.request_with_retry("blob-d", None) == payload
    plain = install_downloads(monkeypatch, payload, {"content-length": str(len(payload))})
    assert qndc.request_with_retry("blob-e", None) == payload
    assert downloads.sleeps == []
    assert plain.sleeps == []


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://h:7070/last/2016-10-28T05/", moment("2016-10-28T05:00:00")),
        ("http://h:7070/last/2016-10-28/", moment("2016-10-28T00:00:00")),
    ],
)
def test_url_limit_is_not_clipped_by_global_limit(
    monkeypatch: pytest.MonkeyPatch, url: str, expected: datetime.datetime
) -> None:
    monkeypatch.setattr(qndc, "MAX_DATETIME_LIMIT", moment("2000-01-01T00:00:00"))
    monkeypatch.setattr(qndc, "MAX_DATE_LIMIT", datetime.date(2000, 1, 1))
    assert qndc.parse_max_datetime_from_url(url) == expected


def test_deprecated_wrapper_names_the_assigned_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = spy_log_err(monkeypatch)

    def target() -> int:
        return 3

    old_name = qndc.deprecated_wrap(target)
    assert old_name() == 3
    assert spy.calls == [(f"WARNING: old_name deprecated, use {__name__}.target",)]


def test_deprecated_wrapper_prefixes_module_name_at_module_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "hl_legacy_module.py"
    source.write_text(
        "from qnt.data.common import deprecated_wrap\n"
        "\n"
        "def target() -> int:\n"
        "    return 7\n"
        "\n"
        "legacy = deprecated_wrap(target)\n"
    )
    spec = importlib.util.spec_from_file_location("hl_legacy_module", source)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    spy = spy_log_err(monkeypatch)
    legacy = module_attr(module, "legacy")
    assert callable(legacy)
    assert legacy() == 7
    assert spy.calls == [
        ("WARNING: hl_legacy_module.legacy deprecated, use hl_legacy_module.target",)
    ]


def module_attr(module: ModuleType, name: str) -> object:
    value: object = getattr(module, name)
    return value


def entry_paths(directory: str) -> tuple[Path, Path]:
    Path(directory).mkdir(exist_ok=True)
    qndc.cache_put(b"value", "k", 1)
    args_file = next(Path(directory).glob("*.args.pickle.gz"))
    value_file = next(Path(directory).glob("*.value.pickle.gz"))
    return args_file, value_file


@pytest.mark.usefixtures("sandbox")
def test_cache_get_needs_both_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qndc, "cache_min_mod_time", None)
    spy = spy_log_err(monkeypatch)
    args_file, value_file = entry_paths(qndc.CACHE_DIR)
    assert qndc.cache_get("k", 1) == b"value"
    args_bytes = args_file.read_bytes()
    value_bytes = value_file.read_bytes()

    args_file.unlink()
    assert qndc.cache_get("k", 1) is None
    args_file.write_bytes(args_bytes)
    value_file.unlink()
    assert qndc.cache_get("k", 1) is None
    value_file.write_bytes(value_bytes)
    assert qndc.cache_get("k", 1) == b"value"
    assert spy.calls == []


@pytest.mark.parametrize("stored", [("k", 2), ("k", 0)])
@pytest.mark.usefixtures("sandbox")
def test_cache_get_ignores_entry_stored_for_other_args(
    monkeypatch: pytest.MonkeyPatch, stored: tuple[str, int]
) -> None:
    monkeypatch.setattr(qndc, "cache_min_mod_time", None)
    args_file, _ = entry_paths(qndc.CACHE_DIR)
    with gzip.open(args_file, "wb") as stream:
        pickle.dump(stored, stream)
    assert qndc.cache_get("k", 1) is None


@pytest.mark.parametrize(
    ("broken", "kind"),
    [
        pytest.param(b"not gzip at all", OSError, id="bad-gzip"),
        pytest.param(gzip.compress(b""), EOFError, id="empty-stream"),
        pytest.param(gzip.compress(b"not a pickle"), pickle.UnpicklingError, id="bad-pickle"),
    ],
)
@pytest.mark.usefixtures("sandbox")
def test_cache_get_logs_unreadable_entry(
    monkeypatch: pytest.MonkeyPatch,
    broken: bytes,
    kind: type[Exception],
) -> None:
    monkeypatch.setattr(qndc, "cache_min_mod_time", None)
    args_file, _ = entry_paths(qndc.CACHE_DIR)
    args_file.write_bytes(broken)
    spy = spy_log_err(monkeypatch)
    assert qndc.cache_get("k", 1) is None
    assert len(spy.calls) == 1
    assert spy.calls[0][0] == "Cache read problem:"
    assert isinstance(spy.calls[0][1], kind)


@pytest.mark.parametrize(("days", "written"), [(-1, True), (0, False), (1, True)])
@pytest.mark.usefixtures("sandbox")
def test_cache_put_writes_unless_retention_is_zero(
    monkeypatch: pytest.MonkeyPatch, days: int, written: bool
) -> None:
    monkeypatch.setattr(qndc, "CACHE_RETENTION", datetime.timedelta(days=days))
    Path(qndc.CACHE_DIR).mkdir()
    qndc.cache_put(b"value", "k", 1)
    assert len(list(Path(qndc.CACHE_DIR).iterdir())) == (2 if written else 0)


class FrozenDatetime(datetime.datetime):
    @classmethod
    @override
    def now(cls, tz: datetime.tzinfo | None = None) -> Self:
        del tz
        return cls.fromtimestamp(NOW.timestamp())


@pytest.fixture
def frozen_cache(sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(
        qndc, "datetime", SimpleNamespace(datetime=FrozenDatetime, timedelta=datetime.timedelta)
    )
    monkeypatch.setattr(qndc, "CACHE_RETENTION", RETENTION)
    monkeypatch.setattr(qndc, "cache_min_mod_time", None)
    directory = sandbox / "data-cache"
    directory.mkdir()
    return directory


def make_entry(directory: Path, name: str, age: datetime.timedelta) -> Path:
    path = directory / name
    path.write_bytes(b"x")
    stamp = (NOW - age).timestamp()
    os.utime(path, (stamp, stamp))
    return path


def force_listing(monkeypatch: pytest.MonkeyPatch, names: list[str]) -> None:
    real_listdir = os.listdir

    def listing(path: str) -> list[str]:
        return list(names) if path == qndc.CACHE_DIR else real_listdir(path)

    monkeypatch.setattr(os, "listdir", listing)


@pytest.mark.parametrize(
    ("min_age", "scanned"),
    [
        (RETENTION - SECOND, False),
        (RETENTION, True),
        (RETENTION + SECOND, True),
        (None, True),
    ],
)
def test_crop_cache_scans_only_when_oldest_entry_may_have_expired(
    frozen_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
    min_age: datetime.timedelta | None,
    scanned: bool,
) -> None:
    expired = make_entry(frozen_cache, "expired.bin", RETENTION + DAY)
    marker = None if min_age is None else NOW - min_age
    monkeypatch.setattr(qndc, "cache_min_mod_time", marker)
    qndc.crop_cache()
    assert expired.exists() is (not scanned)
    assert qndc.cache_min_mod_time == (None if scanned else marker)


def test_crop_cache_removes_only_entries_older_than_retention(
    frozen_cache: Path,
) -> None:
    edge = make_entry(frozen_cache, "edge.bin", RETENTION)
    over = make_entry(frozen_cache, "over.bin", RETENTION + SECOND)
    qndc.crop_cache()
    assert edge.exists()
    assert not over.exists()
    assert qndc.cache_min_mod_time == NOW - RETENTION


def test_crop_cache_remembers_the_oldest_remaining_entry(
    frozen_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, days in (("f1", 2), ("f2", 3), ("f3", 1)):
        make_entry(frozen_cache, name, datetime.timedelta(days=days))
    force_listing(monkeypatch, ["f1", "f2", "f3"])
    qndc.crop_cache()
    assert qndc.cache_min_mod_time == NOW - datetime.timedelta(days=3)


def test_crop_cache_skips_subdirectories_and_continues(
    frozen_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = frozen_cache / "0dir"
    folder.mkdir()
    expired = make_entry(frozen_cache, "1.old", RETENTION + DAY)
    force_listing(monkeypatch, ["0dir", "1.old"])
    qndc.crop_cache()
    assert folder.is_dir()
    assert not expired.exists()


def test_crop_cache_creates_missing_directory(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = sandbox / "fresh-cache"
    monkeypatch.setattr(qndc, "CACHE_DIR", str(target))
    monkeypatch.setattr(qndc, "cache_min_mod_time", NOW)
    qndc.crop_cache()
    assert target.is_dir()
    assert qndc.cache_min_mod_time is None


def test_crop_cache_tolerates_directory_created_concurrently(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = sandbox / "raced-cache"
    target.mkdir()
    monkeypatch.setattr(qndc, "CACHE_DIR", str(target))
    real_isdir = os.path.isdir
    seen: list[str] = []

    def flaky(path: str) -> bool:
        seen.append(path)
        return False if len(seen) == 1 else real_isdir(path)

    monkeypatch.setattr(os.path, "isdir", flaky)
    qndc.crop_cache()
    assert seen[0] == str(target)
    assert target.is_dir()


def test_set_max_datetime_replaces_the_last_segment_of_origin_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qndc, "ORIGIN_BASE_URL", "http://h:7070/last/2020-10-07T10/")
    monkeypatch.setattr(qndc, "BASE_URL", "http://h:7070/last/2020-10-07T10/")
    monkeypatch.setattr(qndc, "MAX_DATETIME_LIMIT", None)
    monkeypatch.setattr(qndc, "MAX_DATE_LIMIT", None)
    qndc.set_max_datetime("2020-05-06T13:45")
    assert qndc.BASE_URL == "http://h:7070/last/2020-05-06T13/"
    limit_time = qndc.MAX_DATETIME_LIMIT
    assert limit_time == moment("2020-05-06T13:00:00")
    qndc.set_max_datetime(None)
    assert qndc.BASE_URL == "http://h:7070/last/2020-10-07T10/"
    limit_date = qndc.MAX_DATE_LIMIT
    assert limit_date == moment("2020-10-07T00:00:00").date()


class AccountProbe:
    def __init__(self, error: urllib.error.HTTPError | None) -> None:
        self.error = error
        self.urls: list[str] = []

    def __call__(self, url: str) -> FakeResponse:
        self.urls.append(url)
        if self.error is not None:
            raise self.error
        return FakeResponse(b"", {})


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://t.local/account", code, "status", Message(), None)


def prepare_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    api_key: str,
    base_url: str = "https://data-api.quantiacs.io/",
    tracking: str | None = None,
    error: urllib.error.HTTPError | None = None,
) -> tuple[SpyLog, AccountProbe]:
    monkeypatch.setenv("API_KEY", api_key)
    monkeypatch.setenv("DATA_BASE_URL", base_url)
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "fresh-cache"))
    monkeypatch.setenv("CACHE_RETENTION", "7")
    if tracking is None:
        monkeypatch.delenv("TRACKING_HOST", raising=False)
    else:
        monkeypatch.setenv("TRACKING_HOST", tracking)
    spy = SpyLog()
    monkeypatch.setattr(qlog, "log_err", spy)
    probe = AccountProbe(error)
    monkeypatch.setattr("urllib.request.urlopen", probe)
    return spy, probe


def load_fresh() -> ModuleType:
    path = qndc.__file__
    assert path is not None
    spec = importlib.util.spec_from_file_location("hl_common_fresh", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_with_default_key_and_plain_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy, probe = prepare_import(monkeypatch, tmp_path, "default")
    module = load_fresh()
    assert probe.urls == []
    assert spy.calls == []
    assert module_attr(module, "BASE_URL") == "https://data-api.quantiacs.io/"
    assert module_attr(module, "MAX_DATETIME_LIMIT") is None
    assert module_attr(module, "MAX_DATE_LIMIT") is None
    assert (tmp_path / "fresh-cache").is_dir()


def test_import_takes_date_limit_from_dated_base_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare_import(monkeypatch, tmp_path, "default", base_url="http://h:7070/last/2020-10-07T10/")
    module = load_fresh()
    assert module_attr(module, "MAX_DATETIME_LIMIT") == moment("2020-10-07T10:00:00")
    assert module_attr(module, "MAX_DATE_LIMIT") == datetime.date(2020, 10, 7)


def test_import_without_api_key_exits_with_instructions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy, probe = prepare_import(monkeypatch, tmp_path, "")
    with pytest.raises(SystemExit) as stop:
        load_fresh()
    assert stop.value.code == 1
    assert spy.calls == [("Please, specify the API_KEY.",), (SEE_DOCS,)]
    assert probe.urls == []


@pytest.mark.parametrize("api_key", ["abc", "x", "  abc  "])
@pytest.mark.parametrize("tracking", [None, "http://t.local"])
def test_import_checks_custom_key_against_tracking_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api_key: str, tracking: str | None
) -> None:
    spy, probe = prepare_import(monkeypatch, tmp_path, api_key, tracking=tracking)
    module = load_fresh()
    host = "https://quantiacs.io" if tracking is None else tracking
    assert probe.urls == [f"{host}/auth/system/account/accountByKey?apiKey={api_key.strip()}"]
    assert spy.calls == []
    assert module_attr(module, "api_key") == api_key.strip()


def test_import_with_unknown_key_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spy, probe = prepare_import(monkeypatch, tmp_path, "abc", error=http_error(404))
    with pytest.raises(SystemExit) as stop:
        load_fresh()
    assert stop.value.code == 1
    assert spy.calls == [("Wrong API_KEY.",), (SEE_DOCS,)]
    assert len(probe.urls) == 1


@pytest.mark.parametrize("code", [401, 403, 500])
def test_import_ignores_other_http_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    spy, probe = prepare_import(monkeypatch, tmp_path, "abc", error=http_error(code))
    module = load_fresh()
    assert spy.calls == []
    assert len(probe.urls) == 1
    assert module_attr(module, "api_key") == "abc"


class InlineThread:
    def __init__(self, target: Callable[[str], None], args: tuple[str]) -> None:
        self.target = target
        self.args = args

    def start(self) -> None:
        self.target(*self.args)


@pytest.fixture
def tracking(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    urls: list[str] = []
    monkeypatch.setattr(qndc, "sent_events", set())
    monkeypatch.setattr(qndc, "get_url_silent", urls.append)
    monkeypatch.setattr(threading, "Thread", InlineThread)
    monkeypatch.setattr(qndc, "api_key", "KEY")
    monkeypatch.setattr(qndc, "tracking_host", "http://t.local")
    monkeypatch.delenv("SUBMISSION_ID", raising=False)
    monkeypatch.delenv("STRATEGY_ID", raising=False)
    return urls


@pytest.mark.parametrize(("submission", "sent"), [(None, True), ("", True), ("run-1", False)])
def test_track_event_is_silent_in_submission_runs(
    tracking: list[str], monkeypatch: pytest.MonkeyPatch, submission: str | None, sent: bool
) -> None:
    if submission is not None:
        monkeypatch.setenv("SUBMISSION_ID", submission)
    qndc.track_event("ev1")
    expected = ["http://t.local/engine/tracklib?apiKey=KEY&event=ev1"] if sent else []
    assert tracking == expected


def test_track_event_sends_each_event_once(tracking: list[str]) -> None:
    qndc.track_event("a")
    qndc.track_event("a")
    qndc.track_event("b")
    assert tracking == [
        "http://t.local/engine/tracklib?apiKey=KEY&event=a",
        "http://t.local/engine/tracklib?apiKey=KEY&event=b",
    ]


def test_track_event_appends_strategy_id(
    tracking: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STRATEGY_ID", "S9")
    qndc.track_event("ev2")
    assert tracking == ["http://t.local/engine/tracklib?apiKey=KEY&event=ev2&strategyId=S9"]


ASSET_NAMES = ["A", "B", "C"]


def assets_cube(liquid: list[list[float]]) -> xr.DataArray:
    times = pd.to_datetime(["2020-01-01", "2020-01-02"])
    close = np.array([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]])
    values = np.stack([close, np.array(liquid)], axis=1)
    return xr.DataArray(
        values,
        dims=["time", "field", "asset"],
        coords={"time": times, "field": ["close", "is_liquid"], "asset": ASSET_NAMES},
    )


def weights_of(values: list[float]) -> xr.DataArray:
    return xr.DataArray(values, dims=["asset"], coords={"asset": ASSET_NAMES})


def test_filter_liquids_keeps_only_flag_equal_to_one() -> None:
    assets = assets_cube([[1.0, 0.0, 2.0], [0.0, 0.0, 2.0]])
    result = qndc.filter_liquids_xarray_assets_dataarray(assets)
    assert result.coords["asset"].values.tolist() == ["A"]
    assert result.shape == (1, 2, 1)
    assert_array_equal(result.values[0, :, 0], [10.0, 1.0])


def test_check_weights_accepts_zero_weights_on_non_liquid() -> None:
    assets = assets_cube([[1.0, 0.0, -1.0], [1.0, 1.0, 1.0]])
    qndc.check_weights_xarray_dataarray_for_nonliquids(weights_of([0.5, 0.0, 0.2]), assets)


@pytest.mark.parametrize("weight", [0.4, -0.4])
def test_check_weights_rejects_any_non_zero_weight_on_non_liquid(weight: float) -> None:
    assets = assets_cube([[1.0, 0.0, -1.0], [1.0, 1.0, 1.0]])
    with pytest.raises(Exception, match=r"\['B'\]"):
        qndc.check_weights_xarray_dataarray_for_nonliquids(weights_of([0.5, weight, 0.2]), assets)
    qndc.check_weights_xarray_dataarray_for_nonliquids(weights_of([0.5, 0.0, 0.2]), assets)


def test_exclude_weights_keeps_only_liquid_assets() -> None:
    assets = assets_cube([[1.0, 0.0, 2.0], [1.0, 1.0, 1.0]])
    result = qndc.exclude_weights_xarray_dataarray_from_nonliquids(
        weights_of([0.5, 0.3, 0.2]), assets
    )
    assert result.coords["asset"].values.tolist() == ["A"]
    assert_array_equal(result.values, [0.5])
