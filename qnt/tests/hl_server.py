import gzip
import io
import json
import urllib.request
from collections.abc import Callable
from types import TracebackType
from typing import TypeAlias
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pandas as pd
import xarray as xr
from typing_extensions import Self


Query: TypeAlias = dict[str, list[str]]
Handler: TypeAlias = Callable[[bytes | None, Query], bytes]


class FakeResponse:
    def __init__(self, body: bytes, headers: dict[str, str]) -> None:
        self.stream = io.BytesIO(body)
        self.headers = headers

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _kind: type[BaseException] | None,
        _error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        return None

    def getheader(self, name: str) -> str | None:
        return self.headers.get(name.lower())

    def read(self, size: int = -1) -> bytes:
        return self.stream.read(size)


class FakeServer:
    """In-process replacement for urllib.request.urlopen that serves registered routes."""

    def __init__(self) -> None:
        self.routes: dict[str, Handler] = {}
        self.calls: list[str] = []
        self.bodies: list[bytes | None] = []
        self.api_keys: list[str | None] = []
        self.failures_left = 0
        self.compress = False

    def route(self, path: str, handler: Handler) -> None:
        self.routes[path] = handler

    def serve_bytes(self, path: str, payload: bytes) -> None:
        self.routes[path] = lambda _body, _query: payload

    def serve_json(self, path: str, payload: object) -> None:
        self.routes[path] = lambda _body, _query: json.dumps(payload).encode()

    def __call__(
        self,
        request: urllib.request.Request | str,
        data: bytes | None = None,
        timeout: float | None = None,
    ) -> FakeResponse:
        del timeout
        if self.failures_left > 0:
            self.failures_left -= 1
            raise OSError("connection reset")
        if isinstance(request, str):
            url, body, api_key = request, data, None
        else:
            url = request.full_url
            body = request.data if isinstance(request.data, bytes) else None
            api_key = request.get_header("X-api-key")
        parts = urlsplit(url)
        self.calls.append(parts.path.lstrip("/") + ("?" + parts.query if parts.query else ""))
        self.bodies.append(body)
        self.api_keys.append(api_key)
        handler = self.routes[parts.path.lstrip("/")]
        payload = handler(body, parse_qs(parts.query))
        headers = {"content-length": str(len(payload))}
        if self.compress:
            payload = gzip.compress(payload)
            headers = {"content-encoding": "gzip", "content-length": str(len(payload))}
        return FakeResponse(payload, headers)


def netcdf_payload(array: xr.DataArray) -> bytes:
    return bytes(array.to_netcdf(compute=True))


def cube(
    fields: list[str],
    assets: list[str],
    times: pd.DatetimeIndex,
    seed: int = 1,
) -> xr.DataArray:
    """Random positive values with dims (field, time, asset)."""
    rng = np.random.default_rng(seed)
    values = rng.uniform(10.0, 20.0, (len(fields), len(times), len(assets)))
    return xr.DataArray(
        values,
        dims=["field", "time", "asset"],
        coords={"field": fields, "time": times, "asset": assets},
    )


def plane(assets: list[str], times: pd.DatetimeIndex, seed: int = 1) -> xr.DataArray:
    """Random positive values with dims (time, asset)."""
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        rng.uniform(1.0, 2.0, (len(times), len(assets))),
        dims=["time", "asset"],
        coords={"time": times, "asset": assets},
    )


STOCK_FIELDS = [
    "open",
    "low",
    "high",
    "close",
    "vol",
    "divs",
    "split",
    "split_cumprod",
    "is_liquid",
]
CATALOG = [
    {"id": "s1", "symbol": "AAA", "exchange": "NYS", "last_point": "2021-01-01"},
    {"id": "s2", "symbol": "BBB", "exchange": "NAS", "last_point": "2021-01-01"},
    {"id": "s3", "symbol": "AAA", "exchange": "NYS", "last_point": "2020-01-01"},
]
DAYS = pd.bdate_range("2021-01-04", periods=30, name="time")


def stock_cube() -> xr.DataArray:
    raw = cube(STOCK_FIELDS, ["s1", "s2", "s3"], DAYS, seed=4)
    values = raw.values.copy()
    values[STOCK_FIELDS.index("split"), :, :] = 1.0
    values[STOCK_FIELDS.index("split_cumprod"), :, :] = 1.0
    values[STOCK_FIELDS.index("split_cumprod"), :10, 0] = 0.5
    values[STOCK_FIELDS.index("is_liquid"), :, :] = 1.0
    return raw.copy(data=values)


def stock_server(server: FakeServer) -> xr.DataArray:
    source = stock_cube()
    server.serve_json("assets", CATALOG)

    def data(body: bytes | None, _query: Query) -> bytes:
        assert body is not None
        params = json.loads(body)
        chosen = [a for a in params["assets"] if a in source.coords["asset"].values]
        window = source.sel(asset=chosen, time=slice(params["min_date"], params["max_date"]))
        return netcdf_payload(window.sortby("time", ascending=False))

    server.route("data", data)
    return source
