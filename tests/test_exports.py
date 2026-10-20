"""CSV/JSONL contracts for types, incremental writes, empty results, and errors."""

from __future__ import annotations

import csv
import io
import json
from datetime import date, datetime, time, timezone
from decimal import Decimal
from typing import Any

import httpx
import pytest

from snowflake_sql_api.exceptions import SnowflakeConfigError, SnowflakeRequestError
from snowflake_sql_api.testing import FakeSnowflake

from .streaming_support import call, fake_client, wire_client
from .support import ok_body

VALUES = {
    "AMOUNT": Decimal("12345678901234567890.123456789"),
    "DATE": date(2023, 1, 2),
    "TIME": time(3, 4, 5, 123456),
    "TIMESTAMP": datetime(2023, 1, 2, 3, 4, 5, 123456, tzinfo=timezone.utc),
    "BINARY": b"\x00\xff",
    "OBJECT": {"nested": [1, None, True]},
    "ARRAY": ["a", 2],
    "TEXT": '雪, "quoted"\nnext line',
    "NULL": None,
    "EMPTY": "",
    "BOOL": True,
    "INT": 42,
    "FLOAT": 1.25,
}

JSON_VALUES = {
    **VALUES,
    "AMOUNT": "12345678901234567890.123456789",
    "DATE": "2023-01-02",
    "TIME": "03:04:05.123456",
    "TIMESTAMP": "2023-01-02T03:04:05.123456+00:00",
    "BINARY": "00ff",
}


@pytest.mark.parametrize("format", ["csv", "jsonl"])
async def test_exports_preserve_values_and_write_before_fetching_all_rows(
    async_mode: bool, format: str
) -> None:
    fake = FakeSnowflake()
    fake.register("SELECT values", [VALUES, VALUES], partitions=2)
    requests_at_write = []

    class ObservedBuffer(io.StringIO):
        def write(self, value: str) -> int:
            requests_at_write.append(len(fake.requests))
            return super().write(value)

    output = ObservedBuffer()
    async with fake_client(fake, async_mode) as client:
        assert (
            await call(client, async_mode, f"export_{format}", "SELECT values", output)
            == 2
        )
    assert not output.closed
    assert requests_at_write == ([1, 1, 2] if format == "csv" else [1, 2])
    if format == "jsonl":
        lines = output.getvalue().splitlines()
        assert len(lines) == 2
        assert [json.loads(line) for line in lines] == [JSON_VALUES, JSON_VALUES]
    else:
        rows = list(csv.DictReader(io.StringIO(output.getvalue(), newline="")))
        expected = {
            **JSON_VALUES,
            "OBJECT": '{"nested": [1, null, true]}',
            "ARRAY": '["a", 2]',
            "NULL": "",
            "BOOL": "true",
            "INT": "42",
            "FLOAT": "1.25",
        }
        assert rows == [expected, expected]


@pytest.mark.parametrize("format", ["csv", "jsonl"])
async def test_empty_exports_preserve_csv_header(async_mode: bool, format: str) -> None:
    fake = FakeSnowflake()
    fake.register("SELECT empty", [], columns=["ID", "NAME"])
    output = io.StringIO()
    async with fake_client(fake, async_mode) as client:
        assert (
            await call(client, async_mode, f"export_{format}", "SELECT empty", output)
            == 0
        )
    assert output.getvalue() == ("ID,NAME\r\n" if format == "csv" else "")
    assert not output.closed


async def test_zero_column_csv_writes_empty_header(async_mode: bool) -> None:
    fake = FakeSnowflake()
    fake.register("SELECT empty", [])
    output = io.StringIO()
    async with fake_client(fake, async_mode) as client:
        assert await call(client, async_mode, "export_csv", "SELECT empty", output) == 0
    assert output.getvalue() == "\r\n"


@pytest.mark.parametrize("format", ["csv", "jsonl"])
async def test_duplicate_column_names_fail_before_writing(
    async_mode: bool, format: str
) -> None:
    fake = FakeSnowflake()
    fake.register("SELECT a, b", [[1, 2]], columns=["X", "X"])
    output = io.StringIO()
    async with fake_client(fake, async_mode) as client:
        with pytest.raises(SnowflakeConfigError, match="unique column names"):
            await call(client, async_mode, f"export_{format}", "SELECT a, b", output)
    assert output.getvalue() == ""
    assert not output.closed


@pytest.mark.parametrize("format", ["csv", "jsonl"])
async def test_later_partition_failure_leaves_partial_output_and_raises(
    async_mode: bool, private_key_pem: bytes, format: str
) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(
                200,
                json=ok_body([{"name": "N", "type": "fixed"}], [["1"]], partitions=2),
            )
        return httpx.Response(200, content=b"damaged JSON")

    output = io.StringIO()
    async with wire_client(handler, private_key_pem, async_mode) as client:
        with pytest.raises(SnowflakeRequestError, match="partition data"):
            await call(client, async_mode, f"export_{format}", "SELECT n", output)
    assert output.getvalue() == ("N\r\n1\r\n" if format == "csv" else '{"N": 1}\n')
    assert len(requests) == 2
    assert not output.closed


@pytest.mark.parametrize("format", ["csv", "jsonl"])
@pytest.mark.parametrize("fail_after", [0, 1])
async def test_sink_failure_stops_fetching_and_closes_internal_iterator(
    async_mode: bool, format: str, fail_after: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeSnowflake()
    fake.register("SELECT n", [{"N": 1}, {"N": 2}], partitions=2)
    captured = []

    class BrokenBuffer(io.StringIO):
        writes = 0

        def write(self, value: str) -> int:
            if self.writes == fail_after:
                raise OSError("disk full")
            self.writes += 1
            return super().write(value)

    output = BrokenBuffer()
    async with fake_client(fake, async_mode) as client:
        original = client._stream_result
        if async_mode:

            async def capture(*args: Any) -> Any:
                result = await original(*args)
                captured.append(result[1])
                return result

        else:

            def capture(*args: Any) -> Any:
                result = original(*args)
                captured.append(result[1])
                return result

        monkeypatch.setattr(client, "_stream_result", capture)
        with pytest.raises(OSError, match="disk full"):
            await call(client, async_mode, f"export_{format}", "SELECT n", output)
        assert not output.closed
        frame = captured[0].ag_frame if async_mode else captured[0].gi_frame
        assert frame is None
    # JSONL has no header: the second write needs partition 1.
    expected_requests = 2 if format == "jsonl" and fail_after == 1 else 1
    assert len(fake.requests) == expected_requests


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), {"nested": float("-inf")}]
)
async def test_jsonl_rejects_nonfinite_numbers(async_mode: bool, value: Any) -> None:
    fake = FakeSnowflake()
    fake.register("SELECT value", [{"VALUE": value}])
    output = io.StringIO()
    async with fake_client(fake, async_mode) as client:
        with pytest.raises(ValueError, match="JSON compliant"):
            await call(client, async_mode, "export_jsonl", "SELECT value", output)
    assert output.getvalue() == ""
    assert not output.closed


@pytest.mark.parametrize("format", ["csv", "jsonl"])
async def test_exports_bind_params_and_notify_once(
    async_mode: bool, format: str
) -> None:
    fake = FakeSnowflake()
    fake.register("SELECT ?", [{"N": 1}])
    seen = []
    async with fake_client(
        fake, async_mode, on_query=lambda sql, params: seen.append((sql, params))
    ) as client:
        await call(
            client, async_mode, f"export_{format}", "SELECT ?", io.StringIO(), [1]
        )
    assert seen == [("SELECT ?", [1])]
    assert json.loads(fake.requests[0].content)["bindings"]["1"]["value"] == "1"
