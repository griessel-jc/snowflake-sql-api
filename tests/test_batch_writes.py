"""Batch INSERT boundaries, progress, and uncertain outcomes on both clients."""

from __future__ import annotations

import asyncio
import json
from dataclasses import FrozenInstanceError
from datetime import date, datetime, time, timezone
from decimal import Decimal
from typing import Any

import httpx
import pytest

from snowflake_sql_api import (
    BatchSummary,
    BatchWriteError,
    BatchWriteResult,
    FailedBatch,
    SnowflakeConfigError,
    SnowflakeError,
    SnowflakeProgrammingError,
    SnowflakeRequestError,
    SnowflakeRetryError,
    SnowflakeTimeoutError,
)
from snowflake_sql_api.bindings import to_bindings

from .streaming_support import wire_client


async def insert(client: Any, async_mode: bool, rows: Any, **kwargs: Any) -> Any:
    result = client.insert_batches("db.s.t", ["v"], rows, **kwargs)
    return await result if async_mode else result


def success(count: int) -> httpx.Response:
    return httpx.Response(200, json={"stats": {"numRowsInserted": count}})


def row_count(request: httpx.Request, width: int = 1) -> int:
    payload = json.loads(request.content)
    count = len(payload["bindings"]) // width
    assert payload["statement"].count("?") == count * width
    assert request.headers["content-type"] == "application/json"
    return count


async def test_row_boundary_is_single_pass_and_submits_before_next_row(
    async_mode: bool, private_key_pem: bytes
) -> None:
    requests = []
    pulled = []
    notifications = []

    def rows():
        for i in range(5):
            assert len(requests) == i // 2
            pulled.append(i)
            yield [i]

    def handler(request):
        requests.append(request)
        assert len(pulled) == min(2 * len(requests), 5)
        return success(row_count(request) + 10)

    async with wire_client(
        handler,
        private_key_pem,
        async_mode,
        on_query=lambda sql, params: notifications.append((sql, list(params))),
    ) as client:
        result = await insert(client, async_mode, rows(), max_rows=2)
    assert result == BatchWriteResult(
        (
            BatchSummary(0, 0, 2, 12),
            BatchSummary(1, 2, 2, 12),
            BatchSummary(2, 4, 1, 11),
        )
    )
    assert (result.input_rows, result.rows_affected, result.completed_batches) == (
        5,
        35,
        3,
    )
    assert [row_count(r) for r in requests] == [2, 2, 1]
    assert [n[1] for n in notifications] == [[0, 1], [2, 3], [4]]
    assert len({r.url.params["requestId"] for r in requests}) == 3
    assert json.loads(requests[0].content)["statement"] == (
        'INSERT INTO "db"."s"."t" ("v") VALUES (?), (?)'
    )
    with pytest.raises(FrozenInstanceError):
        result.batches[0].row_count = 9


VALUES = [
    '雪 🐟 "quoted"\n\\',
    {"nested": ["雪", None, True]},
    b"\x00\xff" * 20,
    Decimal("12345678901234567890.123"),
    date(2023, 1, 2),
    time(1, 2, 3),
    datetime(2023, 1, 2, 3, 4, 5),
    datetime(2023, 1, 2, tzinfo=timezone.utc),
    None,
    True,
    1.25,
]


@pytest.mark.parametrize("value", VALUES)
async def test_exact_wire_byte_cap_and_one_byte_over(
    async_mode: bool, private_key_pem: bytes, value: Any
) -> None:
    requests = []

    def handler(request):
        requests.append(request)
        return success(row_count(request))

    context = {
        "database": "雪db",
        "schema": "s",
        "warehouse": "wh",
        "role": "writer",
        "timezone": "Pacific/Auckland",
        "statement_timeout": 42,
        "parameters": {"QUERY_TAG": 'a"\\雪'},
    }
    async with wire_client(handler, private_key_pem, async_mode, **context) as client:
        await insert(client, async_mode, [[value], [value]])
        two_size = len(requests.pop().content)
        for cap, counts in ((two_size, [2]), (two_size - 1, [1, 1])):
            result = await insert(
                client, async_mode, iter([[value], [value]]), max_bytes=cap
            )
            assert result.input_rows == 2
            assert [row_count(r) for r in requests] == counts
            assert all(len(r.content) <= cap for r in requests)
            for r in requests:
                body = json.loads(r.content)
                assert body["database"] == "雪db"
                assert body["timeout"] == 42
                assert body["parameters"] == {
                    "QUERY_TAG": 'a"\\雪',
                    "TIMEZONE": "Pacific/Auckland",
                }
                assert body["bindings"] == to_bindings([value] * row_count(r))
                assert (
                    r.content
                    == json.dumps(
                        body, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                    ).encode()
                )
            requests.clear()


async def test_binding_key_digit_boundaries_and_byte_lookahead(
    async_mode: bool, private_key_pem: bytes
) -> None:
    requests, pulled = [], []

    def handler(request):
        requests.append(request)
        return success(row_count(request))

    async with wire_client(handler, private_key_pem, async_mode) as client:
        await insert(client, async_mode, [["x"]] * 10)
        cap = len(requests.pop().content)

        def rows():
            for i in range(23):
                assert i <= len(requests) * 10 + 10
                pulled.append(i)
                yield ["x"]

        result = await insert(client, async_mode, rows(), max_bytes=cap)
    assert result.input_rows == len(pulled) == 23
    assert [row_count(r) for r in requests] == [10, 10, 3]
    assert [len(r.content) <= cap for r in requests] == [True] * 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_rows": True},
        {"max_rows": 0},
        {"max_rows": -1},
        {"max_rows": 1.1},
        {"max_rows": 200001},
        {"max_bytes": True},
        {"max_bytes": 0},
        {"max_bytes": "1"},
    ],
)
async def test_invalid_limits_do_not_iterate_or_submit(
    async_mode: bool, private_key_pem: bytes, kwargs: dict
) -> None:
    def rows():
        pytest.fail("invalid arguments consumed input")
        yield []

    def handler(request):
        pytest.fail("invalid arguments submitted")

    async with wire_client(handler, private_key_pem, async_mode) as client:
        with pytest.raises(SnowflakeConfigError):
            await insert(client, async_mode, rows(), **kwargs)
        with pytest.raises(SnowflakeConfigError):
            value = client.insert_batches("t", [], rows())
            if async_mode:
                await value


async def test_empty_input_has_no_http_or_notification(async_mode, private_key_pem):
    def fail(*args):
        pytest.fail("empty input performed work")

    async with wire_client(fail, private_key_pem, async_mode, on_query=fail) as client:
        assert (
            await insert(client, async_mode, iter(()), max_bytes=1)
            == BatchWriteResult()
        )


@pytest.mark.parametrize("completed", [False, True])
@pytest.mark.parametrize(
    "fault", ["iterator", "ragged", "unsupported", "oversized", "record", "json"]
)
async def test_local_fault_discards_unsent_buffer_and_preserves_prefix(
    async_mode, private_key_pem, completed, fault
):
    requests = []
    original = ValueError("input failed")

    def handler(request):
        requests.append(request)
        return success(row_count(request))

    def rows():
        if completed:
            yield [1]
            yield [2]
        yield [3]  # buffered, never flushed after a bad input record
        if fault == "iterator":
            raise original
        yield {
            "ragged": [],
            "unsupported": [object()],
            "oversized": ["x" * 10000],
            "record": {"v": 4},
            "json": [{"bad": object()}],
        }[fault]
        pytest.fail("consumption continued after failure")

    async with wire_client(handler, private_key_pem, async_mode) as client:
        with pytest.raises(BatchWriteError) as info:
            await insert(client, async_mode, rows(), max_rows=2, max_bytes=1000)
    err = info.value
    offset = 2 if completed else 0
    assert isinstance(err, SnowflakeError)
    assert err.completed.input_rows == err.next_row_index == offset
    assert err.completed.completed_batches == int(completed)
    assert err.consumed_rows == offset + (1 if fault == "iterator" else 2)
    assert err.failed_batch is None
    assert err.outcome == "not_executed"
    assert err.phase == ("input" if fault == "iterator" else "validation")
    assert err.__cause__ is not None
    if fault == "iterator":
        assert err.__cause__ is original
    assert len(requests) == int(completed)


async def test_first_oversized_row_is_not_sent(async_mode, private_key_pem):
    requests = []
    async with wire_client(
        lambda r: requests.append(r) or success(1), private_key_pem, async_mode
    ) as client:
        with pytest.raises(BatchWriteError) as info:
            await insert(client, async_mode, [["雪"]], max_bytes=1)
    assert info.value.consumed_rows == 1
    assert info.value.next_row_index == 0
    assert requests == []


@pytest.mark.parametrize(
    "status,outcome,phase",
    [
        (400, "not_executed", "submission"),
        (401, "not_executed", "submission"),
        (403, "not_executed", "submission"),
        (422, "failed", "execution"),
        (404, "unknown", "submission"),
        (503, "unknown", "submission"),
    ],
)
@pytest.mark.parametrize("timeout_first", [False, True])
async def test_batch_two_remote_fault_keeps_prefix_and_attempt_history(
    async_mode, private_key_pem, status, outcome, phase, timeout_first
):
    requests, notes = [], []
    second_attempts = 0

    def handler(request):
        nonlocal second_attempts
        requests.append(request)
        value = json.loads(request.content)["bindings"]["1"]["value"]
        if value == "1":
            return success(1)
        assert value == "2", "batch 3 was submitted"
        second_attempts += 1
        if timeout_first and second_attempts == 1:
            raise httpx.ReadTimeout("lost response", request=request)
        return httpx.Response(
            status,
            json={
                "message": "rejected",
                "code": "000904",
                "sqlState": "42000",
                "statementHandle": "failed-h",
            },
        )

    async with wire_client(
        handler, private_key_pem, async_mode, on_query=lambda *a: notes.append(a)
    ) as client:
        with pytest.raises(BatchWriteError) as info:
            await insert(client, async_mode, iter([[1], [2], [3]]), max_rows=1)
    err = info.value
    assert err.completed == BatchWriteResult((BatchSummary(0, 0, 1, 1),))
    assert err.consumed_rows == 2
    assert err.next_row_index == 1
    assert err.phase == phase
    assert err.outcome == ("unknown" if timeout_first else outcome)
    attempts = 3 if status == 503 else 1 + int(timeout_first)
    assert err.failed_batch.index == 1
    assert err.failed_batch.start_row_index == 1
    assert err.failed_batch.row_count == 1
    assert err.failed_batch.submission_attempts == attempts
    assert err.failed_batch.request_id == requests[1].url.params["requestId"]
    assert err.__cause__.submission_attempts == attempts
    if status not in (401, 503):
        assert err.failed_batch.statement_handle == "failed-h"
        assert err.__cause__.code == "000904"
        assert err.__cause__.sql_state == "42000"
    assert len(notes) == 2
    assert len(requests) == 1 + attempts
    for request in requests[2:]:
        assert request.content == requests[1].content
        assert request.url.params["requestId"] == err.failed_batch.request_id
        assert request.url.params["retry"] == "true"


@pytest.mark.parametrize("post_retry", [False, True])
@pytest.mark.parametrize("failure", ["sql", "timeout", "remote", "success"])
async def test_running_post_retains_parent_context_through_polling(
    async_mode, private_key_pem, post_retry, failure
):
    requests, notes = [], []
    posts, gets = 0, 0

    def handler(request):
        nonlocal posts, gets
        requests.append(request)
        if request.method == "POST":
            posts += 1
            if post_retry and posts == 1:
                raise httpx.ReadTimeout("lost response", request=request)
            return httpx.Response(202, json={"statementHandle": "running-h"})
        gets += 1
        if failure == "timeout" or (failure == "success" and gets == 1):
            return httpx.Response(202, json={"statementHandle": "running-h"})
        if failure == "success":
            return success(1)
        return httpx.Response(
            422 if failure == "sql" else 403,
            json={"message": "poll failed", "statementHandle": "running-h"},
        )

    async with wire_client(
        handler,
        private_key_pem,
        async_mode,
        statement_timeout=0,
        on_query=lambda *a: notes.append(a),
    ) as client:
        if failure == "success":
            # A positive wait permits an intermediate running GET.
            client._statement_timeout = 60
            assert (await insert(client, async_mode, [[1]], max_rows=1)).input_rows == 1
        else:
            with pytest.raises(BatchWriteError) as info:
                await insert(client, async_mode, [[1], [2]], max_rows=1)
            err = info.value
            assert err.phase == "execution"
            assert err.outcome == (
                "failed" if failure == "sql" and not post_retry else "unknown"
            )
            assert err.failed_batch.statement_handle == "running-h"
            assert err.failed_batch.submission_attempts == 1 + int(post_retry)
            assert err.failed_batch.request_id == requests[0].url.params["requestId"]
            assert err.completed.input_rows == 0
            assert err.consumed_rows == 1
            assert err.__cause__.submission_attempts is None
            if failure == "timeout":
                assert isinstance(err.__cause__, SnowflakeTimeoutError)
    assert len(notes) == 1
    assert posts == 1 + int(post_retry)
    assert gets == (2 if failure == "success" else 1)


async def test_retry_exhaustion_unknown_without_consuming_next_batch(
    async_mode, private_key_pem
):
    requests = []

    def handler(request):
        requests.append(request)
        raise httpx.ReadTimeout("lost", request=request)

    async with wire_client(handler, private_key_pem, async_mode) as client:
        with pytest.raises(BatchWriteError) as info:
            await insert(client, async_mode, iter([[1], [2]]), max_rows=1)
    assert info.value.consumed_rows == 1
    assert info.value.outcome == "unknown"
    assert info.value.failed_batch.submission_attempts == 3
    assert isinstance(info.value.__cause__, SnowflakeRetryError)
    assert isinstance(info.value.__cause__.__cause__, httpx.ReadTimeout)
    assert len(requests) == 3


@pytest.mark.parametrize("exc_type", [asyncio.CancelledError, KeyboardInterrupt])
async def test_base_exceptions_propagate(async_mode, private_key_pem, exc_type):
    def handler(request):
        raise exc_type()

    async with wire_client(handler, private_key_pem, async_mode) as client:
        with pytest.raises(exc_type):
            await insert(client, async_mode, [[1]])


async def test_payload_and_callback_faults_are_local(async_mode, private_key_pem):
    requests = []
    async with wire_client(
        lambda r: requests.append(r) or success(1),
        private_key_pem,
        async_mode,
        parameters={"bad": object()},
    ) as client:
        with pytest.raises(BatchWriteError) as info:
            await insert(client, async_mode, [[1]])
        assert info.value.phase == "validation"
        assert info.value.failed_batch is None
        assert info.value.consumed_rows == 1
        client._parameters = {}

        def bad_callback(*args):
            raise ValueError("callback failed")

        client.on_query = bad_callback
        with pytest.raises(BatchWriteError) as info:
            await insert(client, async_mode, [[1]])
        assert info.value.phase == "validation"
        assert info.value.failed_batch is None
    assert not requests


async def test_missing_handle_and_unreadable_affected_count(
    async_mode, private_key_pem
):
    for status, body, outcome in [
        (202, {}, "unknown"),
        (200, {"stats": {"numRowsInserted": "bad"}}, "succeeded"),
    ]:
        async with wire_client(
            lambda r, status=status, body=body: httpx.Response(status, json=body),
            private_key_pem,
            async_mode,
        ) as client:
            with pytest.raises(BatchWriteError) as info:
                await insert(client, async_mode, [[1]])
            assert info.value.phase == ("execution" if status == 200 else "submission")
            assert info.value.outcome == outcome
            assert info.value.failed_batch.submission_attempts == 1


async def test_iter_creation_error_is_wrapped(async_mode, private_key_pem):
    class Broken:
        def __iter__(self):
            raise RuntimeError("no iterator")

    async with wire_client(lambda r: success(1), private_key_pem, async_mode) as client:
        with pytest.raises(BatchWriteError) as info:
            await insert(client, async_mode, Broken())
    assert info.value.phase == "input"
    assert info.value.consumed_rows == 0
    assert info.value.failed_batch is None
    assert info.value.completed == BatchWriteResult()


def test_public_progress_and_error_exports():
    assert issubclass(BatchWriteError, SnowflakeError)
    assert FailedBatch(0, 0, 1, "r").submission_attempts == 0
    assert SnowflakeRequestError("x", submission_attempts=2).submission_attempts == 2
    assert SnowflakeProgrammingError("x").submission_attempts is None


async def test_multicolumn_wire_bytes_and_identifier_quoting(
    async_mode, private_key_pem
):
    requests = []

    def handler(request):
        requests.append(request)
        return success(row_count(request, 3))

    class Once:
        iterated = False

        def __iter__(self):
            assert not self.iterated
            self.iterated = True
            return iter([(i, "雪", b"\x00\xff") for i in range(12)])

    async with wire_client(handler, private_key_pem, async_mode) as client:
        result = client.insert_batches(
            'db.t"q', ["id", 'a"b', "blob"], Once(), max_rows=4
        )
        result = await result if async_mode else result
    assert result.input_rows == 12
    assert len(requests) == 3
    for request in requests:
        body = json.loads(request.content)
        assert body[
            "statement"
        ] == 'INSERT INTO "db"."t""q" ("id", "a""b", "blob") VALUES ' + ", ".join(
            ["(?, ?, ?)"] * 4
        )
        assert len(body["bindings"]) == 12


async def test_pending_lookahead_is_counted_when_submission_fails(
    async_mode, private_key_pem
):
    requests = []
    failure = False

    def handler(request):
        requests.append(request)
        return (
            httpx.Response(400, json={"message": "rejected"})
            if failure
            else success(row_count(request))
        )

    async with wire_client(handler, private_key_pem, async_mode) as client:
        await insert(client, async_mode, [["a"]])
        cap = len(requests.pop().content)
        failure = True
        with pytest.raises(BatchWriteError) as info:
            await insert(client, async_mode, iter([["a"], ["b"], ["c"]]), max_bytes=cap)
    assert len(requests) == 1
    assert info.value.consumed_rows == 2
    assert info.value.next_row_index == 0
    assert info.value.failed_batch.row_count == 1
    assert info.value.outcome == "not_executed"


async def test_auth_generation_failure_counts_zero_actual_requests(
    async_mode, private_key_pem, monkeypatch
):
    from snowflake_sql_api import SnowflakeAuthError

    requests = []

    def broken_token():
        raise SnowflakeAuthError("signing failed")

    async with wire_client(
        lambda r: requests.append(r) or success(1), private_key_pem, async_mode
    ) as client:
        monkeypatch.setattr(client._transport._auth, "token", broken_token)
        with pytest.raises(BatchWriteError) as info:
            await insert(client, async_mode, [[1]])
    assert not requests
    assert info.value.failed_batch.submission_attempts == 0
    assert info.value.outcome == "not_executed"


@pytest.mark.parametrize("failure", [RuntimeError, StopIteration])
async def test_iter_creation_exception_is_not_empty_input(
    async_mode, private_key_pem, failure
):
    class Broken:
        def __iter__(self):
            raise failure("broken source")

    async with wire_client(lambda r: success(1), private_key_pem, async_mode) as client:
        with pytest.raises(BatchWriteError) as info:
            await insert(client, async_mode, Broken())
    assert info.value.phase == "input"
    assert info.value.consumed_rows == 0
    assert isinstance(info.value.__cause__, failure)


async def test_transport_submit_and_get_attempt_metadata(async_mode, private_key_pem):
    from snowflake_sql_api import SnowflakeAuthError

    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("lost", request=request)
        return success(1)

    async with wire_client(handler, private_key_pem, async_mode) as client:
        transport = client._transport
        response = transport.submit(
            {"statement": "INSERT INTO t VALUES (1)"}, request_id="r"
        )
        response = await response if async_mode else response
        assert response.submission_attempts == 2
        response = transport.get_statement("h")
        response = await response if async_mode else response
        assert response.submission_attempts is None
        response = transport.cancel("h")
        response = await response if async_mode else response
        assert response.submission_attempts is None
    assert calls == 4
    assert SnowflakeAuthError("x").submission_attempts is None
