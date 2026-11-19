from __future__ import annotations

import unittest
from dataclasses import dataclass

from rtp_llm.kv_cache_subscriber.source import RtpGrpcCacheStatusSource


@dataclass(frozen=True)
class _Request:
    latest_cache_version: int
    need_cache_keys: bool


class _Pb2:
    CacheVersionPB = _Request


@dataclass(frozen=True)
class _Response:
    cache_keys: dict[int, bool]
    block_size: int
    version: int


class _Stub:
    def __init__(
        self,
        response: _Response | None = None,
        error: Exception | None = None,
    ) -> None:
        self._response = response
        self._error = error
        self.requests: list[tuple[_Request, float]] = []

    async def GetCacheStatus(self, request: _Request, timeout: float) -> _Response:
        self.requests.append((request, timeout))
        if self._error is not None:
            raise self._error
        if self._response is None:
            raise AssertionError("fake stub has no response")
        return self._response


class _Channel:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _source(*stubs: _Stub) -> RtpGrpcCacheStatusSource:
    source = object.__new__(RtpGrpcCacheStatusSource)
    source._pb2 = _Pb2
    source._timeout_s = 2.5
    source._stubs = list(stubs)
    source._channels = []
    return source


class RtpGrpcCacheStatusSourceTest(unittest.IsolatedAsyncioTestCase):
    async def test_merges_all_dp_keys_and_uses_full_snapshot_request(self) -> None:
        first = _Stub(_Response({1: True, 2: False}, block_size=16, version=7))
        second = _Stub(_Response({2: True, 3: True}, block_size=16, version=5))
        source = _source(first, second)

        snapshot = await source.fetch_snapshot()

        self.assertEqual(snapshot.keys, frozenset({1, 2, 3}))
        self.assertEqual(snapshot.block_size, 16)
        self.assertEqual(snapshot.version, 5)
        expected_request = _Request(
            latest_cache_version=-1,
            need_cache_keys=True,
        )
        self.assertEqual(first.requests, [(expected_request, 2.5)])
        self.assertEqual(second.requests, [(expected_request, 2.5)])

    async def test_one_failed_dp_rejects_the_whole_snapshot(self) -> None:
        healthy = _Stub(_Response({1: True}, block_size=16, version=1))
        failed = _Stub(error=RuntimeError("DP unavailable"))

        with self.assertRaisesRegex(RuntimeError, "DP unavailable"):
            await _source(healthy, failed).fetch_snapshot()

    async def test_inconsistent_block_sizes_are_rejected(self) -> None:
        first = _Stub(_Response({1: True}, block_size=16, version=1))
        second = _Stub(_Response({2: True}, block_size=32, version=1))

        with self.assertRaisesRegex(RuntimeError, "inconsistent block sizes"):
            await _source(first, second).fetch_snapshot()

    async def test_close_releases_every_grpc_channel(self) -> None:
        source = _source(_Stub(_Response({}, block_size=16, version=1)))
        channels = [_Channel(), _Channel()]
        source._channels = channels

        await source.close()

        self.assertTrue(all(channel.closed for channel in channels))


if __name__ == "__main__":
    unittest.main()
