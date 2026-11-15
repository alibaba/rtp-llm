"""Select the stream entry point for a real engine or the evaluation mock."""

import asyncio


async def _probe_mock_prefill(stub, pb2):
    peer = await stub.GetPeerInfo(pb2.GetPeerInfoRequestPB(), timeout=1.0)
    return peer.tp_size == 0


async def is_mock_prefill_async(cache, target, stub, pb2):
    # Both evaluation mocks return an empty peer-info response. A real Prefill
    # returns its positive TP size or a gRPC error, which must reach the caller.
    if target not in cache:
        cache[target] = asyncio.create_task(_probe_mock_prefill(stub, pb2))
    probe = cache[target]
    try:
        return await asyncio.shield(probe)
    except Exception:
        if cache.get(target) is probe:
            del cache[target]
        raise


def is_mock_prefill(cache, target, stub, pb2):
    if target not in cache:
        peer = stub.GetPeerInfo(pb2.GetPeerInfoRequestPB(), timeout=1.0)
        cache[target] = peer.tp_size == 0
    return cache[target]
