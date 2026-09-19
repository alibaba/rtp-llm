"""Shared ViT hash lookup/submission for HTTP and gRPC entry points."""

import sys
from array import array

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import MultimodalHashResponsePB

MAX_METADATA_KEYS = 256
MAX_METADATA_ROWS = 1048576
MAX_METADATA_BYTES = 16 * 1024 * 1024


MM_CACHE_SNAPSHOT_MAX_KEYS = 100000
MM_CACHE_SNAPSHOT_MAX_BYTES = 16 * 1024 * 1024


def get_mm_cache_keys(engine, max_keys=MM_CACHE_SNAPSHOT_MAX_KEYS):
    """Snapshot routing hashes and resident tiers without reading tensor data."""
    if engine is None or engine.is_proxy_mode:
        raise NotImplementedError("worker-local cache required")
    hash_key_cache = getattr(engine, "_hash_key_cache", None)
    if hash_key_cache is None:
        cache = engine._embedding_cache
        keys = cache.metadata_keys()
        worker_instance = cache.instance_id
    else:
        keys = hash_key_cache.keys(limit=max_keys)
        worker_instance = hash_key_cache.instance_id
    tiers = engine._embedding_cache.resident_tiers(limit=max_keys)
    # Keep recent routing hashes first, then fill the snapshot with embeddings.
    selected = set(keys)
    for key in tiers:
        if len(selected) >= max_keys:
            break
        selected.add(key)
    tiers = {key: tier for key, tier in tiers.items() if key in selected}
    if len(keys) > max_keys:
        raise OverflowError("cache key snapshot too large")
    return {
        "worker_instance": worker_instance,
        "feature_hash_version": 1,
        "keys": keys,
        "gpu_embedding_keys": [key for key, tier in tiers.items() if tier == "gpu"],
        "cpu_embedding_keys": [key for key, tier in tiers.items() if tier == "cpu"],
    }


def get_mm_cache_metadata(
    engine,
    keys,
    inputs=None,
    timeout_ms=120000,
    user_id="",
    service_name="",
    cancellation_event=None,
    binary_hashes=False,
    model_name="",
):
    if engine is None or engine.is_proxy_mode:
        # Routing metadata belongs to the exact worker selected by FlexLB.
        raise NotImplementedError("worker-local cache required")
    if len(keys) > MAX_METADATA_KEYS:
        raise ValueError("too many multimodal cache keys")
    if any(not key or len(key) > 4096 for key in keys):
        raise ValueError("invalid multimodal cache key")
    if timeout_ms <= 0:
        raise ValueError("multimodal hash timeout must be positive")
    cache = getattr(engine, "_hash_key_cache", None)
    try:
        if cache is None:
            metadata = engine._embedding_cache.metadata(keys)
            if binary_hashes:
                for entry in metadata["entries"]:
                    hashes = array("i", entry.get("feature_hashes", []))
                    if sys.byteorder != "little":
                        hashes.byteswap()
                    entry["feature_hashes"] = hashes.tobytes()
            return metadata
        metadata = cache.metadata(
            keys, engine._embedding_cache, binary_hashes=binary_hashes
        )
    except ValueError as error:
        raise OverflowError(str(error)) from error
    inspection_enabled = (
        engine._greennet_enabled() if hasattr(engine, "_greennet_enabled") else False
    )
    if inspection_enabled:
        for entry in metadata["entries"]:
            if not entry.get("greennet_passed", False):
                entry.update(hit=False, hash_hit=False)
                for field in ("feature_hashes", "split_size", "entry_generation"):
                    entry.pop(field, None)
    missing = {entry["key"] for entry in metadata["entries"] if not entry["hash_hit"]}
    if not missing or inputs is None:
        return metadata

    from rtp_llm.multimodal.multimodal_util import trans_mm_input

    if len(inputs.multimodal_inputs) > MAX_METADATA_KEYS:
        raise ValueError("too many multimodal inputs")
    if any(
        not item.multimodal_url or item.multimodal_tensor.ByteSize()
        for item in inputs.multimodal_inputs
    ):
        raise ValueError("hash submission requires URL inputs without tensors")
    by_key = {item.cache_key(): item for item in trans_mm_input(inputs)}
    if not missing.issubset(by_key) or not by_key.keys() <= set(keys):
        raise ValueError("multimodal inputs do not match requested cache keys")
    missing_keys = [key for key in dict.fromkeys(keys) if key in missing]
    kwargs = (
        {"cancellation_event": cancellation_event}
        if cancellation_event is not None
        else {}
    )
    results = engine.get_embedding_result(
        [by_key[key] for key in missing_keys],
        request_id=inputs.request_id,
        timeout_ms=timeout_ms,
        hashes_only=True,
        user_id=user_id,
        service_name=service_name,
        model_name=model_name,
        **kwargs,
    )
    if len(results) != len(missing_keys):
        raise FtRuntimeException(
            ExceptionType.MM_PROCESS_ERROR, "ViT returned incomplete feature hashes"
        )
    tiers = engine._embedding_cache.resident_tiers(missing_keys)
    completed = {}
    total_rows = sum(sum(entry.get("split_size", [])) for entry in metadata["entries"])
    for key, result in zip(missing_keys, results):
        hashes = result.feature_hashes
        if not hashes or any(h.numel() == 0 for h in hashes):
            raise FtRuntimeException(
                ExceptionType.MM_PROCESS_ERROR, "ViT returned no feature hashes"
            )
        sizes = [h.numel() for h in hashes]
        total_rows += sum(sizes)
        if total_rows > MAX_METADATA_ROWS:
            raise OverflowError("multimodal metadata response exceeds row limit")
        completed[key] = {
            "key": key,
            "hit": True,
            "hash_hit": True,
            "greennet_passed": inspection_enabled,
            "embedding_hit": key in tiers,
            "embedding_tier": tiers.get(key),
            "split_size": sizes,
            "feature_hashes": (
                b"".join(
                    tensor.numpy().astype("<i4", copy=False).tobytes()
                    for tensor in hashes
                )
                if binary_hashes
                else [int(h) for tensor in hashes for h in tensor.tolist()]
            ),
        }
    metadata["entries"] = [
        completed.get(entry["key"], entry) for entry in metadata["entries"]
    ]
    return metadata


def metadata_to_proto(metadata):
    response = MultimodalHashResponsePB(
        worker_instance=metadata["worker_instance"],
        feature_hash_version=metadata["feature_hash_version"],
    )
    for entry in metadata["entries"]:
        response.entries.add(
            key=entry["key"],
            hash_hit=entry.get("hash_hit", entry.get("hit", False)),
            embedding_hit=entry.get("embedding_hit", False),
            embedding_tier=entry.get("embedding_tier") or "",
            greennet_passed=entry.get("greennet_passed", False),
            entry_generation=entry.get("entry_generation") or "",
            split_size=entry.get("split_size", []),
            feature_hashes=entry.get("feature_hashes", b""),
        )
    return response


def metadata_from_proto(response):
    """Keep hashes in a four-byte array instead of allocating Python ints."""
    if len(response.entries) > MAX_METADATA_KEYS:
        raise ValueError("too many multimodal metadata entries")
    entries = []
    total_rows = 0
    for entry in response.entries:
        hashes = array("i")
        hashes.frombytes(entry.feature_hashes)
        if sys.byteorder != "little":
            hashes.byteswap()
        sizes = list(entry.split_size)
        if entry.hash_hit and (
            not sizes or any(n == 0 for n in sizes) or sum(sizes) != len(hashes)
        ):
            raise ValueError("invalid multimodal hash segment lengths")
        total_rows += len(hashes)
        if total_rows > MAX_METADATA_ROWS:
            raise ValueError("multimodal metadata response exceeds row limit")
        entries.append(
            {
                "key": entry.key,
                "hit": entry.hash_hit,
                "hash_hit": entry.hash_hit,
                "embedding_hit": entry.embedding_hit,
                "embedding_tier": entry.embedding_tier or None,
                "greennet_passed": entry.greennet_passed,
                "entry_generation": entry.entry_generation,
                "split_size": sizes,
                "feature_hashes": hashes,
            }
        )
    return {
        "worker_instance": response.worker_instance,
        "feature_hash_version": response.feature_hash_version,
        "entries": entries,
    }
