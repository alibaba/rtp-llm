"""BTC Host rows on the single-node, whole-page Flash V6D/KVCM wire layout."""

from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import closing

Block = tuple[int, bytes]


class V6dStorageBackend:
    """One background executor owns this client and all V6D object leases.

    The Host slot starts with contiguous rank-major physical carrier bytes
    in the deployed KVS order. Any allocation alignment after the last rank
    is excluded from the object and checksums.
    """

    def __init__(self, config: Mapping, rank_bytes: int, ranks: int):
        import torch
        from v6d.data.tensor import Tensor
        from v6d.server.tair_kvcm.store import normalize_kv_cache_groups

        self._v6d_url = config["v6d_url"]
        self._kvcm_url = config["kvcm_url"]
        self._instance = config["instance_id"]
        if not all(
            isinstance(v, str) and v
            for v in (self._v6d_url, self._kvcm_url, self._instance)
        ):
            raise ValueError("BTC V6D requires v6d_url, kvcm_url and instance_id")
        block_size = config["block_size"]
        if any(type(v) is not int or v <= 0 for v in (rank_bytes, ranks, block_size)):
            raise ValueError(
                "BTC V6D requires positive rank_bytes, ranks and block_size"
            )
        self._rank_bytes = rank_bytes
        self._bytes = ranks * rank_bytes
        self._fields = [f"ch3:{rank}" for rank in range(ranks)]
        self._object_type = Tensor.type(shape=[ranks, 1, rank_bytes], dtype=torch.uint8)
        groups = sorted(
            (
                dict(
                    group_id=int(g),
                    spec_type=t,
                    object_size_bytes=self._bytes,
                    meta=self._object_type,
                )
                for g, t in config["group_spec_types"].items()
            ),
            key=lambda item: item["group_id"],
        )
        self._spec_names = {
            (g.group_id, 0): g.spec_name for g in normalize_kv_cache_groups(groups)
        }
        self._spec = {
            "block_size": block_size,
            "groups": groups,
            "node_count": 1,
            "use_eagle_pop": config.get("use_eagle_pop", False),
        }
        self._timeout = float(config.get("commit_timeout_s", 5.0))
        if not math.isfinite(self._timeout) or self._timeout <= 0:
            raise ValueError("BTC V6D commit_timeout_s must be finite and positive")
        self._client = None
        self._manager = None
        self._write_mode = None
        self._closed = False

    def _connect(self) -> None:
        import v6d
        from v6d.server.tair_kvcm.manager_client import HttpKvCacheManagerClient

        if self._closed:
            raise RuntimeError("BTC V6D backend is closed")
        if self._client is not None:
            return
        # An empty peer can accept its first KV spec before it can advertise
        # Tair transport parameters. Register through a control-only client
        # so the data client receives those parameters in its first handshake.
        with closing(v6d.connect(self._v6d_url, control_only=True)) as control:
            status = control.peer_info.get("kvcm_status", {})
            if status.get("mode") != "report-event-and-cache-transfer":
                raise ValueError(
                    "BTC requires a V6D peer in report-event-and-cache-transfer mode"
                )
            # The daemon latches the first registration. An idempotent 'skipped'
            # response is not proof another writer registered the same ABI.
            if not status.get("report_event_ready"):
                result = control.describe_kv_cache(self._spec)
                if result.get("reason") == "already registered":
                    # No previous engine spec existed. This daemon copied our
                    # description without applying it to its running store.
                    raise ValueError(
                        "V6D was registered before an engine declared its KV layout"
                    )
            actual = control.get_kv_cache_spec()
            if (
                actual.get("block_size") != self._spec["block_size"]
                or actual.get("node_count", 1) != 1
                or actual.get("use_eagle_pop", False) != self._spec["use_eagle_pop"]
                or sorted(actual["groups"], key=lambda g: g["group_id"])
                != self._spec["groups"]
            ):
                raise ValueError(
                    "registered V6D KV spec differs from BTC Host row geometry"
                )
            deadline = time.monotonic() + self._timeout
            peer = control.rpc.refresh_peer_info()
            while not peer["kvcm_status"].get("cache_transfer_ready") or (
                peer["is_lazy_strategy"] and not peer.get("tair_kvcm", {}).get("ready")
            ):
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "V6D has not completed its Tair cache registration"
                    )
                time.sleep(0.05)
                peer = control.rpc.refresh_peer_info()

        client = v6d.connect(self._v6d_url)
        try:
            peer = client.peer_info
            status = peer["kvcm_status"]
            if status["mode"] != "report-event-and-cache-transfer" or not status["cache_transfer_ready"]:
                raise RuntimeError("V6D Tair registration changed during data client startup")
            # The lazy data plane advertises the actual Tair SDK registration.
            # Eager peers expose only mode/readiness; the configured manager
            # and instance provide the independent per-key durability proof.
            if peer["is_lazy_strategy"]:
                # V6D catches SDK initialization errors in its constructor.
                # Do not publish this client until it can really cold-read.
                if client.tair_kvcm_runtime is None:
                    raise RuntimeError("V6D lazy Tair data client has no transfer runtime")
                transfer = json.loads(peer["tair_kvcm"]["transfer_client_config"])
                spec_sizes = dict.fromkeys(self._spec_names.values(), self._bytes)
                if (
                    transfer["instance_id"] != self._instance
                    or transfer["block_size"] != self._spec["block_size"]
                    or transfer["location_spec_infos"] != spec_sizes
                ):
                    raise ValueError(
                        "V6D lazy Tair registration differs from the BTC instance or layout"
                    )
            write_mode = peer["kvcm_status"]["write_mode"]
            if write_mode not in {"evict", "seal", "seal-and-evict"}:
                raise ValueError(f"unsupported Tair write mode {write_mode!r}")
            manager = HttpKvCacheManagerClient(
                self._kvcm_url, instance_id=self._instance
            )
        except BaseException:
            client.close()
            raise
        self._client = client
        self._manager = manager
        self._write_mode = write_mode

    def _proofs(self, blocks: Sequence[Block]) -> dict[Block, dict]:
        """Require the exact spec on Tair and every persisted rank digest."""
        from v6d.common.tair_kvcm.checksum import decode_checksum
        from v6d.server.tair_kvcm.store import parse_kvcm_object_identity

        if not blocks:
            return {}
        self._connect()
        confirmed = {}
        for start in range(0, len(blocks), 256):
            batch = blocks[start : start + 256]
            identities = [
                parse_kvcm_object_identity(f"{key.hex()}_{group}", self._spec_names)
                for group, key in batch
            ]
            response = self._manager.get_cache_locations_by_backend(
                {
                    "trace_id": str(uuid.uuid4()),
                    "instance_id": self._instance,
                    "query_type": "QT_BATCH_GET",
                    "block_keys": [i.signed_block_key for i in identities],
                    "block_mask": {"bool_masks": {"values": [False] * len(batch)}},
                    "location_spec_names": [i.spec_name for i in identities],
                    "backend_selectors": [
                        {
                            "backend_type": "ST_TAIRMEMPOOL",
                            "strategy": "LSS_WEIGHTED_RANDOM",
                        }
                    ],
                    "include_checksums": True,
                }
            )
            for block, ident, row in zip(
                batch, identities, response["key_locations"], strict=True
            ):
                for loc in row.get("locations", ()):
                    if loc.get("type") != "ST_TAIRMEMPOOL":
                        continue
                    for spec in loc.get("location_specs", ()):
                        if spec.get("name") == ident.spec_name and spec.get("uri"):
                            digests = decode_checksum(spec.get("checksum", ""))
                            if digests.keys() == set(self._fields):
                                confirmed[block] = digests
        return confirmed

    def lookup(self, keys: Sequence[Block]) -> set[Block]:
        return set(self._proofs(keys))

    def _rank_checksums(self, data: memoryview) -> dict[str, str]:
        import torch
        from kvs_connector.integrity import content_hash

        tensor = torch.frombuffer(data, dtype=torch.uint8, count=self._bytes)
        rows = tensor.view(len(self._fields), self._rank_bytes)
        return {
            field: content_hash(row)
            for field, row in zip(self._fields, rows, strict=True)
        }

    def _matches(self, obj, data: memoryview, digests: dict) -> bool:
        return (
            data.nbytes == self._bytes
            and json.loads(obj.meta["type_content"]) == self._object_type
            and all(obj.meta.get(k) == v for k, v in digests.items())
            and self._rank_checksums(data) == digests
        )

    def read(self, group: int, key: bytes, dest: memoryview) -> bool:
        block = (group, key)
        target = dest.cast("B")
        if target.readonly or target.nbytes < self._bytes:
            raise ValueError("BTC V6D read requires a writable complete Host slot")
        proof = self._proofs([block]).get(block)
        if proof is None:
            return False
        obj = self._client.get(f"{key.hex()}_{group}")
        if obj is None:
            return False
        with obj, obj.buffer.cast("B") as data:
            if not self._matches(obj, data, proof):
                return False
            target[: self._bytes] = data
        return True

    def write(self, group: int, key: bytes, source: memoryview) -> bool:
        block = (group, key)
        data = source.cast("B")[: self._bytes]
        if data.nbytes != self._bytes:
            raise ValueError("BTC V6D write requires a complete Host slot")
        digests = self._rank_checksums(data)
        proof = self._proofs([block]).get(block)
        if proof is not None:
            return proof == digests
        name = f"{key.hex()}_{group}"
        objects = self._client.create(
            keys=[name],
            sizes=[self._bytes],
            object_types=[self._object_type],
            ignore_existing=True,
        )
        obj = objects[0] if objects else None
        if obj is not None:
            with obj:
                try:
                    with obj.buffer.cast("B") as target:
                        target[:] = data
                    obj.update_meta(digests)
                except BaseException:
                    obj.discard()
                    raise
                obj.seal()

        # In evict mode seal only makes the Vineyard object readable.
        # For an existing sealed object this also retries skipped seal writes.
        # Only KVCM readback settles the BTC source pin.
        should_evict = self._write_mode in {"evict", "seal-and-evict"}
        if should_evict and obj is None:
            # A concurrent writer or another storage tier may already own
            # this name. Only evict a sealed local copy of our exact page.
            existing = self._client.get(name, peer="local")
            should_evict = existing is not None
            if existing is not None:
                with existing, existing.buffer.cast("B") as resident:
                    if not self._matches(existing, resident, digests):
                        return False
        if should_evict:
            self._client.rpc_call_with_exception("objects/evict", {"keys": [name]})
        deadline = time.monotonic() + self._timeout
        while True:
            proof = self._proofs([block]).get(block)
            if proof is not None:
                return proof == digests
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.05, remaining))

    def close(self) -> None:
        self._closed = True
        try:
            if self._manager is not None:
                self._manager.close()
        finally:
            if self._client is not None:
                self._client.close()
            self._manager = self._client = None
