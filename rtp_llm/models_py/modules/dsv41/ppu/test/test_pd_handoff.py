"""Owner isolation, TCP integrity and failure cleanup for internal PD bring-up."""

import copy
import json
import socket
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from rtp_llm.models_py.modules.dsv41.ppu.cache import PPUKVCache
from rtp_llm.models_py.modules.dsv41.ppu.pd_handoff import (
    _expect,
    _send_json,
    export_handoffs,
    receive_handoff,
    file_sha256,
    send_handoff,
    validate_handoff,
    runtime_identity,
)
from rtp_llm.models_py.modules.dsv41.ppu.weight import synthetic_config

IDENTITY = {
    "snapshot_id": "test-snapshot",
    "source_sha256": "test-source",
    "dependencies": {"torch": "test"},
}


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def connect(port):
    deadline = time.monotonic() + 5
    while True:
        try:
            return socket.create_connection(("127.0.0.1", port), timeout=5)
        except ConnectionRefusedError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.01)


class PDHandoffTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="v41-pd-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cache = PPUKVCache(synthetic_config())
        self.cache.alloc(0, 9)
        self.cache.alloc(11, 7)
        for index, key in enumerate(self.cache.pool_keys()):
            pool = self.cache._pools[key]
            pool.copy_(
                (torch.arange(pool.numel()).reshape(pool.shape) + index).to(pool.dtype)
            )
        self.paths = export_handoffs(
            self.cache,
            self.root / "out",
            histories={0: [2] * 9, 11: [3] * 7},
            seed_tokens={0: 4, 11: 5},
            owner_map={0: 0, 11: 1},
            identity=IDENTITY,
            model_signature="model",
            run_id="run",
            world=2,
        )
        self.manifest = json.loads((self.paths[0] / "manifest.json").read_text())

    def receive(self, port, **kwargs):
        return receive_handoff(
            self.root / "received",
            "127.0.0.1",
            port,
            identity=IDENTITY,
            model_signature="model",
            run_id="run",
            owner=0,
            world=2,
            timeout=5,
            **kwargs
        )

    def test_select_compacts_planar_blocks_and_does_not_alias(self):
        selected = self.cache.select_requests([11])
        for key in self.cache.pool_keys():
            blocks = self.cache._block_tables[key][11]
            self.assertTrue(
                torch.equal(selected._pools[key], self.cache._pools[key][blocks])
            )
            self.assertEqual(
                selected._block_tables[key], {11: list(range(len(blocks)))}
            )
        self.assertEqual(selected._request_tokens, {11: 7})
        before = selected.pool(2, "indexer_kv").clone()
        self.cache._pools[(2, "indexer_kv")].zero_()
        self.assertTrue(torch.equal(before, selected.pool(2, "indexer_kv")))
        selected.alloc(11, 20)  # the new blocks must not reuse restored live blocks
        self.assertTrue(
            torch.equal(before, selected.pool(2, "indexer_kv")[: before.shape[0]])
        )

    def test_empty_owner_allocates_no_cache(self):
        empty = self.cache.select_requests([])
        self.assertEqual(empty._request_tokens, {})
        self.assertEqual(sum(empty.bytes_summary().values()), 0)

    def test_owner_export_has_exact_request_set(self):
        cache, manifest = PPUKVCache.load_fixture(self.paths[1])
        self.assertEqual(cache._request_tokens, {11: 7})
        self.assertEqual(
            cache.state_checksum(), manifest["meta"]["pd"]["state_checksum"]
        )
        with self.assertRaisesRegex(ValueError, "coverage"):
            export_handoffs(
                self.cache,
                self.root / "bad",
                histories={},
                seed_tokens={},
                owner_map={},
                identity=IDENTITY,
                model_signature="model",
                run_id="run",
                world=2,
            )

    def test_contract_rejects_identity_topology_and_history_mismatch(self):
        mutations = [
            lambda m: m["meta"]["pd"].update(run_id="other"),
            lambda m: m["meta"]["pd"].update(identity={}),
            lambda m: m["meta"]["pd"].update(model_signature="other"),
            lambda m: m["meta"]["pd"].update(target_tp=2),
            lambda m: m["meta"]["pd"].update(owner=1),
            lambda m: m.update(source_cp=1),
            lambda m: m["pool_layouts"].clear(),
            lambda m: m["meta"]["pd"]["histories"].update({"0": [2]}),
            lambda m: m["meta"]["pd"]["seed_tokens"].clear(),
            lambda m: m["meta"]["pd"]["owner_map"].update({"0": 1}),
        ]
        for mutate in mutations:
            manifest = copy.deepcopy(self.manifest)
            mutate(manifest)
            with self.assertRaises(ValueError):
                validate_handoff(
                    manifest,
                    identity=IDENTITY,
                    model_signature="model",
                    run_id="run",
                    owner=0,
                    world=2,
                )

    def test_tcp_roundtrip_waits_for_decoded_result(self):
        port = free_port()

        def consumer():
            session = self.receive(port)
            cache, manifest = PPUKVCache.load_fixture(session.path)
            self.assertEqual(
                cache.state_checksum(), manifest["meta"]["pd"]["state_checksum"]
            )
            session.finish({"ok": True, "generated_tokens": {"0": [4, 6]}})

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(consumer)
            report = send_handoff(self.paths[0], "127.0.0.1", port, timeout=5)
            future.result(timeout=5)
        self.assertEqual(report["decode_result"]["generated_tokens"], {"0": [4, 6]})
        self.assertEqual(report["checksum"], self.manifest["checksum"])
        self.assertFalse(list(self.root.glob(".*.partial-*")))

    def test_partial_and_corrupt_payloads_are_removed(self):
        for corrupt in (False, True):
            port = free_port()
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self.receive, port)
                with connect(port) as sock:
                    _send_json(sock, {"manifest": self.manifest, "payload_bytes": 10})
                    _expect(sock, "ready")
                    sock.sendall(b"bad bytes!" if corrupt else b"short")
                with self.assertRaises((ValueError, ConnectionError)):
                    future.result(timeout=5)
            self.assertFalse((self.root / "received").exists())
            self.assertFalse(list(self.root.glob(".*.partial-*")))

    def test_misrouted_sender_gets_explicit_failure(self):
        port = free_port()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.receive, port)
            with self.assertRaisesRegex(RuntimeError, "owner"):
                send_handoff(self.paths[1], "127.0.0.1", port, timeout=5)
            with self.assertRaisesRegex(ValueError, "owner"):
                future.result(timeout=5)
        self.assertFalse((self.root / "received").exists())

    def test_accept_timeout_removes_staging(self):
        with self.assertRaises(TimeoutError):
            receive_handoff(
                self.root / "received",
                "127.0.0.1",
                free_port(),
                identity=IDENTITY,
                model_signature="model",
                run_id="run",
                owner=0,
                world=2,
                timeout=0.05,
            )
        self.assertFalse(list(self.root.glob(".*.partial-*")))

    def test_existing_destination_is_not_overwritten(self):
        (self.root / "received").mkdir()
        with self.assertRaises(FileExistsError):
            self.receive(free_port())

    def test_invalid_pool_dtype_is_rejected_even_with_matching_payload_hash(self):
        path = self.paths[0]
        payload = torch.load(path / "pools.pt", weights_only=True)
        key = next(iter(payload["pools"]))
        payload["pools"][key] = payload["pools"][key].to(torch.float32)
        torch.save(payload, path / "pools.pt")
        manifest = dict(self.manifest, checksum=file_sha256(path / "pools.pt"))
        (path / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "dtype/stride/block"):
            PPUKVCache.load_fixture(path)

    def test_published_source_tamper_is_rejected(self):
        import hashlib

        source = self.root / "source"
        source.mkdir()
        code = source / "driver.py"
        code.write_text("original")
        files = [{"path": "driver.py", "kind": "file", "value": file_sha256(code)}]
        manifest = {
            "mode": "snapshot",
            "snapshot_id": "test",
            "files": files,
            "source_sha256": hashlib.sha256(
                json.dumps(files, sort_keys=True).encode()
            ).hexdigest(),
        }
        (source / ".v41_snapshot.json").write_text(json.dumps(manifest))
        self.assertEqual(runtime_identity(source)["snapshot_id"], "test")
        code.write_text("changed")
        with self.assertRaisesRegex(ValueError, "source changed"):
            runtime_identity(source)


if __name__ == "__main__":
    unittest.main()
