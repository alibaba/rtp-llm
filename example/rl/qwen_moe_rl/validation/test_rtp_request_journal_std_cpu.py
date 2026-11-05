"""CPU concurrency/lease tests using the candidate's actual strict-client MRO."""
import argparse
import ast
import asyncio
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

_arguments = argparse.ArgumentParser(add_help=False)
_arguments.add_argument("--project-root", type=Path)
_known, _remaining = _arguments.parse_known_args()
sys.argv = [sys.argv[0], *_remaining]
ROOT = _known.project_root or Path(__file__).resolve().parent
if not (ROOT / "adapter").is_dir() and (ROOT.parent / "adapter").is_dir():
    ROOT = ROOT.parent
ROOT = ROOT.resolve()
if not (ROOT / "adapter/rtp_request_journal.py").is_file():
    raise RuntimeError("Use --project-root pointing to the published project's adapter directory")
sys.path.insert(0, str(ROOT / "adapter"))
from rtp_request_journal import NativeRpcJournal, JournaledTokenRpcMixin, NativeRpcAdmissionClosed


@dataclass
class Config:
    max_new_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    do_sample: bool = True
    min_new_tokens: int = 0
    ignore_eos: bool = False
    return_softmax_probs: bool = True
    version: str = "v0"
    fail: bool = False
    invalid_raw: bool = False
    block: object = None

    def HasField(self, name):
        return False


class ProtocolError(RuntimeError):
    pass


class FakeTokenRpcClient:
    def __init__(self):
        self.timeout = 1500
        self.active = self.peak = 0
        self.entered = {}

    async def _request(self, request_id, ids, config):
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.entered.setdefault(request_id, asyncio.Event()).set()
        try:
            if config.block is not None:
                await config.block.wait()
            else:
                await asyncio.sleep(.002)
            if config.fail:
                raise ValueError("fake transport failure")
            infos = [] if config.invalid_raw else [SimpleNamespace(
                DESCRIPTOR=SimpleNamespace(fields_by_name={"raw_logprobs": object()}),
                HasField=lambda name: name == "raw_logprobs")]
            return SimpleNamespace(model_version=config.version, global_step=0,
                                   flatten_output=SimpleNamespace(aux_info=infos))
        finally:
            self.active -= 1

    async def generate(self, request_id, ids, config, *, expected_version=None):
        response = await self._request(request_id, ids, config)
        if expected_version is not None and response.model_version != expected_version:
            raise ProtocolError("post-RPC version validation failed")
        return response

    async def score_tokens(self, request_id, ids, continuation, config, *, expected_version=None):
        response = await self._request(request_id, ids + continuation, config)
        if expected_version is not None and response.model_version != expected_version:
            raise ProtocolError("post-RPC score validation failed")
        return response

    async def close(self):
        return None


# Execute the real nested strict override without importing Ray/Torch/native.
tree = ast.parse((ROOT / "adapter/rtp_multinode_server.py").read_text())
node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
            and n.name == "StrictTokenRpcClient")
namespace = dict(TokenRpcClient=FakeTokenRpcClient, TokenProtocolError=ProtocolError)
exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
             "actual-nested-strict-client", "exec"), namespace)


class ObservedClient(JournaledTokenRpcMixin, namespace["StrictTokenRpcClient"]):
    pass


class Tests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.journal = NativeRpcJournal(self.directory.name, 1,
            lambda: dict(native_boot=0, native_incarnation="cold-or-policy-incarnation"))

    def tearDown(self):
        self.directory.cleanup()

    def client(self, limit=1):
        return ObservedClient(request_journal=self.journal, max_active_requests=limit)

    def rows(self):
        return [json.loads(row) for row in self.journal.path.read_text().splitlines()]

    async def wait_entered(self, client, request_id):
        async def wait():
            while request_id not in client.entered:
                await asyncio.sleep(0)
            await client.entered[request_id].wait()
        await asyncio.wait_for(wait(), 1)

    async def test_gate_one_covers_generate_and_score_without_double_acquire(self):
        c = self.client()
        requests = [c.generate(i, [1, i], Config(version=f"v{i}"), expected_version=f"v{i}")
                    if i % 2 else c.score_tokens(i, [1, i], [99], Config(version=f"v{i}"),
                                                expected_version=f"v{i}") for i in range(12)]
        await asyncio.wait_for(asyncio.gather(*requests), 2)
        self.assertEqual(c.peak, 1)
        rows = self.rows()
        self.assertEqual(len(rows), 36)
        for i in range(12):
            group = [r for r in rows if r["native_request_id"] == str(i)]
            self.assertEqual([r["event"] for r in group], ["queued", "start", "finish"])
            self.assertTrue(all(r["expected_model_version"] == f"v{i}" for r in group))
            self.assertEqual(group[0]["kind"], "generate" if i % 2 else "score_tokens")
            self.assertEqual(group[-1]["outcome"], "success")
            self.assertGreaterEqual(group[1]["queue_wait_seconds"], 0)
            expected = hashlib.sha256(json.dumps(group[0]["prompt_token_ids"]).encode()).hexdigest()
            self.assertEqual(group[1]["prompt_sha256"], expected)
        active = peak = 0
        for row in rows:
            if row["event"] == "start":
                active += 1
                peak = max(peak, active)
            elif row["event"] == "finish" and row["started_unix"] is not None:
                active -= 1
        self.assertEqual((active, peak), (0, 1))

    async def test_default_zero_does_not_serialize_requests(self):
        c = self.client(0)
        await asyncio.gather(*(c.generate(i, [i], Config()) for i in range(5)))
        self.assertGreater(c.peak, 1)

    async def test_exception_and_strict_raw_validation_release_gate(self):
        c = self.client()
        result = await asyncio.gather(c.generate(1, [1], Config(fail=True)),
            c.generate(2, [2], Config(invalid_raw=True)), c.generate(3, [3], Config()),
            return_exceptions=True)
        self.assertIsInstance(result[0], ValueError)
        self.assertIsInstance(result[1], ProtocolError)
        self.assertEqual(c.active, 0)
        self.assertEqual([r["outcome"] for r in self.rows() if r["event"] == "finish"],
                         ["error", "error", "success"])

    async def test_active_cancellation_releases_lease(self):
        c = self.client()
        active = asyncio.create_task(c.generate(1, [1], Config(block=asyncio.Event())))
        await self.wait_entered(c, 1)
        pending = asyncio.create_task(c.generate(2, [2], Config()))
        active.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await active
        await asyncio.wait_for(pending, 1)
        self.assertEqual(c.peak, 1)
        finish = [r for r in self.rows() if r["event"] == "finish"]
        self.assertEqual([r["outcome"] for r in finish], ["cancelled", "success"])

    async def test_queued_cancellation_has_no_native_start(self):
        c = self.client()
        release = asyncio.Event()
        active = asyncio.create_task(c.generate(1, [1], Config(block=release)))
        await self.wait_entered(c, 1)
        queued = asyncio.create_task(c.generate(2, [2], Config()))
        await asyncio.sleep(.002)
        queued.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await queued
        release.set()
        await active
        rows = [r for r in self.rows() if r["native_request_id"] == "2"]
        self.assertEqual([r["event"] for r in rows], ["queued", "finish"])
        self.assertIsNone(rows[-1]["started_unix"])
        self.assertNotIn(2, c.entered)

    async def test_pause_retires_old_queue_even_after_reopen(self):
        c = self.client()
        release = asyncio.Event()
        active = asyncio.create_task(c.generate(1, [1], Config(block=release)))
        await self.wait_entered(c, 1)
        old_queue = asyncio.create_task(c.generate(2, [2], Config()))
        await asyncio.sleep(.002)
        c.close_admission("training_pause")
        with self.assertRaises(NativeRpcAdmissionClosed):
            await c.generate(3, [3], Config())
        c.open_admission()  # A post-refit probe is permitted under the new epoch.
        release.set()
        await active
        with self.assertRaises(NativeRpcAdmissionClosed):
            await old_queue
        await c.score_tokens(4, [4], [5], Config(version="new"), expected_version="new")
        self.assertNotIn(2, c.entered)
        self.assertNotIn(3, c.entered)
        self.assertIn(4, c.entered)

    async def test_finish_catches_post_rpc_validation_and_cold_null_version(self):
        c = self.client()
        with self.assertRaises(ProtocolError):
            await c.generate(1, [1], Config(version="actual"), expected_version="stale")
        await c.score_tokens(2, [1], [2], Config(version="cold"), expected_version=None)
        finishes = [r for r in self.rows() if r["event"] == "finish"]
        self.assertEqual(finishes[0]["transport_outcome"], "response_received")
        self.assertEqual(finishes[0]["outcome"], "error")
        self.assertIsNone(finishes[1]["expected_model_version"])

    async def test_pending_start_is_readable_before_rpc_finishes(self):
        c = self.client()
        pending = asyncio.create_task(c.generate(1, [42, 248046], Config(block=asyncio.Event())))
        await self.wait_entered(c, 1)
        rows = self.rows()
        self.assertEqual([r["event"] for r in rows], ["queued", "start"])
        self.assertEqual(rows[1]["prompt_token_ids"], [42, 248046])
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending


if __name__ == "__main__":
    unittest.main(verbosity=2)
