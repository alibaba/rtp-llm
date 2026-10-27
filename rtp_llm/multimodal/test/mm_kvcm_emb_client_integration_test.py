"""Manual RTP -> live KVCM -> existing PACE integration tests.

Run on the development machine, against an isolated KVCM/PACE deployment::

    RTP_KVCM_RUN_INTEGRATION=1 \
    RTP_KVCM_ENDPOINT=127.0.0.1:19001 \
    RTP_KVCM_INSTANCE_GROUP=emb_it \
    RTP_KVCM_SDK_CONFIG='[{"type":"pace"}]' \
    python -m unittest -v \
      rtp_llm.multimodal.test.mm_kvcm_emb_client_integration_test

Create kve_emb_it on the server first. Alternatively set
RTP_KVCM_CLIENT_CONFIG to the existing RECO_CLIENT_CONFIG JSON map. The tests
do not start/stop services or alter PACE. Each test uses fresh UUID keys and
checks payload bytes; successful Remove only proves logical deletion, not
physical reclamation. Service restart and backend fault injection are separate
deployment checks and are not inferred from these results.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from unittest import TestCase, main, skipUnless

try:
    import torch
except ModuleNotFoundError:
    torch = None

from rtp_llm.multimodal.kvcm import RtpKvMetaObjectClient

_RUN_INTEGRATION = os.environ.get("RTP_KVCM_RUN_INTEGRATION") == "1"
_WORKER_MARKER = "RTP_KVCM_WORKER_RESULT="
_WORKER_MODULE = "rtp_llm.multimodal.test.mm_kvcm_emb_client_integration_test"
_DTYPES = ("uint8", "int32", "float16", "bfloat16", "float32")


def _environment() -> dict[str, str]:
    explicit = os.environ.get("RTP_KVCM_CLIENT_CONFIG", "")
    if explicit:
        return {"RECO_CLIENT_CONFIG": explicit}
    endpoint = os.environ.get("RTP_KVCM_ENDPOINT", "")
    group = os.environ.get("RTP_KVCM_INSTANCE_GROUP", "")
    sdk = os.environ.get("RTP_KVCM_SDK_CONFIG", "")
    if not endpoint or not group or not sdk:
        raise RuntimeError(
            "set RTP_KVCM_CLIENT_CONFIG or all of RTP_KVCM_ENDPOINT, "
            "RTP_KVCM_INSTANCE_GROUP and RTP_KVCM_SDK_CONFIG"
        )
    return {
        "RECO_ENABLE_VIPSERVER": "0",
        "RECO_SERVER_ADDRESS": endpoint,
        "RECO_INSTANCE_GROUP": group,
        "RECO_INSTANCE_ID_SALT": os.environ.get("RTP_KVCM_INSTANCE_ID_SALT", ""),
        "RECO_MODEL_SDK_CONFIG": sdk,
        "RECO_META_CHANNEL_RETRY_TIME": "0",
        "RECO_META_CHANNEL_CONNECTION_TIMEOUT": "1000",
        "RECO_META_CHANNEL_CALL_TIMEOUT": "1500",
        "RECO_STORAGE_THREAD_NUM": "4",
        "RECO_STORAGE_QUEUE_SIZE": "64",
        "RECO_GET_TIMEOUT_MS": "12000",
        "RECO_PUT_TIMEOUT_MS": "12000",
    }


def _new_client() -> RtpKvMetaObjectClient:
    overrides = {}
    for variable, argument in (
        ("RTP_KVCM_CALL_TIMEOUT_MS", "call_timeout_ms"),
        ("RTP_KVCM_PUT_TIMEOUT_MS", "put_timeout_ms"),
        ("RTP_KVCM_GET_TIMEOUT_MS", "get_timeout_ms"),
    ):
        if variable in os.environ:
            overrides[argument] = int(os.environ[variable])
    return RtpKvMetaObjectClient.from_env(environ=_environment(), **overrides)


def _specs(count: int = 67) -> list[dict]:
    return [
        {
            "dtype": _DTYPES[index % len(_DTYPES)],
            "shape": [1 + index % 7, 3 + index],
            "seed": index + 1,
        }
        for index in range(count)
    ]


def _tensor(spec: dict):
    count = 1
    for size in spec["shape"]:
        count *= size
    # Small integral values have an exact representation in every test dtype.
    values = (torch.arange(count, dtype=torch.int64) + spec["seed"]) % 127
    return values.to(getattr(torch, spec["dtype"])).reshape(spec["shape"])


def _digest(tensor) -> str:
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def _assert_payloads(keys: list[str], specs: list[dict], *, trace: str) -> list[dict]:
    expected = [_tensor(spec) for spec in specs]
    actual = [torch.empty_like(tensor) for tensor in expected]
    with _new_client() as client:
        client.load(keys, actual, trace_id=trace)
    records = []
    for index, (source, destination) in enumerate(zip(expected, actual)):
        if not torch.equal(source, destination) or _digest(source) != _digest(
            destination
        ):
            raise AssertionError(f"payload mismatch at object {index}")
        records.append({"nbytes": source.nbytes, "sha256": _digest(destination)})
    return records


def _worker(mode: str) -> None:
    if torch is None:
        raise RuntimeError("torch is required by the live integration worker")
    manifest = json.load(sys.stdin)
    if mode == "save":
        tensors = [_tensor(spec) for spec in manifest["specs"]]
        with _new_client() as client:
            client.save(manifest["keys"], tensors, trace_id=manifest["trace"])
        records = [
            {"nbytes": tensor.nbytes, "sha256": _digest(tensor)} for tensor in tensors
        ]
    elif mode == "load":
        records = _assert_payloads(
            manifest["keys"], manifest["specs"], trace=manifest["trace"]
        )
    else:
        raise ValueError("unsupported integration worker mode")
    print(_WORKER_MARKER + json.dumps({"pid": os.getpid(), "objects": records}))


def _run_worker(mode: str, keys: list[str], specs: list[dict], trace: str) -> dict:
    # The producer exits before the consumer starts: no tensor, native client,
    # registered memory or Python object can be shared between E and P.
    result = subprocess.run(
        [sys.executable, "-m", _WORKER_MODULE, "--kvcm-worker", mode],
        input=json.dumps({"keys": keys, "specs": specs, "trace": trace}),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy(),
        timeout=int(os.environ.get("RTP_KVCM_WORKER_TIMEOUT_SECONDS", "300")),
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"{mode} worker failed ({result.returncode})\n"
            f"{result.stdout[-8192:]}\n{result.stderr[-8192:]}"
        )
    lines = [
        line for line in result.stdout.splitlines() if line.startswith(_WORKER_MARKER)
    ]
    if len(lines) != 1:
        raise AssertionError(f"{mode} worker returned no unique payload result")
    return json.loads(lines[0][len(_WORKER_MARKER) :])


@skipUnless(_RUN_INTEGRATION, "set RTP_KVCM_RUN_INTEGRATION=1 for the live PACE tests")
class RtpKvMetaObjectLiveIntegrationTest(TestCase):
    def setUp(self):
        if torch is None:
            self.fail("the requested live integration test requires torch")
        from kv_cache_manager.client import (
            KV_META_OBJECT_API_VERSION,
            KvMetaObjectClientError,
        )
        from kv_cache_manager.client.pybind import kvcm_py_client

        self.assertEqual(KV_META_OBJECT_API_VERSION, 2)
        self.assertEqual(kvcm_py_client.KV_META_OBJECT_API_VERSION, 2)
        self.client_error = KvMetaObjectClientError
        self.codes = kvcm_py_client.ClientErrorCode
        self.prefix = "rtp-emb-it-" + uuid.uuid4().hex
        self.keys: list[str] = []
        self.client = _new_client()
        self.addCleanup(self._cleanup)

    def _keys(self, count: int) -> list[str]:
        keys = [f"{self.prefix}/{len(self.keys) + index}" for index in range(count)]
        self.keys.extend(keys)
        return keys

    def _cleanup(self):
        errors = []
        try:
            self.client.close()
        except Exception as error:
            errors.append(error)
        # A data-plane timeout permanently invalidates the current native
        # wrapper. A fresh client lets teardown attempt every service batch,
        # even if an earlier Remove batch fails. Avoid paying a deletion grace
        # period for each individual key.
        try:
            with _new_client() as cleanup:
                for begin in range(0, len(self.keys), 64):
                    try:
                        cleanup.remove(
                            self.keys[begin : begin + 64],
                            trace_id=self.prefix + "-cleanup",
                        )
                    except Exception as error:
                        errors.append(error)
        except Exception as error:
            errors.append(error)
        if errors:
            raise RuntimeError(
                f"integration cleanup failed for {len(errors)} operation(s)"
            ) from errors[0]

    def _assert_equal(self, keys, tensors):
        outputs = [torch.empty_like(tensor) for tensor in tensors]
        self.client.load(keys, outputs, trace_id=self.prefix + "-verify")
        for expected, actual in zip(tensors, outputs):
            self.assertTrue(torch.equal(expected, actual))
            self.assertEqual(_digest(expected), _digest(actual))

    def test_cross_process_variable_payload_roundtrip(self):
        specs = _specs()
        keys = self._keys(len(specs))
        producer = _run_worker("save", keys, specs, self.prefix + "-E")
        consumer = _run_worker("load", keys, specs, self.prefix + "-P")
        self.assertNotEqual(producer["pid"], os.getpid())
        self.assertNotEqual(consumer["pid"], os.getpid())
        self.assertEqual(producer["objects"], consumer["objects"])
        self.assertEqual(len(consumer["objects"]), 67)
        self.assertEqual(len({spec["dtype"] for spec in specs}), 5)
        self.assertGreater(len({item["nbytes"] for item in consumer["objects"]}), 30)
        self.assertTrue(any(item["nbytes"] % 2 for item in consumer["objects"]))
        evidence_dir = os.environ.get("RTP_KVCM_EVIDENCE_DIR")
        if evidence_dir:
            path = Path(evidence_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / (self.prefix + "-payloads.json")).write_text(
                json.dumps({"producer": producer, "consumer": consumer}, indent=2),
                encoding="utf-8",
            )

    def test_mixed_existing_and_new_objects_and_partial_load_miss(self):
        keys = self._keys(67)
        tensors = [_tensor(spec) for spec in _specs()]
        self.client.save(keys[::2], tensors[::2], trace_id=self.prefix + "-existing")
        self.client.save(keys, tensors, trace_id=self.prefix + "-mixed")
        self._assert_equal(keys, tensors)

        missing = self._keys(1)[0]
        mixed_outputs = [torch.empty_like(tensors[index]) for index in (0, 1, 2)]
        self.assertFalse(
            self.client.try_load([keys[0], missing, keys[2]], mixed_outputs)
        )
        outputs = [torch.empty_like(tensor) for tensor in tensors[:65]]
        with self.assertRaises(self.client_error) as caught:
            self.client.load(keys[:64] + [missing], outputs)
        self.assertEqual(caught.exception.code, self.codes.ER_SERVICE_NOT_FOUND)
        self.assertEqual(caught.exception.completed_items, 64)
        self.assertEqual(caught.exception.batch_index, 1)
        self.assertEqual(caught.exception.batch_count, 2)
        self.assertFalse(self.client.try_load(keys[:64] + [missing], outputs))
        # Even the already-written prefix is discarded after the logical miss.
        self._assert_equal(keys, tensors)

    def test_size_failure_preserves_existing_value_and_allows_correct_retry(self):
        key = self._keys(1)[0]
        source = torch.arange(13, dtype=torch.uint8)
        self.client.save_one(key, source)
        wrong = torch.full((12,), 255, dtype=torch.uint8)
        before = wrong.clone()
        with self.assertRaises(self.client_error) as caught:
            self.client.load_one(key, wrong)
        self.assertEqual(caught.exception.code, self.codes.ER_SERVICE_SIZE_MISMATCH)
        self.assertEqual(caught.exception.operation, "load")
        self.assertEqual(caught.exception.completed_items, 0)
        self.assertFalse(caught.exception.unknown_outcome)
        self.assertTrue(torch.equal(before, wrong))
        with self.assertRaises(self.client_error) as caught:
            self.client.save_one(key, wrong)
        self.assertEqual(caught.exception.code, self.codes.ER_SERVICE_SIZE_MISMATCH)
        self._assert_equal([key], [source])
        # Same-key/same-size Save is an immutable hit, not an overwrite API.
        self.client.save_one(key, torch.full_like(source, 99))
        self._assert_equal([key], [source])

    def test_equal_byte_size_does_not_validate_tensor_schema(self):
        key = self._keys(1)[0]
        source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        self.client.save_one(key, source)
        output = torch.empty((6,), dtype=torch.int32)
        self.client.load_one(key, output)
        self.assertNotEqual(source.dtype, output.dtype)
        self.assertNotEqual(source.shape, output.shape)
        self.assertEqual(_digest(source), _digest(output))

    def test_later_batch_failure_exposes_prefix_and_controlled_retry(self):
        keys = self._keys(65)
        tensors = [torch.full((3,), index, dtype=torch.uint8) for index in range(65)]
        existing_tail = torch.tensor([11, 12], dtype=torch.uint8)
        self.client.save_one(keys[-1], existing_tail)
        with self.assertRaises(self.client_error) as caught:
            self.client.save(keys, tensors, trace_id=self.prefix + "-partial")
        error = caught.exception
        self.assertEqual(error.code, self.codes.ER_SERVICE_SIZE_MISMATCH)
        self.assertEqual(error.operation, "save")
        self.assertEqual(error.batch_index, 1)
        self.assertEqual(error.batch_start, 64)
        self.assertEqual(error.batch_size, 1)
        self.assertEqual(error.batch_count, 2)
        self.assertEqual(error.completed_items, 64)
        self.assertFalse(error.unknown_outcome)
        self._assert_equal(keys[:64], tensors[:64])
        self._assert_equal(keys[-1:], [existing_tail])
        # This retry is safe only because the failed operation is known and
        # these UUID keys belong to this test, with immutable expected bytes.
        tensors[-1] = existing_tail
        self.client.save(keys, tensors, trace_id=self.prefix + "-retry")
        self._assert_equal(keys, tensors)

    def test_logical_remove_idempotency_and_request_key_reuse(self):
        keys = self._keys(67)
        tensors = [_tensor(spec) for spec in _specs()]
        self.client.save(keys, tensors)
        self.client.remove(keys, trace_id=self.prefix + "-remove")
        self.client.remove(keys, trace_id=self.prefix + "-remove-again")
        for key, tensor in zip(keys, tensors):
            self.assertFalse(self.client.try_load_one(key, torch.empty_like(tensor)))
        replacements = [torch.full_like(tensor, 77) for tensor in tensors]
        self.client.save(keys, replacements, trace_id=self.prefix + "-successor")
        self._assert_equal(keys, replacements)

    def test_close_and_new_client_preserve_committed_payload(self):
        key = self._keys(1)[0]
        source = torch.arange(33, dtype=torch.float32)
        self.client.save_one(key, source)
        self.client.close()
        self.client.close()
        with self.assertRaises(RuntimeError):
            self.client.load_one(key, torch.empty_like(source))
        self.client = _new_client()
        self._assert_equal([key], [source])


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--kvcm-worker":
        _worker(sys.argv[2])
    else:
        main()
