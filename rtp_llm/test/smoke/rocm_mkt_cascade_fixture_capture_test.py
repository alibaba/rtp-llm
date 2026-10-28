"""Manual capture of actual model beam-decode tensors, not an accuracy golden.

Required test_env: MKT_FIXTURE_MODEL_DIR, MKT_FIXTURE_REQUESTS_JSON,
MKT_FIXTURE_ENV_JSON. Optional: MKT_FIXTURE_OUTPUT_DIR,
MKT_FIXTURE_REQUEST_LIMIT (1), MKT_FIXTURE_LAYERS (0,13,27),
MKT_FIXTURE_START_TIMEOUT_SEC (600). All input paths are supplied at execution;
neither private requests nor checkpoint paths belong in the repository.

Run through the standard prechecked ROCm Bazel runner holding the physical GPU
lock. This test inherits visibility and never invokes DeviceResource cleanup.
"""

import hashlib
import json
import logging
import os
import stat
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import torch

from rtp_llm.test.utils.maga_server_manager import MagaServerManager

ATTENTION_MODULE = "rtp_llm.models_py.modules.factory.attention.rocm_impl.aiter"
# These are scalar workload/config settings, never text-bearing prompt options.
CONFIG_KEYS = frozenset(
    """
ACT_TYPE WEIGHT_TYPE MODEL_TYPE QUANTIZATION KV_CACHE_MEM_MB MAX_CONTEXT_BATCH_SIZE
MAX_BATCH_TOKENS_SIZE PDFUSION_SCHEDULER_MODE DECODE_PREFILL_RATIO CONCURRENCY_LIMIT
ENABLE_CUDA_GRAPH ENABLE_FULL_PREFILL_CUDA_GRAPH ENABLE_DECODE_CASCADE_ATTENTION
DECODE_CUDA_GRAPH_MAX_BATCH_SIZE DECODE_CAPTURE_CONFIG ENGINE_ASYNC_WORKER_COUNT
ENABLE_OUTPUT_VOCAB_PRUNING FRONTEND_SERVER_COUNT MAX_SEQ_LEN REUSE_CACHE USE_ASM_PA
USE_TRITON_PA PREFILL_CAPTURE_CONFIG FULL_PREFILL_CUDA_GRAPH_MAX_REQUESTS
FULL_PREFILL_CUDA_GRAPH_MAX_PADDING_RATIO SEQ_SIZE_PER_BLOCK
KERNEL_SEQ_SIZE_PER_BLOCK FP8_KV_CACHE WARM_UP
""".split()
)
CAPTURE_OVERRIDES = {
    "ENABLE_CUDA_GRAPH": "0",
    "ENABLE_FULL_PREFILL_CUDA_GRAPH": "0",
    "ENABLE_DECODE_CASCADE_ATTENTION": "0",
    "WARM_UP": "0",
    "USE_ASM_PA": "1",
    "USE_TRITON_PA": "1",
    "SEQ_SIZE_PER_BLOCK": "16",
    "KERNEL_SEQ_SIZE_PER_BLOCK": "16",
    "FP8_KV_CACHE": "0",
    "ACT_TYPE": "BF16",
    "REUSE_CACHE": "0",
}
PROXY_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "grpc_proxy",
    "GRPC_PROXY",
)
LOCAL_NO_PROXY = ("0.0.0.0", "localhost", "127.0.0.1", "::1")


def _configure_local_service_proxy_environment():
    """Call inside patch.dict so readiness, visit and children share this scope."""
    for name in PROXY_VARIABLES:
        os.environ.pop(name, None)
    for name in ("NO_PROXY", "no_proxy"):
        entries = [value.strip() for value in os.getenv(name, "").split(",")]
        entries = [value for value in entries if value]
        for address in LOCAL_NO_PROXY:
            if address not in entries:
                entries.append(address)
        os.environ[name] = ",".join(entries)


def _sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def _required_path(name, directory=False):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"required test_env is missing: {name}")
    path = Path(value).resolve(strict=True)
    if not (path.is_dir() if directory else path.is_file()):
        raise RuntimeError(f"invalid input type for {name}")
    return path


class _PrivateServerManager(MagaServerManager):
    def print_process_log(self, max_lines=0):
        # The base implementation prints service logs on readiness/request failure.
        # They may contain private text; retain them locally and print no content.
        path = Path(self.log_file_path) if self.log_file_path else None
        if path and path.is_file():
            logging.info("Private service log: %s (sha256=%s)", path, _sha_file(path))
        else:
            logging.info("Private service log is not available")


class _PrivateVisitFilter(logging.Filter):
    def filter(self, record):
        # Covers the base visit() curl/query, response.debug and error-body records.
        return not (
            record.pathname.endswith("/maga_server_manager.py")
            and record.funcName == "visit"
        )


@contextmanager
def _private_manager_logs():
    root = logging.getLogger()
    filter_ = _PrivateVisitFilter()
    root.addFilter(filter_)
    handlers = list(root.handlers)
    for handler in handlers:
        handler.addFilter(filter_)
    try:
        yield
    finally:
        root.removeFilter(filter_)
        for handler in handlers:
            handler.removeFilter(filter_)


def _response_finished(response):
    """Examine completion/error flags only; never format or persist the body."""
    if isinstance(response, str):
        documents = [json.loads(response)]
    elif isinstance(response, list):
        documents = []
        for line in response:
            text = line.decode() if isinstance(line, bytes) else line
            if text.startswith("data:"):
                text = text[5:].strip()
            if text and text != "[DONE]":
                documents.append(json.loads(text))
    else:
        return False
    finished = False
    for document in documents:
        if not isinstance(document, dict):
            return False
        if document.get("error") or document.get("error_code") not in (None, 0):
            return False
        flag = document.get("finished")
        finished = (
            finished
            or flag is True
            or (
                isinstance(flag, list)
                and bool(flag)
                and all(item is True for item in flag)
            )
        )
    return finished


class RocmMktCascadeFixtureCaptureTest(unittest.TestCase):
    def test_capture_real_first_decode(self):
        model = _required_path("MKT_FIXTURE_MODEL_DIR", directory=True)
        requests_path = _required_path("MKT_FIXTURE_REQUESTS_JSON")
        env_path = _required_path("MKT_FIXTURE_ENV_JSON")
        config = json.loads((model / "config.json").read_text())
        requests = json.loads(requests_path.read_text())
        env_source = json.loads(env_path.read_text())
        self.assertTrue(
            isinstance(requests, list), "private requests must be a JSON list"
        )
        self.assertTrue(
            isinstance(env_source, dict), "environment input must be a JSON object"
        )
        self.assertTrue(
            set(env_source) <= CONFIG_KEYS,
            "unsupported environment keys; extend the explicit config allowlist",
        )
        self.assertTrue(
            all(isinstance(v, (str, int, float, bool)) for v in env_source.values()),
            "environment values must be scalars",
        )
        server_env = {
            key: (str(int(value)) if isinstance(value, bool) else str(value))
            for key, value in env_source.items()
        }
        self.assertEqual(
            server_env.get("QUANTIZATION"),
            "FP8_PER_CHANNEL_COMPRESSED",
            "this fixture target requires FP8 PTPC weights",
        )
        self.assertEqual(server_env.get("MODEL_TYPE"), "qwen_3")
        self.assertTrue(
            int(server_env.get("KV_CACHE_MEM_MB", "0")) > 0,
            "an explicit KV cache budget is required for reproducible capture",
        )
        model_layers = int(config["num_hidden_layers"])
        layers = tuple(
            int(value)
            for value in os.getenv("MKT_FIXTURE_LAYERS", "0,13,27").split(",")
        )
        limit = int(os.getenv("MKT_FIXTURE_REQUEST_LIMIT", "1"))
        self.assertTrue(
            layers
            and len(set(layers)) == len(layers)
            and all(0 <= layer < model_layers for layer in layers),
            "invalid selected layers",
        )
        self.assertTrue(0 < limit <= len(requests), "invalid request limit")
        self.assertEqual(
            (
                config["num_attention_heads"],
                config["num_key_value_heads"],
                config["head_dim"],
            ),
            (16, 8, 128),
            "checkpoint attention shape differs from this fixture target",
        )
        selected = []
        for ordinal, request in enumerate(requests[:limit]):
            self.assertTrue(isinstance(request, dict), "request must be an object")
            generation = request.get("generate_config", {})
            self.assertTrue(
                isinstance(generation, dict), "generation config must be an object"
            )
            beams = generation.get("variable_num_beams", [])
            self.assertTrue(
                isinstance(beams, list)
                and len(beams) == 2
                and all(isinstance(n, int) and n > 1 for n in beams),
                "expected two-step dynamic beam request",
            )
            self.assertTrue(
                generation.get("max_new_tokens") == 2,
                "capture expects exactly one decode after prefill",
            )
            selected.append(
                {
                    "ordinal": ordinal,
                    "sha256": _json_hash(request),
                    "expected_decode_rows": beams[0],
                    "variable_num_beams": beams,
                }
            )

        output_value = os.getenv("MKT_FIXTURE_OUTPUT_DIR")
        if not output_value:
            output_value = str(
                Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]) / "real_cascade_fixture"
            )
        output = Path(output_value).absolute()
        self.assertFalse(output.is_symlink(), "output directory must not be a symlink")
        output.mkdir(parents=True, mode=0o700, exist_ok=True)
        self.assertEqual(
            output.stat().st_uid,
            os.getuid(),
            "output directory must belong to this test user",
        )
        self.assertFalse(
            any(output.iterdir()), "use a fresh empty fixture output directory"
        )
        output.chmod(0o700)
        previous_umask = os.umask(0o077)
        try:
            self._capture(
                model,
                config,
                requests,
                env_source,
                server_env,
                selected,
                layers,
                output,
                requests_path,
                env_path,
            )
        finally:
            os.umask(previous_umask)

    def _capture(
        self,
        model,
        config,
        requests,
        env_source,
        server_env,
        selected,
        layers,
        output,
        requests_path,
        env_path,
    ):
        source_root = Path(__file__).resolve().parents[2]
        hook = source_root / "test/smoke/data/hooks/rocm_cascade_capture.py"
        source_names = [
            "start_backend_server.py",
            "utils/hot_hook_runtime.py",
            "models_py/modules/factory/attention/rocm_impl/aiter.py",
            "models_py/triton_kernels/attention/beam_search_cascade.py",
            "test/smoke/rocm_mkt_cascade_fixture_capture_test.py",
            "test/smoke/data/hooks/rocm_cascade_capture.py",
        ]
        model_files = sorted(
            set(model.glob("*.safetensors"))
            | set(model.glob("*.bin"))
            | set(model.glob("*.json"))
        )
        self.assertTrue(
            any(path.suffix in (".safetensors", ".bin") for path in model_files),
            "checkpoint weight files are missing",
        )
        server_env.update(CAPTURE_OVERRIDES)
        provenance = {
            "schema_version": 1,
            "capture_mode": "eager",
            "model_config": {
                key: config[key]
                for key in (
                    "num_hidden_layers",
                    "num_attention_heads",
                    "num_key_value_heads",
                    "head_dim",
                    "hidden_size",
                    "vocab_size",
                )
            },
            "checkpoint_files": {path.name: _sha_file(path) for path in model_files},
            "requests_file_sha256": _sha_file(requests_path),
            "environment_file_sha256": _sha_file(env_path),
            "source_sha256": {
                name: _sha_file(source_root / name) for name in source_names
            },
            "configured_environment": env_source,
            "capture_overrides": CAPTURE_OVERRIDES,
            "effective_environment": dict(server_env),
            "proxy_policy": {
                "scope": "capture_test_and_service_children",
                "unset_variables": PROXY_VARIABLES,
                "no_proxy_variables": ["NO_PROXY", "no_proxy"],
                "no_proxy_additions": LOCAL_NO_PROXY,
                "preserve_existing_no_proxy": True,
                "proxy_values_recorded": False,
            },
            "selected_requests": selected,
            "selected_layers": layers,
            "visibility": {
                name: os.getenv(name)
                for name in (
                    "HIP_VISIBLE_DEVICES",
                    "CUDA_VISIBLE_DEVICES",
                    "GPU_COUNT",
                    "WORLD_SIZE",
                )
            },
            "batch_semantics": "sequential_requests; not a real multi-request scheduler batch",
            "kernel_replay_accuracy": "NOT_RUN",
            "end_to_end_on_off_accuracy": "NOT_RUN",
        }
        provenance_path = output / "provenance.json"
        _write_json(provenance_path, provenance)
        provenance_hash = _sha_file(provenance_path)
        control = output / "control.json"
        _write_json(control, {"armed": False})
        hook_config = output / "hook_config.json"
        _write_json(
            hook_config,
            {
                "case": "real_cascade_capture",
                "capture_root": str(output),
                "control_file": str(control),
                "function_hooks": [
                    {
                        "target": ATTENTION_MODULE + ".AiterDecodeImplTriton.forward",
                        "before": "enter_layer",
                        "after": "leave_layer",
                        "exception": "leave_layer",
                    },
                    {
                        "target": ATTENTION_MODULE + ".AiterDecodeAttnOpTriton.forward",
                        "after": "capture_after",
                    },
                ],
            },
        )
        server_env.update(
            {
                "RTP_HOT_HOOK": "1",
                "RTP_HOT_HOOK_FILE": str(hook),
                "RTP_HOT_HOOK_CONFIG": str(hook_config),
                "RTP_HOT_HOOK_DUMP_DIR": str(output / "hook_runtime"),
                "RTP_HOT_HOOK_ALLOW_LOCAL_MUTATION": "0",
                "MAGA_SERVER_WORK_DIR": str(output),
            }
        )
        private_env = {
            "TEST_UNDECLARED_OUTPUTS_DIR": str(output),
            "MAGA_SERVER_WORK_DIR": str(output),
        }
        with mock.patch.dict(os.environ, private_env), _private_manager_logs():
            _configure_local_service_proxy_environment()
            manager = _PrivateServerManager(
                env_args=server_env,
                device_ids=[],
                smoke_args_str=os.getenv("SMOKE_ARGS", ""),
            )
            try:
                started = manager.start_server(
                    model_path=str(model),
                    model_type="qwen_3",
                    tokenizer_path=str(model),
                    timeout=int(os.getenv("MKT_FIXTURE_START_TIMEOUT_SEC", "600")),
                )
                self.assertTrue(
                    started, "capture service failed; inspect its private log"
                )
                for request in selected:
                    _write_json(
                        control,
                        {
                            "armed": True,
                            "request_ordinal": request["ordinal"],
                            "request_sha256": request["sha256"],
                            "layers": layers,
                            "expected_decode_rows": request["expected_decode_rows"],
                            "provenance_sha256": provenance_hash,
                            "model_num_layers": config["num_hidden_layers"],
                            "kv_cache_mem_mb_configured": int(
                                server_env["KV_CACHE_MEM_MB"]
                            ),
                        },
                    )
                    ok, response = manager.visit(
                        requests[request["ordinal"]], retry_times=1
                    )
                    _write_json(control, {"armed": False})
                    self.assertTrue(ok, "request failed; inspect its private log")
                    try:
                        finished = _response_finished(response)
                    except (ValueError, TypeError, UnicodeError):
                        finished = False
                    self.assertTrue(
                        finished, "generation response has an error or lacks completion"
                    )
                    del response
                    self._validate_request(output, request, layers, provenance_hash)
            finally:
                _write_json(control, {"armed": False})
                manager.stop_server()
        self.assertEqual(
            _sha_file(requests_path),
            provenance["requests_file_sha256"],
            "request file changed during capture",
        )
        self.assertEqual(
            _sha_file(env_path),
            provenance["environment_file_sha256"],
            "environment file changed during capture",
        )
        self.assertEqual(
            {path.name: _sha_file(path) for path in model_files},
            provenance["checkpoint_files"],
            "checkpoint changed during capture",
        )
        self.assertEqual(
            {name: _sha_file(source_root / name) for name in source_names},
            provenance["source_sha256"],
            "capture sources changed during execution",
        )
        for path in output.rglob("*"):
            if path.is_file():
                path.chmod(0o600)
            elif path.is_dir():
                path.chmod(0o700)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)
        _write_json(
            output / "capture_summary.json",
            {
                "evidence": "RTP_MEASURED",
                "capture_status": "PASS",
                "fixture_kind": "real_model_beam_capture",
                "capture_mode": "eager",
                "request_count": len(selected),
                "layers": layers,
                "fixture_count": len(selected) * len(layers),
                "provenance_sha256": provenance_hash,
                "kernel_replay_accuracy": "NOT_RUN",
                "end_to_end_on_off_accuracy": "NOT_RUN",
            },
        )
        logging.info(
            "Captured %d requests x %d layers; kernel replay and "
            "end-to-end comparison remain NOT_RUN",
            len(selected),
            len(layers),
        )

    def _validate_request(self, output, request, layers, provenance_hash):
        for layer in layers:
            directory = (
                output / f"request_{request['ordinal']:04d}" / f"layer_{layer:03d}"
            )
            self.assertFalse(
                (directory / "error.json").exists(),
                "capture hook failed; inspect its data-free error.json",
            )
            self.assertTrue(
                (directory / "manifest.json").is_file(),
                "capture hook did not produce every required layer",
            )
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(manifest["request_sha256"], request["sha256"])
            self.assertEqual(manifest["provenance_sha256"], provenance_hash)
            self.assertEqual(manifest["layer_idx"], layer)
            self.assertEqual(manifest["rows"], request["expected_decode_rows"])
            self.assertNotEqual(
                manifest["pid"], os.getpid(), "expected backend-process capture"
            )
            self.assertEqual(
                _sha_file(directory / "tensors.pt"), manifest["tensor_file_sha256"]
            )
            tensors = torch.load(
                directory / "tensors.pt", map_location="cpu", weights_only=True
            )
            self.assertEqual(set(tensors), set(manifest["tensors"]))
            for name, value in tensors.items():
                payload = value.contiguous().view(torch.uint8).numpy().tobytes()
                self.assertEqual(
                    hashlib.sha256(payload).hexdigest(),
                    manifest["tensors"][name]["sha256"],
                    "saved tensor bytes differ from manifest",
                )
            self.assertTrue(
                torch.equal(tensors["history"] + 1, tensors["seq_lens"]),
                "saved history+1 mismatch",
            )
            self.assertTrue(
                bool(torch.isfinite(tensors["q"]).all())
                and bool(torch.isfinite(tensors["off_output"]).all()),
                "nonfinite saved query/output",
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
