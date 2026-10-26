import gc
import json
import os
import resource
import subprocess
import sys
import unittest
from types import SimpleNamespace


class _PythonStartupFailure:
    @property
    def http_port(self):
        raise NotImplementedError("startup Python exception sentinel")


def _run_failure_case(case):
    # The old detached startup aborts the child; keep that failure bounded and
    # do not create a core file in the test workspace.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    import torch

    import rtp_llm.ops as ops

    ops.ensure_engine_ops_loaded()
    config_types = {
        "parallelism_config": "ParallelismConfig",
        "runtime_config": "RuntimeConfig",
        "nccl_comm_config": "NcclCommConfig",
        "pd_sep_config": "PDSepConfig",
        "concurrency_config": "ConcurrencyConfig",
        "fmha_config": "FMHAConfig",
        "kv_cache_config": "KVCacheConfig",
        "profiling_debug_logging_config": "ProfilingDebugLoggingConfig",
        "hw_kernel_config": "HWKernelConfig",
        "device_resource_config": "DeviceResourceConfig",
        "moe_config": "MoeConfig",
        "model_specific_config": "ModelSpecificConfig",
        "sp_config": "SpeculativeExecutionConfig",
        "cache_store_config": "CacheStoreConfig",
        "misc_config": "MiscellaneousConfig",
        "arpc_config": "ArpcConfig",
        "grpc_config": "GrpcConfig",
        "grammar_config": "GrammarConfig",
    }
    engine_config = SimpleNamespace(
        **{name: getattr(ops, cls)() for name, cls in config_types.items()}
    )
    if case == "python":
        engine_config.server_config = _PythonStartupFailure()
        error_type = NotImplementedError
        error_text = "startup Python exception sentinel"
    else:
        # LocalRpcServer rejects this before constructing NormalEngine. The
        # real initRPCServer translates its failed Status into a C++ exception.
        engine_config.server_config = SimpleNamespace(
            http_port=0,
            rpc_server_port=-1,
            torch_allocator_dump_cooldown_seconds=-1.0,
        )
        error_type = RuntimeError
        error_text = "allocator dump cooldown must be finite and non-negative"
    model = SimpleNamespace(
        model_config=ops.ModelConfig(),
        weight=SimpleNamespace(weights=[], global_weights={}),
        py_model=None,
        weight_manager=None,
    )
    op = ops.RtpLLMOp()
    assert not torch.cuda.is_initialized()
    try:
        op.init(model, engine_config, None, None, None, None)
    except error_type as error:
        assert error_text in str(error), str(error)
    else:
        raise AssertionError("startup unexpectedly succeeded")
    # Also catch publication of a partially initialized service: stop() must
    # not dereference its absent engine, and destruction must remain safe.
    op.stop()
    op.stop()
    del op
    gc.collect()
    assert not torch.cuda.is_initialized()
    print(json.dumps({"case": case, "exception_returned": True, "cleanup": True}))


class RtpLLMStartupTest(unittest.TestCase):
    def _assert_startup_failure_returns(self, case):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        result = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--failure-case", case],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        expected = {"case": case, "exception_returned": True, "cleanup": True}
        self.assertIn(json.dumps(expected), result.stdout.splitlines())

    def test_python_startup_exception_returns_to_caller(self):
        self._assert_startup_failure_returns("python")

    def test_native_startup_exception_returns_and_cleans_partial_service(self):
        self._assert_startup_failure_returns("native")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--failure-case":
        _run_failure_case(sys.argv[2])
    else:
        unittest.main()
