"""Image-build gate: native ABI and vLLM hook registration must both work."""

from importlib import metadata
from pathlib import Path
import sys
import sysconfig
import subprocess
import os
import tempfile
import socket
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch


def verify_observer_startup(navi_sched, torch):
    import importlib

    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        MambaSpec,
    )

    module = importlib.import_module("navi_sched.adapters.vllm.VLLMEngineObserver")
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=16,
            enable_prefix_caching=True,
            prefix_caching_hash_algo="sha256_cbor",
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=True,
            max_num_partial_prefills=1,
            max_long_partial_prefills=1,
            long_prefill_token_threshold=0,
            max_num_scheduled_tokens=8192,
            max_num_batched_tokens=8192,
        ),
    )
    cache = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["attention"],
                kv_cache_spec=FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=1,
                    head_size=128,
                    dtype=torch.bfloat16,
                ),
            ),
            KVCacheGroupSpec(
                layer_names=["linear"],
                kv_cache_spec=MambaSpec(
                    block_size=16,
                    shapes=((1, 128),),
                    dtypes=(torch.bfloat16,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {
        "VLLM_NAVI_SCHED_BIZ_NAME": "ppu_observer_build_smoke",
        "VLLM_NAVI_SCHED_SERVER_IP": "127.0.0.1",
        "VLLM_NAVI_SCHED_SERVER_PORT": str(port),
        "VLLM_NAVI_SCHED_AGENT_GIG_PORT": "0",
    }
    agent_factory = partial(
        navi_sched.NaviSchedAgent,
        runtime_config={"thread_num": 2},
        enable_kmon=False,
        enable_snapshot=False,
    )
    with patch.dict(os.environ, env), patch.object(
        module, "NaviSchedAgent", agent_factory
    ):
        observer = module.VLLMEngineObserver(config, cache)
        try:
            assert observer._cache_policy.hybrid_tail_required
            assert observer._cache_policy.total_blocks == cache.num_blocks
            print("PPU_NAVI_SCHED_OBSERVER_STARTUP_OK", flush=True)
        finally:
            observer.shutdown()


def verify_frontend_cache_config():
    import msgspec
    from vllm.v1.engine import EngineCoreReadyResponse
    from vllm.v1.engine.core_client import MPClient

    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=4096),
        cache_config=SimpleNamespace(num_gpu_blocks=None, block_size=16),
    )
    frontend = SimpleNamespace(vllm_config=config)
    response = EngineCoreReadyResponse(
        max_model_len=4096,
        num_gpu_blocks=1024,
        dp_stats_address=None,
        block_size=512,
    )
    MPClient._apply_ready_response(frontend, msgspec.msgpack.encode(response))
    assert config.cache_config.block_size == 512
    assert config.cache_config.num_gpu_blocks == 1024
    print("PPU_NAVI_SCHED_FRONTEND_CACHE_CONFIG_OK", flush=True)


def main():
    assert sys.version_info[:2] == (3, 12), sys.version
    package = Path(sysconfig.get_path("purelib")) / "navi_sched"
    # glibc modules must come from the PPU runtime as one matched set.
    for pattern in (
        "librt.so*",
        "libpthread.so*",
        "libc.so*",
        "libdl.so*",
        "libm.so*",
        "libresolv.so*",
        "ld-linux*.so*",
    ):
        for path in package.glob(pattern):
            path.unlink()

    try:
        import navi_sched
    except ImportError:
        subprocess.run(["ldd", str(package / "_navi_sched_core.so")], check=False)
        for line in Path("/proc/self/maps").read_text().splitlines():
            if "libtbb" in line:
                print("PPU_NAVI_LOADED_TBB", line, flush=True)
        raise
    from vllm import plugins

    entries = [
        entry
        for entry in metadata.entry_points(group="vllm.general_plugins")
        if entry.value == "navi_sched_vllm_plugin:register_vllm_plugin"
    ]
    assert len(entries) == 1, entries
    entries[0].load()()
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.output_processor import OutputProcessor
    from navi_sched.adapters.vllm.VLLMRequestRegistration import (
        install_request_registration_rollback,
    )

    install_request_registration_rollback(AsyncLLM, OutputProcessor)
    print("PPU_NAVI_SCHED_REGISTRATION_GUARD_OK", flush=True)
    adapter = plugins.get_engine_adapter()
    assert adapter is not None
    assert adapter._engine_transport_factory is not None
    assert adapter._engine_observer_factory is not None
    assert adapter._renderer_transport_factory is not None
    assert adapter._frontend_app_configurator is not None
    text_renderer = SimpleNamespace(mm_processor=None)
    with patch.dict(os.environ, {"VLLM_NAVI_SCHED_DEFER_RENDER": "1"}):
        assert adapter._renderer_transport_factory(text_renderer) is text_renderer
    print("PPU_NAVI_SCHED_TEXT_RENDERER_STARTUP_OK", flush=True)
    import torch

    verify_frontend_cache_config()
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("test_ppu_ready_response.py"))],
        check=True,
    )
    print("PPU_NAVI_SCHED_FORWARD_ADMISSION_CAPABILITY_OK", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("test_ppu_concurrency_response.py")),
        ],
        check=True,
    )
    print("PPU_NAVI_SCHED_STREAM_CONCURRENCY_CLEANUP_OK", flush=True)

    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="navi-ppu-startup-") as work:
        try:
            os.chdir(work)
            with navi_sched.NaviSchedAgent(
                "ppu_build_smoke",
                total_blocks=16,
                block_size=16,
                linear_step=1,
                hybrid_tail_required=False,
                runtime_config={"thread_num": 2},
                enable_kmon=False,
                enable_snapshot=False,
            ) as agent:
                agent.init()
                assert agent.grpc_port > 0
                print("PPU_NAVI_SCHED_AGENT_STARTUP_OK", torch.__version__, flush=True)
            verify_observer_startup(navi_sched, torch)
        finally:
            os.chdir(previous)
    print("PPU_NAVI_SCHED_NATIVE_AND_PLUGIN_OK", navi_sched.__file__)


if __name__ == "__main__":
    main()
