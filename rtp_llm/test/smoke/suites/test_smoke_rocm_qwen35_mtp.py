"""Dedicated ROCm Qwen3.6 dense MTP PD CUDA Graph smoke."""

import pytest

from rtp_llm.test.smoke_framework.manifest import build_smoke_params
from rtp_llm.test.smoke_framework.runner import run_smoke_test


_COMMON_ARGS = (
    "--load_cache_timeout_ms 120000 --load_method scratch --warm_up 0 "
    "--act_type BF16 --seq_size_per_block 1024 --kernel_seq_size_per_block 16 "
    "--test_block_num 64 --max_seq_len 12800 --tp_size 1 --world_size 1 "
    "--ep_size 1 --reuse_cache 1 --use_asm_pa 1 --use_aiter_pa 1 "
    "--use_triton_pa 1 --reserver_runtime_mem_mb 40480 "
    "--quantization FP8_PER_CHANNEL_COMPRESSED --use_swizzleA 1 "
    "--sp_model_type qwen35_dense_mtp --gen_num_per_cycle 3 --sp_type eagle "
    "--sp_checkpoint_path /mnt/nas1/hf/Qwen3.6-27B --sp_act_type bf16 "
    "--cache_store_rdma_mode 0 --use_local 1"
)

SMOKE_CASES = {
    "rocm_qwen35_dense_mtp_pd_fp8_tp1_step3_cg": {
        "task_info": "data/model/qwen3_next/qwen35_dense_fp8_tp1_mtp_pd.json",
        "smoke_args": {
            "prefill": _COMMON_ARGS + " --role_type PREFILL",
            "decode": _COMMON_ARGS
            + " --role_type DECODE --concurrency_limit 4 --enable_cuda_graph 1 "
            "--enable_cuda_graph_debug_mode 1 --decode_capture_config '1,2,3,4'",
        },
        "gpu_type": "MI308X-ROCM7",
        "platform": "rocm",
        "markers": ["smoke", "rocm", "MI308X_ROCM7", "dedicated"],
        "timeout": 1200,
    }
}

SUITE_NAME = "smoke_rocm_qwen35_mtp"
_test_params = build_smoke_params(
    pytest, {SUITE_NAME: SMOKE_CASES}, composite_suites={}
)


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("test_name,test_config", _test_params)
def test_smoke_rocm_qwen35_mtp(test_name: str, test_config: dict):
    run_smoke_test(test_name, test_config)
