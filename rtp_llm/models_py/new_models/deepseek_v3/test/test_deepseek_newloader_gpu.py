"""Device-only DeepSeek newloader numerical tests.

Only these cases are exported by the GPU targets, so CUDA and ROCm lanes do
not repeat the CPU loader/configuration suite.
"""

import unittest

from test_deepseek_newloader import DeepSeekNewloaderTest as _SharedTests


class DeepSeekNewloaderGpuTest(unittest.TestCase):
    test_moe_cuda_noaux_router_matches_reference = (
        _SharedTests._gpu_moe_cuda_noaux_router_matches_reference
    )
    test_moe_cuda_fast_select_topk_matches_reference = (
        _SharedTests._gpu_moe_cuda_fast_select_topk_matches_reference
    )
    test_moe_rocm_noaux_reference_router_matches_expected = (
        _SharedTests._gpu_moe_rocm_noaux_reference_router_matches_expected
    )
    test_mla_online_fp8_uses_models_py_quantizer = (
        _SharedTests._gpu_mla_online_fp8_uses_models_py_quantizer
    )
    test_mla_online_fp8_rocm_keeps_exact_bf16_kv_b_views = (
        _SharedTests._gpu_mla_online_fp8_rocm_keeps_exact_bf16_kv_b_views
    )
    test_mla_online_fused_fp8_projection_matches_bf16_reference = (
        _SharedTests._gpu_mla_online_fused_fp8_projection_matches_bf16_reference
    )
    test_mla_online_fp8_kc_vc_use_bf16_checkpoint_source = (
        _SharedTests._gpu_mla_online_fp8_kc_vc_use_bf16_checkpoint_source
    )
    test_mla_prequantized_fp8_kernel_views_execute_numerically = (
        _SharedTests._gpu_mla_prequantized_fp8_kernel_views_execute_numerically
    )


del _SharedTests


if __name__ == "__main__":
    unittest.main()
