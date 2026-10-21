import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors import (
    mega_moe,
    mega_moe_fp8,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.layer import (
    Fp8Fp4MoeRuntimeConfig,
)


class MegaMoeFp8ImplTest(unittest.TestCase):
    def test_launch_supports_routed_and_gated_shared(self):
        kernel = Mock()
        deep_gemm = SimpleNamespace(mega_fp8=SimpleNamespace(fp8_fp8_mega_moe=kernel))
        executor = SimpleNamespace(
            config=SimpleNamespace(layer_id=0),
            l1=object(),
            l2=object(),
            _mega_buf=object(),
            _maybe_pre_kernel_barrier=Mock(),
        )
        gates = object()
        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}), patch.dict(
            os.environ, {"MEGA_MOE_FP8_RESERVE_SM": "0"}
        ), patch.object(mega_moe_fp8, "sync_cuda_graph_warmup_ranks"):
            mega_moe_fp8.MegaMoeFp8Executor._launch(executor, object(), 1, "cuda:0")
            mega_moe_fp8.MegaMoeFp8Executor._launch(
                executor, object(), 1, "cuda:0", shared_expert_gates=gates
            )
            self.assertIs(kernel.call_args.kwargs["shared_expert_gates"], gates)
            self.assertEqual(kernel.call_count, 2)
            executor._jit_pack_only = True
            mega_moe_fp8.MegaMoeFp8Executor._launch(executor, object(), 1, "cuda:0")
            self.assertEqual(kernel.call_count, 2)
            self.assertEqual(executor._maybe_pre_kernel_barrier.call_count, 2)

    def test_shared_backend_requires_gate_api_without_private_marker(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors import (
            mega_moe_fp8_se,
        )

        def gated(*args, shared_expert_gates=None):
            pass

        def ungated(*args):
            pass

        backend = SimpleNamespace(fp8_fp8_mega_moe=gated)
        with patch.dict(
            sys.modules, {"deep_gemm": SimpleNamespace(mega_fp8=backend)}
        ), patch.object(mega_moe_fp8_se, "mega_moe_fp8_available", return_value=True):
            self.assertTrue(mega_moe_fp8_se.mega_moe_fp8_se_available())
            backend.fp8_fp8_mega_moe = ungated
            self.assertFalse(mega_moe_fp8_se.mega_moe_fp8_se_available())

    def test_warmup_cache_tracks_packer_and_buffer(self):
        executor = mega_moe_fp8.MegaMoeFp8Executor.__new__(
            mega_moe_fp8.MegaMoeFp8Executor
        )
        torch.nn.Module.__init__(executor)
        executor.cfg = Fp8Fp4MoeRuntimeConfig(
            layer_id=1,
            hidden_size=4096,
            moe_inter_dim=1024,
            expert_num=16,
            moe_k=4,
            n_shared_experts=0,
            swiglu_limit=0.0,
            ep_size=2,
            ep_rank=0,
            max_tokens_per_rank=32,
            moe_strategy="mega_moe_fp8",
        )
        executor.uses_shared_expert_gates = False
        executor._input_packer = SimpleNamespace(name="fused")
        executor._mega_buf = SimpleNamespace(num_max_tokens_per_rank=1440)
        executor.warmup_jit = Mock()
        executor._resolve_jit_warmup_token_counts = Mock(return_value=[1, 32])
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ, {"MODEL_WARM_UP": "1", "MEGA_MOE_NVCC_TMPDIR": tmpdir}
        ), patch.dict(
            sys.modules, {"deep_gemm": SimpleNamespace(get_num_sms=lambda: 148)}
        ), patch(
            "torch.cuda.is_current_stream_capturing", return_value=False
        ), patch(
            "torch.distributed.is_initialized", return_value=False
        ), patch.object(
            mega_moe, "_MEGA_MOE_JIT_WARMED_KEYS", set()
        ):
            executor._maybe_warmup_jit_once()
            executor._maybe_warmup_jit_once()
            self.assertEqual(executor.warmup_jit.call_count, 1)
            self.assertEqual(len(mega_moe._MEGA_MOE_JIT_WARMED_KEYS), 1)
            os.environ["MEGA_MOE_INPUT_PACKER_IMPL"] = "legacy"
            executor._maybe_warmup_jit_once()
            os.environ["MEGA_MOE_PACK_BLOCK_M"] = "1"
            executor._maybe_warmup_jit_once()
            executor._mega_buf.num_max_tokens_per_rank = 2880
            executor._maybe_warmup_jit_once()
            self.assertEqual(executor.warmup_jit.call_count, 4)


if __name__ == "__main__":
    unittest.main()
