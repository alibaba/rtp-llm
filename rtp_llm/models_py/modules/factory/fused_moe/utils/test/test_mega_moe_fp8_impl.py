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
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fp8_se import (
    MegaMoeFp8SEExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.layer import (
    Fp8Fp4MoeRuntimeConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.fp8_impl import (
    mega_moe_fp8_impl,
)


class MegaMoeFp8ImplTest(unittest.TestCase):
    def test_values_match_deepgemm(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(mega_moe_fp8_impl(), "optimized")
            for value, expected in (
                ("", "optimized"),
                ("optimized", "optimized"),
                ("legacy", "legacy"),
            ):
                os.environ["DG_MEGA_MOE_FP8_IMPL"] = value
                self.assertEqual(mega_moe_fp8_impl(), expected)
                self.assertEqual(os.environ["DG_MEGA_MOE_FP8_IMPL"], value)
            for value in ("LEGACY", "0", "1", " optimized "):
                os.environ["DG_MEGA_MOE_FP8_IMPL"] = value
                with self.assertRaisesRegex(ValueError, "expected optimized or legacy"):
                    mega_moe_fp8_impl()

    def test_launch_delegates_same_env_and_rejects_invalid_modes(self):
        seen = []
        kernel = Mock(
            side_effect=lambda *args, **kwargs: seen.append(
                os.environ.get("DG_MEGA_MOE_FP8_IMPL")
            )
        )
        deep_gemm = SimpleNamespace(mega_fp8=SimpleNamespace(fp8_fp8_mega_moe=kernel))
        executor = SimpleNamespace(
            config=SimpleNamespace(layer_id=0),
            l1=object(),
            l2=object(),
            _mega_buf=object(),
            _maybe_pre_kernel_barrier=Mock(),
        )
        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}), patch.dict(
            os.environ, {"MEGA_MOE_FP8_RESERVE_SM": "0"}
        ), patch.object(
            mega_moe_fp8, "mega_moe_snapshot_active", return_value=False
        ), patch.object(
            mega_moe_fp8, "sync_cuda_graph_warmup_ranks"
        ):
            for value in ("optimized", "legacy", "", "optimized"):
                os.environ["DG_MEGA_MOE_FP8_IMPL"] = value
                mega_moe_fp8.MegaMoeFp8Executor._launch(executor, object(), 1, "cuda:0")
            self.assertEqual(seen, ["optimized", "legacy", "", "optimized"])
            os.environ["DG_MEGA_MOE_FP8_IMPL"] = "invalid"
            with self.assertRaisesRegex(ValueError, "expected optimized or legacy"):
                mega_moe_fp8.MegaMoeFp8Executor._launch(executor, object(), 1, "cuda:0")
            os.environ["DG_MEGA_MOE_FP8_IMPL"] = "legacy"
            with self.assertRaisesRegex(
                ValueError, "does not support shared_expert_gates"
            ):
                mega_moe_fp8.MegaMoeFp8Executor._launch(
                    executor, object(), 1, "cuda:0", shared_expert_gates=object()
                )
            self.assertEqual(kernel.call_count, 4)
            with self.assertRaisesRegex(
                ValueError, "does not support shared_expert_gates"
            ):
                MegaMoeFp8SEExecutor(None, None, {})
            os.environ["DG_MEGA_MOE_FP8_IMPL"] = "optimized"
            mega_moe_fp8.MegaMoeFp8Executor._launch(
                executor, object(), 1, "cuda:0", shared_expert_gates=object()
            )
            self.assertEqual(kernel.call_count, 5)

    def test_warmup_cache_separates_implementations(self):
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
            for value in ("optimized", "", "legacy", "legacy", "optimized"):
                os.environ["DG_MEGA_MOE_FP8_IMPL"] = value
                executor._maybe_warmup_jit_once()
            self.assertEqual(executor.warmup_jit.call_count, 2)
            self.assertEqual(len(mega_moe._MEGA_MOE_JIT_WARMED_KEYS), 2)


if __name__ == "__main__":
    unittest.main()
