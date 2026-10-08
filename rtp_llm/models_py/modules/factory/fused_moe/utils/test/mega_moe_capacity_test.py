"""Startup communication capacity is separate from the scheduler token budget."""

import os
import sys
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
    MegaMoeExecutor,
    _mega_output_capacity,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se import (
    MegaMoeSEExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.config import (
    resolve_mega_moe_max_tokens_per_rank,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.buffer import (
    _MEGA_OUTPUT_CACHE,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.se_buffer import (
    _MEGA_SE_OUTPUT_CACHE,
)
from rtp_llm.utils.model_weight import W


class MegaMoECapacityTest(unittest.TestCase):
    def tearDown(self):
        _MEGA_OUTPUT_CACHE.clear()
        _MEGA_SE_OUTPUT_CACHE.clear()

    def test_weight_setup_passes_override_to_allocator_and_aligns_output(self):
        for executor_type, module_name, allocator_name in (
            (MegaMoeExecutor, "mega_moe", "_get_or_create_mega_buf"),
            (MegaMoeSEExecutor, "mega_moe_se", "_get_or_create_mega_se_buf"),
        ):
            for requested, aligned in ((8192, 9600), (32, 1920)):
                # Prefill and Decode have independent processes/output caches.
                _MEGA_OUTPUT_CACHE.clear()
                _MEGA_SE_OUTPUT_CACHE.clear()
                prefix = (
                    "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors."
                    + module_name
                )
                tensor = torch.empty(1)
                backend = SimpleNamespace(
                    transform_weights_for_mega_moe=mock.Mock(
                        return_value=((tensor, tensor), (tensor, tensor))
                    )
                )
                config = SimpleNamespace(
                    max_tokens_per_rank=131072,
                    mega_moe_backend=backend,
                    expert_activation="swiglu",
                    n_local_experts=2,
                    n_routed_experts=16,
                    n_activated_experts=4,
                    n_shared_experts=1,
                    dim=128,
                    moe_inter_dim=128,
                    moe_w1_layout="gate_up",
                )
                weights = {
                    W.moe_w1: torch.empty(2, 256, 64),
                    W.moe_s1: torch.empty(2, 256, 1),
                    W.moe_w2: torch.empty(2, 128, 64),
                    W.moe_s2: torch.empty(2, 128, 1),
                }
                with self.subTest(
                    executor=executor_type, capacity=requested
                ), ExitStack() as stack:
                    stack.enter_context(
                        mock.patch.dict(
                            os.environ,
                            {"MEGA_MOE_MAX_TOKENS_PER_RANK": str(requested)},
                            clear=True,
                        )
                    )
                    stack.enter_context(
                        mock.patch.dict(sys.modules, {"deep_gemm": backend})
                    )
                    stack.enter_context(mock.patch("torch.cuda.empty_cache"))
                    stack.enter_context(
                        mock.patch(
                            prefix + ".prepare_fp4_weight_scale_for_deepgemm",
                            return_value=tensor,
                        )
                    )
                    stack.enter_context(
                        mock.patch(
                            prefix + "._get_validated_world_ep_group",
                            return_value=object(),
                        )
                    )
                    allocator = stack.enter_context(
                        mock.patch(
                            prefix + "." + allocator_name,
                            return_value=SimpleNamespace(
                                num_max_tokens_per_rank=aligned
                            ),
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(executor_type, "_maybe_warmup_jit_once")
                    )
                    if executor_type is MegaMoeSEExecutor:
                        stack.enter_context(
                            mock.patch.object(
                                executor_type, "_setup_shared_expert_weights"
                            )
                        )
                    executor = executor_type(config, None, weights)
                    self.assertEqual(
                        allocator.call_args.kwargs["num_max_tokens_per_rank"], requested
                    )
                    self.assertEqual(tuple(executor._mega_y.shape), (aligned, 128))
                    self.assertEqual(config.max_tokens_per_rank, 131072)

    def test_unset_or_blank_preserves_runtime_capacity(self):
        for value in (None, "", " "):
            env = {} if value is None else {"MEGA_MOE_MAX_TOKENS_PER_RANK": value}
            with self.subTest(value=value), mock.patch.dict(
                os.environ, env, clear=True
            ):
                self.assertEqual(resolve_mega_moe_max_tokens_per_rank(131072), 131072)

    def test_override_is_input_tokens_per_rank_without_another_parallel_split(self):
        for executor_type in (MegaMoeExecutor, MegaMoeSEExecutor):
            # Prefill is chunked upstream; Decode includes speculative verify rows.
            for runtime_capacity, requested in ((131072, 8192), (40, 32)):
                config = SimpleNamespace(max_tokens_per_rank=runtime_capacity)
                with self.subTest(
                    executor=executor_type, capacity=requested
                ), mock.patch.dict(
                    os.environ,
                    {"MEGA_MOE_MAX_TOKENS_PER_RANK": str(requested)},
                    clear=True,
                ), mock.patch.object(
                    executor_type, "setup_weights"
                ) as setup:
                    executor = executor_type(config, None, {})
                    setup.assert_called_once_with({})
                    self.assertEqual(executor._max_tokens_per_rank, requested)
                    self.assertEqual(config.max_tokens_per_rank, runtime_capacity)
                # Capacity is fixed at construction, including subsequent JIT warmup.
                with mock.patch.dict(
                    os.environ, {"MEGA_MOE_JIT_WARMUP_TOKENS": "1,1000000"}, clear=True
                ):
                    counts = executor._resolve_jit_warmup_token_counts(148)
                    self.assertTrue(counts)
                    self.assertLessEqual(max(counts), requested)

    def test_invalid_override_fails_before_weight_setup(self):
        for value in ("0", "-1", "1.5", "invalid"):
            for executor_type in (MegaMoeExecutor, MegaMoeSEExecutor):
                with self.subTest(value=value, executor=executor_type), mock.patch.dict(
                    os.environ, {"MEGA_MOE_MAX_TOKENS_PER_RANK": value}, clear=True
                ), mock.patch.object(executor_type, "setup_weights") as setup:
                    with self.assertRaisesRegex(
                        ValueError, "MEGA_MOE_MAX_TOKENS_PER_RANK"
                    ):
                        executor_type(SimpleNamespace(max_tokens_per_rank=40), None, {})
                    setup.assert_not_called()

    def test_input_guard_and_output_use_allocated_aligned_capacity(self):
        # Alignment is owned by DeepGEMM. Use its returned capacity, rather
        # than treating the requested 32 as a hard limit on execution rows.
        buf = SimpleNamespace(num_max_tokens_per_rank=1920)
        self.assertEqual(_mega_output_capacity(buf, 32), 1920)
        for executor_type in (MegaMoeExecutor, MegaMoeSEExecutor):
            with mock.patch.dict(
                os.environ, {"MEGA_MOE_MAX_TOKENS_PER_RANK": "32"}, clear=True
            ), mock.patch.object(executor_type, "setup_weights"):
                executor = executor_type(
                    SimpleNamespace(max_tokens_per_rank=40), None, {}
                )
            executor._mega_buf = buf
            executor._mega_y = torch.empty(1920, 1)
            executor._validate_capacity(40)
            executor._validate_capacity(1920)
            with self.assertRaisesRegex(RuntimeError, "exceeds"):
                executor._validate_capacity(1921)
            executor._mega_y = torch.empty(31, 1)
            with self.assertRaisesRegex(RuntimeError, "output rows"):
                executor._validate_capacity(40)


if __name__ == "__main__":
    unittest.main()
