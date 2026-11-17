"""CPU/real-Gloo coverage of GLM MegaMoE host and EP contracts."""

import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.parallelism import (
    get_world_ep_group,
)
from rtp_llm.models_py.modules.glm5_mega_moe import (
    GLM5MegaMoE,
    GLM5MegaMoEFP8,
    GLM5MegaMoEFP8SE,
    GLM5MegaMoEFused,
    GLM5MegaMoESE,
)
from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe import _activation_clamp
from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_fp8_se_wrapper import (
    MegaMoeFp8SEWrapper,
)
from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_fp8_wrapper import (
    MegaMoeFp8Wrapper,
)
from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_se_wrapper import MegaMoeSEWrapper
from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_wrapper import MegaMoeWrapper
from rtp_llm.utils.model_weight import W


def _wrapper(cls=MegaMoeWrapper, *, decode=False, capacity=4):
    wrapper = cls.__new__(cls)
    torch.nn.Module.__init__(wrapper)
    wrapper._is_decode_role = decode
    wrapper.expert_num = 8
    wrapper.mega_moe = SimpleNamespace(
        cfg=SimpleNamespace(max_tokens_per_rank=capacity),
        _mega_buf=SimpleNamespace(num_max_tokens_per_rank=8),
        _mega_group=dist.group.WORLD,
    )
    return wrapper


def _distributed_chunks(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=3,
        timeout=timedelta(seconds=20),
    )
    try:
        for lengths in ((0, 3, 9), (0, 0, 0), (3, 4, 1)):
            for cls in (
                MegaMoeWrapper,
                MegaMoeFp8Wrapper,
                MegaMoeSEWrapper,
                MegaMoeFp8SEWrapper,
            ):
                wrapper = _wrapper(cls)
                wrapper.mega_moe._mega_group = get_world_ep_group(
                    SimpleNamespace(ep_size=3, ep_rank=rank), dist
                )
                tokens = lengths[rank]
                x = torch.arange(tokens * 2, dtype=torch.float32).view(tokens, 2)
                weights = torch.arange(tokens, dtype=torch.float32).view(tokens, 1)
                ids = torch.arange(tokens, dtype=torch.int64).view(tokens, 1)
                calls = []

                def kernel(chunk, chunk_weights, chunk_ids):
                    # A missing empty call or rank-local early return changes
                    # the communication sequence and fails the bounded group.
                    participants = torch.ones((), dtype=torch.int64)
                    dist.all_reduce(participants, group=wrapper.mega_moe._mega_group)
                    assert participants.item() == 3
                    calls.append(chunk.size(0))
                    assert chunk.size(0) <= 4
                    return chunk * 3 + chunk_weights + chunk_ids

                with mock.patch("torch.cuda.is_available", return_value=False):
                    result = wrapper._forward_chunked(x, weights, ids, kernel)
                expected_calls = max((max(lengths) + 3) // 4, 1)
                assert calls == [
                    max(min(tokens - i * 4, 4), 0) for i in range(expected_calls)
                ], (rank, lengths, calls)
                torch.testing.assert_close(result, x * 3 + weights + ids)
    finally:
        dist.destroy_process_group()


class GlmMegaMoeContractTest(unittest.TestCase):
    def test_unequal_ep_tokens_execute_empty_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = (Path(directory) / "gloo-rendezvous").as_uri()
            mp.start_processes(
                _distributed_chunks,
                args=(rendezvous,),
                nprocs=3,
                join=True,
                start_method="fork",
            )

    def test_unsupported_config_rejected_before_weight_setup(self):
        for world_size, world_rank, ep_size, ep_rank in (
            (8, 0, 4, 0),
            (8, 4, 8, 0),
            (1, 0, 1, 0),
        ):
            parallelism = SimpleNamespace(
                world_size=world_size,
                world_rank=world_rank,
                ep_size=ep_size,
                ep_rank=ep_rank,
            )
            with mock.patch.object(MegaMoeWrapper, "_get_mega_moe_cls") as factory:
                with self.assertRaisesRegex(ValueError, "EP == WORLD"):
                    MegaMoeWrapper(SimpleNamespace(), parallelism, {})
                factory.assert_not_called()

    def test_runtime_group_checked_before_symmetric_allocation(self):
        for cls in (
            GLM5MegaMoE,
            GLM5MegaMoEFP8,
            GLM5MegaMoEFP8SE,
            GLM5MegaMoESE,
            GLM5MegaMoEFused,
        ):
            module = cls.__new__(cls)
            torch.nn.Module.__init__(module)
            module.cfg = SimpleNamespace(ep_size=4, ep_rank=0, n_activated_experts=1)
            module._mega_l1_w = torch.empty(1)
            module._num_shared_experts = 1
            with (
                mock.patch("torch.distributed.is_initialized", return_value=True),
                mock.patch.dict("sys.modules", {"deep_gemm": SimpleNamespace()}),
                mock.patch("torch.distributed.get_world_size", return_value=8),
                mock.patch("torch.distributed.get_rank", return_value=0),
                self.subTest(backend=cls.__name__),
            ):
                # The deliberately incomplete cfg would fail at allocation
                # arguments if validation did not run first.
                with self.assertRaisesRegex(RuntimeError, "EP group to equal WORLD"):
                    module._setup_buffer_and_warmup()

    def test_runtime_group_requires_initialization_and_rank_match(self):
        cfg = SimpleNamespace(ep_size=8, ep_rank=0)
        for initialized, rank in ((False, 0), (True, 4)):
            with (
                mock.patch(
                    "torch.distributed.is_initialized", return_value=initialized
                ),
                mock.patch("torch.distributed.get_world_size", return_value=8),
                mock.patch("torch.distributed.get_rank", return_value=rank),
                self.assertRaises(RuntimeError),
            ):
                get_world_ep_group(cfg, dist)

    def test_explicit_zero_clamp_reaches_routed_and_shared_config(self):
        config = SimpleNamespace(hidden_size=4, expert_num=8, moe_k=1, moe_inter_size=4)
        parallelism = SimpleNamespace(ep_size=2, ep_rank=0, world_size=2, world_rank=0)
        for limit, expected in ((None, 10.0), (0.0, 0.0), (3.0, 3.0)):
            if limit is not None:
                config.swiglu_limit = limit
            weights = {
                W.moe_w1: torch.empty((4, 8, 4), dtype=torch.int8),
                W.moe_w2: torch.empty((4, 4, 4), dtype=torch.int8),
                W.moe_s1: torch.ones(1),
                W.moe_s2: torch.ones(1),
            }
            factory = mock.Mock()
            with (
                mock.patch.object(
                    MegaMoeWrapper, "_get_mega_moe_cls", return_value=factory
                ),
                mock.patch("torch.cuda.empty_cache"),
            ):
                MegaMoeWrapper(config, parallelism, weights)
            actual = factory.from_params.call_args.kwargs["swiglu_limit"]
            self.assertEqual(actual, expected)
            self.assertEqual(
                _activation_clamp(SimpleNamespace(swiglu_limit=actual)),
                None if expected == 0 else expected,
            )

    def test_decode_and_graph_capture_do_not_synchronize_or_chunk(self):
        for decode, capturing in ((True, False), (False, True)):
            wrapper = _wrapper(decode=decode)
            kernel = mock.Mock(side_effect=lambda x, weights, ids: x * 3)
            with (
                mock.patch("torch.cuda.is_available", return_value=capturing),
                mock.patch(
                    "torch.cuda.is_current_stream_capturing", return_value=capturing
                ),
                mock.patch("torch.distributed.all_reduce") as collective,
            ):
                x = torch.ones(3, 2)
                torch.testing.assert_close(
                    wrapper._forward_chunked(x, x, x, kernel), x * 3
                )
                with self.assertRaisesRegex(ValueError, "exceeds capacity"):
                    wrapper._forward_chunked(torch.ones(5, 2), x, x, kernel)
            collective.assert_not_called()
            kernel.assert_called_once()

    def test_prefill_constructor_preserves_configured_chunk_budget(self):
        config = SimpleNamespace(
            hidden_size=4,
            expert_num=8,
            moe_k=1,
            moe_inter_size=4,
            max_seq_len=1048576,
        )
        parallelism = SimpleNamespace(
            ep_size=8,
            ep_rank=0,
            world_size=8,
            world_rank=0,
            tp_size=8,
            prefill_cp_config=SimpleNamespace(is_enabled=lambda: True),
        )
        cases = (
            ({"DSV4_MOE_CHUNK_TOKENS": "131072"}, 131072),
            ({"DSV4_MOE_CHUNK_TOKENS": "32768"}, 32768),
            ({"DSV4_MOE_CHUNK_TOKENS": "131072", "DSV4_CHUNK_TOKENS": "65536"}, 65536),
            ({"DSV4_MOE_CHUNK_TOKENS": "4096", "DSV4_MOE_CHUNK_PREFILL": "0"}, 131072),
            ({"DSV4_CHUNK_TOKENS": "0"}, 131072),
        )
        for env, expected in cases:
            weights = {
                W.moe_w1: torch.empty((1, 8, 4), dtype=torch.int8),
                W.moe_w2: torch.empty((1, 4, 4), dtype=torch.int8),
                W.moe_s1: torch.ones(1),
                W.moe_s2: torch.ones(1),
            }
            factory = mock.Mock()
            with (
                self.subTest(env=env),
                mock.patch.dict("os.environ", env, clear=True),
                mock.patch.object(
                    MegaMoeWrapper, "_get_mega_moe_cls", return_value=factory
                ),
                mock.patch("torch.cuda.empty_cache"),
            ):
                MegaMoeWrapper(config, parallelism, weights)
                self.assertEqual(
                    factory.from_params.call_args.kwargs["max_tokens_per_rank"],
                    expected,
                )

    def test_decode_capacity_is_independent_of_prefill_chunk_override(self):
        from rtp_llm.ops import RoleType

        config = SimpleNamespace(
            hidden_size=4,
            expert_num=8,
            moe_k=1,
            moe_inter_size=4,
            max_seq_len=1048576,
            gen_num_per_cycle=4,
        )
        parallelism = SimpleNamespace(
            ep_size=8,
            ep_rank=0,
            world_size=8,
            world_rank=0,
            tp_size=1,
            role_type=RoleType.DECODE,
        )
        weights = {
            W.moe_w1: torch.empty((1, 8, 4), dtype=torch.int8),
            W.moe_w2: torch.empty((1, 4, 4), dtype=torch.int8),
            W.moe_s1: torch.ones(1),
            W.moe_s2: torch.ones(1),
        }
        factory = mock.Mock()
        with (
            mock.patch.dict("os.environ", {"DSV4_MOE_CHUNK_TOKENS": "128"}, clear=True),
            mock.patch.object(
                MegaMoeWrapper, "_get_mega_moe_cls", return_value=factory
            ),
            mock.patch("torch.cuda.empty_cache"),
        ):
            MegaMoeWrapper(config, parallelism, weights, max_generate_batch_size=512)
        self.assertEqual(
            factory.from_params.call_args.kwargs["max_tokens_per_rank"], 2560
        )

    def test_prefill_cannot_bypass_synchronized_execution_via_prepacking(self):
        wrapper = _wrapper()
        self.assertEqual(wrapper.prepacked_input_views(3), (None,) * 4)
        with self.assertRaisesRegex(ValueError, "only for decode"):
            wrapper.forward_prepacked(torch.ones(3, 2))
        shared = _wrapper(MegaMoeFp8SEWrapper)
        self.assertIsNone(shared.prepare_fused_mhc_inputs(torch.ones(48, 4096)))

    def test_graph_clones_keep_decode_contract(self):
        for cls in (
            MegaMoeWrapper,
            MegaMoeFp8Wrapper,
            MegaMoeSEWrapper,
            MegaMoeFp8SEWrapper,
        ):
            wrapper = _wrapper(cls, decode=True)
            wrapper.mega_moe.clone_for_cuda_graph = mock.Mock(return_value=object())
            clone = wrapper.clone_for_cuda_graph()
            self.assertIs(type(clone), cls)
            self.assertTrue(clone._is_decode_role)
            self.assertEqual(clone.expert_num, wrapper.expert_num)


if __name__ == "__main__":
    unittest.main()
