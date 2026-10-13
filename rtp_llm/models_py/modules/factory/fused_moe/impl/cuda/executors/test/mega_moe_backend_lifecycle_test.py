import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_backend import (
    MegaMoeBackend,
    MegaMoeBackendConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fp8 import (
    MegaMoeFp8Backend,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fp8_runtime import (
    MegaMoeFp8Wrapper,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fp8_se import (
    MegaMoeFp8SharedBackend,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fp8_se_runtime import (
    MegaMoeFp8SEWrapper,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fused import (
    MegaMoeFusedBackend,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fused_runtime import (
    MegaMoeFusedWrapper,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_runtime import (
    MegaMoeWrapper,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_shared import (
    MegaMoeSharedBackend,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_shared_runtime import (
    MegaMoeSEWrapper,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.shared_weights import (
    prepare_shared_expert_weights,
)
from rtp_llm.ops import RoleType
from rtp_llm.utils.model_weight import W

PACKAGE = "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors"
BACKENDS = (
    (MegaMoeBackend, "mega_moe_backend._get_or_create_cuda_graph_clone_buf"),
    (MegaMoeFp8Backend, "mega_moe_fp8._get_or_create_cuda_graph_clone_buf_fp8"),
    (MegaMoeSharedBackend, "mega_moe_shared.get_or_create_mega_moe_se_clone_buf"),
    (MegaMoeFp8SharedBackend, "mega_moe_fp8._get_or_create_cuda_graph_clone_buf_fp8"),
    (MegaMoeFusedBackend, "mega_moe_fused._get_or_create_cuda_graph_clone_buf_fused"),
)


class FakeKernel(nn.Module):
    def __init__(self):
        super().__init__()
        self._mega_buf = SimpleNamespace(num_max_tokens_per_rank=2)
        self.calls = []

    def forward(self, hidden, weights, ids):
        self.calls.append(("routed", hidden.shape[0]))
        return hidden + 1

    def forward_with_shared_expert(self, hidden, weights, ids):
        self.calls.append(("fused", hidden.shape[0]))
        return hidden + 2


class MegaMoeLifecycleTest(unittest.TestCase):
    def test_generic_and_glm5_layers_share_routed_arena(self):
        from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe import buffer
        from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.backend_buffer import (
            get_or_create_mega_buf,
        )

        group = SimpleNamespace(size=lambda: 4)
        arena = SimpleNamespace(buffer=torch.empty(1))
        params = dict(
            group=group,
            num_experts=8,
            num_max_tokens_per_rank=16,
            num_topk=2,
            hidden=128,
            intermediate_hidden=256,
        )
        with patch.dict(buffer._MEGA_BUF_CACHE, {}, clear=True):
            with patch(
                "deep_gemm.get_symm_buffer_for_mega_moe", return_value=arena
            ) as allocate:
                generic = buffer._get_or_create_mega_buf(
                    **params, use_fp8_dispatch=True, activation="swiglu"
                )
                glm = get_or_create_mega_buf(**params)
                self.assertIs(glm, generic)
                allocate.assert_called_once()
                # Another group must retain a separate collective arena.
                get_or_create_mega_buf(
                    **{**params, "group": SimpleNamespace(size=lambda: 4)}
                )
                self.assertEqual(allocate.call_count, 2)

    def test_warmup_retains_automatic_capacity_buckets(self):
        from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.backend_jit_warmup import (
            generate_mega_moe_jit_token_counts,
        )

        counts = generate_mega_moe_jit_token_counts(
            num_ranks=4,
            num_experts=256,
            num_experts_per_rank=64,
            num_topk=8,
            intermediate_hidden=2048,
            num_sms=132,
            max_tokens_per_rank=64,
        )
        self.assertEqual(counts, [1, 64])

    def backend(self, backend_type):
        cfg = MegaMoeBackendConfig(
            layer_id=0,
            dim=8,
            moe_inter_dim=4,
            n_routed_experts=4,
            n_activated_experts=2,
            ep_size=1,
            ep_rank=0,
            max_tokens_per_rank=8,
            n_local_experts=4,
            local_expert_start=0,
            local_expert_end=4,
        )
        model = backend_type(cfg)
        model._mega_group = object()
        model._mega_buf = object()
        model._mega_y = torch.ones(3, 8)
        model._mega_l1_w = torch.ones(2, 2)
        model._mega_l1_sf = torch.ones(2, 2)
        model._mega_l2_w = torch.ones(2, 2)
        model._mega_l2_sf = torch.ones(2, 2)
        if backend_type in (
            MegaMoeSharedBackend,
            MegaMoeFp8SharedBackend,
            MegaMoeFusedBackend,
        ):
            model._shared_l1_w = torch.ones(2, 2)
            model._shared_l1_sf = torch.ones(2, 2)
            model._shared_l2_w = torch.ones(2, 2)
            model._shared_l2_sf = torch.ones(2, 2)
            model._num_shared_experts = 2
        if backend_type is MegaMoeFusedBackend:
            model._shared_mid_fp8 = torch.ones(2, 2)
            model._shared_mid_sf = torch.ones(2, 2)
        return model

    def test_each_clone_allocates_only_its_backend_arena(self):
        for backend_type, allocator_name in BACKENDS:
            with self.subTest(backend=backend_type.__name__):
                model = self.backend(backend_type)
                arena = object()
                with patch(
                    PACKAGE + "." + allocator_name, return_value=arena
                ) as allocator:
                    clone = model.clone_for_cuda_graph()
                    allocator.assert_called_once()
                self.assertIs(type(clone), backend_type)
                self.assertIs(clone._mega_buf, arena)
                self.assertIs(clone.cfg, model.cfg)
                self.assertIs(clone._mega_l1_w, model._mega_l1_w)
                self.assertEqual(clone._num_shared_experts, model._num_shared_experts)
                self.assertNotEqual(clone._mega_y.data_ptr(), model._mega_y.data_ptr())
                if backend_type is MegaMoeFusedBackend:
                    self.assertNotEqual(
                        clone._shared_mid_fp8.data_ptr(),
                        model._shared_mid_fp8.data_ptr(),
                    )
                clone._mega_y.fill_(5)
                self.assertEqual(model._mega_y.tolist(), [[1] * 8] * 3)

    def test_fused_clone_never_allocates_an_ordinary_arena(self):
        model = self.backend(MegaMoeFusedBackend)
        with patch(
            PACKAGE + ".mega_moe_backend._get_or_create_cuda_graph_clone_buf",
            side_effect=AssertionError("wrong allocator"),
        ):
            with patch(
                PACKAGE + ".mega_moe_fused._get_or_create_cuda_graph_clone_buf_fused",
                return_value=object(),
            ):
                model.clone_for_cuda_graph()

    def test_ordered_shared_clones_reuse_arena_but_isolate_outputs(self):
        for backend_type, allocator_name in BACKENDS:
            with self.subTest(backend=backend_type.__name__):
                model = self.backend(backend_type)
                with patch(
                    PACKAGE + "." + allocator_name,
                    side_effect=AssertionError("shared clone must not allocate"),
                ):
                    clone = model.clone_for_cuda_graph(share_mega_buf=True)
                self.assertIs(clone._mega_buf, model._mega_buf)
                self.assertNotEqual(clone._mega_y.data_ptr(), model._mega_y.data_ptr())

    def test_uninitialized_shared_arena_is_rejected(self):
        for backend_type, _ in BACKENDS:
            with self.subTest(backend=backend_type.__name__):
                model = self.backend(backend_type)
                model._mega_buf = None
                self.assertRaises(
                    RuntimeError, model.clone_for_cuda_graph, share_mega_buf=True
                )

    def test_wrapper_rejects_deferred_tp_allreduce_before_kernel_execution(self):
        wrapper = MegaMoeWrapper.__new__(MegaMoeWrapper)
        nn.Module.__init__(wrapper)
        wrapper.mega_moe = FakeKernel()
        with self.assertRaisesRegex(ValueError, "complete EP output"):
            wrapper(
                torch.zeros(2, 8),
                torch.ones(2, 2),
                torch.zeros(2, 2, dtype=torch.int64),
                skip_tp_allreduce=True,
            )
        self.assertEqual(wrapper.mega_moe.calls, [])

    def test_uneven_ranks_execute_the_same_number_of_chunks(self):
        wrapper = MegaMoeWrapper.__new__(MegaMoeWrapper)
        nn.Module.__init__(wrapper)
        wrapper.mega_moe = FakeKernel()
        hidden = torch.zeros(3, 8)
        with patch(
            PACKAGE + ".mega_moe_runtime.synchronize_moe_chunk_tokens",
            return_value=7,
        ):
            result = wrapper(
                hidden, torch.ones(3, 2), torch.zeros(3, 2, dtype=torch.int64)
            )
        self.assertTrue(torch.equal(result, hidden + 1))
        self.assertEqual(
            wrapper.mega_moe.calls,
            [("routed", 2), ("routed", 1), ("routed", 0), ("routed", 0)],
        )

    def test_decode_overflow_fails_before_any_collective(self):
        wrapper = MegaMoeWrapper.__new__(MegaMoeWrapper)
        nn.Module.__init__(wrapper)
        wrapper.mega_moe = FakeKernel()
        wrapper._is_decode_role = True
        with patch(
            PACKAGE + ".mega_moe_runtime.synchronize_moe_chunk_tokens"
        ) as synchronize:
            with self.assertRaisesRegex(ValueError, "workspace capacity"):
                wrapper(
                    torch.zeros(3, 8),
                    torch.ones(3, 2),
                    torch.zeros(3, 2, dtype=torch.int64),
                )
        synchronize.assert_not_called()
        self.assertEqual(wrapper.mega_moe.calls, [])

    def test_wrapper_chunking_preserves_kernel_and_hook_dispatch(self):
        for wrapper_type in (
            MegaMoeWrapper,
            MegaMoeFp8Wrapper,
            MegaMoeSEWrapper,
            MegaMoeFp8SEWrapper,
            MegaMoeFusedWrapper,
        ):
            with self.subTest(wrapper=wrapper_type.__name__):
                wrapper = wrapper_type.__new__(wrapper_type)
                nn.Module.__init__(wrapper)
                wrapper.mega_moe = FakeKernel()
                hooks = []
                wrapper.mega_moe.register_forward_pre_hook(
                    lambda *args: hooks.append(1)
                )
                hidden = torch.arange(32, dtype=torch.float32).reshape(4, 8)
                result = wrapper(
                    hidden, torch.ones(4, 2), torch.zeros(4, 2, dtype=torch.int64)
                )
                fused = wrapper_type is MegaMoeFusedWrapper
                self.assertTrue(torch.equal(result, hidden + (2 if fused else 1)))
                self.assertEqual(
                    wrapper.mega_moe.calls, [("fused" if fused else "routed", 2)] * 2
                )
                self.assertEqual(
                    len(hooks),
                    2 if wrapper_type in (MegaMoeWrapper, MegaMoeFp8Wrapper) else 0,
                )

    def test_shared_weight_transform_selects_routed_backend(self):
        cfg = self.backend(MegaMoeBackend).cfg
        w1 = torch.empty(8, 8, dtype=torch.float8_e4m3fn)
        w2 = torch.empty(8, 4, dtype=torch.float8_e4m3fn)
        sf = torch.ones(2, 2, dtype=torch.int32)
        for fp8 in (False, True):
            with self.subTest(fp8=fp8):
                with patch(
                    "rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.shared_weights.prepare_fp8_weight_scale_for_deepgemm",
                    return_value=sf,
                ):
                    with patch(
                        "deep_gemm.transform_weights_for_mega_moe",
                        return_value=((w1, sf), (w2, sf)),
                    ) as fp4_transform:
                        with patch(
                            "deep_gemm.transform_weights_for_mega_moe_fp8",
                            return_value=((w1, sf), (w2, sf)),
                        ) as fp8_transform:
                            result = prepare_shared_expert_weights(
                                cfg, w1, sf, w2, sf, fp8_routed=fp8
                            )
                self.assertEqual(result.num_experts, 1)
                self.assertIs(result.l1[0], w1)
                self.assertEqual(fp4_transform.call_count, int(not fp8))
                self.assertEqual(fp8_transform.call_count, int(fp8))

    def test_shared_weight_mixed_scale_formats_are_rejected(self):
        cfg = self.backend(MegaMoeBackend).cfg
        self.assertRaises(
            TypeError,
            prepare_shared_expert_weights,
            cfg,
            torch.empty(8, 8, dtype=torch.float8_e4m3fn),
            torch.ones(2, 2),
            torch.empty(8, 4, dtype=torch.float8_e4m3fn),
            torch.ones(2, 2, dtype=torch.int32),
            fp8_routed=False,
        )

    def test_shared_wrapper_preserves_checkpoint_ownership(self):
        class SetupKernel(FakeKernel):
            @classmethod
            def from_params(cls, **kwargs):
                return cls()

            def setup_weights_from_fp8(self, **kwargs):
                self.routed = kwargs

            def setup_shared_expert_from_fp8(self, **kwargs):
                self.shared = kwargs

            def maybe_warmup_fused_shared_jit_once(self):
                self.warmed = True

        config = SimpleNamespace(
            hidden_size=8,
            expert_num=4,
            moe_k=2,
            moe_inter_size=4,
            max_seq_len=128,
            gen_num_per_cycle=0,
        )
        parallel = SimpleNamespace(
            ep_size=1,
            ep_rank=0,
            role_type=RoleType.PDFUSION,
            prefill_cp_config=SimpleNamespace(is_enabled=lambda: False),
            get_ffn_tp_size=lambda: 1,
        )
        for wrapper_type in (
            MegaMoeSEWrapper,
            MegaMoeFp8SEWrapper,
            MegaMoeFusedWrapper,
        ):
            with self.subTest(wrapper=wrapper_type.__name__):
                weights = {
                    W.moe_w1: torch.empty(4, 8, 8, dtype=torch.float8_e4m3fn),
                    W.moe_w2: torch.empty(4, 8, 4, dtype=torch.float8_e4m3fn),
                    W.moe_s1: torch.ones(4, 8, 2),
                    W.moe_s2: torch.ones(4, 8, 2),
                    W.ffn_w13: torch.empty(8, 8, dtype=torch.float8_e4m3fn),
                    W.ffn_w2: torch.empty(8, 4, dtype=torch.float8_e4m3fn),
                    W.ffn_s13: torch.ones(2, 2),
                    W.ffn_s2: torch.ones(2, 2),
                }
                shared_w1 = weights[W.ffn_w13]
                with patch.object(
                    wrapper_type, "_get_mega_moe_cls", return_value=SetupKernel
                ):
                    with patch("torch.cuda.empty_cache"):
                        wrapper = wrapper_type(config, parallel, weights)
                self.assertIs(wrapper.mega_moe.shared["w1_w"], shared_w1)
                self.assertTrue(wrapper.mega_moe.warmed)
                self.assertNotIn(W.moe_w1, weights)
                self.assertEqual(
                    W.ffn_w13 in weights, wrapper_type is MegaMoeFusedWrapper
                )


if __name__ == "__main__":
    unittest.main()
