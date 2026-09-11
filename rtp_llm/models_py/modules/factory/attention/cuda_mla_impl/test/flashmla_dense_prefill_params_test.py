import os
import weakref
from types import SimpleNamespace
from typing import Sequence
from unittest import TestCase, main, skipUnless
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.factory.attention import attn_factory
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
    flashinfer_mla_wrapper,
    flashmla_dense_prefill,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import (
    MlaFlashMLAPrefillImpl,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_dense_prefill import (
    FlashMLADeviceParams,
    MlaFlashMLAPrefillOp,
    build_flashmla_device_params,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_forward_plan import (
    FlashMLAForwardRoute,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_page_rr_cache import (
    MlaPageRRCacheAdapter,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import MlaImplBase
from rtp_llm.models_py.modules.factory.linear.quantized_activation import (
    QuantizedActivation,
)
from rtp_llm.ops import AttentionConfigs
from rtp_llm.ops.compute_ops import rtp_llm_ops

_TEST_TMPDIR = os.environ.get("TEST_TMPDIR")
if _TEST_TMPDIR:
    os.environ.setdefault("DG_JIT_CACHE_DIR", os.path.join(_TEST_TMPDIR, "deep_gemm"))

CUDA_AVAILABLE = torch.cuda.is_available()


class FlashMlaWorkspaceLifetimeTest(TestCase):
    def test_release_drops_scratch_but_keeps_plan_and_consumed_output(self) -> None:
        op = object.__new__(MlaFlashMLAPrefillOp)
        op._forward_workspace = SimpleNamespace(
            packed_kv=torch.empty(128), attention_output=torch.arange(16)
        )
        op._fp8_prefix_rope = torch.empty(64)
        plan = object()
        op._forward_plan = plan
        consumed_output = op._forward_workspace.attention_output.clone()
        refs = [
            weakref.ref(op._forward_workspace.packed_kv),
            weakref.ref(op._forward_workspace.attention_output),
            weakref.ref(op._fp8_prefix_rope),
        ]
        wrapper = object.__new__(MlaFlashMLAPrefillImpl)
        wrapper.fmha_impl = op
        with patch("torch.cuda.empty_cache") as flush:
            wrapper.release_forward_workspace()
            wrapper.release_forward_workspace()
        flush.assert_not_called()
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertIs(op._forward_plan, plan)
        torch.testing.assert_close(consumed_output, torch.arange(16))


class FlashMlaCanonicalPrefixQuantizedActivationTest(TestCase):
    def setUp(self) -> None:
        self.op = object.__new__(MlaFlashMLAPrefillOp)
        self.op.kv_lora_rank = 512
        self.op.qk_rope_head_dim = 64
        self.op.q_lens = [1, 2]
        self.op.batch_reuse_info_host = ((0, 2, 0, 0), (0, 1, 0, 0))
        self.op.external_prefix_cache = True
        self.op.has_reuse_cache = True
        self.op._canonical_prefix_offsets = (0, 2, 3)
        self.op._forward_plan = SimpleNamespace(route=FlashMLAForwardRoute.FULL)
        self.latent = torch.tensor([3, 5, 6], dtype=torch.bfloat16)[:, None].repeat(
            1, 512
        )
        self.quantized = QuantizedActivation(
            torch.zeros((3, 512), dtype=torch.float8_e4m3fn),
            torch.zeros((1, 4), dtype=torch.int32),
            self.latent,
        )
        self.k_pe = self.latent[:, :64] * 10
        prefix = torch.tensor([1, 2, 4], dtype=torch.bfloat16)[:, None]
        self.prefix = torch.cat((prefix.repeat(1, 512), prefix.repeat(1, 64) * 10), 1)

    def test_materialized_prefix_interleaves_retained_bf16_query_rows(self) -> None:
        latent, rope = self.op._gather_reused_kv(
            self.quantized, self.k_pe, None, self.prefix
        )
        expected = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.bfloat16)[:, None]
        torch.testing.assert_close(latent, expected.repeat(1, 512), rtol=0, atol=0)
        torch.testing.assert_close(rope, expected.repeat(1, 64) * 10, rtol=0, atol=0)

    def test_external_prefix_validation_preserves_quantized_projection_input(
        self,
    ) -> None:
        q = torch.empty((3, 1, 192), dtype=torch.bfloat16)
        with (
            patch.object(self.op, "_create_kv_b_proj", return_value=None),
            patch.object(self.op, "_packed_kv_projection", return_value=None),
            patch.object(
                self.op,
                "_forward_full",
                side_effect=lambda q, compressed_kv, *args, **kwargs: compressed_kv,
            ),
        ):
            actual = self.op.forward(q, self.quantized, self.k_pe, None, 0, self.prefix)
        self.assertIs(actual, self.quantized)

    def test_external_prefix_still_rejects_non_bf16_cache_rows(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "canonical prefix mismatch"):
            self.op.forward(
                torch.empty((3, 1, 192), dtype=torch.bfloat16),
                self.quantized,
                self.k_pe,
                None,
                0,
                self.prefix.float(),
            )


class FlashMlaDensePrefillConfigForwardingTest(TestCase):
    def test_wrapper_forwards_expanded_kv_budget(self) -> None:
        configs = AttentionConfigs()
        configs.head_num = 96
        configs.kv_lora_rank = 512
        configs.rope_head_dim = 64
        configs.nope_head_dim = 128
        configs.v_head_dim = 128
        configs.kernel_tokens_per_block = 4096
        configs.softmax_extra_scale = 1.0
        configs.use_mla = True
        configs.mla_prefill_expanded_kv_budget_bytes = 5 * 1024**3
        captured: dict[str, object] = {}

        def make_op(*args: object, **kwargs: object) -> object:
            captured["expanded_kv_budget_bytes"] = int(
                kwargs["expanded_kv_budget_bytes"]
            )
            return object()

        with (
            patch.object(
                flashmla_dense_prefill,
                "MlaFlashMLAPrefillOp",
                side_effect=make_op,
            ),
            patch.object(
                flashinfer_mla_wrapper,
                "NewMlaRotaryEmbeddingOp",
                return_value=object(),
            ),
            patch.object(
                flashinfer_mla_wrapper,
                "MlaKVCacheWriteOp",
                return_value=object(),
            ),
            patch.object(
                flashinfer_mla_wrapper.MlaFlashInferImplBase,
                "__init__",
                return_value=None,
            ),
        ):
            MlaFlashMLAPrefillImpl(
                configs,
                SimpleNamespace(),
                [],
                torch.empty(0),
            )

        self.assertEqual(captured["expanded_kv_budget_bytes"], 5 * 1024**3)

    def test_wrapper_does_not_expand_prefill_cp_config(self) -> None:
        configs = AttentionConfigs()
        configs.head_num = 96
        configs.kv_lora_rank = 512
        configs.rope_head_dim = 64
        configs.nope_head_dim = 128
        configs.v_head_dim = 128
        configs.kernel_tokens_per_block = 128
        configs.softmax_extra_scale = 1.0
        configs.use_mla = True
        configs.mla_prefill_expanded_kv_budget_bytes = 5 * 1024**3
        parallelism = SimpleNamespace(
            tp_size=8,
            tp_rank=5,
            kv_page_rr_enabled=lambda: False,
            prefill_cp_config=SimpleNamespace(kv_cache_sharded=True),
        )
        captured: dict[str, object] = {}

        def make_op(*args: object, **kwargs: object) -> object:
            captured.update(kwargs)
            return object()

        with patch.object(
            flashmla_dense_prefill,
            "MlaFlashMLAPrefillOp",
            side_effect=make_op,
        ), patch.object(
            flashinfer_mla_wrapper,
            "NewMlaRotaryEmbeddingOp",
            return_value=object(),
        ), patch.object(
            flashinfer_mla_wrapper,
            "MlaKVCacheWriteOp",
            return_value=object(),
        ), patch.object(
            flashinfer_mla_wrapper.MlaFlashInferImplBase,
            "__init__",
            return_value=None,
        ):
            impl = MlaFlashMLAPrefillImpl(
                configs,
                SimpleNamespace(),
                [],
                torch.empty(0),
                parallelism_config=parallelism,
            )

        self.assertFalse(captured["external_prefix_cache"])
        self.assertIsNone(impl.page_rr_cache_adapter)

    def test_factory_skips_mla_impl_without_page_rr_prefill_capability(self) -> None:
        class UnsupportedImpl(MlaImplBase):
            @staticmethod
            def support(attn_configs: object, attn_inputs: object) -> bool:
                return True

            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

        class SupportedImpl(UnsupportedImpl):
            @classmethod
            def support_page_rr_prefill(cls) -> bool:
                return True

        weight = SimpleNamespace(
            weights=[],
            get_global_weight=lambda _name: torch.empty(0),
        )
        attn_inputs = SimpleNamespace(
            is_prefill=True,
            is_target_verify=False,
            is_mtp_draft_update=False,
            input_lengths_host=torch.tensor([1], dtype=torch.int32),
            prefix_lengths_host=torch.tensor([0], dtype=torch.int32),
        )
        configs = SimpleNamespace(
            indexer_topk=128,
            is_sparse=False,
            mla_fp8_compute=False,
        )
        parallelism = SimpleNamespace(
            kv_page_rr_enabled=lambda: True,
            prefill_cp_config=SimpleNamespace(is_enabled=lambda: False),
        )

        with patch.object(
            attn_factory,
            "PREFILL_MLA_IMPS",
            [UnsupportedImpl, SupportedImpl],
        ):
            impl = attn_factory.get_mla_impl(
                configs,
                weight,
                attn_inputs,
                parallelism_config=parallelism,
            )

        self.assertIsInstance(impl, SupportedImpl)


def _indptr(lengths: list[int]) -> torch.Tensor:
    values = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    return torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            values.cumsum(0, dtype=torch.int32),
        )
    )


def _padding_offset(lengths: list[int]) -> torch.Tensor:
    max_length = max(lengths)
    offsets = [
        batch * max_length - sum(lengths[:batch])
        for batch, length in enumerate(lengths)
        for _ in range(length)
    ]
    return torch.tensor(offsets, dtype=torch.int32, device="cuda")


def _attention_inputs(
    q_lens: list[int],
    prefix_lens: list[int],
    block_tables: list[torch.Tensor],
    current_group: int,
) -> SimpleNamespace:
    if len(q_lens) != len(prefix_lens):
        raise ValueError("q_lens and prefix_lens must have the same batch size")
    return SimpleNamespace(
        is_prefill=True,
        total_tokens=sum(q_lens),
        input_lengths_host=torch.tensor(q_lens, dtype=torch.int32),
        prefix_lengths_host=torch.tensor(prefix_lens, dtype=torch.int32),
        input_lengths=torch.tensor(q_lens, dtype=torch.int32, device="cuda"),
        prefix_lengths=torch.tensor(prefix_lens, dtype=torch.int32, device="cuda"),
        cu_seqlens=_indptr(q_lens),
        cu_kv_seqlens=_indptr(
            [q_len + prefix_len for q_len, prefix_len in zip(q_lens, prefix_lens)]
        ),
        padding_offset=_padding_offset(q_lens),
        kv_cache_kernel_block_id_device_by_group=block_tables,
        kv_cache_kernel_block_id_device=block_tables[current_group],
    )


def _assert_cuda_i32(test: TestCase, tensor: torch.Tensor) -> None:
    test.assertTrue(tensor.is_cuda)
    test.assertEqual(tensor.dtype, torch.int32)


@skipUnless(CUDA_AVAILABLE, "requires CUDA")
class FlashMlaDensePrefillParamsTest(TestCase):
    page_size = 128

    def _read_page_rr_prefix(
        self,
        *,
        raw_dtype: torch.dtype,
        mla_fp8_compute: bool,
        kv_scale: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        canonical = (
            torch.arange(2 * 576, dtype=torch.float32, device="cuda")
            .remainder(9)
            .sub(4)
            .reshape(2, 576)
            .to(raw_dtype)
        )
        adapter = SimpleNamespace(
            page_tokens=self.page_size,
            read_prefix=Mock(return_value=canonical),
        )
        forward = Mock(return_value=torch.empty(0, device="cuda"))
        impl = object.__new__(MlaFlashMLAPrefillImpl)
        impl.fmha_impl = SimpleNamespace(forward=forward)
        impl.fmha_params = SimpleNamespace(prefix_lens_host=(2,))
        impl.page_rr_cache_adapter = adapter
        impl.attn_inputs = SimpleNamespace(
            kv_cache_kernel_block_id_device=torch.ones(
                (1, 1), dtype=torch.int32, device="cuda"
            )
        )
        impl.attn_configs = SimpleNamespace(
            kv_lora_rank=512,
            rope_head_dim=64,
            mla_fp8_compute=mla_fp8_compute,
            mla_fp8_kv_scale=kv_scale,
        )
        kv_cache = SimpleNamespace(
            kv_cache_base=torch.empty(
                (1, self.page_size, 576), dtype=raw_dtype, device="cuda"
            )
        )
        impl.compute_prefill_context(
            torch.empty((1, 1, 192), dtype=torch.bfloat16, device="cuda"),
            torch.empty((1, 512), dtype=torch.bfloat16, device="cuda"),
            torch.empty((1, 1, 64), dtype=torch.bfloat16, device="cuda"),
            kv_cache,
            0,
        )
        adapter.read_prefix.assert_called_once()
        return canonical, forward.call_args.kwargs["canonical_prefix_kv"]

    def test_page_rr_fp8_prefix_is_dequantized_with_fixed_kv_scale(self) -> None:
        raw, actual = self._read_page_rr_prefix(
            raw_dtype=torch.float8_e4m3fn,
            mla_fp8_compute=True,
            kv_scale=0.5,
        )

        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(
            actual,
            raw.to(torch.bfloat16) * 0.5,
            rtol=0,
            atol=0,
        )

    def test_page_rr_bf16_prefix_is_forwarded_without_copy(self) -> None:
        raw, actual = self._read_page_rr_prefix(
            raw_dtype=torch.bfloat16,
            mla_fp8_compute=False,
        )

        self.assertIs(actual, raw)

    def test_page_rr_prefix_cache_dtype_must_match_fp8_mode(self) -> None:
        for raw_dtype, mla_fp8_compute in (
            (torch.bfloat16, True),
            (torch.float8_e4m3fn, False),
        ):
            with self.subTest(
                raw_dtype=raw_dtype, mla_fp8_compute=mla_fp8_compute
            ), self.assertRaisesRegex(RuntimeError, "raw cache"):
                self._read_page_rr_prefix(
                    raw_dtype=raw_dtype,
                    mla_fp8_compute=mla_fp8_compute,
                )

    def _make_unplanned_op(
        self,
        *,
        expanded_kv_budget_bytes: int = 5 * 1024**3,
    ) -> MlaFlashMLAPrefillOp:
        op = object.__new__(MlaFlashMLAPrefillOp)
        op.num_heads = 12
        op.kv_lora_rank = 512
        op.qk_rope_head_dim = 64
        op.qk_nope_head_dim = 128
        op.v_head_dim = 128
        op.page_size = self.page_size
        op.expanded_kv_budget_bytes = expanded_kv_budget_bytes
        op.external_prefix_cache = False
        op.flash_mla_cuda = SimpleNamespace(dense_prefill_fwd=lambda *args: None)
        op.fp8_compute = False
        op._prefix_producer = None
        op._forward_plan = None
        op._prefix_runtime_launches = ()
        op._forward_workspace = None
        return op

    def _make_plan_params(
        self,
        *,
        q_lens: Sequence[int],
        reuse_lens: Sequence[int],
    ) -> FlashMLADeviceParams:
        self.assertEqual(len(q_lens), len(reuse_lens))
        q_lens = list(q_lens)
        reuse_lens = list(reuse_lens)
        max_blocks = max(
            (q_len + reuse_len + self.page_size - 1) // self.page_size
            for q_len, reuse_len in zip(q_lens, reuse_lens, strict=True)
        )
        block_table = torch.arange(
            len(q_lens) * max_blocks, dtype=torch.int32, device="cuda"
        ).view(len(q_lens), max_blocks)
        return build_flashmla_device_params(
            _attention_inputs(q_lens, reuse_lens, [block_table], current_group=0),
            self.page_size,
        )

    def test_plan_builds_full_route_once_without_prefix_metadata(self) -> None:
        op = self._make_unplanned_op(expanded_kv_budget_bytes=0)
        params = self._make_plan_params(q_lens=(128,), reuse_lens=(1024,))
        with patch.object(
            flashmla_dense_prefill,
            "plan_flashmla_forward",
            wraps=flashmla_dense_prefill.plan_flashmla_forward,
        ) as planner:
            op.plan(params)

        self.assertEqual(planner.call_count, 1)
        self.assertIs(op._forward_plan.route, FlashMLAForwardRoute.FULL)
        self.assertEqual(op._prefix_runtime_launches, ())
        self.assertIsNone(op._forward_workspace)

    def test_plan_materializes_contiguous_prefix_launch_in_one_storage(self) -> None:
        op = self._make_unplanned_op(
            expanded_kv_budget_bytes=256 * 12 * (128 + 64 + 128) * 2
        )
        params = self._make_plan_params(q_lens=(2, 3), reuse_lens=(128, 128))
        with patch.object(
            op,
            "_materialize_prefix_runtime_launches",
            wraps=op._materialize_prefix_runtime_launches,
        ) as materializer:
            op.plan(params)

        self.assertEqual(materializer.call_count, 1)
        self.assertEqual(len(op._prefix_runtime_launches), 1)
        launch = op._prefix_runtime_launches[0]
        self.assertEqual(launch.qo_indptr.cpu().tolist(), [0, 2, 5])
        self.assertEqual(launch.kv_indptr.cpu().tolist(), [0, 128, 256])
        self.assertEqual(launch.gather_qo_indptr.cpu().tolist(), [0, 0, 0])
        self.assertEqual(
            launch.batch_reuse_info.cpu().tolist(),
            [[0, 128, 0, 1], [1, 128, 2, 1]],
        )
        self.assertEqual(launch.destination_starts.cpu().tolist(), [0, 2])
        self.assertEqual(launch.q_range, (0, 5))

    def test_plan_materializes_b1_then_noncontiguous_launches_once(self) -> None:
        op = self._make_unplanned_op(
            expanded_kv_budget_bytes=256 * 12 * (128 + 64 + 128) * 2
        )
        params = self._make_plan_params(q_lens=(2, 3, 1), reuse_lens=(384, 0, 128))
        with patch.object(
            flashmla_dense_prefill,
            "plan_flashmla_forward",
            wraps=flashmla_dense_prefill.plan_flashmla_forward,
        ) as planner:
            op.plan(params)

        self.assertEqual(planner.call_count, 1)
        self.assertTrue(op._forward_plan.requires_fp32_accumulator)
        self.assertEqual(len(op._prefix_runtime_launches), 2)
        b1, noncontiguous = op._prefix_runtime_launches
        self.assertEqual(b1.qo_indptr.cpu().tolist(), [0, 2])
        self.assertEqual(b1.kv_indptr.cpu().tolist(), [0, 256])
        self.assertEqual(b1.gather_qo_indptr.cpu().tolist(), [0, 0])
        self.assertEqual(b1.batch_reuse_info.cpu().tolist(), [[0, 256, 0, 2]])
        self.assertEqual(b1.destination_starts.cpu().tolist(), [0])
        self.assertEqual(b1.q_range, (0, 2))
        self.assertEqual(noncontiguous.qo_indptr.cpu().tolist(), [0, 2, 3])
        self.assertEqual(noncontiguous.kv_indptr.cpu().tolist(), [0, 128, 256])
        self.assertEqual(noncontiguous.gather_qo_indptr.cpu().tolist(), [0, 0, 0])
        self.assertEqual(
            noncontiguous.batch_reuse_info.cpu().tolist(),
            [[0, 128, 2, 1], [1, 128, 8, 1]],
        )
        self.assertEqual(noncontiguous.destination_starts.cpu().tolist(), [0, 5])
        self.assertIsNone(noncontiguous.q_range)

    def test_fixed_q4_uses_row_stride_block_table(self) -> None:
        block_table = torch.tensor(
            [[11, 12, 13, 14], [21, 22, 23, 24]],
            dtype=torch.int32,
            device="cuda",
        )
        attn_inputs = _attention_inputs(
            q_lens=[4, 4],
            prefix_lens=[130, 5],
            block_tables=[block_table],
            current_group=0,
        )

        params = build_flashmla_device_params(attn_inputs, self.page_size)

        self.assertEqual(list(params.q_lens_host), [4, 4])
        self.assertEqual(list(params.prefix_lens_host), [130, 5])
        self.assertEqual(list(params.kv_lens_host), [134, 9])
        self.assertIs(params.attn_inputs, attn_inputs)
        self.assertIsNone(params.slot_mapping)

        expected = {
            "qo_indptr_d": [0, 4, 8],
            "prefill_ragged_kv_len_indptr_d": [0, 134, 143],
            "positions_d": [130, 131, 132, 133, 5, 6, 7, 8],
            "batch_indice_d": [0, 0, 0, 0, 1, 1, 1, 1],
            # Column 2 is an offset into the fully flattened page table.  The
            # second request therefore starts at the row stride (4), not at
            # the first request's live-page count (2).
            "batch_reuse_info_vec_d": [[0, 130, 0, 2], [1, 5, 4, 1]],
        }
        for name, values in expected.items():
            actual = getattr(params, name)
            _assert_cuda_i32(self, actual)
            torch.testing.assert_close(
                actual.cpu(), torch.tensor(values, dtype=torch.int32), rtol=0, atol=0
            )

        self.assertEqual(
            params.qo_indptr_d.data_ptr(), attn_inputs.cu_seqlens.data_ptr()
        )
        self.assertEqual(
            params.prefill_ragged_kv_len_indptr_d.data_ptr(),
            attn_inputs.cu_kv_seqlens.data_ptr(),
        )

    def test_ragged_q_positions_and_prefix_pages(self) -> None:
        block_table = torch.arange(100, 115, dtype=torch.int32, device="cuda").reshape(
            3, 5
        )
        attn_inputs = _attention_inputs(
            q_lens=[2, 5, 1],
            prefix_lens=[0, 128, 257],
            block_tables=[block_table],
            current_group=0,
        )

        params = build_flashmla_device_params(attn_inputs, self.page_size)

        expected = {
            "qo_indptr_d": [0, 2, 7, 8],
            "prefill_ragged_kv_len_indptr_d": [0, 2, 135, 393],
            "positions_d": [0, 1, 128, 129, 130, 131, 132, 257],
            "batch_indice_d": [0, 0, 1, 1, 1, 1, 1, 2],
            "batch_reuse_info_vec_d": [
                [0, 0, 0, 0],
                [1, 128, 5, 1],
                [2, 257, 10, 3],
            ],
        }
        self.assertEqual(list(params.q_lens_host), [2, 5, 1])
        self.assertEqual(list(params.prefix_lens_host), [0, 128, 257])
        self.assertEqual(list(params.kv_lens_host), [2, 133, 258])
        for name, values in expected.items():
            actual = getattr(params, name)
            _assert_cuda_i32(self, actual)
            torch.testing.assert_close(
                actual.cpu(), torch.tensor(values, dtype=torch.int32), rtol=0, atol=0
            )

    def test_consecutive_plans_do_not_overwrite_prior_metadata(self) -> None:
        group_zero = torch.tensor(
            [[10, 11, 12], [20, 21, 22]],
            dtype=torch.int32,
            device="cuda",
        )
        group_one = torch.tensor(
            [[110, 111, 112], [120, 121, 122]],
            dtype=torch.int32,
            device="cuda",
        )
        groups = [group_zero, group_one]
        first_inputs = _attention_inputs([4, 4], [129, 1], groups, current_group=0)
        first = build_flashmla_device_params(first_inputs, self.page_size)
        first_snapshot = {
            name: getattr(first, name).clone()
            for name in (
                "qo_indptr_d",
                "prefill_ragged_kv_len_indptr_d",
                "positions_d",
                "batch_indice_d",
                "batch_reuse_info_vec_d",
            )
        }

        # A subsequent planner invocation models the next forward selecting a
        # different HybridCache group.  It must produce fresh metadata rather
        # than update storage still owned by the earlier forward in place.
        second_inputs = _attention_inputs([1, 3], [260, 64], groups, current_group=1)
        second = build_flashmla_device_params(second_inputs, self.page_size)

        self.assertIsNot(first, second)
        for name, snapshot in first_snapshot.items():
            torch.testing.assert_close(getattr(first, name), snapshot, rtol=0, atol=0)
        torch.testing.assert_close(
            second.batch_reuse_info_vec_d.cpu(),
            torch.tensor([[0, 260, 0, 3], [1, 64, 3, 1]], dtype=torch.int32),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            second.positions_d.cpu(),
            torch.tensor([260, 64, 65, 66], dtype=torch.int32),
            rtol=0,
            atol=0,
        )

    def test_slot_mapping_reads_live_hybrid_group_alias(self) -> None:
        initial_group = torch.tensor(
            [[10, 11, 12], [20, 21, 22]],
            dtype=torch.int32,
            device="cuda",
        )
        live_group = torch.tensor(
            [[110, 111, 112], [120, 121, 122]],
            dtype=torch.int32,
            device="cuda",
        )
        attn_inputs = _attention_inputs(
            [4, 4], [130, 5], [initial_group, live_group], current_group=0
        )
        params = build_flashmla_device_params(attn_inputs, self.page_size)

        impl = object.__new__(MlaFlashMLAPrefillImpl)
        impl.fmha_params = params
        impl.attn_inputs = attn_inputs
        impl.seq_size_per_block = self.page_size

        # K3 selects the physical HybridCache group immediately before each
        # layer.  Cache write must use that live alias, not the group that was
        # visible when the per-forward plan was first built.
        attn_inputs.kv_cache_kernel_block_id_device = live_group
        slot_mapping = impl._device_slot_mapping()

        self.assertIsNotNone(slot_mapping)
        assert slot_mapping is not None
        torch.testing.assert_close(
            slot_mapping.cpu(),
            torch.tensor(
                [
                    111 * 128 + 2,
                    111 * 128 + 3,
                    111 * 128 + 4,
                    111 * 128 + 5,
                    120 * 128 + 5,
                    120 * 128 + 6,
                    120 * 128 + 7,
                    120 * 128 + 8,
                ],
                dtype=torch.int64,
            ),
            rtol=0,
            atol=0,
        )

    def test_reuse_gather_reads_live_hybrid_group_alias(self) -> None:
        initial_group = torch.tensor(
            [[10, 11, 12], [20, 21, 22]],
            dtype=torch.int32,
            device="cuda",
        )
        live_group = torch.tensor(
            [[110, 111, 112], [120, 121, 122]],
            dtype=torch.int32,
            device="cuda",
        )
        attn_inputs = _attention_inputs(
            [4, 4], [130, 5], [initial_group, live_group], current_group=0
        )
        params = build_flashmla_device_params(attn_inputs, self.page_size)

        op = self._make_unplanned_op(expanded_kv_budget_bytes=0)
        op.plan(params)

        # Model-layer dispatch switches this alias after the per-forward plan.
        # Both cache write and reused-KV gather must observe the same live group.
        attn_inputs.kv_cache_kernel_block_id_device = live_group
        compressed_kv = torch.empty((8, 512), dtype=torch.bfloat16, device="cuda")
        k_pe = torch.empty((8, 1, 64), dtype=torch.bfloat16, device="cuda")
        kv_cache = SimpleNamespace(
            kv_cache_base=torch.empty(1, dtype=torch.uint8, device="cuda")
        )
        captured: dict[str, torch.Tensor] = {}

        def fake_reuse_gather(
            final_compressed_kv: torch.Tensor,
            final_k_pe: torch.Tensor,
            suffix_compressed_kv: torch.Tensor,
            suffix_k_pe: torch.Tensor,
            kv_cache_base: torch.Tensor,
            page_indices: torch.Tensor,
            batch_reuse_info: torch.Tensor,
            qo_indptr: torch.Tensor,
            page_size: int,
        ) -> None:
            captured["page_indices"] = page_indices
            captured["batch_reuse_info"] = batch_reuse_info
            captured["qo_indptr"] = qo_indptr
            self.assertEqual(page_size, self.page_size)

        with patch.object(
            rtp_llm_ops,
            "reuse_kv_cache_indexed_batched",
            side_effect=fake_reuse_gather,
        ):
            gathered_compressed_kv, gathered_k_pe = op._gather_reused_kv(
                compressed_kv, k_pe, kv_cache
            )

        self.assertEqual(captured["page_indices"].data_ptr(), live_group.data_ptr())
        self.assertNotEqual(
            captured["page_indices"].data_ptr(), initial_group.data_ptr()
        )
        self.assertIs(captured["batch_reuse_info"], params.batch_reuse_info_vec_d)
        self.assertIs(captured["qo_indptr"], params.qo_indptr_d)
        self.assertEqual(tuple(gathered_compressed_kv.shape), (143, 512))
        self.assertEqual(tuple(gathered_k_pe.shape), (143, 64))

    def test_cacheless_prefill_does_not_require_block_table(self) -> None:
        empty_table = torch.empty((1, 0), dtype=torch.int32, device="cuda")
        attn_inputs = _attention_inputs(
            q_lens=[4],
            prefix_lens=[0],
            block_tables=[empty_table],
            current_group=0,
        )

        params = build_flashmla_device_params(attn_inputs, self.page_size)

        self.assertFalse(params.has_reuse_cache)

    def test_rejects_query_write_past_block_table(self) -> None:
        block_table = torch.tensor([[17]], dtype=torch.int32, device="cuda")
        attn_inputs = _attention_inputs(
            q_lens=[4],
            prefix_lens=[127],
            block_tables=[block_table],
            current_group=0,
        )

        params = build_flashmla_device_params(attn_inputs, self.page_size)
        impl = object.__new__(MlaFlashMLAPrefillImpl)
        impl.page_rr_cache_adapter = None
        impl.seq_size_per_block = self.page_size
        with self.assertRaisesRegex(RuntimeError, "query write exceeds"):
            impl._validate_direct_cache_capacity(params, block_table)

    def test_page_rr_validates_rank_local_block_table_width(self) -> None:
        rank_zero_table = torch.tensor([[17, 18]], dtype=torch.int32, device="cuda")
        rank_seven_table = torch.tensor([[27]], dtype=torch.int32, device="cuda")
        # Nine global pages: rank 0 owns pages 0 and 8, while rank 7 owns only
        # page 7.  The cache adapter must validate those local widths rather
        # than requiring nine columns on every rank.
        for rank, table in ((0, rank_zero_table), (7, rank_seven_table)):
            with self.subTest(rank=rank):
                attn_inputs = _attention_inputs(
                    q_lens=[1],
                    prefix_lens=[8 * self.page_size],
                    block_tables=[table],
                    current_group=0,
                )
                params = build_flashmla_device_params(attn_inputs, self.page_size)
                adapter = MlaPageRRCacheAdapter(
                    page_tokens=self.page_size,
                    shard_size=8,
                    shard_rank=rank,
                )
                adapter.validate_block_table_capacity(table, params.kv_lens_host)
                self.assertIs(params.attn_inputs.kv_cache_kernel_block_id_device, table)
                self.assertEqual(params.batch_reuse_info_host[0], (0, 1024, 0, 8))

        too_narrow = _attention_inputs(
            q_lens=[1],
            prefix_lens=[8 * self.page_size],
            block_tables=[rank_zero_table[:, :1]],
            current_group=0,
        )
        with self.assertRaisesRegex(RuntimeError, "rank-local block table"):
            MlaPageRRCacheAdapter(
                page_tokens=self.page_size,
                shard_size=8,
                shard_rank=0,
            ).validate_block_table_capacity(
                too_narrow.kv_cache_kernel_block_id_device,
                (8 * self.page_size + 1,),
            )

    def test_page_rr_slot_mapping_writes_only_owner_pages(self) -> None:
        block_table = torch.tensor([[11, 12]], dtype=torch.int32, device="cuda")
        attn_inputs = _attention_inputs(
            q_lens=[4],
            prefix_lens=[127],
            block_tables=[block_table],
            current_group=0,
        )
        params = build_flashmla_device_params(attn_inputs, self.page_size)
        impl = object.__new__(MlaFlashMLAPrefillImpl)
        impl.fmha_params = params
        impl.attn_inputs = attn_inputs
        impl.seq_size_per_block = self.page_size
        impl.page_rr_cache_adapter = MlaPageRRCacheAdapter(
            page_tokens=self.page_size,
            shard_size=8,
            shard_rank=0,
        )

        slot_mapping = impl._device_slot_mapping()

        self.assertIsNotNone(slot_mapping)
        assert slot_mapping is not None
        torch.testing.assert_close(
            slot_mapping.cpu(),
            torch.tensor([11 * 128 + 127, -1, -1, -1], dtype=torch.int64),
            rtol=0,
            atol=0,
        )

    def test_page_rr_unused_tail_rank_accepts_empty_local_block_table(self) -> None:
        block_table = torch.empty((1, 0), dtype=torch.int32, device="cuda")
        attn_inputs = _attention_inputs(
            q_lens=[4],
            prefix_lens=[1],
            block_tables=[block_table],
            current_group=0,
        )
        params = build_flashmla_device_params(attn_inputs, self.page_size)
        impl = object.__new__(MlaFlashMLAPrefillImpl)
        impl.fmha_params = params
        impl.attn_inputs = attn_inputs
        impl.seq_size_per_block = self.page_size
        impl.page_rr_cache_adapter = MlaPageRRCacheAdapter(
            page_tokens=self.page_size,
            shard_size=8,
            shard_rank=7,
        )

        slot_mapping = impl._device_slot_mapping()

        self.assertIsNotNone(slot_mapping)
        assert slot_mapping is not None
        torch.testing.assert_close(
            slot_mapping.cpu(),
            torch.full((4,), -1, dtype=torch.int64),
            rtol=0,
            atol=0,
        )

        with self.assertRaisesRegex(RuntimeError, "rank-local block table"):
            MlaPageRRCacheAdapter(
                page_tokens=self.page_size,
                shard_size=8,
                shard_rank=0,
            ).validate_block_table_capacity(
                block_table,
                (5,),
            )


if __name__ == "__main__":
    main()
