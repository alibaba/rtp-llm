import os
import weakref
from types import SimpleNamespace
from typing import Sequence
from unittest import TestCase, main, skipUnless
from unittest.mock import patch

import torch

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


if __name__ == "__main__":
    main()
