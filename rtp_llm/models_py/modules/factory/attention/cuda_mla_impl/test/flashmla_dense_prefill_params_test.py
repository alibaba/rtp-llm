from types import SimpleNamespace
from unittest import TestCase, main, skipUnless
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.attention.attn_factory import DECODE_MLA_IMPS
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
    flashinfer_mla_wrapper,
    flashmla_dense_prefill,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import (
    MlaFlashInferPrefillImpl,
    MlaFlashMLAPrefillImpl,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_dense_prefill import (
    MlaFlashMLAPrefillOp,
    build_flashmla_device_params,
)
from rtp_llm.ops import AttentionConfigs
from rtp_llm.ops.compute_ops import rtp_llm_ops

CUDA_AVAILABLE = torch.cuda.is_available()


class FlashMlaDensePrefillConfigForwardingTest(TestCase):
    def test_sparse_mla_decode_is_registered_without_cp_dependency(self) -> None:
        self.assertIn("SparseMlaImpl", [impl.__name__ for impl in DECODE_MLA_IMPS])

    def test_prefill_backend_support_routes_glm53_to_flashinfer(self) -> None:
        attn_inputs = SimpleNamespace(is_prefill=True)
        k3 = SimpleNamespace(
            use_mla=True,
            nope_head_dim=128,
            rope_head_dim=64,
            v_head_dim=128,
        )
        glm53 = SimpleNamespace(
            use_mla=True,
            nope_head_dim=256,
            rope_head_dim=0,
            v_head_dim=256,
        )

        self.assertTrue(MlaFlashMLAPrefillImpl.support(k3, attn_inputs))
        self.assertTrue(MlaFlashInferPrefillImpl.support(k3, attn_inputs))
        self.assertFalse(MlaFlashMLAPrefillImpl.support(glm53, attn_inputs))
        self.assertTrue(MlaFlashInferPrefillImpl.support(glm53, attn_inputs))

    def test_wrapper_forwards_explicit_prefix_chunk_capacity(self) -> None:
        configs = AttentionConfigs()
        configs.head_num = 96
        configs.kv_lora_rank = 512
        configs.rope_head_dim = 64
        configs.nope_head_dim = 128
        configs.v_head_dim = 128
        configs.kernel_tokens_per_block = 4096
        configs.softmax_extra_scale = 1.0
        configs.use_mla = True
        configs.mla_prefill_kv_chunk_tokens = 32768
        captured: dict[str, int] = {}

        def make_op(*args: object, **kwargs: object) -> object:
            captured["prefix_chunk_tokens"] = int(kwargs["prefix_chunk_tokens"])
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
            MlaFlashMLAPrefillImpl(
                configs,
                SimpleNamespace(),
                [],
                torch.empty(0),
            )

        self.assertEqual(captured["prefix_chunk_tokens"], 32768)


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
            "reuse_cache_page_indice_d": [11, 12, 13, 14, 21, 22, 23, 24],
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
        self.assertEqual(
            params.reuse_cache_page_indice_d.data_ptr(), block_table.data_ptr()
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

    def test_consecutive_plans_do_not_overwrite_prior_group(self) -> None:
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
                "reuse_cache_page_indice_d",
            )
        }

        # A subsequent planner invocation models the next forward selecting a
        # different HybridCache group.  It must produce fresh metadata rather
        # than update storage still owned by the earlier forward in place.
        second_inputs = _attention_inputs([1, 3], [260, 64], groups, current_group=1)
        second = build_flashmla_device_params(second_inputs, self.page_size)

        self.assertIsNot(first, second)
        self.assertEqual(
            first.reuse_cache_page_indice_d.data_ptr(), group_zero.data_ptr()
        )
        self.assertEqual(
            second.reuse_cache_page_indice_d.data_ptr(), group_one.data_ptr()
        )
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

        op = object.__new__(MlaFlashMLAPrefillOp)
        op.qk_rope_head_dim = 64
        op.kv_lora_rank = 512
        op.page_size = self.page_size
        op.has_reuse_cache = True
        op.reuse_cache_page_indice = params.reuse_cache_page_indice_d
        op.batch_reuse_info_vec = params.batch_reuse_info_vec_d
        op.qo_indptr = params.qo_indptr_d
        op.total_kv_lens = sum(params.kv_lens_host)
        op.batch_size = len(params.q_lens_host)
        op._direct_attn_inputs = attn_inputs
        op._direct_block_table_width = initial_group.shape[1]

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

    def test_rejects_cuda_host_mirrors(self) -> None:
        block_table = torch.arange(8, dtype=torch.int32, device="cuda").reshape(2, 4)
        for field in ("input_lengths_host", "prefix_lengths_host"):
            with self.subTest(field=field):
                attn_inputs = _attention_inputs(
                    q_lens=[4, 4],
                    prefix_lens=[128, 16],
                    block_tables=[block_table],
                    current_group=0,
                )
                setattr(attn_inputs, field, getattr(attn_inputs, field).cuda())
                with self.assertRaisesRegex(RuntimeError, rf"CPU.*{field}"):
                    build_flashmla_device_params(attn_inputs, self.page_size)

    def test_rejects_query_write_past_block_table(self) -> None:
        block_table = torch.tensor([[17]], dtype=torch.int32, device="cuda")
        attn_inputs = _attention_inputs(
            q_lens=[4],
            prefix_lens=[127],
            block_tables=[block_table],
            current_group=0,
        )

        with self.assertRaisesRegex(RuntimeError, "query write exceeds"):
            build_flashmla_device_params(attn_inputs, self.page_size)

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
        self.assertEqual(params.block_table_width, 0)
        self.assertEqual(params.reuse_cache_page_indice_d.numel(), 0)

    def test_rejects_non_i32_host_mirror(self) -> None:
        block_table = torch.tensor([[17, 18]], dtype=torch.int32, device="cuda")
        attn_inputs = _attention_inputs(
            q_lens=[4],
            prefix_lens=[8],
            block_tables=[block_table],
            current_group=0,
        )
        attn_inputs.input_lengths_host = attn_inputs.input_lengths_host.to(torch.int64)

        with self.assertRaisesRegex(RuntimeError, "int32 input_lengths_host"):
            build_flashmla_device_params(attn_inputs, self.page_size)


if __name__ == "__main__":
    main()
