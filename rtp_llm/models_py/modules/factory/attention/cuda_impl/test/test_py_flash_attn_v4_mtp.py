import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flash_attn_v4 import (
    FlashAttn4MTPImpl,
    FlashAttn4MTPOp,
    FlashAttn4MTPParams,
    _get_num_splits,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.atten_test_util import (
    write_kv_cache,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.attention_ref import (
    apply_base_rope_to_qkv_reference,
    compute_paged_prefill_reference,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.base_attention_test import (
    BaseAttentionTest,
    create_attention_config,
)
from rtp_llm.ops import AttentionConfigs, KvCacheDataType, RopeStyle
from rtp_llm.ops.compute_ops import LayerKVCache

PAGE_SIZE = 64
HEAD_DIM = 256
NUM_Q_HEADS = 12
NUM_KV_HEADS = 2
QUERY_LEN = 5
MAX_KV_LEN = 320


def _make_config() -> AttentionConfigs:
    config = create_attention_config(
        head_num=NUM_Q_HEADS,
        head_num_kv=NUM_KV_HEADS,
        size_per_head=HEAD_DIM,
        seq_size_per_block=PAGE_SIZE,
        dtype=torch.bfloat16,
    )
    config.max_seq_len = MAX_KV_LEN
    config.kv_cache_dtype = KvCacheDataType.BASE
    config.is_causal = True
    config.need_rope_kv_cache = True
    config.q_scaling = 1.0
    config.softmax_extra_scale = 1.0
    config.use_mla = False
    config.gen_num_per_cycle = QUERY_LEN - 1
    config.rope_config.style = RopeStyle.Base
    config.rope_config.dim = HEAD_DIM
    config.rope_config.base = 10000
    config.rope_config.max_pos = MAX_KV_LEN
    return config


def _full_impl_reference(
    qkv: torch.Tensor,
    cache_snapshot: torch.Tensor,
    page_table: torch.Tensor,
    kv_lengths: list[int],
) -> torch.Tensor:
    prefix_lengths = [kv_len - QUERY_LEN for kv_len in kv_lengths]
    rope_qkv = apply_base_rope_to_qkv_reference(
        qkv,
        [QUERY_LEN] * len(kv_lengths),
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        position_offsets=prefix_lengths,
    )
    q_size = NUM_Q_HEADS * HEAD_DIM
    kv_size = NUM_KV_HEADS * HEAD_DIM
    query, key, value = torch.split(rope_qkv, [q_size, kv_size, kv_size], dim=-1)
    query = query.reshape(-1, NUM_Q_HEADS, HEAD_DIM)
    key = key.reshape(-1, NUM_KV_HEADS, HEAD_DIM)
    value = value.reshape(-1, NUM_KV_HEADS, HEAD_DIM)
    reference_cache = LayerKVCache()
    reference_cache.kv_cache_base = cache_snapshot.clone()
    write_kv_cache(
        key,
        value,
        reference_cache,
        torch.full((len(kv_lengths),), QUERY_LEN, dtype=torch.int32),
        page_table,
        start_positions=torch.tensor(prefix_lengths, dtype=torch.int32),
    )
    return compute_paged_prefill_reference(
        query,
        reference_cache,
        page_table,
        kv_lengths,
        [QUERY_LEN] * len(kv_lengths),
    )


class TestFlashAttn4MTPContract(unittest.TestCase):
    def test_num_splits_matches_upstream_resource_limits(self) -> None:
        # Upstream uses min(SM capacity, N tiles, 128). Its heuristic returns
        # zero to disable SplitKV when no extra split fits; this explicit
        # num_splits API represents the same non-split execution with one.
        self.assertEqual(
            _get_num_splits(
                sm_count=132,
                batch_size=1,
                query_len=QUERY_LEN,
                num_q_heads=NUM_Q_HEADS,
                num_kv_heads=NUM_KV_HEADS,
                max_kv_len=4 * 32,
            ),
            1,
        )
        self.assertEqual(
            _get_num_splits(
                sm_count=20,
                batch_size=2,
                query_len=QUERY_LEN,
                num_q_heads=NUM_Q_HEADS,
                num_kv_heads=NUM_KV_HEADS,
                max_kv_len=10 * 32,
            ),
            5,
        )
        self.assertEqual(
            _get_num_splits(
                sm_count=132,
                batch_size=1,
                query_len=QUERY_LEN,
                num_q_heads=NUM_Q_HEADS,
                num_kv_heads=NUM_KV_HEADS,
                max_kv_len=10 * 32,
            ),
            10,
        )
        self.assertEqual(
            _get_num_splits(
                sm_count=512,
                batch_size=1,
                query_len=QUERY_LEN,
                num_q_heads=NUM_Q_HEADS,
                num_kv_heads=NUM_KV_HEADS,
                max_kv_len=256 * 32,
            ),
            128,
        )
        self.assertEqual(
            _get_num_splits(
                sm_count=2,
                batch_size=2,
                query_len=QUERY_LEN,
                num_q_heads=NUM_Q_HEADS,
                num_kv_heads=NUM_KV_HEADS,
                max_kv_len=10 * 32,
            ),
            1,
        )
        self.assertEqual(
            _get_num_splits(
                sm_count=132,
                batch_size=0,
                query_len=QUERY_LEN,
                num_q_heads=NUM_Q_HEADS,
                num_kv_heads=NUM_KV_HEADS,
                max_kv_len=10 * 32,
            ),
            1,
        )

    def test_prepare_builds_fixed_width_kernel_metadata(self) -> None:
        op = object.__new__(FlashAttn4MTPOp)
        op.page_size = PAGE_SIZE
        op.max_kv_len = MAX_KV_LEN
        op.head_num = NUM_Q_HEADS
        op.kv_head_num = NUM_KV_HEADS
        op.query_len = QUERY_LEN
        op.fmha_params = mock.Mock()
        op.mtp_params = None
        page_table = torch.tensor([[0, 1, 0], [2, 3, 4]], dtype=torch.int32)
        inputs = SimpleNamespace(
            is_target_verify=True,
            input_lengths=torch.full((2,), QUERY_LEN, dtype=torch.int32),
            prefix_lengths_device=torch.tensor([60, 124], dtype=torch.int32),
            total_tokens=2 * QUERY_LEN,
            cu_seqlens_device=torch.tensor([0, 5, 10], dtype=torch.int32),
            cu_kv_seqlens_device=torch.tensor([0, 65, 194], dtype=torch.int32),
            kv_cache_kernel_block_id_device=page_table,
            prefill_cuda_graph_copy_params=None,
        )

        with (
            mock.patch(
                "rtp_llm.models_py.modules.factory.attention.cuda_impl."
                "py_flash_attn_v4.check_attention_inputs"
            ),
            mock.patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(multi_processor_count=132),
            ),
        ):
            op.prepare(inputs)

        args = op.fmha_params.fill_params_mha_device.call_args.args
        self.assertEqual(op.mtp_params.batch_size, 2)
        self.assertEqual(args[0].tolist(), [60, 124])
        self.assertEqual(args[1].tolist(), [65, 129])
        self.assertEqual(args[2].tolist(), [QUERY_LEN, QUERY_LEN])
        self.assertEqual(args[3].tolist(), page_table.tolist())
        self.assertFalse(args[5])

    def test_cuda_graph_prepare_refreshes_metadata_in_place(self) -> None:
        op = object.__new__(FlashAttn4MTPOp)
        op.page_size = PAGE_SIZE
        op.max_kv_len = MAX_KV_LEN
        op.head_num = NUM_Q_HEADS
        op.kv_head_num = NUM_KV_HEADS
        op.query_len = QUERY_LEN
        op.fmha_params = mock.Mock()
        op.mtp_params = None
        page_table = torch.tensor([[0, 1, 0], [2, 3, 4]], dtype=torch.int32)
        prefix_lengths = torch.tensor([60, 124], dtype=torch.int32)
        inputs = SimpleNamespace(
            total_tokens=2 * QUERY_LEN,
            prefix_lengths_device=prefix_lengths,
            kv_cache_kernel_block_id_device=page_table,
        )

        with (
            mock.patch(
                "rtp_llm.models_py.modules.factory.attention.cuda_impl."
                "py_flash_attn_v4.check_attention_inputs"
            ),
            mock.patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(multi_processor_count=132),
            ),
        ):
            op.prepare(inputs)
            mtp_params = op.mtp_params
            input_lengths_ptr = mtp_params.input_lengths.data_ptr()
            kv_lengths_ptr = mtp_params.kv_lengths.data_ptr()
            num_splits = mtp_params.num_splits
            prefix_lengths.copy_(torch.tensor([100, 200], dtype=torch.int32))
            op.fmha_params.reset_mock()

            op.prepare_cuda_graph(inputs)

        self.assertIs(op.mtp_params, mtp_params)
        self.assertEqual(op.mtp_params.input_lengths.data_ptr(), input_lengths_ptr)
        self.assertEqual(op.mtp_params.kv_lengths.data_ptr(), kv_lengths_ptr)
        self.assertEqual(op.mtp_params.num_splits, num_splits)
        self.assertEqual(op.mtp_params.kv_lengths.tolist(), [105, 205])
        args = op.fmha_params.fill_params_mha_device.call_args.args
        self.assertIs(args[0], op.mtp_params.prefix_lengths)
        self.assertIs(args[1], op.mtp_params.kv_lengths)
        self.assertIs(args[2], op.mtp_params.input_lengths)
        self.assertIs(args[3], op.mtp_params.page_table)
        self.assertTrue(args[5])

    def test_cuda_graph_prepare_rejects_bucket_or_buffer_changes(self) -> None:
        op = object.__new__(FlashAttn4MTPOp)
        op.page_size = PAGE_SIZE
        op.query_len = QUERY_LEN
        op.fmha_params = mock.Mock()
        prefix_lengths = torch.tensor([60, 124], dtype=torch.int32)
        page_table = torch.tensor([[0, 1, 0], [2, 3, 4]], dtype=torch.int32)
        op.mtp_params = FlashAttn4MTPParams(
            batch_size=2,
            query_len=QUERY_LEN,
            num_splits=1,
            input_lengths=torch.full((2,), QUERY_LEN, dtype=torch.int32),
            prefix_lengths=prefix_lengths,
            kv_lengths=torch.empty_like(prefix_lengths),
            page_table=page_table,
        )

        with mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl."
            "py_flash_attn_v4.check_attention_inputs"
        ):
            changed_tokens = SimpleNamespace(
                total_tokens=QUERY_LEN,
                prefix_lengths_device=prefix_lengths,
                kv_cache_kernel_block_id_device=page_table,
            )
            with self.assertRaisesRegex(RuntimeError, "token count"):
                op.prepare_cuda_graph(changed_tokens)

            changed_buffers = SimpleNamespace(
                total_tokens=2 * QUERY_LEN,
                prefix_lengths_device=prefix_lengths.clone(),
                kv_cache_kernel_block_id_device=page_table.clone(),
            )
            with self.assertRaisesRegex(RuntimeError, "buffers cannot change"):
                op.prepare_cuda_graph(changed_buffers)

    def test_prepare_rejects_token_count_outside_fixed_width_batches(self) -> None:
        op = object.__new__(FlashAttn4MTPOp)
        op.query_len = QUERY_LEN
        inputs = SimpleNamespace(total_tokens=2 * QUERY_LEN - 1)
        with mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl."
            "py_flash_attn_v4.check_attention_inputs"
        ):
            with self.assertRaisesRegex(ValueError, "divisible by query_len"):
                op.prepare(inputs)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestFlashAttn4MTPHopper(BaseAttentionTest):
    @classmethod
    def setUpClass(cls) -> None:
        if torch.cuda.get_device_capability()[0] != 9:
            raise unittest.SkipTest("FA4 MTP currently targets SM90")
        torch.manual_seed(2026)

    def _make_mtp_inputs(
        self,
        kv_lengths: list[int],
        page_table: torch.Tensor,
        is_target_verify: bool,
        is_cuda_graph: bool = False,
    ):
        inputs = self._create_chunked_prefill_attention_inputs(
            input_lengths=[QUERY_LEN] * len(kv_lengths),
            prefix_lengths=[kv_len - QUERY_LEN for kv_len in kv_lengths],
            seq_size_per_block=PAGE_SIZE,
            dtype=torch.bfloat16,
            kv_cache_block_id=page_table,
            max_seq_len=MAX_KV_LEN,
            is_cuda_graph=is_cuda_graph,
        )
        inputs.prefix_lengths = inputs.prefix_lengths.to(self.device)
        inputs.is_target_verify = is_target_verify
        return inputs

    def _make_page_table(self, kv_lengths: list[int]) -> tuple[torch.Tensor, int]:
        page_count = self._calculate_total_blocks(kv_lengths, PAGE_SIZE)
        page_table = self._create_kv_cache_block_ids(
            len(kv_lengths),
            kv_lengths,
            PAGE_SIZE,
            max_seq_len=MAX_KV_LEN,
        ).to(self.device)
        return page_table, page_count

    def _assert_eager_impl_matches_reference(
        self, is_target_verify: bool, kv_lengths: list[int]
    ) -> None:
        page_table, page_count = self._make_page_table(kv_lengths)
        inputs = self._make_mtp_inputs(kv_lengths, page_table, is_target_verify)
        cache, _, _ = self._create_kv_cache(
            page_count,
            PAGE_SIZE,
            NUM_KV_HEADS,
            HEAD_DIM,
            dtype=torch.bfloat16,
        )
        qkv = torch.randn(
            len(kv_lengths) * QUERY_LEN,
            (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        cache_snapshot = cache.kv_cache_base.clone()
        expected = _full_impl_reference(qkv, cache_snapshot, page_table, kv_lengths)
        impl = FlashAttn4MTPImpl(_make_config(), inputs)
        actual = impl.forward(qkv.clone(), cache)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def _copy_replay_metadata(self, inputs, replay_inputs) -> None:
        inputs.kv_cache_kernel_block_id_device.copy_(
            replay_inputs.kv_cache_kernel_block_id_device
        )
        inputs.kv_cache_kernel_block_id.copy_(replay_inputs.kv_cache_kernel_block_id)
        inputs.cu_kv_seqlens_device.copy_(replay_inputs.cu_kv_seqlens_device)
        inputs.input_lengths.copy_(replay_inputs.input_lengths)
        inputs.prefix_lengths.copy_(replay_inputs.prefix_lengths)
        inputs.sequence_lengths.copy_(replay_inputs.sequence_lengths)

    def test_target_verify_eager_impl_matches_paged_attention_reference(self) -> None:
        self._assert_eager_impl_matches_reference(True, [193, 257])

    def test_draft_prefill_eager_impl_matches_paged_attention_reference(self) -> None:
        self._assert_eager_impl_matches_reference(False, [132, 259])

    def test_cuda_graph_replay_matches_paged_attention_reference(self) -> None:
        capture_lengths = [193, 257]
        replay_lengths = [130, 319]
        page_table, page_count = self._make_page_table(capture_lengths)
        replay_page_table, replay_page_count = self._make_page_table(replay_lengths)
        self.assertLessEqual(replay_page_count, page_count)

        inputs = self._make_mtp_inputs(
            capture_lengths, page_table, True, is_cuda_graph=True
        )
        cache, _, _ = self._create_kv_cache(
            page_count,
            PAGE_SIZE,
            NUM_KV_HEADS,
            HEAD_DIM,
            dtype=torch.bfloat16,
        )
        cache_snapshot = cache.kv_cache_base.clone()
        capture_qkv = torch.randn(
            len(capture_lengths) * QUERY_LEN,
            (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        static_qkv = capture_qkv.clone()
        impl = FlashAttn4MTPImpl(_make_config(), inputs)

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            impl.forward(static_qkv, cache)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        static_qkv.copy_(capture_qkv)
        cache.kv_cache_base.copy_(cache_snapshot)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = impl.forward(static_qkv, cache)

        replay_qkv = torch.randn_like(static_qkv)
        static_qkv.copy_(replay_qkv)
        cache.kv_cache_base.copy_(cache_snapshot)
        page_table.copy_(replay_page_table)
        replay_inputs = self._make_mtp_inputs(
            replay_lengths, replay_page_table, True, is_cuda_graph=True
        )
        self._copy_replay_metadata(inputs, replay_inputs)
        impl.prepare_cuda_graph(inputs)
        graph.replay()
        torch.cuda.synchronize()

        expected = _full_impl_reference(
            replay_qkv, cache_snapshot, page_table, replay_lengths
        )
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_cuda_graph_replay_is_bitwise_equal_to_eager_impl(self) -> None:
        capture_lengths = [193, 257]
        replay_lengths = [130, 319]
        capture_page_table, page_count = self._make_page_table(capture_lengths)
        replay_page_table, replay_page_count = self._make_page_table(replay_lengths)
        self.assertLessEqual(replay_page_count, page_count)

        capture_inputs = self._make_mtp_inputs(
            capture_lengths, capture_page_table, True, is_cuda_graph=True
        )
        replay_inputs = self._make_mtp_inputs(
            replay_lengths, replay_page_table, True, is_cuda_graph=True
        )
        graph_cache, _, _ = self._create_kv_cache(
            page_count,
            PAGE_SIZE,
            NUM_KV_HEADS,
            HEAD_DIM,
            dtype=torch.bfloat16,
        )
        cache_snapshot = graph_cache.kv_cache_base.clone()
        eager_cache, _, _ = self._create_kv_cache(
            page_count,
            PAGE_SIZE,
            NUM_KV_HEADS,
            HEAD_DIM,
            dtype=torch.bfloat16,
        )
        eager_cache.kv_cache_base.copy_(cache_snapshot)

        capture_qkv = torch.randn(
            len(capture_lengths) * QUERY_LEN,
            (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM,
            dtype=torch.bfloat16,
            device=self.device,
        )
        replay_qkv = torch.randn_like(capture_qkv)
        eager_impl = FlashAttn4MTPImpl(_make_config(), replay_inputs)
        expected = eager_impl.forward(replay_qkv.clone(), eager_cache).clone()

        static_qkv = capture_qkv.clone()
        graph_impl = FlashAttn4MTPImpl(_make_config(), capture_inputs)
        self.assertTrue(graph_impl.support_cuda_graph())
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            graph_impl.forward(static_qkv, graph_cache)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        static_qkv.copy_(capture_qkv)
        graph_cache.kv_cache_base.copy_(cache_snapshot)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = graph_impl.forward(static_qkv, graph_cache)

        static_qkv.copy_(replay_qkv)
        graph_cache.kv_cache_base.copy_(cache_snapshot)
        self._copy_replay_metadata(capture_inputs, replay_inputs)
        graph_impl.prepare_cuda_graph(capture_inputs)
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
