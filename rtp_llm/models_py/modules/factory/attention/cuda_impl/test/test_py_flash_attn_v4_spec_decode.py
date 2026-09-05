import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models_py.modules.factory.attention import attn_factory
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flash_attn_v4 import (
    _FA4_TILE_N,
    FlashAttn4SpecDecodeImpl,
    FlashAttn4SpecDecodeOp,
    FlashAttn4SpecDecodeParams,
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
from rtp_llm.ops import (
    AttentionConfigs,
    FMHAConfig,
    KvCacheDataType,
    RopeStyle,
    get_rope_cache,
)
from rtp_llm.ops.compute_ops import LayerKVCache

_MODULE = "rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flash_attn_v4"

PAGE_SIZE = 64
HEAD_DIM = 256
NUM_Q_HEADS = 12
NUM_KV_HEADS = 2
QUERY_LEN = 5
MAX_KV_LEN = 320
# Row count MhaRotaryEmbeddingOp asks getRopeCacheOnce for.
ROPE_MAX_POS = MAX_KV_LEN + (QUERY_LEN - 1) + 1


def _make_config(
    page_size: int = PAGE_SIZE, head_dim: int = HEAD_DIM
) -> AttentionConfigs:
    config = create_attention_config(
        head_num=NUM_Q_HEADS,
        head_num_kv=NUM_KV_HEADS,
        size_per_head=head_dim,
        seq_size_per_block=page_size,
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
    config.rope_config.dim = head_dim
    config.rope_config.base = 10000
    config.rope_config.max_pos = MAX_KV_LEN
    return config


def _full_impl_reference(
    qkv: torch.Tensor,
    cache_snapshot: torch.Tensor,
    page_table: torch.Tensor,
    kv_lengths: list[int],
    head_dim: int = HEAD_DIM,
    position_offsets: list[int] | None = None,
    rotary_dim: int | None = None,
) -> tuple[torch.Tensor, LayerKVCache]:
    prefix_lengths = [kv_len - QUERY_LEN for kv_len in kv_lengths]
    rope_qkv = apply_base_rope_to_qkv_reference(
        qkv,
        [QUERY_LEN] * len(kv_lengths),
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        head_dim,
        position_offsets=position_offsets or prefix_lengths,
        rotary_dim=rotary_dim,
    )
    q_size = NUM_Q_HEADS * head_dim
    kv_size = NUM_KV_HEADS * head_dim
    query, key, value = torch.split(rope_qkv, [q_size, kv_size, kv_size], dim=-1)
    query = query.reshape(-1, NUM_Q_HEADS, head_dim)
    key = key.reshape(-1, NUM_KV_HEADS, head_dim)
    value = value.reshape(-1, NUM_KV_HEADS, head_dim)
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
    output = compute_paged_prefill_reference(
        query,
        reference_cache,
        page_table,
        kv_lengths,
        [QUERY_LEN] * len(kv_lengths),
    )
    return output, reference_cache


class TestFlashAttn4SpecDecodeDisableGating(unittest.TestCase):
    def test_enable_fa4_spec_decode_gates_the_impl(self) -> None:
        config = FMHAConfig()
        impl_name = FlashAttn4SpecDecodeImpl.__name__

        self.assertTrue(config.enable_fa4_spec_decode)
        self.assertFalse(attn_factory._is_fmha_impl_disabled(impl_name, config))

        config.enable_fa4_spec_decode = False
        self.assertTrue(attn_factory._is_fmha_impl_disabled(impl_name, config))

    def test_flashinfer_switches_do_not_govern_fa4(self) -> None:
        # FlashAttn4SpecDecodeImpl subclasses a FlashInfer prefill base, but its kernel
        # comes from vllm_flash_attention, so the FlashInfer switches must not
        # capture it.
        config = FMHAConfig()
        impl_name = FlashAttn4SpecDecodeImpl.__name__

        config.disable_flashinfer_native = True
        config.disable_flashinfer_hybrid_prefill = True
        self.assertFalse(attn_factory._is_fmha_impl_disabled(impl_name, config))


class TestFlashAttn4SpecDecodeSupportEnvelope(unittest.TestCase):
    def _support(
        self,
        page_size: int = PAGE_SIZE,
        head_dim: int = HEAD_DIM,
        is_target_verify: bool = True,
        is_spec_draft_prefill: bool = False,
    ) -> bool:
        inputs = SimpleNamespace(
            is_target_verify=is_target_verify,
            is_spec_draft_prefill=is_spec_draft_prefill,
        )
        with (
            mock.patch(f"{_MODULE}.is_sm90", return_value=True),
            mock.patch(f"{_MODULE}._load_fa_forward", return_value=object()),
        ):
            return FlashAttn4SpecDecodeImpl.support(
                _make_config(page_size, head_dim), inputs
            )

    def test_head_dim_envelope_matches_upstream_sm90_limit(self) -> None:
        self.assertTrue(self._support(head_dim=256))
        self.assertTrue(self._support(head_dim=272))
        self.assertTrue(self._support(head_dim=512))
        self.assertFalse(self._support(head_dim=520))
        self.assertFalse(self._support(head_dim=4))

    def test_irregular_head_dim_requires_the_tma_page_size(self) -> None:
        # The cp.async paged-KV kernel asserts on head dims that pad to a wider tile.
        self.assertTrue(self._support(page_size=_FA4_TILE_N, head_dim=72))
        self.assertFalse(self._support(page_size=64, head_dim=72))

    def test_page_sizes_beyond_the_tested_shapes_stay_supported(self) -> None:
        # kernel_tokens_per_block follows seq_size_per_block, so 128 is a live
        # deployment value; the cp.async path indexes pages by divmod and does not
        # constrain page size.
        for page_size in (_FA4_TILE_N, 64, 128, 256):
            with self.subTest(page_size=page_size):
                self.assertTrue(self._support(page_size=page_size, head_dim=128))

    def test_only_spec_decode_inputs_are_claimed(self) -> None:
        self.assertTrue(
            self._support(is_target_verify=False, is_spec_draft_prefill=True)
        )
        self.assertFalse(
            self._support(is_target_verify=False, is_spec_draft_prefill=False)
        )


class TestFlashAttn4SpecDecodeContract(unittest.TestCase):
    def test_num_splits_follows_fa2_wave_efficiency(self) -> None:
        # This is the FA2 wave-efficiency heuristic:
        # it over-schedules splits to fill idle SMs.
        # Two M blocks on 132 SMs leaves the GPU almost empty,
        # so all four N tiles get their own split.
        self.assertEqual(
            _get_num_splits(
                sm_count=132,
                batch_size=1,
                query_len=QUERY_LEN,
                num_q_heads=NUM_Q_HEADS,
                num_kv_heads=NUM_KV_HEADS,
                max_kv_len=4 * 32,
            ),
            4,
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
        op = object.__new__(FlashAttn4SpecDecodeOp)
        op.page_size = PAGE_SIZE
        op.max_kv_len = MAX_KV_LEN
        op.head_num = NUM_Q_HEADS
        op.kv_head_num = NUM_KV_HEADS
        op.query_len = QUERY_LEN
        op.fmha_params = mock.Mock()
        op.spec_params = None
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
        self.assertEqual(op.spec_params.batch_size, 2)
        self.assertEqual(args[0].tolist(), [60, 124])
        self.assertEqual(args[1].tolist(), [65, 129])
        self.assertEqual(args[2].tolist(), [QUERY_LEN, QUERY_LEN])
        self.assertEqual(args[3].tolist(), page_table.tolist())
        self.assertFalse(args[5])

    def test_cuda_graph_prepare_refreshes_metadata_in_place(self) -> None:
        op = object.__new__(FlashAttn4SpecDecodeOp)
        op.page_size = PAGE_SIZE
        op.max_kv_len = MAX_KV_LEN
        op.head_num = NUM_Q_HEADS
        op.kv_head_num = NUM_KV_HEADS
        op.query_len = QUERY_LEN
        op.fmha_params = mock.Mock()
        op.spec_params = None
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
            spec_params = op.spec_params
            input_lengths_ptr = spec_params.input_lengths.data_ptr()
            kv_lengths_ptr = spec_params.kv_lengths.data_ptr()
            num_splits = spec_params.num_splits
            prefix_lengths.copy_(torch.tensor([100, 200], dtype=torch.int32))
            op.fmha_params.reset_mock()

            op.prepare_cuda_graph(inputs)

        self.assertIs(op.spec_params, spec_params)
        self.assertEqual(op.spec_params.input_lengths.data_ptr(), input_lengths_ptr)
        self.assertEqual(op.spec_params.kv_lengths.data_ptr(), kv_lengths_ptr)
        self.assertEqual(op.spec_params.num_splits, num_splits)
        self.assertEqual(op.spec_params.kv_lengths.tolist(), [105, 205])
        args = op.fmha_params.fill_params_mha_device.call_args.args
        self.assertIs(args[0], op.spec_params.prefix_lengths)
        self.assertIs(args[1], op.spec_params.kv_lengths)
        self.assertIs(args[2], op.spec_params.input_lengths)
        self.assertIs(args[3], op.spec_params.page_table)
        self.assertTrue(args[5])

    def test_cuda_graph_prepare_rejects_buffer_changes(self) -> None:
        op = object.__new__(FlashAttn4SpecDecodeOp)
        op.page_size = PAGE_SIZE
        op.query_len = QUERY_LEN
        op.fmha_params = mock.Mock()
        prefix_lengths = torch.tensor([60, 124], dtype=torch.int32)
        page_table = torch.tensor([[0, 1, 0], [2, 3, 4]], dtype=torch.int32)
        op.spec_params = FlashAttn4SpecDecodeParams(
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
            changed_buffers = SimpleNamespace(
                total_tokens=2 * QUERY_LEN,
                prefix_lengths_device=prefix_lengths.clone(),
                kv_cache_kernel_block_id_device=page_table.clone(),
            )
            with self.assertRaisesRegex(RuntimeError, "buffers cannot change"):
                op.prepare_cuda_graph(changed_buffers)

    def test_prepare_rejects_token_count_outside_fixed_width_batches(self) -> None:
        op = object.__new__(FlashAttn4SpecDecodeOp)
        op.query_len = QUERY_LEN
        inputs = SimpleNamespace(total_tokens=2 * QUERY_LEN - 1)
        with mock.patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl."
            "py_flash_attn_v4.check_attention_inputs"
        ):
            with self.assertRaisesRegex(ValueError, "divisible by query_len"):
                op.prepare(inputs)


class _Fa4HopperCases:
    """FA4 SM90 numerics for one shape; concrete subclasses bind page_size/head_dim."""

    page_size = PAGE_SIZE
    head_dim = HEAD_DIM

    @classmethod
    def setUpClass(cls) -> None:
        if torch.cuda.get_device_capability()[0] != 9:
            raise unittest.SkipTest("FA4 spec-decode currently targets SM90")
        torch.manual_seed(2026)

    def _make_config(self) -> AttentionConfigs:
        return _make_config(self.page_size, self.head_dim)

    def _make_impl(
        self, attn_inputs, config: AttentionConfigs | None = None
    ) -> FlashAttn4SpecDecodeImpl:
        """Build the impl, then rebind its cos/sin cache to this shape's rope dim.

        getRopeCacheOnce is a std::call_once singleton keyed only on interleave, so
        whichever shape constructs first hands its rope dim to all the others. In
        production one process serves one rope dim; these shapes only share a process
        because they are one test target, so replacing the cache restores the
        isolation each of them would really have. Safe after construction because
        __init__ only stores the tensor.
        """
        config = config or self._make_config()
        impl = FlashAttn4SpecDecodeImpl(config, attn_inputs)
        self.assertEqual(
            ROPE_MAX_POS, config.max_seq_len + config.gen_num_per_cycle + 1
        )
        rope_style = config.rope_config.style
        config.rope_config.style = RopeStyle.Base
        try:
            impl.rope_impl.cos_sin_cache = get_rope_cache(
                config.rope_config, ROPE_MAX_POS, False
            )
        finally:
            config.rope_config.style = rope_style
        return impl

    def _make_qkv(self, batch_size: int) -> torch.Tensor:
        return torch.randn(
            batch_size * QUERY_LEN,
            (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * self.head_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )

    def _make_spec_inputs(
        self,
        kv_lengths: list[int],
        page_table: torch.Tensor,
        is_target_verify: bool,
        is_cuda_graph: bool = False,
    ):
        inputs = self._create_chunked_prefill_attention_inputs(
            input_lengths=[QUERY_LEN] * len(kv_lengths),
            prefix_lengths=[kv_len - QUERY_LEN for kv_len in kv_lengths],
            seq_size_per_block=self.page_size,
            dtype=torch.bfloat16,
            kv_cache_block_id=page_table,
            max_seq_len=MAX_KV_LEN,
            is_cuda_graph=is_cuda_graph,
        )
        inputs.prefix_lengths = inputs.prefix_lengths.to(self.device)
        inputs.is_target_verify = is_target_verify
        return inputs

    def _set_text_mrope_positions(self, inputs, position_offsets: list[int]) -> None:
        positions = torch.cat(
            [
                torch.arange(offset, offset + QUERY_LEN, device=self.device)
                for offset in position_offsets
            ]
        ).to(torch.int32)
        inputs.combo_position_ids = positions.repeat_interleave(3)

    def _make_page_table(self, kv_lengths: list[int]) -> tuple[torch.Tensor, int]:
        page_count = self._calculate_total_blocks(kv_lengths, self.page_size)
        page_table = self._create_kv_cache_block_ids(
            len(kv_lengths),
            kv_lengths,
            self.page_size,
            max_seq_len=MAX_KV_LEN,
        ).to(self.device)
        return page_table, page_count

    def _make_kv_cache(self, page_count: int):
        cache, _, _ = self._create_kv_cache(
            page_count,
            self.page_size,
            NUM_KV_HEADS,
            self.head_dim,
            dtype=torch.bfloat16,
        )
        return cache

    def _reference(self, qkv, cache_snapshot, page_table, kv_lengths):
        return _full_impl_reference(
            qkv, cache_snapshot, page_table, kv_lengths, self.head_dim
        )[0]

    def _assert_eager_impl_matches_reference(
        self, is_target_verify: bool, kv_lengths: list[int]
    ) -> None:
        page_table, page_count = self._make_page_table(kv_lengths)
        inputs = self._make_spec_inputs(kv_lengths, page_table, is_target_verify)
        cache = self._make_kv_cache(page_count)
        qkv = self._make_qkv(len(kv_lengths))
        cache_snapshot = cache.kv_cache_base.clone()
        expected = self._reference(qkv, cache_snapshot, page_table, kv_lengths)
        impl = self._make_impl(inputs)
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

    def test_support_admits_the_shape_under_test(self) -> None:
        inputs = SimpleNamespace(is_target_verify=True, is_spec_draft_prefill=False)
        self.assertTrue(FlashAttn4SpecDecodeImpl.support(self._make_config(), inputs))

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

        inputs = self._make_spec_inputs(
            capture_lengths, page_table, True, is_cuda_graph=True
        )
        cache = self._make_kv_cache(page_count)
        cache_snapshot = cache.kv_cache_base.clone()
        capture_qkv = self._make_qkv(len(capture_lengths))
        static_qkv = capture_qkv.clone()
        impl = self._make_impl(inputs)

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
        replay_inputs = self._make_spec_inputs(
            replay_lengths, replay_page_table, True, is_cuda_graph=True
        )
        self._copy_replay_metadata(inputs, replay_inputs)
        impl.prepare_cuda_graph(inputs)
        graph.replay()
        torch.cuda.synchronize()

        expected = self._reference(
            replay_qkv, cache_snapshot, page_table, replay_lengths
        )
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_cuda_graph_replay_is_bitwise_equal_to_eager_impl(self) -> None:
        capture_lengths = [193, 257]
        replay_lengths = [130, 319]
        capture_page_table, page_count = self._make_page_table(capture_lengths)
        replay_page_table, replay_page_count = self._make_page_table(replay_lengths)
        self.assertLessEqual(replay_page_count, page_count)

        capture_inputs = self._make_spec_inputs(
            capture_lengths, capture_page_table, True, is_cuda_graph=True
        )
        replay_inputs = self._make_spec_inputs(
            replay_lengths, replay_page_table, True, is_cuda_graph=True
        )
        graph_cache = self._make_kv_cache(page_count)
        cache_snapshot = graph_cache.kv_cache_base.clone()
        eager_cache = self._make_kv_cache(page_count)
        eager_cache.kv_cache_base.copy_(cache_snapshot)

        capture_qkv = self._make_qkv(len(capture_lengths))
        replay_qkv = torch.randn_like(capture_qkv)
        eager_impl = self._make_impl(replay_inputs)
        expected = eager_impl.forward(replay_qkv.clone(), eager_cache).clone()

        static_qkv = capture_qkv.clone()
        graph_impl = self._make_impl(capture_inputs)
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


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestFlashAttn4SpecDecodeHopperPage64HeadDim256(
    _Fa4HopperCases, BaseAttentionTest
):
    page_size = 64
    head_dim = 256

    def test_mrope_target_verify_uses_logical_positions(self) -> None:
        kv_lengths = [193, 257]
        position_offsets = [17, 29]
        page_table, page_count = self._make_page_table(kv_lengths)
        inputs = self._make_spec_inputs(kv_lengths, page_table, True)
        self._set_text_mrope_positions(inputs, position_offsets)
        config = self._make_config()
        config.rope_config.style = RopeStyle.Mrope
        config.rope_config.dim = 64
        config.rope_config.index_factor = 3
        cache = self._make_kv_cache(page_count)
        cache_snapshot = cache.kv_cache_base.clone()
        qkv = self._make_qkv(len(kv_lengths))
        self.assertTrue(FlashAttn4SpecDecodeImpl.support(config, inputs))
        expected, expected_cache = _full_impl_reference(
            qkv,
            cache_snapshot,
            page_table,
            kv_lengths,
            self.head_dim,
            position_offsets,
            config.rope_config.dim,
        )

        actual = self._make_impl(inputs, config).forward(qkv.clone(), cache)

        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(
            cache.kv_cache_base,
            expected_cache.kv_cache_base,
            rtol=2e-2,
            atol=2e-2,
        )

    def test_mrope_cuda_graph_replay_refreshes_logical_positions(self) -> None:
        kv_lengths = [193, 257]
        capture_offsets = [17, 29]
        replay_offsets = [47, 59]
        page_table, page_count = self._make_page_table(kv_lengths)
        inputs = self._make_spec_inputs(
            kv_lengths, page_table, True, is_cuda_graph=True
        )
        self._set_text_mrope_positions(inputs, capture_offsets)
        config = self._make_config()
        config.rope_config.style = RopeStyle.Mrope
        config.rope_config.dim = 64
        config.rope_config.index_factor = 3
        cache = self._make_kv_cache(page_count)
        cache_snapshot = cache.kv_cache_base.clone()
        capture_qkv = self._make_qkv(len(kv_lengths))
        static_qkv = capture_qkv.clone()
        impl = self._make_impl(inputs, config)

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
        self._set_text_mrope_positions(inputs, replay_offsets)
        impl.prepare_cuda_graph(inputs)
        graph.replay()
        torch.cuda.synchronize()

        expected, expected_cache = _full_impl_reference(
            replay_qkv,
            cache_snapshot,
            page_table,
            kv_lengths,
            self.head_dim,
            replay_offsets,
            config.rope_config.dim,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(
            cache.kv_cache_base,
            expected_cache.kv_cache_base,
            rtol=2e-2,
            atol=2e-2,
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestFlashAttn4SpecDecodeHopperPage64HeadDim128(
    _Fa4HopperCases, BaseAttentionTest
):
    page_size = 64
    head_dim = 128


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestFlashAttn4SpecDecodeHopperPage64HeadDim512(
    _Fa4HopperCases, BaseAttentionTest
):
    page_size = 64
    head_dim = 512


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestFlashAttn4SpecDecodeHopperTmaHeadDim64(_Fa4HopperCases, BaseAttentionTest):
    # page_size == tile N selects the TMA paged-KV kernel instead of cp.async.
    page_size = _FA4_TILE_N
    head_dim = 64


if __name__ == "__main__":
    unittest.main()
