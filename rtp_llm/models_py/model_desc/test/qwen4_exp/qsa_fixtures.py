from types import SimpleNamespace

import torch

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    INDEXER_KV_TAG,
    INDEXER_STATE_TAG,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_context import (
    Qwen4ExpQSARuntimeContext,
)
from rtp_llm.models_py.modules.base.common.zero_centered_rms_norm import (
    exact_head_rms_norm,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import apply_partial_rope
from rtp_llm.ops import RopeStyle


class QSARuntimeFixture:
    D = 4
    RATIO = 4
    KV_TOKENS_PER_BLOCK = 8
    STATE_TOKENS_PER_BLOCK = 8

    def setUp(self):
        torch.manual_seed(17)
        self.lengths = [5, 8]
        self.total_tokens = sum(self.lengths)
        positions = []
        for seq_len in self.lengths:
            positions.extend([[pos, pos, pos] for pos in range(seq_len)])
        self.position_ids = torch.tensor(positions, dtype=torch.int32).reshape(-1)
        self.cu_seqlens = torch.tensor([0, 5, 13], dtype=torch.int32)
        self.input_lengths = torch.tensor(self.lengths, dtype=torch.int32)
        self.prefix_lengths = torch.zeros(2, dtype=torch.int32)
        self.sequence_lengths = torch.zeros(2, dtype=torch.int32)
        self.kv_base = torch.zeros(7, 2 * self.D * 2, dtype=torch.uint8)
        self.state_base = torch.zeros(8, 2 * self.RATIO * self.D, dtype=torch.float32)
        self.kv_table = torch.tensor([[3], [1]], dtype=torch.int32)
        self.state_table = torch.tensor([[6], [2]], dtype=torch.int32)
        self.rope_config = SimpleNamespace(
            style=RopeStyle.Mrope,
            index_factor=3,
            dim=4,
            mrope_dim1=1,
            mrope_dim2=1,
            mrope_dim3=0,
            mrope_interleaved=True,
            base=10_000,
        )
        self.indexer = SimpleNamespace(
            head_dim=self.D,
            compress_ratio=self.RATIO,
            k_norm_gamma=torch.rand(self.D),
            norm_eps=1e-6,
        )

    def _inputs(self, *, kv_table=None, state_table=None, **overrides):
        values = dict(
            is_prefill=True,
            is_target_verify=False,
            is_cuda_graph=False,
            is_s_padded=False,
            context_parallel_info=None,
            cache_store_inputs=None,
            input_lengths=self.input_lengths,
            prefix_lengths=self.prefix_lengths,
            sequence_lengths=self.sequence_lengths,
            cu_seqlens_device=self.cu_seqlens,
            cu_kv_seqlens_device=self.cu_seqlens,
            combo_position_ids=self.position_ids,
            kv_cache_kernel_block_id_device=kv_table,
            kv_cache_block_id_device=state_table,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def _context(self, *, kv_base=None, state_base=None, **state_overrides):
        main_inputs = self._inputs()
        kv_inputs = self._inputs(kv_table=self.kv_table)
        state_inputs = self._inputs(state_table=self.state_table, **state_overrides)
        return Qwen4ExpQSARuntimeContext(
            main_cache=SimpleNamespace(tag="full"),
            main_inputs=main_inputs,
            indexer_kv_cache=SimpleNamespace(
                tag=INDEXER_KV_TAG,
                kv_cache_base=self.kv_base if kv_base is None else kv_base,
                seq_size_per_block=self.KV_TOKENS_PER_BLOCK,
            ),
            indexer_kv_inputs=kv_inputs,
            indexer_state_cache=SimpleNamespace(
                tag=INDEXER_STATE_TAG,
                kv_cache_base=self.state_base if state_base is None else state_base,
                seq_size_per_block=self.STATE_TOKENS_PER_BLOCK,
            ),
            indexer_state_inputs=state_inputs,
        )

    def _expected(self, raw_keys, cos, sin, start):
        pooled = raw_keys.float().mean(dim=0).to(raw_keys.dtype)
        pooled = exact_head_rms_norm(
            pooled, self.indexer.k_norm_gamma, self.indexer.norm_eps
        )
        return apply_partial_rope(pooled, cos, sin).to(torch.bfloat16)

    def _single_request_context(
        self,
        kv_base,
        state_base,
        *,
        is_prefill,
        input_length,
        sequence_length,
        position_ids,
        target_verify=False,
        prefix_length=0,
    ):
        device = kv_base.device
        input_lengths = torch.tensor([input_length], dtype=torch.int32, device=device)
        sequence_lengths = torch.tensor(
            [sequence_length], dtype=torch.int32, device=device
        )
        prefix_lengths = torch.tensor([prefix_length], dtype=torch.int32, device=device)
        cu_seqlens = torch.tensor(
            [0, input_length if is_prefill else 1],
            dtype=torch.int32,
            device=device,
        )
        position_ids = torch.as_tensor(
            position_ids, dtype=torch.int32, device=device
        ).reshape(-1)
        kv_table = torch.tensor([[3]], dtype=torch.int32, device=device)
        state_table = torch.tensor([[6]], dtype=torch.int32, device=device)

        def _inputs(*, kv_table=None, state_table=None):
            return SimpleNamespace(
                is_prefill=is_prefill,
                is_target_verify=target_verify,
                is_cuda_graph=False,
                is_s_padded=False,
                context_parallel_info=None,
                cache_store_inputs=None,
                input_lengths=input_lengths,
                prefix_lengths=prefix_lengths,
                sequence_lengths=sequence_lengths,
                cu_seqlens_device=cu_seqlens,
                cu_kv_seqlens_device=cu_seqlens,
                combo_position_ids=position_ids,
                kv_cache_kernel_block_id_device=kv_table,
                kv_cache_block_id_device=state_table,
            )

        return Qwen4ExpQSARuntimeContext(
            main_cache=SimpleNamespace(tag="full"),
            main_inputs=_inputs(),
            indexer_kv_cache=SimpleNamespace(
                tag=INDEXER_KV_TAG,
                kv_cache_base=kv_base,
                seq_size_per_block=self.KV_TOKENS_PER_BLOCK,
            ),
            indexer_kv_inputs=_inputs(kv_table=kv_table),
            indexer_state_cache=SimpleNamespace(
                tag=INDEXER_STATE_TAG,
                kv_cache_base=state_base,
                seq_size_per_block=self.STATE_TOKENS_PER_BLOCK,
            ),
            indexer_state_inputs=_inputs(state_table=state_table),
        )

    def _decode_indexer(self, heads=2):
        return SimpleNamespace(
            head_dim=self.D,
            n_heads=heads,
            compress_ratio=self.RATIO,
            token_budget=8,
            k_norm_gamma=self.indexer.k_norm_gamma,
            norm_eps=self.indexer.norm_eps,
        )

    @staticmethod
    def _base_rope_config(
        index_factor, *, style=RopeStyle.Base, indexer_is_neox_style=True
    ):
        return SimpleNamespace(
            style=style,
            index_factor=index_factor,
            dim=4,
            base=10_000,
            scale=1.0,
            indexer_is_neox_style=indexer_is_neox_style,
        )

    @staticmethod
    def _set_position_ids(context, position_ids):
        position_ids = torch.as_tensor(
            position_ids,
            dtype=torch.int32,
            device=context.main_inputs.input_lengths.device,
        ).reshape(-1)
        for inputs in (
            context.main_inputs,
            context.indexer_kv_inputs,
            context.indexer_state_inputs,
        ):
            inputs.combo_position_ids = position_ids

    def _target_verify_context(self, *, q_len=4, committed_length=7):
        """Mirror the target-verify ABI produced by MtpExecutor.

        Target verification is represented as a context-style invocation:
        ``input_lengths=[gamma + 1]``, ``prefix_lengths=[committed]`` and an
        empty ``sequence_lengths`` tensor.
        """
        device = self.kv_base.device
        input_lengths = torch.tensor([q_len], dtype=torch.int32, device=device)
        prefix_lengths = torch.tensor(
            [committed_length], dtype=torch.int32, device=device
        )
        sequence_lengths = torch.empty(0, dtype=torch.int32, device=device)
        cu_seqlens = torch.tensor([0, q_len], dtype=torch.int32, device=device)
        cu_kv_seqlens = torch.tensor(
            [0, committed_length + q_len], dtype=torch.int32, device=device
        )
        position_ids = torch.arange(
            committed_length,
            committed_length + q_len,
            dtype=torch.int32,
            device=device,
        )
        position_ids = position_ids.unsqueeze(1).expand(-1, 3).reshape(-1)
        kv_table = torch.tensor([[3, 4]], dtype=torch.int32, device=device)
        state_table = torch.tensor([[6, 7]], dtype=torch.int32, device=device)

        def _inputs(*, kv_table=None, state_table=None):
            return SimpleNamespace(
                is_prefill=True,
                is_target_verify=True,
                is_cuda_graph=False,
                is_s_padded=False,
                context_parallel_info=None,
                cache_store_inputs=None,
                input_lengths=input_lengths,
                prefix_lengths=prefix_lengths,
                sequence_lengths=sequence_lengths,
                cu_seqlens_device=cu_seqlens,
                cu_kv_seqlens_device=cu_kv_seqlens,
                combo_position_ids=position_ids,
                kv_cache_kernel_block_id_device=kv_table,
                kv_cache_block_id_device=state_table,
            )

        return Qwen4ExpQSARuntimeContext(
            main_cache=SimpleNamespace(tag="full"),
            main_inputs=_inputs(),
            indexer_kv_cache=SimpleNamespace(
                tag=INDEXER_KV_TAG,
                kv_cache_base=self.kv_base,
                seq_size_per_block=self.KV_TOKENS_PER_BLOCK,
            ),
            indexer_kv_inputs=_inputs(kv_table=kv_table),
            indexer_state_cache=SimpleNamespace(
                tag=INDEXER_STATE_TAG,
                kv_cache_base=self.state_base,
                seq_size_per_block=self.STATE_TOKENS_PER_BLOCK,
            ),
            indexer_state_inputs=_inputs(state_table=state_table),
        )

    def _draft_incremental_context(
        self,
        kv_base,
        state_base,
        *,
        prefixes,
        lengths,
        kv_table=None,
        state_table=None,
        position_factor=3,
        is_mtp_draft=True,
        position_overrides=None,
    ):
        device = kv_base.device
        batch_size = len(lengths)
        input_lengths = torch.tensor(lengths, dtype=torch.int32, device=device)
        prefix_lengths = torch.tensor(prefixes, dtype=torch.int32, device=device)
        sequence_lengths = torch.empty(0, dtype=torch.int32, device=device)
        cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        cu_seqlens[1:] = input_lengths.cumsum(0)
        cu_kv_seqlens = torch.zeros_like(cu_seqlens)
        cu_kv_seqlens[1:] = (prefix_lengths + input_lengths).cumsum(0)
        positions = []
        for prefix, length in zip(prefixes, lengths):
            positions.extend(range(prefix, prefix + length))
        if position_overrides is not None:
            positions = position_overrides
        position_ids = torch.tensor(positions, dtype=torch.int32, device=device)
        if position_factor == 3:
            position_ids = position_ids.unsqueeze(1).expand(-1, 3)
        position_ids = position_ids.reshape(-1)
        if kv_table is None:
            kv_table = torch.arange(
                1,
                1 + batch_size * 3,
                dtype=torch.int32,
                device=device,
            ).view(batch_size, 3)
        if state_table is None:
            state_table = torch.arange(
                1 + batch_size * 3,
                1 + batch_size * 6,
                dtype=torch.int32,
                device=device,
            ).view(batch_size, 3)

        def _inputs(*, kv_table=None, state_table=None):
            return SimpleNamespace(
                is_prefill=True,
                is_target_verify=False,
                is_cuda_graph=False,
                is_s_padded=False,
                context_parallel_info=None,
                cache_store_inputs=None,
                input_lengths=input_lengths,
                prefix_lengths=prefix_lengths,
                sequence_lengths=sequence_lengths,
                cu_seqlens_device=cu_seqlens,
                cu_kv_seqlens_device=cu_kv_seqlens,
                combo_position_ids=position_ids,
                kv_cache_kernel_block_id_device=kv_table,
                kv_cache_block_id_device=state_table,
            )

        return Qwen4ExpQSARuntimeContext(
            main_cache=SimpleNamespace(tag="full"),
            main_inputs=_inputs(),
            indexer_kv_cache=SimpleNamespace(
                tag=INDEXER_KV_TAG,
                kv_cache_base=kv_base,
                seq_size_per_block=self.KV_TOKENS_PER_BLOCK,
            ),
            indexer_kv_inputs=_inputs(kv_table=kv_table),
            indexer_state_cache=SimpleNamespace(
                tag=INDEXER_STATE_TAG,
                kv_cache_base=state_base,
                seq_size_per_block=self.STATE_TOKENS_PER_BLOCK,
            ),
            indexer_state_inputs=_inputs(state_table=state_table),
            is_mtp_draft=is_mtp_draft,
        )

    def _check_draft_graph_replay(self, lanes):
        device = torch.device("cuda")
        self.D = self.KV_TOKENS_PER_BLOCK = self.STATE_TOKENS_PER_BLOCK = 128
        self.indexer.k_norm_gamma = torch.randn(128, device=device).bfloat16()
        indexer = self._decode_indexer()
        rope = self._base_rope_config(3)
        rope.dim = 64
        kv_initial = torch.randn(
            12, 32 * 128, device=device, dtype=torch.bfloat16
        ).view(torch.uint8)
        state_initial = torch.randn(12, 8 * 128, device=device)
        kv = kv_initial.clone()
        state = state_initial.clone()
        kv_table = torch.tensor([[1, 2], [3, 4]], device=device, dtype=torch.int32)
        state_table = torch.tensor([[5, 6], [7, 8]], device=device, dtype=torch.int32)
        ctx = self._draft_incremental_context(
            kv,
            state,
            prefixes=[5, 127],
            lengths=[lanes, lanes],
            kv_table=kv_table,
            state_table=state_table,
        )
        for inputs in (
            ctx.main_inputs,
            ctx.indexer_kv_inputs,
            ctx.indexer_state_inputs,
        ):
            inputs.is_cuda_graph = True
            inputs.is_s_padded = True
            inputs.input_lengths_device = inputs.input_lengths
            inputs.prefix_lengths_device = inputs.prefix_lengths
        q = torch.randn(2 * lanes, 2, 128, device=device, dtype=torch.bfloat16)
        raw = torch.randn(2 * lanes, 128, device=device, dtype=torch.bfloat16)

        def forward():
            result = ctx.select_draft_graph_tokens(
                q, raw, indexer=indexer, rope_config=rope
            )
            ctx.finalize_side_cache()
            return result

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                forward()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            selected = forward()
        for prefixes, lengths, swap in (
            ([7, 127], [1, lanes], False),
            ([125, 5], [min(3, lanes), min(2, lanes)], True),
            ([127, 127], [lanes, lanes], False),
        ):
            with self.subTest(prefixes=prefixes, lengths=lengths):
                kv.copy_(kv_initial)
                state.copy_(state_initial)
                q.normal_()
                raw.normal_()
                ctx.main_inputs.input_lengths.copy_(
                    torch.tensor(lengths, device=device, dtype=torch.int32)
                )
                ctx.main_inputs.prefix_lengths.copy_(
                    torch.tensor(prefixes, device=device, dtype=torch.int32)
                )
                ctx.main_inputs.cu_seqlens_device.copy_(
                    torch.tensor(
                        [0, lengths[0], sum(lengths)], device=device, dtype=torch.int32
                    )
                )
                kv_table.copy_(
                    torch.tensor(
                        [[3, 4], [1, 2]] if swap else [[1, 2], [3, 4]],
                        device=device,
                        dtype=torch.int32,
                    )
                )
                state_table.copy_(
                    torch.tensor(
                        [[7, 8], [5, 6]] if swap else [[5, 6], [7, 8]],
                        device=device,
                        dtype=torch.int32,
                    )
                )
                ref_kv, ref_state = kv_initial.clone(), state_initial.clone()
                ref = self._draft_incremental_context(
                    ref_kv,
                    ref_state,
                    prefixes=prefixes,
                    lengths=lengths,
                    kv_table=kv_table,
                    state_table=state_table,
                )
                expected = ref.select_draft_incremental_prefill_tokens(
                    q[: sum(lengths)],
                    raw[: sum(lengths)],
                    indexer=indexer,
                    rope_config=rope,
                )
                ref.finalize_side_cache()
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    selected[: sum(lengths)], expected, rtol=0, atol=0
                )
                self.assertTrue(torch.all(selected[sum(lengths) :] == -1).item())
                torch.testing.assert_close(state, ref_state, rtol=0, atol=0)
                torch.testing.assert_close(kv, ref_kv, rtol=0, atol=0)
