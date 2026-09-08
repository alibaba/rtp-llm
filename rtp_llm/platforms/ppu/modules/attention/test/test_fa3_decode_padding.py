import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from rtp_llm.platforms.ppu.modules.attention.fa3_mha import (
    FA3DecodeAttnOp,
    FA3DecodeImpl,
    FA3PrefillPagedAttnOp,
    FA3PrefillPagedImpl,
)


class TestFA3DecodeGraphPadding(unittest.TestCase):
    def test_padded_cache_lengths_use_one_token_dummy_sequences(self):
        op = object.__new__(FA3DecodeAttnOp)
        op._graph_cache_seqlens = torch.empty(4, dtype=torch.int32)
        attn_inputs = SimpleNamespace(
            sequence_lengths=torch.tensor([7, 9, 0, 0], dtype=torch.int32)
        )

        op._refresh_graph_cache_seqlens(attn_inputs)

        torch.testing.assert_close(
            op._graph_cache_seqlens,
            torch.tensor([8, 10, 1, 1], dtype=torch.int32),
        )

    def test_scheduler_metadata_is_refreshed_in_place(self):
        op = object.__new__(FA3DecodeAttnOp)
        op._graph_scheduler_metadata = torch.tensor([1, 2, 3], dtype=torch.int32)
        original_data_ptr = op._graph_scheduler_metadata.data_ptr()
        op._get_scheduler_metadata = lambda *args: torch.tensor(
            [4, 5, 6], dtype=torch.int32
        )

        metadata = op._refresh_graph_scheduler_metadata(
            batch_size=4,
            max_seqlen_k=64,
            cache_seqlens=torch.tensor([8, 10, 1, 1], dtype=torch.int32),
            cu_seqlens_q=torch.arange(5, dtype=torch.int32),
        )

        self.assertEqual(metadata.data_ptr(), original_data_ptr)
        torch.testing.assert_close(metadata, torch.tensor([4, 5, 6], dtype=torch.int32))


class TestFA3GraphReplayLeavesBlockTableAlone(unittest.TestCase):
    """The host block-table padding contract belongs to CudaGraphRunner.

    prepareInputs() clears both the device and the host mirror before copying the
    real rows, so padded rows already resolve to the reserved block 0. The FA3
    backends must not second-guess which rows are padding: MTP target verify
    supplies an empty ``sequence_lengths`` upstream, so any predicate derived from
    it would misclassify real rows.
    """

    @staticmethod
    def _runner_prepared_inputs():
        # What prepareInputs() hands to prepare_cuda_graph(): rows 0-1 real,
        # rows 2-3 padding already zeroed by the runner. sequence_lengths carries
        # the target-verify shape, where no real value is ever written.
        return SimpleNamespace(
            sequence_lengths=torch.tensor([0, 0, 0, 0], dtype=torch.int32),
            kv_cache_kernel_block_id=torch.tensor(
                [[11, 12], [21, 22], [0, 0], [0, 0]], dtype=torch.int32
            ),
        )

    def test_target_verify_replay_preserves_real_rows(self):
        attn_inputs = self._runner_prepared_inputs()
        expected = attn_inputs.kv_cache_kernel_block_id.clone()

        class FakePrefillAttnOp:
            def __init__(self):
                self.block_table_seen_by_fill_params = None

            def prepare_for_cuda_graph_replay(self, inputs):
                self.block_table_seen_by_fill_params = (
                    inputs.kv_cache_kernel_block_id.clone()
                )

        impl = object.__new__(FA3PrefillPagedImpl)
        impl.fmha_impl = FakePrefillAttnOp()

        impl.prepare_cuda_graph(attn_inputs)

        torch.testing.assert_close(
            impl.fmha_impl.block_table_seen_by_fill_params, expected
        )
        torch.testing.assert_close(attn_inputs.kv_cache_kernel_block_id, expected)

    def test_mrope_decode_replay_preserves_real_rows(self):
        attn_inputs = self._runner_prepared_inputs()
        attn_inputs.prefix_lengths = torch.empty(0, dtype=torch.int32)
        attn_inputs.input_lengths = torch.ones(4, dtype=torch.int32)
        expected = attn_inputs.kv_cache_kernel_block_id.clone()

        class FakeDecodeAttnOp:
            def prepare_for_cuda_graph_replay(self, inputs):
                pass

        class FakeRopeParams:
            def __init__(self):
                self.block_table_seen_by_fill_params = None

            def fill_params(self, prefix, seq, inp, block_table, block, **kwargs):
                self.block_table_seen_by_fill_params = block_table.clone()

        impl = object.__new__(FA3DecodeImpl)
        impl.fmha_impl = FakeDecodeAttnOp()
        impl.use_ppu_mrope = True
        impl.rope_params = FakeRopeParams()
        impl.attn_configs = SimpleNamespace(kernel_tokens_per_block=64)

        impl.prepare_cuda_graph(attn_inputs)

        torch.testing.assert_close(
            impl.rope_params.block_table_seen_by_fill_params, expected
        )
        torch.testing.assert_close(attn_inputs.kv_cache_kernel_block_id, expected)


class TestRunnerHostBlockTableContract(unittest.TestCase):
    """Regression guard for the runner-side contract FA3 depends on.

    Mirrors CudaGraphRunner::prepareInputs host bookkeeping. Every graph instance
    slices one shared capture buffer (prepareCaptureInputs), so a replay must not
    be able to observe rows written by a replay of a different graph size.
    """

    MAX_BS = 32
    MAX_BLOCKS = 4
    CAPTURE = (1, 8, 16, 32)

    def _graph_bs(self, real_bs):
        return next(bs for bs in self.CAPTURE if bs >= real_bs)

    def test_replays_never_expose_stale_or_real_rows_as_padding(self):
        shared_block_table = torch.zeros(
            (self.MAX_BS, self.MAX_BLOCKS), dtype=torch.int32
        )

        # Descending then ascending real batches over shared graph buffers is the
        # pattern that used to leave a previous replay's block ids behind.
        for real_bs in (20, 10, 14, 25, 12, 30, 9, 16, 6, 31):
            graph_bs = self._graph_bs(real_bs)
            real_rows = torch.arange(
                1, real_bs * self.MAX_BLOCKS + 1, dtype=torch.int32
            ).view(real_bs, self.MAX_BLOCKS)

            # prepareInputs(): clear the whole host mirror, then copy real rows.
            shared_block_table.fill_(0)
            shared_block_table[:real_bs] = real_rows

            view = shared_block_table[:graph_bs]
            torch.testing.assert_close(
                view[:real_bs],
                real_rows,
                msg=f"real rows corrupted at real_bs={real_bs}",
            )
            torch.testing.assert_close(
                view[real_bs:],
                torch.zeros((graph_bs - real_bs, self.MAX_BLOCKS), dtype=torch.int32),
                msg=f"padding rows not on block 0 at real_bs={real_bs}",
            )


class TestTargetVerifyUniformInputLengths(unittest.TestCase):
    """Target verify must reach fill_params with the capture-time uniform N.

    CudaGraphRunner::prepareInputs zeroes the padded input_lengths rows once
    num_tokens_per_bs_ > 1. fillParamsInternal iterates ``for j < input_length``, so
    a zeroed row emits neither a per-token entry nor a page_indice entry, leaving the
    persistent ragged buffers short of the length the captured kernels replay.
    """

    N = 5

    def _op(self, enable_cuda_graph=True):
        op = object.__new__(FA3PrefillPagedAttnOp)
        op.enable_cuda_graph = enable_cuda_graph
        return op

    def _verify_inputs(self, real_bs=3, graph_bs=8):
        lengths = torch.zeros(graph_bs, dtype=torch.int32)
        lengths[:real_bs] = self.N
        return SimpleNamespace(
            input_lengths=lengths,
            is_target_verify=True,
            prefill_cuda_graph_copy_params=None,
        )

    def test_padded_rows_are_restored_to_uniform_n(self):
        attn_inputs = self._verify_inputs()

        self._op()._restore_uniform_verify_input_lengths(attn_inputs)

        torch.testing.assert_close(
            attn_inputs.input_lengths,
            torch.full((8,), self.N, dtype=torch.int32),
        )

    def test_draft_prefill_keeps_inactive_rows_at_zero(self):
        # Draft prefill is driven by prefill_cuda_graph_copy_params and its inactive
        # slots are legitimately 0; restoring N there would invent query tokens.
        attn_inputs = self._verify_inputs()
        attn_inputs.prefill_cuda_graph_copy_params = object()
        expected = attn_inputs.input_lengths.clone()

        self._op()._restore_uniform_verify_input_lengths(attn_inputs)

        torch.testing.assert_close(attn_inputs.input_lengths, expected)

    def test_eager_verify_is_untouched(self):
        attn_inputs = self._verify_inputs()
        expected = attn_inputs.input_lengths.clone()

        self._op(enable_cuda_graph=False)._restore_uniform_verify_input_lengths(
            attn_inputs
        )

        torch.testing.assert_close(attn_inputs.input_lengths, expected)

    def test_plain_decode_is_untouched(self):
        attn_inputs = self._verify_inputs()
        attn_inputs.is_target_verify = False
        expected = attn_inputs.input_lengths.clone()

        self._op()._restore_uniform_verify_input_lengths(attn_inputs)

        torch.testing.assert_close(attn_inputs.input_lengths, expected)

    def test_prepare_hands_fill_params_no_zero_rows(self):
        # Guards the call site, not just the helper: fill_params is what turns
        # input_lengths into the ragged batch_indice/positions/page_indice buffers.
        graph_bs, real_bs = 8, 3
        attn_inputs = self._verify_inputs(real_bs=real_bs, graph_bs=graph_bs)
        attn_inputs.prefix_lengths = torch.zeros(graph_bs, dtype=torch.int32)
        attn_inputs.sequence_lengths = torch.empty(0, dtype=torch.int32)
        attn_inputs.kv_cache_kernel_block_id = torch.zeros(
            (graph_bs, 2), dtype=torch.int32
        )
        attn_inputs.kv_cache_kernel_block_id_device = torch.zeros(
            (graph_bs, 2), dtype=torch.int32
        )
        attn_inputs.cu_seqlens_device = torch.arange(
            0, (graph_bs + 1) * self.N, self.N, dtype=torch.int32
        )
        attn_inputs.dtype = torch.bfloat16

        class FakeFmhaParams:
            def __init__(self):
                self.input_lengths_seen = None
                self.kvlen_d = torch.full((graph_bs,), 1, dtype=torch.int32)

            def fill_params(self, prefix, seq, inp, block_table, block, realloc):
                self.input_lengths_seen = inp.clone()

        op = object.__new__(FA3PrefillPagedAttnOp)
        op.enable_cuda_graph = True
        op.fmha_params = FakeFmhaParams()
        op.seq_size_per_block = 64
        op.max_seq_len = 4096
        op._graph_max_seqlen_k = None
        op._graph_scheduler_metadata = None
        op._get_scheduler_metadata = lambda *args: torch.zeros(2, dtype=torch.int32)

        # prepare() pulls get_scalar_type from the engine ops module; stub it so this
        # stays a logic-only test instead of depending on the compiled extension.
        compute_ops = types.ModuleType("rtp_llm.ops.compute_ops")
        compute_ops.get_scalar_type = lambda dtype: dtype
        ops = types.ModuleType("rtp_llm.ops")
        ops.compute_ops = compute_ops
        with mock.patch.dict(
            sys.modules,
            {"rtp_llm.ops": ops, "rtp_llm.ops.compute_ops": compute_ops},
        ):
            op.prepare(attn_inputs, forbid_realloc=True)

        torch.testing.assert_close(
            op.fmha_params.input_lengths_seen,
            torch.full((graph_bs,), self.N, dtype=torch.int32),
        )


class TestRaggedParamsBufferContract(unittest.TestCase):
    """Why uniform N matters, and why the runner must also zero the block table.

    Mirrors FlashInferMlaAttnParams::fillParamsInternal. The buffers are persistent
    (fill_params(forbid_realloc=True)) and refreshBuffer only resizes tensor views,
    which is a no-op for an already captured kernel: the graph keeps replaying the
    capture-time token and page counts. Anything the current replay does not rewrite
    is read back from the previous replay.
    """

    N = 5
    GRAPH_BS = 16
    BLOCK = 2048
    MAX_SEQ_LEN = 32768
    RESERVED_BLOCK = 0

    def _capture_token_count(self):
        return self.GRAPH_BS * self.N

    def _fill_params_internal(self, input_lengths, prefix_lengths, block_table):
        batch_indice, positions, page_indice = [], [], []
        for i in range(self.GRAPH_BS):
            input_length = int(input_lengths[i])
            prefix_length = int(prefix_lengths[i])
            for j in range(input_length):
                batch_indice.append(i)
                positions.append(j + prefix_length)
            seq_len = input_length + prefix_length
            for j in range((seq_len + self.BLOCK - 1) // self.BLOCK):
                page_indice.append(int(block_table[i][j]))
        return batch_indice, positions, page_indice

    def _replay_inputs(self, real_bs, zero_padded_input_lengths, zero_block_table):
        blocks = self.MAX_SEQ_LEN // self.BLOCK + 2
        input_lengths = torch.full((self.GRAPH_BS,), self.N, dtype=torch.int32)
        prefix_lengths = torch.zeros(self.GRAPH_BS, dtype=torch.int32)
        prefix_lengths[:real_bs] = 4096
        # A previous, larger replay left its real block ids behind.
        block_table = torch.arange(
            1, self.GRAPH_BS * blocks + 1, dtype=torch.int32
        ).view(self.GRAPH_BS, blocks)
        if zero_block_table:
            block_table.fill_(0)
            block_table[:real_bs] = torch.arange(
                1, real_bs * blocks + 1, dtype=torch.int32
            ).view(real_bs, blocks)
        if zero_padded_input_lengths:
            input_lengths[real_bs:] = 0
        return input_lengths, prefix_lengths, block_table, real_bs

    def test_uniform_n_rewrites_every_captured_token_slot(self):
        for real_bs in (14, 10, 12, 6, 10):
            args = self._replay_inputs(
                real_bs, zero_padded_input_lengths=False, zero_block_table=True
            )
            batch_indice, _, page_indice = self._fill_params_internal(*args[:3])
            self.assertEqual(
                len(batch_indice),
                self._capture_token_count(),
                msg=f"ragged token buffer left a stale tail at real_bs={real_bs}",
            )
            padded_pages = page_indice[
                sum(
                    (int(args[0][i]) + int(args[1][i]) + self.BLOCK - 1) // self.BLOCK
                    for i in range(real_bs)
                ) :
            ]
            self.assertTrue(
                all(page == self.RESERVED_BLOCK for page in padded_pages),
                msg=f"padded pages left real blocks at real_bs={real_bs}",
            )

    def test_zeroed_padding_shortens_the_ragged_buffers(self):
        # Documents the defect the restore closes: the captured kernels still walk
        # graph_bs * N tokens, so the untouched tail is the previous replay's data.
        args = self._replay_inputs(
            10, zero_padded_input_lengths=True, zero_block_table=True
        )
        batch_indice, _, _ = self._fill_params_internal(*args[:3])

        self.assertLess(len(batch_indice), self._capture_token_count())

    def test_uniform_n_alone_is_not_enough_without_a_zeroed_block_table(self):
        # Guards the runner-side half of the contract: with the host block table
        # left dirty, restoring N routes padded tokens onto real blocks instead.
        args = self._replay_inputs(
            10, zero_padded_input_lengths=False, zero_block_table=False
        )
        _, _, page_indice = self._fill_params_internal(*args[:3])

        self.assertTrue(any(page != self.RESERVED_BLOCK for page in page_indice[-3:]))


if __name__ == "__main__":
    unittest.main()
