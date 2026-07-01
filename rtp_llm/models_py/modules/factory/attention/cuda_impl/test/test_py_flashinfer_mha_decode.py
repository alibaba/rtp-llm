import logging
import math
import sys
import unittest
from typing import List

import torch
from attention_ref import compute_flashinfer_decode_reference
from base_attention_test import BaseAttentionTest, compare_tensors

from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferDecodeAttnOp,
)
from rtp_llm.ops.compute_ops import (
    PyAttentionInputs,
    fill_mla_params,
    get_typemeta,
    rtp_llm_ops,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")


class TestPyFlashinferDecodeAttnOp(BaseAttentionTest):
    """Test suite for PyFlashinferDecodeAttnOp with correctness verification"""

    def _create_attention_inputs(
        self,
        batch_size: int,
        sequence_lengths: List[int],
        seq_size_per_block: int,
        dtype: torch.dtype = torch.float16,
    ) -> PyAttentionInputs:
        """Helper to create PyAttentionInputs for decode"""
        attn_inputs = self._create_attention_inputs_base(
            batch_size=batch_size,
            sequence_lengths=sequence_lengths,
            seq_size_per_block=seq_size_per_block,
        )
        attn_inputs.dtype = get_typemeta(torch.zeros([1], dtype=dtype))
        return attn_inputs

    def _check_params(
        self,
        attn_inputs: PyAttentionInputs,
        batch_size: int,
        sequence_lengths: List[int],
        seq_size_per_block: int,
    ):
        """Check that the prepared parameters match expected values

        This validates that fill_mla_params correctly generates:
        - decode_page_indptr: cumulative count of pages per sequence
        - page_indice: sequential block IDs for all sequences
        - paged_kv_last_page_len: last page length for each sequence
        """
        # Call fill_mla_params to get the actual params
        mla_params = fill_mla_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_block_id,
            seq_size_per_block,
        )

        # Calculate expected values
        expected_page_indptr = [0]
        expected_page_indices = []
        expected_last_page_len = []

        block_offset = 0
        for seq_len in sequence_lengths:
            num_blocks = math.ceil(seq_len / seq_size_per_block)
            expected_page_indptr.append(expected_page_indptr[-1] + num_blocks)

            # Add all block indices for this sequence
            for j in range(num_blocks):
                expected_page_indices.append(block_offset + j)

            # Last page length is the remainder, or full block size if perfectly aligned
            expected_last_page_len.append(
                seq_len % seq_size_per_block or seq_size_per_block
            )
            block_offset += num_blocks

        # Get actual values from mla_params
        actual_page_indptr = mla_params.decode_page_indptr_h.tolist()
        actual_page_indices = mla_params.page_indice_h.tolist()[
            : len(expected_page_indices)
        ]
        actual_last_page_len = mla_params.paged_kv_last_page_len_h.tolist()

        # Verify each parameter
        if actual_page_indptr != expected_page_indptr:
            error_msg = f"page_indptr mismatch:\n  Expected: {expected_page_indptr}\n  Got: {actual_page_indptr}"
            logging.error(error_msg)
            raise AssertionError(error_msg)

        if actual_page_indices != expected_page_indices:
            error_msg = f"page_indices mismatch:\n  Expected: {expected_page_indices}\n  Got: {actual_page_indices}"
            logging.error(error_msg)
            raise AssertionError(error_msg)

        if actual_last_page_len != expected_last_page_len:
            error_msg = f"last_page_len mismatch:\n  Expected: {expected_last_page_len}\n  Got: {actual_last_page_len}"
            logging.error(error_msg)
            raise AssertionError(error_msg)

        # All checks passed
        logging.info(f"✓ fill_mla_params check passed:")
        logging.info(f"  decode_page_indptr: {actual_page_indptr}")
        logging.info(f"  page_indice: {actual_page_indices}")
        logging.info(f"  paged_kv_last_page_len: {actual_last_page_len}")

    def _test_decode_correctness(
        self,
        batch_size: int,
        sequence_lengths: List[int],
        head_num: int = 32,
        head_num_kv: int = 8,
        size_per_head: int = 128,
        seq_size_per_block: int = 64,
    ):
        """Test decode correctness by comparing with flashinfer reference implementation"""

        config = self._create_config(
            head_num=head_num,
            head_num_kv=head_num_kv,
            size_per_head=size_per_head,
            seq_size_per_block=seq_size_per_block,
        )

        attn_inputs = self._create_attention_inputs(
            batch_size, sequence_lengths, config.seq_size_per_block
        )

        # Create PyFlashinferDecodeAttnOp instance
        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, attn_inputs)

        # Check that prepared parameters match expected values BEFORE calling prepare
        self._check_params(
            attn_inputs, batch_size, sequence_lengths, config.seq_size_per_block
        )

        fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        attn_op.set_params(fmha_params)
        params = attn_op.prepare(attn_inputs)

        # Create query input [batch_size, head_num, head_dim]
        local_head_num = config.head_num // config.tp_size
        local_kv_head_num = config.head_num_kv // config.tp_size
        q = self._create_query_tensor(batch_size, local_head_num, config.size_per_head)

        # Create KV cache
        total_blocks = self._calculate_total_blocks(
            sequence_lengths, config.seq_size_per_block
        )
        kv_cache, k_cache, v_cache = self._create_kv_cache(
            total_blocks,
            config.seq_size_per_block,
            local_kv_head_num,
            config.size_per_head,
            dtype=torch.float16,
        )

        # Forward pass through PyFlashinferDecodeAttnOp
        output = attn_op.forward(q, kv_cache, params)

        # Generate block_id_list from attn_inputs for reference computation
        block_id_list = self._generate_block_id_list(
            attn_inputs, sequence_lengths, config.seq_size_per_block
        )

        # Compute reference outputs using flashinfer's single_decode_with_kv_cache
        ref_output_stacked = compute_flashinfer_decode_reference(
            q,
            k_cache,
            v_cache,
            sequence_lengths,
            block_id_list,
            config.seq_size_per_block,
        )

        # Compare outputs
        compare_tensors(
            output,
            ref_output_stacked,
            rtol=1e-2,
            atol=1e-2,
            name=f"Decode output (batch={batch_size}, seq_lens={sequence_lengths})",
        )

        logging.info(
            f"✓ Test passed: batch_size={batch_size}, sequence_lengths={sequence_lengths}"
        )

    def test_single_batch_decode(self):
        """Test decode for a single batch"""
        logging.info("\n=== Testing single batch decode ===")
        for head_dim in [128, 256]:
            logging.info(f"\n--- Testing head_dim={head_dim} ---")
            self._test_decode_correctness(
                batch_size=1,
                sequence_lengths=[128],
                size_per_head=head_dim,
            )

    def test_multi_batch_decode(self):
        """Test decode for multiple batches with varying sequence lengths"""
        logging.info("\n=== Testing multi-batch decode ===")
        for head_dim in [128, 256]:
            logging.info(f"\n--- Testing head_dim={head_dim} ---")
            self._test_decode_correctness(
                batch_size=4,
                sequence_lengths=[64, 128, 256, 512],
                size_per_head=head_dim,
            )

    def test_different_block_sizes(self):
        """Test with different block sizes"""
        logging.info("\n=== Testing different block sizes ===")
        for head_dim in [128, 256]:
            for block_size in [16, 32, 64, 128]:
                logging.info(
                    f"\n--- Testing head_dim={head_dim}, block_size={block_size} ---"
                )
                self._test_decode_correctness(
                    batch_size=2,
                    sequence_lengths=[100, 200],
                    size_per_head=head_dim,
                    seq_size_per_block=block_size,
                )

    def test_different_head_configurations(self):
        """Test with different head configurations (GQA)"""
        logging.info("\n=== Testing different head configurations ===")
        test_cases = [
            (32, 32, "MHA"),  # MHA: head_num == head_num_kv (group_size=1)
            (32, 8, "GQA"),  # GQA: head_num > head_num_kv (group_size=4)
            (32, 4, "GQA-4"),  # GQA with group_size=8
        ]

        for head_dim in [128, 256]:
            for head_num, head_num_kv, name in test_cases:
                logging.info(
                    f"\n--- Testing {name}: head_num={head_num}, head_num_kv={head_num_kv}, head_dim={head_dim} ---"
                )
                self._test_decode_correctness(
                    batch_size=2,
                    sequence_lengths=[100, 200],
                    head_num=head_num,
                    head_num_kv=head_num_kv,
                    size_per_head=head_dim,
                )

    def test_edge_case_sequence_lengths(self):
        """Test edge cases with sequence lengths"""
        logging.info("\n=== Testing edge case sequence lengths ===")

        for head_dim in [128, 256]:
            # Sequence length exactly equal to block size
            logging.info(
                f"\n--- Testing seq_len == block_size, head_dim={head_dim} ---"
            )
            self._test_decode_correctness(
                batch_size=1,
                sequence_lengths=[64],
                size_per_head=head_dim,
                seq_size_per_block=64,
            )

            # Sequence length slightly more than block size
            logging.info(f"\n--- Testing seq_len > block_size, head_dim={head_dim} ---")
            self._test_decode_correctness(
                batch_size=1,
                sequence_lengths=[65],
                size_per_head=head_dim,
                seq_size_per_block=64,
            )

            # Very short sequences
            logging.info(f"\n--- Testing short sequences, head_dim={head_dim} ---")
            self._test_decode_correctness(
                batch_size=2,
                sequence_lengths=[10, 20],
                size_per_head=head_dim,
                seq_size_per_block=64,
            )


class TestPyFlashinferDecodeCudaGraph(BaseAttentionTest):
    """Test CUDA graph buffer management for PyFlashinferDecodeAttnOp.

    Verifies the critical invariants that the C++ CUDA graph runner depends on:
    1. prepare() with is_cuda_graph=True sets _fixed_batch_size and wires up
       the decode_wrapper's internal buffers for graph capture.
    2. prepare_for_cuda_graph_replay() refreshes the page tables AND calls
       plan() to update workspace scheduling metadata for ALL decode paths
       (both FA2 tensor-core and non-tensor-core), without reallocating buffers.

    Full forward correctness under CUDA graph capture/replay cannot be tested
    at the Python UT level — that path is exercised by smoke tests with real
    model inference via cuda_graph_runner.cc.
    """

    def _create_cuda_graph_inputs(
        self,
        batch_size: int,
        sequence_lengths: List[int],
        seq_size_per_block: int,
        dtype: torch.dtype = torch.float16,
    ) -> PyAttentionInputs:
        """Create inputs with is_cuda_graph=True, mimicking cuda_graph_runner.cc."""
        attn_inputs = PyAttentionInputs()
        attn_inputs.is_prefill = False
        attn_inputs.is_cuda_graph = True

        seq_t = torch.tensor(sequence_lengths, dtype=torch.int32)
        attn_inputs.sequence_lengths = (seq_t - 1).pin_memory()
        attn_inputs.input_lengths = torch.ones(
            batch_size, dtype=torch.int32
        ).pin_memory()
        attn_inputs.prefix_lengths = torch.empty(0, dtype=torch.int32).pin_memory()

        kv_cache_block_id = self._create_kv_cache_block_ids(
            batch_size, sequence_lengths, seq_size_per_block
        )
        attn_inputs.kv_cache_kernel_block_id = kv_cache_block_id
        attn_inputs.kv_cache_kernel_block_id_device = kv_cache_block_id.cuda()

        attn_inputs.cu_seqlens_device = torch.arange(
            0, batch_size + 1, dtype=torch.int32, device="cuda"
        )
        attn_inputs.dtype = get_typemeta(torch.zeros([1], dtype=dtype))
        return attn_inputs

    def test_capture_sets_fixed_batch_size(self):
        """prepare() with is_cuda_graph=True must set _fixed_batch_size."""
        config = self._create_config()
        capture_bs = 4
        seq_lens = [64, 128, 256, 512]
        inputs = self._create_cuda_graph_inputs(
            capture_bs,
            seq_lens,
            config.seq_size_per_block,
        )

        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, inputs)
        self.assertTrue(attn_op.enable_cuda_graph)
        self.assertEqual(attn_op.decode_wrapper._fixed_batch_size, 0)

        fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        attn_op.set_params(fmha_params)
        attn_op.prepare(inputs)

        self.assertEqual(attn_op.decode_wrapper._fixed_batch_size, capture_bs)
        self.assertTrue(attn_op.decode_wrapper._use_cuda_graph)
        logging.info("_fixed_batch_size correctly set after prepare()")

    def test_replay_refreshes_plan_metadata(self):
        """prepare_for_cuda_graph_replay() MUST call plan() during replay.

        FlashInfer's design contract: plan() must be called before every graph
        replay to update workspace scheduling metadata. Skipping plan() leaves
        stale data in the workspace buffer → kernel hang/wrong results.

        The previous hang was caused by FlashInfer decode.py L960-961 forcing
        non_blocking=False for host indices. Fix: pass page_indice_d (device)
        so all copies inside plan() are non-blocking.
        """
        config = self._create_config()
        capture_bs = 8
        capture_seq_lens = [64, 128, 256, 512, 64, 128, 256, 512]

        capture_inputs = self._create_cuda_graph_inputs(
            capture_bs,
            capture_seq_lens,
            config.seq_size_per_block,
        )
        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, capture_inputs)
        self.assertTrue(attn_op.use_tensor_core)
        self.assertTrue(attn_op._requires_fa2_cuda_graph_replan())
        fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        attn_op.set_params(fmha_params)
        attn_op.prepare(capture_inputs)

        self.assertEqual(attn_op.decode_wrapper._fixed_batch_size, capture_bs)

        original_plan = attn_op.decode_wrapper.plan
        plan_calls = []

        def counted_plan(*args, **kwargs):
            plan_calls.append((args, kwargs))
            return original_plan(*args, **kwargs)

        attn_op.decode_wrapper.plan = counted_plan

        run_seq_lens = [100, 200, 300, 400, 64, 128, 256, 512]
        run_inputs = self._create_cuda_graph_inputs(
            capture_bs,
            run_seq_lens,
            config.seq_size_per_block,
        )
        attn_op.prepare_for_cuda_graph_replay(run_inputs)

        # plan() MUST be called during replay (FlashInfer design contract).
        self.assertEqual(len(plan_calls), 1)

        # Verify plan() was called with device indices (to avoid sync hang)
        # and disable_split_kv=True (to match capture-time grid config).
        _, plan_kwargs = plan_calls[0]
        self.assertTrue(plan_kwargs.get("disable_split_kv", False))
        self.assertTrue(plan_kwargs.get("non_blocking", False))
        # Second positional arg is indices — must be on CUDA device
        plan_args = plan_calls[0][0]
        indices_arg = plan_args[1]  # (indptr, indices, last_page_len, ...)
        self.assertTrue(
            indices_arg.is_cuda,
            "indices must be device tensor to avoid sync hang",
        )

        # _fixed_batch_size must stay at the captured graph size.
        self.assertEqual(attn_op.decode_wrapper._fixed_batch_size, capture_bs)

        # Device buffers should be updated by fill_params (verify they exist).
        page_indptr = fmha_params.decode_page_indptr_h
        self.assertIsNotNone(page_indptr)
        self.assertGreaterEqual(len(page_indptr), capture_bs + 1)
        logging.info(
            f"Replay OK: _fixed_batch_size={attn_op.decode_wrapper._fixed_batch_size}, "
            f"page_indptr={page_indptr.tolist()}"
        )

    def test_non_fa2_replay_also_replans(self):
        """Non-fa2 decode also calls plan() during replay (FlashInfer contract).

        Both FA2 and non-FA2 paths need plan() to update workspace scheduling
        metadata before each replay. The non-FA2 path previously skipped plan(),
        causing stale workspace hang on models with GQA ratio < 4 (e.g. Qwen3-1.7B).
        """
        config = self._create_config(head_num=32, head_num_kv=32)
        capture_bs = 4
        capture_seq_lens = [64, 128, 256, 512]

        capture_inputs = self._create_cuda_graph_inputs(
            capture_bs,
            capture_seq_lens,
            config.seq_size_per_block,
        )
        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, capture_inputs)
        self.assertFalse(attn_op.use_tensor_core)
        self.assertFalse(attn_op._requires_fa2_cuda_graph_replan())
        fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        attn_op.set_params(fmha_params)
        attn_op.prepare(capture_inputs)

        self.assertTrue(attn_op.decode_wrapper._use_cuda_graph)
        self.assertEqual(attn_op.decode_wrapper._fixed_batch_size, capture_bs)
        self.assertEqual(
            attn_op.decode_wrapper._paged_kv_indptr_buf.data_ptr(),
            fmha_params.decode_page_indptr_d.data_ptr(),
        )
        self.assertEqual(
            attn_op.decode_wrapper._paged_kv_indices_buf.data_ptr(),
            fmha_params.page_indice_d.data_ptr(),
        )
        self.assertEqual(
            attn_op.decode_wrapper._paged_kv_last_page_len_buf.data_ptr(),
            fmha_params.paged_kv_last_page_len_d.data_ptr(),
        )
        self.assertFalse(hasattr(attn_op.decode_wrapper, "_qo_indptr_buf"))

        plan_calls = []

        original_plan = attn_op.decode_wrapper.plan

        def counted_plan(*args, **kwargs):
            plan_calls.append((args, kwargs))
            return original_plan(*args, **kwargs)

        attn_op.decode_wrapper.plan = counted_plan

        run_seq_lens = [100, 200, 256, 512]
        run_inputs = self._create_cuda_graph_inputs(
            capture_bs,
            run_seq_lens,
            config.seq_size_per_block,
        )
        attn_op.prepare_for_cuda_graph_replay(run_inputs)

        # Non-FA2 path ALSO calls plan() during replay (FlashInfer contract).
        self.assertEqual(len(plan_calls), 1)
        # Verify CG-safe tensor placement: indices on device, indptr on host
        plan_args = plan_calls[0][0]
        self.assertTrue(plan_args[1].is_cuda, "indices must be device")
        self.assertFalse(plan_args[0].is_cuda, "indptr must be host")

    def test_replay_updates_page_tables(self):
        """Page table buffers must reflect the replay inputs, not capture inputs."""
        import math

        config = self._create_config(seq_size_per_block=64)
        capture_bs = 4
        capture_seq_lens = [64, 128, 256, 512]
        active_bs = 2
        run_seq_lens = [100, 200, 256, 512]

        capture_inputs = self._create_cuda_graph_inputs(
            capture_bs,
            capture_seq_lens,
            config.seq_size_per_block,
        )
        attn_op = PyFlashinferDecodeAttnOp(config.attn_configs, capture_inputs)
        fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
        attn_op.set_params(fmha_params)
        attn_op.prepare(capture_inputs)

        run_inputs = self._create_cuda_graph_inputs(
            capture_bs,
            run_seq_lens,
            config.seq_size_per_block,
        )
        attn_op.prepare_for_cuda_graph_replay(run_inputs)

        # Verify page_indptr matches run_seq_lens
        page_indptr = fmha_params.decode_page_indptr_h.tolist()
        expected_blocks = [math.ceil(s / 64) for s in run_seq_lens[:active_bs]]
        expected_indptr = [0]
        for nb in expected_blocks:
            expected_indptr.append(expected_indptr[-1] + nb)

        for i in range(active_bs + 1):
            self.assertEqual(
                page_indptr[i],
                expected_indptr[i],
                f"page_indptr[{i}] mismatch: expected {expected_indptr[i]}, got {page_indptr[i]}",
            )

        # Verify last_page_len
        last_page_len = fmha_params.paged_kv_last_page_len_h.tolist()
        for i, seq_len in enumerate(run_seq_lens[:active_bs]):
            expected = seq_len % 64 or 64
            self.assertEqual(
                last_page_len[i],
                expected,
                f"last_page_len[{i}] mismatch: expected {expected}, got {last_page_len[i]}",
            )
        expected_last_page_lens = [s % 64 or 64 for s in run_seq_lens[:active_bs]]
        logging.info(
            f"Page table update OK: indptr={expected_indptr}, "
            f"last_page_len={expected_last_page_lens}"
        )

    # (head_num_kv, use_tensor_core, label) — covers BOTH decode CG paths:
    #   - kv=8  → ratio 4 → use_tensor_core=True  → FA2 tensor-core path
    #   - kv=16 → ratio 2 → use_tensor_core=False → non-FA2 path
    #     (this is the Qwen3-1.7B config that actually hung: the non-FA2 replay
    #      previously skipped plan(), leaving stale workspace → deadlock)
    _NO_HANG_CONFIGS = [(8, True, "fa2"), (16, False, "non_fa2")]

    def _run_no_hang_subprocess(self, head_num_kv: int, expect_tc: bool, label: str):
        """Run one CG capture+replay+hang-detection scenario in a subprocess."""
        import subprocess

        test_dir = str(__import__("pathlib").Path(__file__).resolve().parent)
        # test file is at github-opensource/rtp_llm/models_py/modules/factory/attention/cuda_impl/test/
        project_root = str(__import__("pathlib").Path(__file__).resolve().parents[7])

        script_tpl = """
import math, signal, sys, logging, torch
logging.basicConfig(level=logging.INFO, format="%(message)s")
sys.path.insert(0, ".")
sys.path.insert(0, "rtp_llm/models_py/modules/factory/attention/cuda_impl/test")

from base_attention_test import BaseAttentionTest
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import PyFlashinferDecodeAttnOp
from rtp_llm.ops.compute_ops import PyAttentionInputs, get_typemeta, rtp_llm_ops, LayerKVCache

def make_cg_inputs(bt, bs, seq_lens, spb):
    ai = PyAttentionInputs()
    ai.is_prefill = False
    ai.is_cuda_graph = True
    st = torch.tensor(seq_lens, dtype=torch.int32)
    ai.sequence_lengths = (st - 1).pin_memory()
    ai.input_lengths = torch.ones(bs, dtype=torch.int32).pin_memory()
    ai.prefix_lengths = torch.empty(0, dtype=torch.int32).pin_memory()
    bid = bt._create_kv_cache_block_ids(bs, seq_lens, spb)
    ai.kv_cache_kernel_block_id = bid
    ai.kv_cache_kernel_block_id_device = bid.cuda()
    ai.cu_seqlens_device = torch.arange(0, bs + 1, dtype=torch.int32, device="cuda")
    ai.dtype = get_typemeta(torch.zeros(1, dtype=torch.float16))
    return ai

def alarm_handler(signum, frame):
    raise TimeoutError("HANG")

bt = BaseAttentionTest()
bt.setUp()
cfg = bt._create_config(head_num=32, head_num_kv=__HEAD_NUM_KV__, size_per_head=128, seq_size_per_block=32)
bs = 4
cap_sl = [128, 256, 192, 224]
run_sl = [64, 128, 96, 160]
lhn = cfg.head_num // cfg.tp_size
lkvhn = cfg.head_num_kv // cfg.tp_size

cap_inp = make_cg_inputs(bt, bs, cap_sl, cfg.seq_size_per_block)
attn_op = PyFlashinferDecodeAttnOp(cfg.attn_configs, cap_inp)
assert attn_op.use_tensor_core is __EXPECT_TC__, f"use_tensor_core={attn_op.use_tensor_core}, expected __EXPECT_TC__"
fp = rtp_llm_ops.FlashInferMlaAttnParams()
attn_op.set_params(fp)
attn_op.prepare(cap_inp)

blocks = sum(math.ceil(s / cfg.seq_size_per_block) for s in cap_sl)
kvc = torch.randn(blocks, 2, lkvhn, cfg.seq_size_per_block, cfg.size_per_head, dtype=torch.float16, device="cuda")
q = torch.randn(bs, lhn, cfg.size_per_head, dtype=torch.float16, device="cuda")
lkv = LayerKVCache()
lkv.kv_cache_base = kvc

torch.cuda.synchronize()
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    _ = attn_op.forward(q, lkv, fp)
stream.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=stream):
    gout = attn_op.forward(q, lkv, fp)
stream.synchronize()
logging.info("capture OK")

run_inp = make_cg_inputs(bt, bs, run_sl, cfg.seq_size_per_block)
signal.signal(signal.SIGALRM, alarm_handler)
signal.alarm(10)
try:
    attn_op.prepare_for_cuda_graph_replay(run_inp)
    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()
except TimeoutError:
    print("HANG_DETECTED", flush=True)
    sys.exit(1)
finally:
    signal.alarm(0)

assert gout.shape == (bs, lhn, cfg.size_per_head)
assert gout.abs().sum().item() > 0
logging.info(f"replay OK: shape={gout.shape}, sum={gout.abs().sum().item():.4f}")
print("ALL_PASSED", flush=True)
"""
        script = script_tpl.replace("__HEAD_NUM_KV__", str(head_num_kv)).replace(
            "__EXPECT_TC__", "True" if expect_tc else "False"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=30,
            env={
                **__import__("os").environ,
                # Prepend our paths but KEEP the inherited PYTHONPATH: the bazel
                # wrapper injects torch/flashinfer site-packages (_JIT_CACHE_PATHS)
                # there. Overwriting it makes the subprocess fail to import torch.
                "PYTHONPATH": f"{project_root}:{test_dir}:"
                + __import__("os").environ.get("PYTHONPATH", ""),
            },
        )
        combined = result.stdout + result.stderr
        if "HANG_DETECTED" in combined:
            self.fail(
                f"[{label}] CUDA graph replay hung! plan() likely deadlocked. "
                "Check that page_indice_d (device) is passed to avoid "
                "FlashInfer's forced synchronous H2D copy."
            )
        self.assertEqual(
            result.returncode,
            0,
            f"[{label}] Subprocess failed (rc={result.returncode}):\n{combined[-2000:]}",
        )
        self.assertIn("ALL_PASSED", combined, f"[{label}] did not finish")

    def test_cuda_graph_capture_replay_no_hang(self):
        """End-to-end CUDA graph capture + replay with hang detection.

        Runs each config in a subprocess to avoid FlashInfer JIT cache
        contamination across configs. A 10s signal.alarm() watchdog inside the
        subprocess (plus a 30s subprocess timeout) detects hangs.

        Covers BOTH decode CG paths — see _NO_HANG_CONFIGS:
        - FA2 tensor-core path (GQA 4:1, use_tensor_core=True)
        - non-FA2 path (GQA 2:1, use_tensor_core=False) — the Qwen3-1.7B config
          that actually deadlocked because non-FA2 replay skipped plan().
        """
        for head_num_kv, expect_tc, label in self._NO_HANG_CONFIGS:
            with self.subTest(path=label):
                self._run_no_hang_subprocess(head_num_kv, expect_tc, label)


if __name__ == "__main__":
    unittest.main()
