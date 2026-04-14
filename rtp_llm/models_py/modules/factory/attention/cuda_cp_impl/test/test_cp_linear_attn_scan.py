"""
Unit tests for Qwen3NextGatedDeltaNet CP prefill output correctness.

Verifies that CP prefill output matches single-GPU non-CP reference (per-token).
Uses real NCCL communication via multiprocessing.Process.

Run with:
  bazelisk test //rtp_llm/models_py/modules/factory/attention/cuda_cp_impl/test:test_cp_linear_attn_scan
"""

import logging
import math
import multiprocessing as mp
import os
import unittest
from typing import Dict, List, Optional
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.test.cp_test_utils import (
    build_cp_attn_inputs,
    build_padding_mask,
    build_restore_indices,
    compute_rank_positions,
)
from rtp_llm.models_py.triton_kernels.causal_conv1d import (
    prepare_causal_conv1d_metadata,
)
from rtp_llm.test.utils.port_util import PortManager

logging.basicConfig(level=logging.INFO, format="%(message)s")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _AttnInputsWrapper:
    """Thin wrapper to override readonly pybind11 attributes for testing."""

    def __init__(self, wrapped, overrides: dict):
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_overrides", overrides)

    def __getattr__(self, name):
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        return getattr(object.__getattribute__(self, "_wrapped"), name)

    def __setattr__(self, name, value):
        try:
            setattr(object.__getattribute__(self, "_wrapped"), name, value)
        except AttributeError:
            object.__getattribute__(self, "_overrides")[name] = value


def _add_device_tensors(inputs, device: torch.device):
    return _AttnInputsWrapper(
        inputs,
        {
            "prefix_lengths_d": inputs.prefix_lengths.to(device),
            "input_lengths_d": inputs.input_lengths.to(device),
        },
    )


def _noop_write_cache_store():
    """Patch compute_ops.write_cache_store to no-op."""
    import rtp_llm.ops.compute_ops as _co

    return patch.object(_co, "write_cache_store", lambda *a, **kw: None)


class MockLayerKVCache:
    def __init__(self, kv_cache_base: torch.Tensor, seq_size_per_block: int):
        self.kv_cache_base = kv_cache_base
        self.seq_size_per_block = seq_size_per_block


def allocate_kv_cache(
    sequence_lengths: List[int],
    tokens_per_block: int,
    num_v_heads: int,
    head_v_dim: int,
    head_k_dim: int,
    conv_kernel_dim: int,
    qkv_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    total_blocks = sum(math.ceil(sl / tokens_per_block) for sl in sequence_lengths)
    item_size = {torch.bfloat16: 2, torch.float16: 2, torch.float32: 4}[dtype]
    ssm_bytes = num_v_heads * head_v_dim * head_k_dim * item_size
    conv_bytes = (conv_kernel_dim - 1) * qkv_size * item_size
    block_elems = (ssm_bytes + conv_bytes) // item_size
    return torch.zeros(total_blocks, block_elems, dtype=dtype, device=device)


def _build_layer_weights(
    hidden_size: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Optional[torch.Tensor]]:
    from rtp_llm.utils.model_weight import W

    qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads
    z_dim = head_v_dim * num_v_heads
    qkvz_scale = hidden_size**-0.5
    ba_scale = hidden_size**-0.5
    out_scale = (num_v_heads * head_v_dim) ** -0.5
    return {
        W.linear_attn_conv1d_w: torch.randn(
            qkv_dim, 1, conv_kernel_dim, device=device, dtype=dtype
        )
        * (conv_kernel_dim**-0.5),
        W.linear_attn_dt_b: torch.randn(num_v_heads, device=device, dtype=dtype) * 0.1,
        W.linear_attn_alog: torch.randn(num_v_heads, device=device, dtype=dtype) * 0.1,
        W.linear_attn_norm_w: torch.ones(head_v_dim, device=device, dtype=dtype),
        W.linear_attn_qkvz_w: torch.randn(
            hidden_size, qkv_dim + z_dim, device=device, dtype=dtype
        )
        * qkvz_scale,
        W.linear_attn_qkvz_s: None,
        W.linear_attn_ba_w: torch.randn(
            hidden_size, num_v_heads * 2, device=device, dtype=dtype
        )
        * ba_scale,
        W.linear_attn_out_w: torch.randn(
            num_v_heads * head_v_dim, hidden_size, device=device, dtype=dtype
        )
        * out_scale,
        W.linear_attn_out_s: None,
    }


def _build_nocp_attn_inputs(
    sequence_lengths: List[int],
    tokens_per_block: int,
    device: torch.device,
):
    from rtp_llm.ops.compute_ops import PyAttentionInputs, get_typemeta

    batch_size = len(sequence_lengths)
    inp = PyAttentionInputs()
    inp.is_prefill = True
    inp.input_lengths = torch.tensor(sequence_lengths, dtype=torch.int32, device="cpu")
    inp.prefix_lengths = torch.zeros(batch_size, dtype=torch.int32, device="cpu")
    inp.cache_store_inputs = None

    cu = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for i, sl in enumerate(sequence_lengths):
        cu[i + 1] = cu[i] + sl
    inp.cu_seqlens = cu

    max_blocks = max(math.ceil(sl / tokens_per_block) for sl in sequence_lengths)
    block_ids = torch.zeros(batch_size, max_blocks, dtype=torch.int32)
    offset = 0
    for i, sl in enumerate(sequence_lengths):
        nb = math.ceil(sl / tokens_per_block)
        block_ids[i, :nb] = torch.arange(offset, offset + nb, dtype=torch.int32)
        offset += nb
    inp.kv_cache_block_id_host = block_ids
    inp.kv_cache_kernel_block_id_host = block_ids
    inp.kv_cache_block_id_device = block_ids.to(device)
    inp.kv_cache_kernel_block_id_device = block_ids.to(device)
    inp.context_parallel_info = None
    inp.dtype = get_typemeta(torch.zeros(1, dtype=torch.bfloat16))
    return inp


def _build_cp_metadata(
    sequence_lengths: List[int],
    cp_chunk_lengths: List[int],
    cp_size: int,
    cp_rank: int,
    cp_attn_inputs,
    device: torch.device,
):
    from rtp_llm.models_py.model_desc.qwen3_next import (
        CpChunkAlignInfo,
        Qwen3NextMetadata,
    )

    batch_size = len(sequence_lengths)
    cp_info = cp_attn_inputs.context_parallel_info
    restore_indices = cp_info.prefill_qkv_restore_indice
    padding_mask = cp_info.prefill_qkv_padding_mask
    unpad_restore = restore_indices[padding_mask == 1]

    total_ag = padding_mask.shape[0]
    local_chunk_total = total_ag // cp_size
    local_start = cp_rank * local_chunk_total

    inv_restore = torch.full((total_ag,), -1, dtype=torch.long, device=device)
    inv_restore[unpad_restore.long()] = torch.arange(
        unpad_restore.shape[0], device=device
    )
    local_inv = inv_restore[local_start : local_start + local_chunk_total]
    cp_local_valid_mask = local_inv >= 0
    cp_local_extract_idx = local_inv[cp_local_valid_mask]

    full_cu = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    full_cu[1:] = torch.tensor(sequence_lengths, device=device).cumsum(0)

    full_conv_meta = prepare_causal_conv1d_metadata(
        query_start_loc=full_cu, device=device
    )

    chunk_align_info = CpChunkAlignInfo.build(
        full_cu=full_cu,
        cp_size=cp_size,
        cp_rank=cp_rank,
        device=device,
    )

    return Qwen3NextMetadata(
        full_prefill_conv1d_meta=full_conv_meta,
        full_prefill_cu_seqlens=full_cu,
        cp_restore_indices=restore_indices,
        cp_local_extract_indices=cp_local_extract_idx,
        cp_local_valid_mask=cp_local_valid_mask,
        cp_chunk_align_info=chunk_align_info,
    )


# ---------------------------------------------------------------------------
# Worker function (runs on each rank)
# ---------------------------------------------------------------------------


def _worker(
    rank: int,
    world_size: int,
    nccl_port: int,
    sequence_lengths: List[int],
    num_layers: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_dim: int,
    tokens_per_block: int,
):
    """Per-rank worker: runs non-CP reference and CP path, then compares."""
    import torch.distributed as dist

    from rtp_llm.models_py.distributed import collective_torch as ct
    from rtp_llm.models_py.distributed.collective_torch import Group
    from rtp_llm.models_py.model_desc.qwen3_next import (
        Qwen3NextGatedDeltaNet,
        Qwen3NextMetadata,
    )
    from rtp_llm.models_py.triton_kernels.fla.block import store_ssm_state_to_block_map
    from rtp_llm.ops import DataType, LinearAttentionConfig, ParallelismConfig

    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(nccl_port)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

        par = ParallelismConfig()
        par.world_rank = rank
        par.world_size = world_size
        par.local_rank = rank
        par.tp_size = world_size
        par.dp_size = 1

        ct._parallelism_config = par
        ct._initialized = True
        ct._group_map[Group.DP_AND_TP] = dist.group.WORLD

        device = torch.device(f"cuda:{rank}")
        cp_size = world_size
        hidden_size = num_v_heads * head_v_dim
        batch_size = len(sequence_lengths)
        total_tokens = sum(sequence_lengths)
        cp_chunk_lengths = [sl // cp_size for sl in sequence_lengths]
        dtype = torch.bfloat16
        qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads

        linear_cfg = LinearAttentionConfig()
        linear_cfg.linear_num_key_heads = num_k_heads
        linear_cfg.linear_num_value_heads = num_v_heads
        linear_cfg.linear_key_head_dim = head_k_dim
        linear_cfg.linear_value_head_dim = head_v_dim
        linear_cfg.linear_conv_kernel_dim = conv_kernel_dim
        linear_cfg.ssm_state_dtype = DataType.TYPE_BF16
        linear_cfg.conv_state_dtype = DataType.TYPE_BF16

        torch.manual_seed(123)
        layers_cp = []
        layers_nocp = []
        for _ in range(num_layers):
            weights = _build_layer_weights(
                hidden_size,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                conv_kernel_dim,
                device,
            )
            m_cp = Qwen3NextGatedDeltaNet(
                linear_cfg, ParallelismConfig(), weights, layernorm_eps=1e-6
            ).to(device)
            m_nocp = Qwen3NextGatedDeltaNet(
                linear_cfg, ParallelismConfig(), weights, layernorm_eps=1e-6
            ).to(device)
            layers_cp.append(m_cp)
            layers_nocp.append(m_nocp)

        for m in layers_cp:
            m.parallelism_config.tp_size = cp_size
            m.parallelism_config.tp_rank = rank

        torch.manual_seed(42)
        full_hidden = torch.randn(total_tokens, hidden_size, device=device, dtype=dtype)

        # ==============================================================
        # 1. Non-CP reference
        # ==============================================================
        nocp_inputs = _build_nocp_attn_inputs(
            sequence_lengths, tokens_per_block, device
        )
        nocp_inputs = _add_device_tensors(nocp_inputs, device)

        nocp_conv_meta = prepare_causal_conv1d_metadata(
            query_start_loc=nocp_inputs.cu_seqlens, device=device
        )
        nocp_meta = Qwen3NextMetadata(prefill_conv1d_meta=nocp_conv_meta)

        with torch.no_grad(), _noop_write_cache_store():
            ref_h = full_hidden
            for layer in layers_nocp:
                kv = MockLayerKVCache(
                    allocate_kv_cache(
                        sequence_lengths,
                        tokens_per_block,
                        num_v_heads,
                        head_v_dim,
                        head_k_dim,
                        conv_kernel_dim,
                        qkv_dim,
                        dtype,
                        device,
                    ),
                    tokens_per_block,
                )
                ref_h = layer(ref_h, None, kv, nocp_inputs, nocp_meta)
        ref_output = ref_h

        # ==============================================================
        # 2. CP path (real NCCL)
        # ==============================================================
        all_rank_pos = compute_rank_positions(sequence_lengths, cp_size)
        rank_positions = all_rank_pos[rank]
        rank_idx = torch.tensor(rank_positions, device=device)
        local_hidden = full_hidden[rank_idx].contiguous()

        cp_attn_inputs = build_cp_attn_inputs(
            sequence_lengths,
            cp_chunk_lengths,
            cp_size,
            tokens_per_block,
            device=device,
        )
        # cp_test_utils doesn't set kv_cache_kernel_block_id_device
        cp_attn_inputs.kv_cache_kernel_block_id_device = (
            cp_attn_inputs.kv_cache_block_id_device
        )
        cp_attn_inputs = _add_device_tensors(cp_attn_inputs, device)

        cp_meta = _build_cp_metadata(
            sequence_lengths,
            cp_chunk_lengths,
            cp_size,
            rank,
            cp_attn_inputs,
            device,
        )

        with torch.no_grad(), _noop_write_cache_store():
            cp_h = local_hidden
            for layer in layers_cp:
                kv = MockLayerKVCache(
                    allocate_kv_cache(
                        sequence_lengths,
                        tokens_per_block,
                        num_v_heads,
                        head_v_dim,
                        head_k_dim,
                        conv_kernel_dim,
                        qkv_dim,
                        dtype,
                        device,
                    ),
                    tokens_per_block,
                )
                cp_h = layer(cp_h, None, kv, cp_attn_inputs, cp_meta)
        cp_output = cp_h

        # ==============================================================
        # 3. Compare output
        # ==============================================================
        ref_local = ref_output[rank_idx]
        output_ok = torch.allclose(
            cp_output.float(), ref_local.float(), rtol=1e-2, atol=1e-2
        )
        diff = (cp_output.float() - ref_local.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # ==============================================================
        # 4. Verify full_h and global_final (patch to capture args)
        # ==============================================================
        from unittest.mock import patch as mock_patch

        captured_nocp = []
        captured_cp = []

        original_store = store_ssm_state_to_block_map

        def _capture_nocp(h, final_states, *args, **kwargs):
            captured_nocp.append((h.clone(), final_states.clone()))
            return original_store(h, final_states, *args, **kwargs)

        def _capture_cp(h, final_states, *args, **kwargs):
            captured_cp.append((h.clone(), final_states.clone()))
            return original_store(h, final_states, *args, **kwargs)

        STORE_PATH = (
            "rtp_llm.models_py.model_desc.qwen3_next.store_ssm_state_to_block_map"
        )

        with torch.no_grad(), _noop_write_cache_store():
            ref_h2 = full_hidden
            for layer in layers_nocp:
                kv = MockLayerKVCache(
                    allocate_kv_cache(
                        sequence_lengths,
                        tokens_per_block,
                        num_v_heads,
                        head_v_dim,
                        head_k_dim,
                        conv_kernel_dim,
                        qkv_dim,
                        dtype,
                        device,
                    ),
                    tokens_per_block,
                )
                with mock_patch(STORE_PATH, side_effect=_capture_nocp):
                    ref_h2 = layer(ref_h2, None, kv, nocp_inputs, nocp_meta)

        with torch.no_grad(), _noop_write_cache_store():
            cp_h2 = local_hidden
            for layer in layers_cp:
                kv = MockLayerKVCache(
                    allocate_kv_cache(
                        sequence_lengths,
                        tokens_per_block,
                        num_v_heads,
                        head_v_dim,
                        head_k_dim,
                        conv_kernel_dim,
                        qkv_dim,
                        dtype,
                        device,
                    ),
                    tokens_per_block,
                )
                with mock_patch(STORE_PATH, side_effect=_capture_cp):
                    cp_h2 = layer(cp_h2, None, kv, cp_attn_inputs, cp_meta)

        ssm_ok = True
        ssm_max_diff = 0.0
        for layer_idx in range(num_layers):
            nocp_h, nocp_fs = captured_nocp[layer_idx]
            cp_full_h, cp_global_fs = captured_cp[layer_idx]

            h_diff_val = (cp_full_h.float() - nocp_h.float()).abs().max().item()
            fs_diff_val = (cp_global_fs.float() - nocp_fs.float()).abs().max().item()
            ssm_max_diff = max(ssm_max_diff, h_diff_val, fs_diff_val)

            logging.info(
                f"  rank {rank} layer {layer_idx}: "
                f"full_h cp={cp_full_h.shape} nocp={nocp_h.shape} "
                f"h_diff={h_diff_val:.6f} fs_diff={fs_diff_val:.6f}"
            )

            if not (
                torch.allclose(cp_full_h.float(), nocp_h.float(), rtol=1e-2, atol=1e-2)
                and torch.allclose(
                    cp_global_fs.float(), nocp_fs.float(), rtol=1e-2, atol=1e-2
                )
            ):
                ssm_ok = False

        passed = output_ok and ssm_ok

        dist.barrier()
        logging.info(
            f"  rank {rank}: max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} "
            f"ssm_max={ssm_max_diff:.6f} {'PASS' if passed else 'FAIL'}"
        )
        dist.barrier()
        torch.cuda.synchronize()
        dist.destroy_process_group()

        assert passed, (
            f"rank {rank} failed: max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} "
            f"ssm_max={ssm_max_diff:.6f}"
        )

    except Exception as e:
        print(f"Rank {rank} error: {e}")
        import traceback

        traceback.print_exc()
        raise


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

GPU_COUNT = int(os.environ.get("GPU_COUNT", "4"))


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.device_count() >= GPU_COUNT,
    f"Requires >= {GPU_COUNT} GPUs",
)
class TestGDNCPPrefillOutput(unittest.TestCase):
    """Verify CP GatedDeltaNet prefill output matches non-CP reference with real NCCL."""

    def setUp(self):
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        self.port_manager = PortManager()

    def _run_test(
        self,
        cp_size: int,
        sequence_lengths: List[int],
        num_layers: int = 1,
        num_k_heads: int = 16,
        num_v_heads: int = 32,
        head_k_dim: int = 128,
        head_v_dim: int = 128,
        conv_kernel_dim: int = 4,
        tokens_per_block: int = 16,
    ):
        assert all(
            sl % (cp_size * 2) == 0 for sl in sequence_lengths
        ), f"Seq lengths must be divisible by cp_size*2={cp_size * 2}"

        ports, locks = self.port_manager.get_consecutive_ports(1)
        nccl_port = ports[0]

        try:
            processes = []
            for rank in range(cp_size):
                p = mp.Process(
                    target=_worker,
                    args=(
                        rank,
                        cp_size,
                        nccl_port,
                        sequence_lengths,
                        num_layers,
                        num_k_heads,
                        num_v_heads,
                        head_k_dim,
                        head_v_dim,
                        conv_kernel_dim,
                        tokens_per_block,
                    ),
                    name=f"rank-{rank}",
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join(timeout=300)
                if p.exitcode != 0:
                    raise RuntimeError(
                        f"Process {p.name} exited with code {p.exitcode}"
                    )
        finally:
            for lock in locks:
                lock.__exit__(None, None, None)

    # --- Single layer ---

    def test_single_layer_single_seq(self):
        self._run_test(cp_size=GPU_COUNT, sequence_lengths=[256], num_layers=1)

    def test_single_layer_multi_batch(self):
        self._run_test(cp_size=GPU_COUNT, sequence_lengths=[256, 512], num_layers=1)

    def test_single_layer_2gpu(self):
        self._run_test(cp_size=2, sequence_lengths=[128], num_layers=1)

    # --- Multi layer ---

    def test_multi_layer_single_seq(self):
        self._run_test(cp_size=GPU_COUNT, sequence_lengths=[256], num_layers=4)

    def test_multi_layer_multi_batch(self):
        self._run_test(cp_size=GPU_COUNT, sequence_lengths=[256, 512], num_layers=4)

    # --- Larger sequence ---

    def test_single_layer_64k(self):
        self._run_test(cp_size=GPU_COUNT, sequence_lengths=[64 * 1024], num_layers=1)

    def test_single_layer_unaligned(self):
        self._run_test(cp_size=2, sequence_lengths=[200], num_layers=1)

    def test_single_layer_unaligned_multi_batch(self):
        self._run_test(cp_size=2, sequence_lengths=[200, 392], num_layers=1)


if __name__ == "__main__":
    unittest.main()
