import itertools
import logging
import os
import time
from typing import Any, cast

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.modules.fmha import FlashInferPythonDecodeImpl, FlashInferDecodeImpl, FMHADecodeImplBase, XQAImpl
from rtp_llm.models_py.modules.flashinfer_python import FlashInferPythonDecodeOp
from rtp_llm.models_py.standalone.qwen3_sp import Qwen2MtpModel
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCacheDecodeOp,
    KVCache,
    PyAttentionInputs,
    PyModelInputs,
)
from rtp_llm.utils.model_weight import W
from librtp_compute_ops import init_device  # type: ignore[import]

USE_FAKE_FMHA = os.environ.get("QWEN3_SP_PERF_USE_FAKE_FMHA", "0") != "0"


class DummyFMHADecodeImpl(FMHADecodeImplBase):
    class _NoOp:
        def support(self, _attn_inputs):
            return True

        def prepare(self, _attn_inputs):
            return None

        def forward(self, qkv, *args, **kwargs):
            return qkv

    def __init__(self, config, attn_inputs):
        self.config = config
        super().__init__(
            DummyFMHADecodeImpl._NoOp(),
            DummyFMHADecodeImpl._NoOp(),
            attn_inputs,
        )
        self.attn_inputs = attn_inputs
        self.support_ = True
        self.fmha_params = cast(Any, None)
        self.rope_params = cast(Any, None)

    def forward(self, qkv: torch.Tensor, kv_cache):
        hidden = self.config.hidden_size
        return qkv[..., :hidden]

class MyFlashInferPythonDecodeImpl(FMHADecodeImplBase):
    def __init__(self, config, attn_inputs):
        super().__init__(
            FlashInferPythonDecodeOp(config.gpt_init_params),
            FusedRopeKVCacheDecodeOp(config.gpt_init_params),
            attn_inputs,
        )

def mock_get_fmha_impl(self, attn_inputs):
    if USE_FAKE_FMHA:
        logging.info("Using DummyFMHADecodeImpl fallback for qwen3_sp_perf_test")
        return DummyFMHADecodeImpl(self.config, attn_inputs)
    return FlashInferPythonDecodeImpl(self.config, attn_inputs)

# Monkeypatch get_fmha_impl
Qwen2MtpModel.get_fmha_impl = mock_get_fmha_impl

class MockPyModelInputs:
    def __init__(self, input_ids, attention_inputs, input_hiddens):
        self.input_ids = input_ids
        self.attention_inputs = attention_inputs
        self.input_hiddens = input_hiddens
        self.bert_embedding_inputs = None

class MockLayerWeights:
    def __init__(self, config):
        self.weights = {
            W.pre_ln_gamma: torch.ones(config.hidden_size, dtype=torch.bfloat16, device='cuda'),
            W.post_ln_gamma: torch.ones(config.hidden_size, dtype=torch.bfloat16, device='cuda'),
            W.attn_qkv_w: torch.randn(config.hidden_size, (config.head_num + 2 * config.head_num_kv) * config.size_per_head, dtype=torch.bfloat16, device='cuda'),
            W.attn_o_w: torch.randn(config.head_num * config.size_per_head, config.hidden_size, dtype=torch.bfloat16, device='cuda'),
            W.ffn_w1: torch.randn(config.hidden_size, config.inter_size, dtype=torch.bfloat16, device='cuda'), # gate
            W.ffn_w3: torch.randn(config.hidden_size, config.inter_size, dtype=torch.bfloat16, device='cuda'), # up
            W.ffn_w2: torch.randn(config.inter_size, config.hidden_size, dtype=torch.bfloat16, device='cuda'), # down
        }

    def __getitem__(self, key):
        return self.weights.get(key)

    def get(self, key, default=None):
        return self.weights.get(key, default)

    def __contains__(self, key):
        return key in self.weights

class MockModelWeights:
    def __init__(self, config):
        self.config = config
        self.weights = [MockLayerWeights(config) for _ in range(config.layer_num)]
        self.global_weights = {
            W.embedding: torch.randn(config.vocab_size, config.hidden_size, dtype=torch.bfloat16, device='cuda'),
            W.multi_tokens_predict_eh_proj: torch.randn(config.hidden_size * 2, config.hidden_size, dtype=torch.bfloat16, device='cuda'),
            W.multi_tokens_predict_enorm: torch.ones(config.hidden_size, dtype=torch.bfloat16, device='cuda'),
            W.multi_tokens_predict_hnorm: torch.ones(config.hidden_size, dtype=torch.bfloat16, device='cuda'),
            W.multi_tokens_predict_final_ln_gamma: torch.ones(config.hidden_size, dtype=torch.bfloat16, device='cuda'),
            W.lm_head: torch.randn(config.hidden_size, config.vocab_size, dtype=torch.bfloat16, device='cuda'),
        }
        # Add global weights to layer 0 weights as Qwen2MtpModel accesses them via weights.weights[0]
        self.weights[0].weights.update(self.global_weights)

    def get_global_weight(self, key):
        return self.global_weights.get(key)

def run_benchmark(model, config, batch_size, seq_len):
    try:
        # Decode phase: 1 token per sequence
        input_ids = torch.randint(0, config.vocab_size, (batch_size,), device='cuda')
        input_hiddens = torch.randn(
            (batch_size, config.head_num * config.size_per_head),
            dtype=torch.bfloat16,
            device='cuda'
        )

        # Setup KV Cache (following rtp_auto_model.py format)
        block_size = 64
        # Set seq_size_per_block on config (required by attention kernels)
        config.seq_size_per_block = block_size
        # Calculate blocks needed (matching rtp_auto_model.py: +1 for extra buffer)
        num_blocks_per_seq = (seq_len + block_size - 1) // block_size + 1
        total_blocks = num_blocks_per_seq * batch_size

        # FP8 cache with layer dimension
        # Shape: [layer_num * 2, total_blocks+1, head_num_kv, block_size, size_per_head]
        # Note: +1 because block IDs start from 1 (block 0 is unused)
        kv_shape = [
            config.layer_num * 2,
            total_blocks + 1,  # +1 because block IDs start from 1
            config.head_num_kv,
            block_size,
            config.size_per_head,
        ]
        kv_cache_total = torch.ones(
            kv_shape,
            dtype=torch.float8_e4m3fn,
            device='cuda'
        )
        kv_cache_scale = torch.ones(kv_shape, dtype=torch.float8_e4m3fn, device='cuda')

        # Split into k_cache and v_cache
        k_cache = kv_cache_total[:config.layer_num, :, :, :, :]
        v_cache = kv_cache_total[config.layer_num:, :, :, :, :]
        k_cache_scale = kv_cache_scale[:config.layer_num, :, :, :, :]
        v_cache_scale = kv_cache_scale[config.layer_num:, :, :, :, :]

        kv_cache = cast(Any, KVCache())
        kv_cache.k_cache_base = k_cache
        kv_cache.v_cache_base = v_cache
        kv_cache.k_scale_base = k_cache_scale
        kv_cache.v_scale_base = v_cache_scale

        # Set model kv cache
        model.kv_cache = kv_cache        # Setup Attention Inputs (following rtp_auto_model.py decode format)
        attention_inputs = cast(Any, PyAttentionInputs())
        host_seq_lens = torch.full(
            (batch_size,),
            seq_len,
            dtype=torch.int32,
            device='cpu',
        ).pin_memory()
        attention_inputs.sequence_lengths = host_seq_lens

        host_input_lens = torch.full(
            (batch_size,),
            1,
            dtype=torch.int32,
            device='cpu',
        ).pin_memory()
        attention_inputs.input_lengths = host_input_lens  # q_len=1

        # Add prefix_lengths (required for decode)
        attention_inputs.prefix_lengths = torch.zeros(batch_size, dtype=torch.int32)

        # Add padding_offset for decode (one per query token)
        attention_inputs.padding_offset = torch.zeros(batch_size, dtype=torch.int32, device='cuda')

        attention_inputs.set_dtype_from_tensor(torch.empty(0, dtype=torch.bfloat16))

        # Block tables (block IDs start from 1, not 0, following rtp_auto_model.py)
        host_block_tables = torch.arange(
            1, total_blocks + 1,  # Start from 1, not 0
            dtype=torch.int32,
            device='cpu',
        ).view(batch_size, num_blocks_per_seq).pin_memory()
        device_block_tables = host_block_tables.to('cuda')
        attention_inputs.kv_cache_block_id_device = device_block_tables
        attention_inputs.kv_cache_block_id_host = host_block_tables
        # kv_block_offset should be layer_num * (total_blocks+1) (matching rtp_auto_model.py pattern)
        attention_inputs.kv_block_offset = config.layer_num * (total_blocks + 1)
        attention_inputs.is_prefill = False  # This is decode phase

        # cu_seqlens for Q (batch)
        attention_inputs.cu_seqlens = torch.arange(
            batch_size + 1,
            dtype=torch.int32,
            device='cuda',
        )

        # cu_kv_seqlens for KV
        attention_inputs.cu_kv_seqlens = torch.arange(
            0,
            (batch_size + 1) * seq_len,
            seq_len,
            dtype=torch.int32,
            device='cuda',
        )

        inputs = MockPyModelInputs(
            input_ids=input_ids,
            attention_inputs=attention_inputs,
            input_hiddens=input_hiddens
        )

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                model(inputs)
        torch.cuda.synchronize()

        # Benchmark using kernel time measurements
        num_iters = 10
        kernel_times = []

        print(f"\n  Measuring kernel times for bs={batch_size}, sl={seq_len}...")
        for iter_idx in range(num_iters):
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                with_stack=False,
            ) as prof:
                with torch.no_grad():
                    model(inputs)

            # Collect CUDA kernel times for this iteration
            all_events = prof.events()
            if all_events is None:
                all_events = []
            cuda_events = [evt for evt in all_events
                          if evt.device_type == torch.profiler.DeviceType.CUDA
                          and evt.name and evt.cuda_time > 0]
            total_cuda_time_us = sum(evt.cuda_time for evt in cuda_events)
            kernel_times.append(total_cuda_time_us)

            # Print detailed kernel info only for the first iteration
            if iter_idx == 0:
                print(f"\n  Detailed kernel profiling (iteration 1/{num_iters}):")
                # Sort by start time (chronological order)
                cuda_events.sort(key=lambda x: x.time_range.start)

                print(f"  {'#':<4} {'Kernel Name':<76} {'Time (us)':<12} {'Start (us)':<12}")
                print(f"  {'-' * 106}")
                for idx, evt in enumerate(cuda_events, 1):
                    kernel_name = evt.name
                    cuda_time_us = evt.cuda_time
                    start_time_us = evt.time_range.start
                    print(f"  {idx:<4} {kernel_name:<76} {cuda_time_us:<12.2f} {start_time_us:<12.2f}")

                print(f"  {'-' * 106}")
                print(f"  {'Total CUDA Time':<92} {total_cuda_time_us:<12.2f} us")
                print()

        # Calculate average kernel time in milliseconds
        avg_latency = sum(kernel_times) / len(kernel_times) / 1000.0  # Convert us to ms

        print(f"  Kernel time statistics over {num_iters} iterations:")
        print(f"    Mean:   {avg_latency:.4f} ms")
        print(f"    Min:    {min(kernel_times)/1000.0:.4f} ms")
        print(f"    Max:    {max(kernel_times)/1000.0:.4f} ms")
        print(f"    StdDev: {(sum((t - avg_latency*1000)**2 for t in kernel_times) / len(kernel_times))**0.5 / 1000.0:.4f} ms")
        print()

        torch.cuda.empty_cache()

        return avg_latency
    except Exception as e:
        import traceback
        print(f"Failed for bs={batch_size}, sl={seq_len}: {e} {traceback.format_exc()}")
        return None

if __name__ == "__main__":
    # Hardcoded Config
    head_num = 48
    hidden_size = 6144
    size_per_head = hidden_size // head_num
    layer_num = 1
    max_seq_len = 32768
    vocab_size = 151936
    inter_size = 13824
    rms_norm_eps = 1e-06
    num_key_value_heads = 8

    config = GptInitModelParameters(
        head_num=head_num,
        size_per_head=size_per_head,
        layer_num=layer_num,
        max_seq_len=max_seq_len,
        vocab_size=vocab_size
    )
    config.inter_size = inter_size
    config.layernorm_eps = rms_norm_eps
    config.head_num_kv = num_key_value_heads
    config.activation_type = "SiGLU"
    config.layernorm_type = "pre_layernorm"
    config.norm_type = "rmsnorm"
    config.has_post_decoder_layernorm = True
    config.has_pre_decoder_layernorm = False
    config.kv_cache_data_type = "fp8"

    # Initialize device
    init_device(config.gpt_init_params)

    # Create Mock Weights
    weights = cast(ModelWeights, MockModelWeights(config))

    # Initialize Model
    model = Qwen2MtpModel(config, weights)
    model.cuda()
    model.eval()

    # Define test ranges
    batch_sizes = list(range(16, 550, 16))
    seq_lens = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

    # For decode scenario: input_ids has batch_size tokens, seq_len is KV cache length
    # Current kernel limitations (with block_size=64):
    #   - bs=16, seq_len<=768 works
    #   - bs=32, seq_len<=768 works
    #   - bs=64 or seq_len>1024 fails with illegal memory access
    # This appears to be a kernel-level limitation in the attention implementation

    # batch_sizes = [32]
    # seq_lens = [768]  # Maximum verified stable configuration

    # Target configuration (exceeds current kernel limits):
    # batch_sizes = [32, 64]
    # seq_lens = [65536]

    results = []

    print("Running benchmarks...")
    for bs, sl in itertools.product(batch_sizes, seq_lens):
        try:
            print(f"Testing Batch Size: {bs}, Seq Len: {sl}...", end=' ')
            avg_latency = run_benchmark(model, config, bs, sl)
            print(f"Avg Latency: {avg_latency:.4f} ms" if avg_latency is not None else "FAILED")
        except Exception as e:
            import traceback
            print(f"FAILED with exception: {e} {traceback.format_exc()}")
            avg_latency = None
        if avg_latency is not None:
            results.append((bs, sl, avg_latency))
        else:
            results.append((bs, sl, float('nan')))

    print("\n" + "=" * 45)
    print("Summary Table")
    print("=" * 45)
    print(f"{'Batch Size':<12} \t {'Seq Len':<10} \t {'Latency (ms)':<15}")
    # print("-" * 45)
    for bs, sl, lat in results:
        lat_str = f"{lat:.4f}" if not torch.isnan(torch.tensor(lat)) else "FAILED"
        print(f"{bs:<12} \t {sl:<10} \t {lat_str:<15}")
    print("=" * 45)
