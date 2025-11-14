import itertools
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional
from unittest import SkipTest, TestCase, main

import torch

MAX_ITERATIONS = 100000

device = torch.device(f"cuda")

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.rotary_embedding.deepseek_rotary_embedding import (
    DeepseekV3YarnRotaryEmbedding,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import (
    MlaFlashInferPrefillImpl,
)
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs
from rtp_llm.utils.model_weight import W


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_kv_from_cache(ckv, kpe, kv_len, batch_size, num_heads):
    bs_page_num, page_size, ckv_dim = ckv.shape
    page_num = bs_page_num // batch_size
    _, _, kpe_dim = kpe.shape
    ckv = ckv.view(batch_size, page_num * page_size, ckv_dim)
    kpe = kpe.view(batch_size, page_num * page_size, kpe_dim)
    ckv = ckv[:, :kv_len, :]
    kpe = kpe[:, :kv_len, :]
    k = (
        torch.cat([ckv, kpe], dim=-1)
        .view(-1, 1, ckv_dim + kpe_dim)
        .repeat_interleave(num_heads, dim=1)
    )
    v = ckv.repeat_interleave(num_heads, dim=1)

    return k, v


def create_cos_sin_cache():
    rotary_emb = DeepseekV3YarnRotaryEmbedding(
        64,
        163840,
        10000,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=0.707,
        mscale_all_dim=0.707,
    )
    half_rope_dim = 64 // 2
    cos_cache = rotary_emb.cos_cached[:, :half_rope_dim]
    sin_cache = rotary_emb.sin_cached[:, :half_rope_dim]
    # cos sin cache must be float32
    cos_sin_cache = (
        torch.cat([cos_cache, sin_cache], dim=-1)
        .contiguous()
        .to(device)
        .to(torch.float32)
    )
    return cos_sin_cache


class MLABenchmark(TestCase):
    # 扩展测试参数范围以进行更全面的性能测试
    NUM_TOKENS = [1, 4, 7, 16, 32, 64]
    HIDDEN_SIZES = [1024, 2048, 4096]
    PAGE_SIZE = [32, 64, 128]
    BATCH_SIZES = [1, 2, 4, 8]

    # Benchmark配置
    WARMUP_ITERATIONS = 5
    BENCHMARK_ITERATIONS = 20

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 创建函数映射字典
        self.fmha_function_map = {
            "forward": self._call_forward,
            "reuse_kv_cache_indexed": self._call_reuse_kv_cache_indexed,
        }

        # 函数描述映射
        self.fmha_function_descriptions = {
            "forward": "Forward pass only",
            "reuse_kv_cache_indexed": "_reuse_kv_cache_indexed function only",
        }

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device(device)
        # 确保CUDA同步
        torch.cuda.synchronize()

    def _call_forward(
        self, fmha_impl, q, compressed_kv, k_pe, kv_cache, reuse_len, config
    ):
        """调用forward函数"""
        fmha_impl.forward(q, compressed_kv, k_pe, kv_cache, 0)

    def _call_reuse_kv_cache_indexed(
        self, fmha_impl, q, compressed_kv, k_pe, kv_cache, reuse_len, config
    ):
        """调用_reuse_kv_cache_indexed函数"""
        fmha_impl.fmha_impl._reuse_kv_cache_indexed_batched(
            compressed_kv, k_pe, kv_cache
        )

    def _benchmark_mla_implementation(
        self,
        num_tokens: int,
        hidden_size: int,
        page_size: int,
        batch_size: int = 1,
        reuse_len: int = 128,
        absorb_opt_len: int = 1,
        function_key: str = "forward",
    ):
        """基准测试MLA实现的性能"""
        # print(f"\n=== Benchmarking MLA Implementation ===")
        # print(f"Parameters: tokens={num_tokens}, hidden_size={hidden_size}, page_size={page_size}, batch_size={batch_size}, reuse_len={reuse_len}, absorb_opt_len={absorb_opt_len}, function_key={function_key}")

        # 生成测试数据
        input_lengths = [num_tokens] * batch_size
        mock_page_num = 2048
        page_num = math.ceil(reuse_len + num_tokens + page_size - 1 / page_size)
        block_list = [i for i in range(1, page_num + 1)]
        # print(f"block_list: {block_list}")
        kvcache_block_id = torch.tensor(
            [block_list],
            dtype=torch.int32,
            device=torch.device("cpu"),
        )

        # 配置参数
        config = ModelConfig()
        config.attn_config.head_num = 16
        config.hidden_size = hidden_size
        config.attn_config.nope_head_dim = 128
        config.attn_config.rope_head_dim = 64
        config.attn_config.kv_lora_rank = 512
        config.attn_config.v_head_dim = 128
        config.attn_config.q_lora_rank = 0
        config.attn_config.tokens_per_block = 64
        config.attn_config.softmax_extra_scale = 1.0
        config.attn_config.use_mla = True
        config.attn_config.size_per_head = 192

        torch.manual_seed(0)
        # sequence_lengths_mius_1 = [x for x in sequence_lengths]
        input_lengths_t = torch.tensor(
            input_lengths, dtype=torch.int32, device=torch.device("cpu")
        )
        prefix_lengths_t = torch.tensor(
            [reuse_len],
            dtype=torch.int32,
            device=torch.device("cpu"),
        )

        # 创建attention inputs
        attn_inputs: PyAttentionInputs = PyAttentionInputs()
        attn_inputs.is_prefill = True
        attn_inputs.prefix_lengths = prefix_lengths_t
        attn_inputs.sequence_lengths = torch.tensor(
            [], dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs.input_lengths = input_lengths_t
        attn_inputs.kv_cache_block_id_host = kvcache_block_id

        # 创建权重
        weights = self._create_weights(config, config.hidden_size)
        layer_weights: List[Dict[str, torch.Tensor]] = [weights]

        # 创建输入数据
        q = torch.randn(
            [num_tokens, config.attn_config.head_num, config.attn_config.nope_head_dim + config.attn_config.rope_head_dim],
            dtype=torch.bfloat16,
            device=device,
        )

        compressed_kv = torch.randn(
            [num_tokens, config.attn_config.kv_lora_rank],
            dtype=torch.bfloat16,
            device=device,
        )

        k_pe = torch.randn(
            [num_tokens, config.attn_config.rope_head_dim],
            dtype=torch.bfloat16,
            device=device,
        )

        cache = torch.randn(
            [mock_page_num, page_size, config.attn_config.kv_lora_rank + config.attn_config.rope_head_dim],
            dtype=torch.bfloat16,
            device=device,
        )

        kv_cache: Optional[KVCache] = KVCache()
        kv_cache.k_cache_base = cache

        # 创建cos_sin_cache
        cos_sin_cache = create_cos_sin_cache()

        # 预热阶段
        # print("Warming up...")
        for i in range(self.WARMUP_ITERATIONS):
            fmha_impl = MlaFlashInferPrefillImpl(
                config.attn_config, attn_inputs, layer_weights, cos_sin_cache, absorb_opt_len, quant_config=config.quant_config
            )
            # fmha_impl.forward(q, compressed_kv, k_pe, kv_cache, 0)
            self.fmha_function_map[function_key](
                fmha_impl, q, compressed_kv, k_pe, kv_cache, reuse_len, config
            )
            torch.cuda.synchronize()

        # 基准测试阶段
        # print("Running benchmark...")
        times = []

        for i in range(self.BENCHMARK_ITERATIONS):
            # 重新创建实现对象以确保公平测试
            fmha_impl = MlaFlashInferPrefillImpl(
                config.attn_config, attn_inputs, layer_weights, cos_sin_cache, absorb_opt_len, quant_config=config.quant_config
            )
            # 开始计时
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            # 执行核心计算
            # fmha_impl.forward(q, compressed_kv, k_pe, kv_cache, 0)
            self.fmha_function_map[function_key](
                fmha_impl, q, compressed_kv, k_pe, kv_cache, reuse_len, config
            )
            # 结束计时
            torch.cuda.synchronize()
            end_time = time.perf_counter()

            times.append(end_time - start_time)

            # if (i + 1) % 5 == 0:
            #     print(f"Completed {i + 1}/{self.BENCHMARK_ITERATIONS} iterations")

        # 计算统计信息
        times = torch.tensor(times)
        mean_time = times.mean().item()
        std_time = times.std().item()
        min_time = times.min().item()
        max_time = times.max().item()

        # 计算吞吐量 (tokens per second)
        tokens_per_second = num_tokens * batch_size / mean_time

        # print(f"\n=== Benchmark Results ===")
        # print(f"Mean execution time: {mean_time*1000:.3f} ms")
        # print(f"Std deviation: {std_time*1000:.3f} ms")
        # print(f"Min time: {min_time*1000:.3f} ms")
        # print(f"Max time: {max_time*1000:.3f} ms")
        # print(f"Throughput: {tokens_per_second:.2f} tokens/sec")
        # print(f"Total tokens processed: {num_tokens * batch_size}")

        return {
            "mean_time": mean_time,
            "std_time": std_time,
            "min_time": min_time,
            "max_time": max_time,
            "throughput": tokens_per_second,
            "tokens": num_tokens * batch_size,
        }

    def _create_weights(self, config, hidden_size):
        """创建测试权重"""
        weights = {}
        weights[W.mla_fusedqkrope_no_lora_w] = torch.randn(
            [
                config.hidden_size,
                config.attn_config.size_per_head * config.attn_config.head_num
                + config.attn_config.kv_lora_rank
                + config.attn_config.rope_head_dim,
            ],
            dtype=torch.bfloat16,
            device=device,
        )

        weights[W.mla_kv_a_ln_gamma] = torch.randn(
            [config.attn_config.kv_lora_rank], dtype=torch.bfloat16, device=device
        )

        weights[W.mla_kc] = torch.randn(
            [config.attn_config.head_num, config.attn_config.nope_head_dim, config.attn_config.kv_lora_rank],
            dtype=torch.bfloat16,
            device=device,
        )

        weights[W.mla_vc] = torch.randn(
            [config.attn_config.head_num, config.attn_config.kv_lora_rank, config.attn_config.v_head_dim],
            dtype=torch.bfloat16,
            device=device,
        )

        weights[W.mla_v_w] = torch.randn(
            [config.attn_config.kv_lora_rank, hidden_size],
            dtype=torch.bfloat16,
            device=device,
        )

        weights[W.mla_k_nope_w] = torch.randn(
            [config.attn_config.kv_lora_rank, hidden_size],
            dtype=torch.bfloat16,
            device=device,
        )

        weights[W.attn_o_w] = torch.randn(
            [
                config.attn_config.head_num * config.attn_config.v_head_dim,
                config.hidden_size,
            ],
            dtype=torch.bfloat16,
            device=device,
        )

        return weights

    '''
    def test_mla_benchmark_forward(self):
        """小规模基准测试"""
        results = []
        for params in itertools.product(
            [1, 64, 128, 144, 160, 176, 192, 256, 512, 1024, 2048, 4096, 8192, 10240],
            [2048],
            [64],
            [1],
            [
                0,
                64,
                128,
                256,
                512,
                1024,
                2048,
                4096,
                8192,
                10240,
                20480,
                40960,
                81920,
                102400,
            ],
            [1, MAX_ITERATIONS],
            ["forward"],
        ):
            result = self._benchmark_mla_implementation(*params)
            result.update(
                {
                    "num_tokens": params[0],
                    "hidden_size": params[1],
                    "page_size": params[2],
                    "batch_size": params[3],
                    "reuse_len": params[4],
                    "absorb_opt_len": params[5] == MAX_ITERATIONS,
                }
            )
            results.append(result)

        self._print_fmha_summary(results, "forward Benchmark")
    '''

    def test_mla_benchmark_reuse_kv_cache_indexed(self):
        """小规模基准测试"""
        results = []
        for params in itertools.product(
            [4096],
            [2048],
            [64],
            [1],
            [
                0,
                64,
                128,
                256,
                512,
                1024,
                2048,
                4096,
                8192,
                10240,
                20480,
                40960,
                81920,
                102400,
            ],
            [1],
            ["reuse_kv_cache_indexed"],
        ):
            result = self._benchmark_mla_implementation(*params)
            result.update(
                {
                    "num_tokens": params[0],
                    "hidden_size": params[1],
                    "page_size": params[2],
                    "batch_size": params[3],
                    "reuse_len": params[4],
                    "absorb_opt_len": params[5],
                }
            )
            results.append(result)

        self._print_summary(results, "reuse_kv_cache_indexed Benchmark")

    def test_mla_benchmark_focused(self):
        """针对特定配置的详细基准测试"""
        # 测试你关注的特定配置
        result = self._benchmark_mla_implementation(
            num_tokens=80,
            hidden_size=2048,
            page_size=64,
            batch_size=1,
            reuse_len=0,
            absorb_opt_len=100000,
        )

        print(f"\n=== Focused Benchmark Summary ===")
        print(
            f"Configuration: tokens=7, hidden_size=2048, page_size=64, batch_size=1, reuse_len=0, absorb_opt_len=1"
        )
        print(f"Average execution time: {result['mean_time']*1000:.3f} ms")
        print(f"Throughput: {result['throughput']:.2f} tokens/sec")

    def _print_summary(self, results, title):
        """打印基准测试结果摘要"""
        print(f"\n{'='*80}")
        print(f"{title} Summary")
        print(f"{'='*80}")

        # 按吞吐量排序
        results.sort(key=lambda x: x["absorb_opt_len"], reverse=True)

        # 更新表头以包含新参数
        print(
            f"{'Rank':<4} {'Tokens':<6} {'Hidden':<6} {'Page':<4} {'Batch':<5} {'Reuse':<5} {'Absorb':<6} {'Time(ms)':<10} {'Throughput':<12}"
        )
        print(f"{'-'*80}")

        for i, result in enumerate(results, 1):
            print(
                f"{i:<4} {result['num_tokens']:<6} {result['hidden_size']:<6} "
                f"{result['page_size']:<4} {result['batch_size']:<5} "
                f"{result['reuse_len']:<5} {result['absorb_opt_len']:<6} "
                f"{result['mean_time']*1000:<10.3f} {result['throughput']:<12.2f}"
            )

        # 找出最佳配置
        best = results[0]
        print(f"\nBest configuration:")
        print(
            f"  Tokens: {best['num_tokens']}, Hidden: {best['hidden_size']}, "
            f"Page: {best['page_size']}, Batch: {best['batch_size']}, "
            f"Reuse: {best['reuse_len']}, Absorb: {best['absorb_opt_len']}"
        )
        print(
            f"  Time: {best['mean_time']*1000:.3f} ms, Throughput: {best['throughput']:.2f} tokens/sec"
        )

    def _print_fmha_summary(self, results, title):
        """打印fmha函数benchmark结果摘要 - 按Absorb值分组显示"""
        print(f"\n{'='*120}")
        print(f"{title} Summary")
        print(f"{'='*120}")

        # 按Absorb值分组
        absorb_groups = {}
        for result in results:
            absorb_val = result["absorb_opt_len"]
            if absorb_val not in absorb_groups:
                absorb_groups[absorb_val] = []
            absorb_groups[absorb_val].append(result)

        # 为每个Absorb值组排序（按执行时间）
        for absorb_val in absorb_groups:
            absorb_groups[absorb_val].sort(key=lambda x: x["mean_time"])

        # 获取所有唯一的参数组合（除了Absorb）
        param_combinations = set()
        for result in results:
            key = (
                result["num_tokens"],
                result["hidden_size"],
                result["page_size"],
                result["batch_size"],
                result["reuse_len"],
            )
            param_combinations.add(key)

        param_combinations = sorted(param_combinations)

        # 打印表头
        print(
            f"{'Rank':<4} {'Tokens':<6} {'Hidden':<6} {'Page':<4} {'Batch':<5} {'Reuse':<5} "
            f"{'Time(ms)_Absorb0':<15} {'Throughput_Absorb0':<18} "
            f"{'Time(ms)_Absorb1':<15} {'Throughput_Absorb1':<18} {'Improvement':<12}"
        )
        print(f"{'-'*120}")

        rank = 1
        for params in param_combinations:
            tokens, hidden, page, batch, reuse = params

            # 获取该参数组合下不同Absorb值的结果
            absorb0_result = None
            absorb1_result = None

            for result in results:
                if (
                    result["num_tokens"] == tokens
                    and result["hidden_size"] == hidden
                    and result["page_size"] == page
                    and result["batch_size"] == batch
                    and result["reuse_len"] == reuse
                ):

                    if result["absorb_opt_len"] == 0:
                        absorb0_result = result
                    elif result["absorb_opt_len"] == 1:
                        absorb1_result = result

            # 格式化输出
            if absorb0_result and absorb1_result:
                time0 = f"{absorb0_result['mean_time']*1000:.3f}"
                throughput0 = f"{absorb0_result['throughput']:.2f}"
                time1 = f"{absorb1_result['mean_time']*1000:.3f}"
                throughput1 = f"{absorb1_result['throughput']:.2f}"

                # 计算性能改进百分比
                improvement = (
                    (absorb0_result["mean_time"] - absorb1_result["mean_time"])
                    / absorb0_result["mean_time"]
                    * 100
                )
                improvement_str = f"{improvement:+.1f}%"

            elif absorb0_result:
                time0 = f"{absorb0_result['mean_time']*1000:.3f}"
                throughput0 = f"{absorb0_result['throughput']:.2f}"
                time1 = "N/A"
                throughput1 = "N/A"
                improvement_str = "N/A"

            elif absorb1_result:
                time0 = "N/A"
                throughput0 = "N/A"
                time1 = f"{absorb1_result['mean_time']*1000:.3f}"
                throughput1 = f"{absorb1_result['throughput']:.2f}"
                improvement_str = "N/A"
            else:
                continue

            print(
                f"{rank:<4} {tokens:<6} {hidden:<6} {page:<4} {batch:<5} {reuse:<5} "
                f"{time0:<15} {throughput0:<18} {time1:<15} {throughput1:<18} {improvement_str:<12}"
            )
            rank += 1

        # 打印统计摘要
        print(f"\n{'='*120}")
        print(f"Performance Comparison Summary")
        print(f"{'='*120}")

        # 计算平均性能改进
        improvements = []
        for params in param_combinations:
            tokens, hidden, page, batch, reuse = params

            absorb0_result = None
            absorb1_result = None

            for result in results:
                if (
                    result["num_tokens"] == tokens
                    and result["hidden_size"] == hidden
                    and result["page_size"] == page
                    and result["batch_size"] == batch
                    and result["reuse_len"] == reuse
                ):

                    if result["absorb_opt_len"] == 0:
                        absorb0_result = result
                    elif result["absorb_opt_len"] == 1:
                        absorb1_result = result

            if absorb0_result and absorb1_result:
                improvement = (
                    (absorb0_result["mean_time"] - absorb1_result["mean_time"])
                    / absorb0_result["mean_time"]
                    * 100
                )
                improvements.append(improvement)

        if improvements:
            avg_improvement = sum(improvements) / len(improvements)
            print(
                f"Average performance improvement (Absorb1 vs Absorb0): {avg_improvement:+.2f}%"
            )
            print(f"Best improvement: {max(improvements):+.2f}%")
            print(f"Worst improvement: {min(improvements):+.2f}%")

        # 按Absorb值分别显示最佳配置
        print(f"\n{'='*120}")
        print(f"Best Configurations by Absorb Value")
        print(f"{'='*120}")

        for absorb_val in sorted(absorb_groups.keys()):
            best_result = absorb_groups[absorb_val][0]  # 最快的
            print(f"\nAbsorb={absorb_val} (Best Performance):")
            print(
                f"  Tokens: {best_result['num_tokens']}, Hidden: {best_result['hidden_size']}, "
                f"Page: {best_result['page_size']}, Batch: {best_result['batch_size']}, "
                f"Reuse: {best_result['reuse_len']}"
            )
            print(
                f"  Time: {best_result['mean_time']*1000:.3f} ms, "
                f"Throughput: {best_result['throughput']:.2f} tokens/sec"
            )

    def _print_fmha_comparison_summary(self, results):
        """打印不同fmha函数的性能比较 - 按Absorb值分组"""
        print(f"\n{'='*140}")
        print(f"FMHA Function Performance Comparison Summary (by Absorb Value)")
        print(f"{'='*140}")

        # 按函数和Absorb值分组
        function_absorb_groups = {}
        for result in results:
            func = result["function"]
            absorb = result["absorb_opt_len"]
            key = (func, absorb)
            if key not in function_absorb_groups:
                function_absorb_groups[key] = []
            function_absorb_groups[key].append(result)

        # 计算每个函数-Absorb组合的平均性能
        print(
            f"{'Function':<20} {'Absorb':<6} {'Avg Time(ms)':<12} {'Avg Throughput':<15} {'Count':<6} {'Description'}"
        )
        print(f"{'-'*140}")

        summary_data = []
        for (func, absorb), group in function_absorb_groups.items():
            avg_time = sum(r["mean_time"] for r in group) / len(group) * 1000
            avg_throughput = sum(r["throughput"] for r in group) / len(group)
            description = group[0]["description"]
            summary_data.append(
                (func, absorb, avg_time, avg_throughput, len(group), description)
            )

        # 按函数名和Absorb值排序
        summary_data.sort(key=lambda x: (x[0], x[1]))

        for func, absorb, avg_time, avg_throughput, count, description in summary_data:
            print(
                f"{func:<20} {absorb:<6} {avg_time:<12.3f} {avg_throughput:<15.2f} {count:<6} {description}"
            )

        # 计算Absorb对每个函数的影响
        print(f"\n{'='*140}")
        print(f"Absorb Impact Analysis")
        print(f"{'='*140}")

        function_impact = {}
        for func in set(r["function"] for r in results):
            absorb0_group = [
                r for r in results if r["function"] == func and r["absorb_opt_len"] == 0
            ]
            absorb1_group = [
                r for r in results if r["function"] == func and r["absorb_opt_len"] == 1
            ]

            if absorb0_group and absorb1_group:
                avg_time0 = sum(r["mean_time"] for r in absorb0_group) / len(
                    absorb0_group
                )
                avg_time1 = sum(r["mean_time"] for r in absorb1_group) / len(
                    absorb1_group
                )
                improvement = (avg_time0 - avg_time1) / avg_time0 * 100
                function_impact[func] = improvement

        print(f"{'Function':<20} {'Absorb Impact':<15} {'Interpretation'}")
        print(f"{'-'*140}")

        for func, impact in function_impact.items():
            if impact > 0:
                interpretation = f"Absorb1 is {impact:.1f}% faster"
            elif impact < 0:
                interpretation = f"Absorb0 is {abs(impact):.1f}% faster"
            else:
                interpretation = "No significant difference"

            print(f"{func:<20} {impact:+.2f}%{'':<8} {interpretation}")


if __name__ == "__main__":
    main()
