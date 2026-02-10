"""
测试 SparseMlaOp 的正确性
"""

import math
from unittest import SkipTest, TestCase, main

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
    SparseMlaOp,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TestParam:
    """测试参数配置"""

    def __init__(
        self,
        num_tokens: int,
        total_cache_len: int,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        page_size: int,
        top_k: int,
        batch_size: int = 1,
        softmax_extra_scale: float = 1.0,
        seed: int = 42,
        check_correctness: bool = True,
    ):
        self.num_tokens = num_tokens
        self.total_cache_len = total_cache_len
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.page_size = page_size
        self.top_k = top_k
        self.batch_size = batch_size
        self.softmax_extra_scale = softmax_extra_scale
        self.seed = seed
        self.check_correctness = check_correctness

    def __str__(self):
        return (
            f"TestParam(num_tokens={self.num_tokens}, "
            f"total_cache_len={self.total_cache_len}, "
            f"num_heads={self.num_heads}, "
            f"kv_lora_rank={self.kv_lora_rank}, "
            f"qk_head_dim={self.qk_head_dim}, "
            f"page_size={self.page_size}, "
            f"top_k={self.top_k}, "
            f"batch_size={self.batch_size})"
        )


class Testcase:
    """测试用例数据"""

    def __init__(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        topk_indices: torch.Tensor,
        block_table: torch.Tensor,
        mla_params: rtp_llm_ops.FlashInferMlaAttnParams,
        scale: float,
    ):
        self.q = q
        self.kv = kv
        self.topk_indices = topk_indices
        self.block_table = block_table
        self.mla_params = mla_params
        self.scale = scale


def generate_block_table(
    batch_size: int, total_cache_len: int, page_size: int
) -> torch.Tensor:
    """生成 block table"""
    num_blocks_per_seq = math.ceil(total_cache_len / page_size)

    # 创建 block table: [batch_size, num_blocks_per_seq]
    block_table = torch.zeros(
        [batch_size, num_blocks_per_seq],
        dtype=torch.int32,
        device=torch.device("cpu"),
    )

    bias = 0
    for i in range(batch_size):
        block_table[i, :] = torch.arange(
            bias,
            bias + num_blocks_per_seq,
            dtype=torch.int32,
            device=torch.device("cpu"),
        )
        bias += num_blocks_per_seq

    return block_table


def generate_testcase(p: TestParam) -> Testcase:
    """生成测试用例"""
    set_seed(p.seed)
    device = torch.device("cuda")

    # 生成 Q: [num_tokens, num_heads, qk_head_dim]
    q = (
        torch.randn(
            [p.num_tokens, p.num_heads, p.qk_head_dim],
            dtype=torch.bfloat16,
            device=device,
        )
        / 10.0
    )
    q.clamp_(-10, 10)

    # 生成 KV cache: [total_cache_len, num_kv_heads, d_qk]
    # 对于 MLA sparse attention，KV 的最后一个维度是 d_qk
    # 对于 MLA，通常 num_kv_heads = 1
    kv = (
        torch.randn(
            [p.total_cache_len, 1, p.qk_head_dim],
            dtype=torch.bfloat16,
            device=device,
        )
        / 10.0
    )
    kv.clamp_(-10, 10)

    # 生成 topk_indices: [num_tokens, top_k] (request-local indices)
    # 注意：每个 token 一行，所有 head 共享相同的 topk_indices
    # 随机选择 top_k 个位置
    topk_indices_2d = torch.randint(
        0,
        p.total_cache_len // p.batch_size,  # request-local range
        [p.num_tokens, p.top_k],
        dtype=torch.int32,
        device=device,
    )

    # 扩展为 3D: [num_tokens, h_kv, top_k]，其中 h_kv=1 for MLA
    topk_indices = topk_indices_2d.unsqueeze(1)  # [num_tokens, 1, topk]

    # 生成 block table
    block_table_host = generate_block_table(
        p.batch_size, p.total_cache_len // p.batch_size, p.page_size
    )
    block_table_device = block_table_host.to(device)

    # 创建 MLA params
    mla_params = rtp_llm_ops.FlashInferMlaAttnParams()

    # 根据是否是 decode 阶段来设置参数
    if p.num_tokens == p.batch_size:
        # Decode 阶段: 每个 batch 只有一个 token
        sequence_lengths = torch.tensor(
            [p.total_cache_len // p.batch_size] * p.batch_size,
            dtype=torch.int32,
            device=torch.device("cpu"),
        )
        input_lengths = torch.ones(
            p.batch_size, dtype=torch.int32, device=torch.device("cpu")
        )
        # Decode 阶段没有 prefix
        prefix_lengths = torch.tensor([], dtype=torch.int32, device=torch.device("cpu"))
    else:
        # Prefill 阶段: 每个 batch 的 input_lengths 不同
        sequence_lengths = torch.tensor(
            [p.total_cache_len // p.batch_size] * p.batch_size,
            dtype=torch.int32,
            device=torch.device("cpu"),
        )
        # 为每个 batch 生成不同的 input_lengths，总和等于 num_tokens
        base_tokens = p.num_tokens // p.batch_size
        remainder = p.num_tokens % p.batch_size
        input_lengths = torch.tensor(
            [base_tokens + (1 if i < remainder else 0) for i in range(p.batch_size)],
            dtype=torch.int32,
            device=torch.device("cpu"),
        )

        prefix_lengths = torch.zeros(
            p.batch_size, dtype=torch.int32, device=torch.device("cpu")
        )

    mla_params.fill_params(
        prefix_lengths,
        sequence_lengths,
        input_lengths,
        block_table_host,
        p.page_size,
    )

    scale = (p.qk_head_dim**-0.5) * p.softmax_extra_scale

    return Testcase(q, kv, topk_indices, block_table_device, mla_params, scale)


def ref_sparse_mla_forward(
    q: torch.Tensor,
    kv: torch.Tensor,
    topk_indices_global: torch.Tensor,
    scale: float,
    kv_lora_rank: int,
) -> torch.Tensor:
    """
    参考实现：使用 PyTorch 实现 sparse MLA attention

    Args:
        q: [num_tokens, num_heads, qk_head_dim]
        kv: [total_cache_len, h_kv, d_qk]
        topk_indices_global: [num_tokens, top_k] - 全局索引，所有 head 共享
        scale: softmax scale
        kv_lora_rank: KV lora rank (output dimension)

    Returns:
        output: [num_tokens, num_heads, kv_lora_rank]
    """
    num_tokens, num_heads, qk_head_dim = q.shape
    top_k = topk_indices_global.shape[1]

    # 转换为 float32 以提高精度
    q_fp32 = q.float()
    kv_fp32 = kv.float()

    # Squeeze h_kv dimension if it's 1
    if kv_fp32.shape[1] == 1:
        kv_fp32 = kv_fp32.squeeze(1)  # [total_cache_len, d_qk]

    # Gather KV: [num_tokens * top_k, d_qk]
    indices_flat = topk_indices_global.reshape(-1)  # [num_tokens * top_k]

    # 处理无效索引
    invalid_mask = (indices_flat < 0) | (indices_flat >= kv_fp32.shape[0])
    indices_flat_clamped = torch.clamp(indices_flat, 0, kv_fp32.shape[0] - 1)

    gathered_kv = kv_fp32.index_select(
        0, indices_flat_clamped
    )  # [num_tokens * top_k, d_qk]
    gathered_kv = gathered_kv.view(
        num_tokens, 1, top_k, -1
    )  # [num_tokens, 1, top_k, d_qk]
    gathered_kv = gathered_kv.expand(
        num_tokens, num_heads, top_k, -1
    )  # [num_tokens, num_heads, top_k, d_qk]

    # 计算注意力分数: Q @ K^T
    # q: [num_tokens, num_heads, qk_head_dim]
    # gathered_kv: [num_tokens, num_heads, top_k, d_qk]
    attn_scores = torch.matmul(
        q_fp32.unsqueeze(2), gathered_kv.transpose(-1, -2)
    )  # [num_tokens, num_heads, 1, top_k]
    attn_scores = attn_scores.squeeze(2) * scale  # [num_tokens, num_heads, top_k]

    # 将无效索引的分数设为 -inf
    invalid_mask_2d = invalid_mask.view(num_tokens, 1, top_k).expand(
        num_tokens, num_heads, top_k
    )
    attn_scores[invalid_mask_2d] = float("-inf")

    # Softmax
    attn_weights = torch.softmax(attn_scores, dim=-1)  # [num_tokens, num_heads, top_k]

    # 处理全为 -inf 的情况
    attn_weights = torch.nan_to_num(attn_weights, 0.0)

    # 计算输出: attn_weights @ V
    # attn_weights: [num_tokens, num_heads, top_k]
    # gathered_kv: [num_tokens, num_heads, top_k, d_qk]
    # 只取 V 部分 (前 kv_lora_rank 维)
    gathered_v = gathered_kv[
        ..., :kv_lora_rank
    ]  # [num_tokens, num_heads, top_k, kv_lora_rank]

    output = torch.matmul(
        attn_weights.unsqueeze(2), gathered_v
    )  # [num_tokens, num_heads, 1, kv_lora_rank]
    output = output.squeeze(2)  # [num_tokens, num_heads, kv_lora_rank]

    return output.to(q.dtype)


class SparseMlaOpTest(TestCase):
    """SparseMlaOp 测试类"""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device(torch.device("cuda"))

    def _run_test(self, p: TestParam):
        """运行单个测试"""
        print(f"\n{'='*80}")
        print(f"Running test: {p}")
        print(f"{'='*80}")

        # 生成测试数据
        torch.cuda.empty_cache()
        testcase = generate_testcase(p)

        # 创建 SparseMlaOp
        sparse_mla_op = SparseMlaOp(
            num_heads=p.num_heads,
            kv_lora_rank=p.kv_lora_rank,
            qk_rope_head_dim=p.qk_rope_head_dim,
            qk_nope_head_dim=p.qk_nope_head_dim,
            page_size=p.page_size,
            softmax_extra_scale=p.softmax_extra_scale,
            top_k=p.top_k,
        )

        # Plan
        sparse_mla_op.plan(testcase.mla_params, testcase.block_table)

        # 执行前向传播
        def run_forward():
            return sparse_mla_op.forward(testcase.q, testcase.kv, testcase.topk_indices)

        output = run_forward()
        torch.cuda.synchronize()

        # 正确性检查
        if p.check_correctness:
            # 将 request-local indices 转换为 global indices 用于参考实现
            # 注意: 这里直接调用内部方法仅用于测试目的
            global_indices = sparse_mla_op._convert_topk_indices_to_global(  # type: ignore[reportPrivateUsage]
                testcase.topk_indices
            )
            # global_indices 是 [num_tokens, h_kv, top_k] 格式，其中 h_kv=1
            # 参考实现需要 2D [num_tokens, top_k]
            global_indices_2d = global_indices[:, 0, :]

            # 运行参考实现
            ref_output = ref_sparse_mla_forward(
                testcase.q,
                testcase.kv,
                global_indices_2d,
                testcase.scale,
                p.kv_lora_rank,
            )
            torch.cuda.synchronize()

            # 比较结果
            output_norm = output / (torch.norm(output) + 1e-8)
            ref_output_norm = ref_output / (torch.norm(ref_output) + 1e-8)

            # 计算各种误差指标
            abs_diff = torch.abs(output - ref_output)
            rel_diff = abs_diff / (torch.abs(ref_output) + 1e-8)
            max_abs_error = torch.max(abs_diff).item()
            max_rel_error = torch.max(rel_diff).item()
            mean_abs_error = torch.mean(abs_diff).item()

            # 余弦相似度
            output_flat = output.flatten()
            ref_output_flat = ref_output.flatten()
            cosine_sim = F.cosine_similarity(
                output_flat.unsqueeze(0), ref_output_flat.unsqueeze(0), dim=1
            ).item()

            print(f"\n正确性检查:")
            print(f"  Max absolute error: {max_abs_error:.6f}")
            print(f"  Max relative error: {max_rel_error:.6f}")
            print(f"  Mean absolute error: {mean_abs_error:.6f}")
            print(f"  Cosine similarity: {cosine_sim:.6f}")

            # 断言
            self.assertTrue(
                torch.allclose(output_norm, ref_output_norm, atol=1e-2, rtol=1e-2),
                f"Output mismatch! Max abs error: {max_abs_error}, Max rel error: {max_rel_error}",
            )
            self.assertGreater(
                cosine_sim,
                0.99,
                f"Cosine similarity too low: {cosine_sim}",
            )

            print(f"✓ Test passed!")
            return True
        else:
            return True

    def test_sparse_mla_op_prefill(self):
        """
        测试 Prefill 阶段的 Sparse MLA 操作

        测试场景包括：
        1. 不同的序列长度 (num_tokens, total_cache_len)
        2. 不同的 top_k 配置 (128, 256, 384)
        3. 不同的 batch_size (1, 2)
        4. 不同的 num_heads (64, 128) - FlashMLA 支持 64 或 128
        5. 不同的维度配置 (d_qk=512 或 576, kv_lora_rank=512)

        Prefill 特点:
        - 每个 batch 的 input_lengths 不同
        - prefix_lengths 为全 0
        """
        test_cases = [
            # (num_tokens, total_cache_len, top_k, batch_size, num_heads, qk_nope_dim, kv_lora_rank, 描述)
            # 基础配置测试
            (7, 128, 128, 1, 64, 448, 512, "小规模单batch"),
            (128, 2048, 128, 2, 64, 448, 512, "中等规模多batch"),
            (64, 4096, 256, 1, 128, 448, 512, "大 top_k 配置"),
            # 不规则形状测试
            (7, 592, 128, 1, 64, 448, 512, "不规则形状1"),
            (62, 1840, 256, 1, 64, 448, 512, "不规则形状2"),
            (213, 1592, 384, 1, 64, 448, 512, "不规则形状3"),
            # 不同维度配置测试
            (32, 512, 128, 1, 64, 448, 512, "d_qk=512配置"),
            (32, 512, 128, 1, 64, 512, 512, "d_qk=576配置"),
        ]

        for (
            num_tokens,
            total_cache_len,
            top_k,
            batch_size,
            num_heads,
            qk_nope_dim,
            kv_lora_rank,
            desc,
        ) in test_cases:
            with self.subTest(desc=desc):
                p = TestParam(
                    num_tokens=num_tokens,
                    total_cache_len=total_cache_len,
                    num_heads=num_heads,
                    kv_lora_rank=kv_lora_rank,
                    qk_rope_head_dim=64,
                    qk_nope_head_dim=qk_nope_dim,
                    page_size=64,
                    top_k=top_k,
                    batch_size=batch_size,
                )
                self._run_test(p)

    def test_sparse_mla_op_decode(self):
        """
        测试 Decode 阶段的 Sparse MLA 操作

        测试场景包括：
        1. 不同的 batch_size (1, 2, 4)
        2. 不同的 cache 长度
        3. 不同的 top_k 配置
        4. 不同的 num_heads (64, 128)

        Decode 特点:
        - block_table.size(0) == num_tokens
        - 每个 batch 只有一个 token (num_tokens == batch_size)
        - prefix_lengths.size(0) = 0 (空张量)
        """
        test_cases = [
            # (batch_size, total_cache_len, top_k, num_heads, qk_nope_dim, kv_lora_rank, 描述)
            (1, 128, 128, 64, 448, 512, "单batch解码"),
            (2, 512, 128, 64, 448, 512, "2batch解码"),
            (4, 1024, 256, 64, 448, 512, "4batch解码"),
            (1, 2048, 256, 128, 448, 512, "大cache解码"),
            (2, 4096, 384, 64, 448, 512, "大top_k解码"),
            (1, 512, 128, 64, 512, 512, "d_qk=576解码"),
        ]

        for (
            batch_size,
            total_cache_len,
            top_k,
            num_heads,
            qk_nope_dim,
            kv_lora_rank,
            desc,
        ) in test_cases:
            with self.subTest(desc=desc):
                p = TestParam(
                    num_tokens=batch_size,  # Decode: num_tokens == batch_size
                    total_cache_len=total_cache_len,
                    num_heads=num_heads,
                    kv_lora_rank=kv_lora_rank,
                    qk_rope_head_dim=64,
                    qk_nope_head_dim=qk_nope_dim,
                    page_size=64,
                    top_k=top_k,
                    batch_size=batch_size,
                )
                self._run_test(p)


if __name__ == "__main__":
    # 设置默认设备
    device = torch.device("cuda:0")
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_default_dtype(torch.bfloat16)

    main()
