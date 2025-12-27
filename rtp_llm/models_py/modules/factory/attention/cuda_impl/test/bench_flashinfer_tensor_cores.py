"""
性能测试脚本：比较不同配置下 use_tensor_cores=True/False 的性能差异

测试维度：
- head_num (num_qo_heads)
- head_num_kv (num_kv_heads)
- head_dim
- use_tensor_cores (True/False)

输出指标：
- 执行时间 (ms)
- 内存带宽 (GB/s)
"""

import csv
from datetime import datetime

import flashinfer
import numpy as np
import torch
from bench_utils import bench_gpu_time


def bench_single_case(
    wrapper,
    batch_size,
    seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_block_size,
    q_dtype,
    kv_dtype,
):
    """
    使用已创建的wrapper对单个batch+seq配置进行测试

    Args:
        wrapper: 已创建的BatchDecodeWithPagedKVCacheWrapper
        batch_size: 批次大小
        seq_len: 序列长度
        num_qo_heads: Query/Output头数量
        num_kv_heads: Key/Value头数量
        head_dim: 每个头的维度
        page_block_size: 页块大小
        q_dtype: Query数据类型
        kv_dtype: Key/Value数据类型

    Returns:
        tuple: (time_ms, bandwidth_gbps, status)
    """
    try:
        np.random.seed(42)

        # 准备输入数据
        seq_lens = torch.full((batch_size,), seq_len)
        seq_lens_blocks = torch.ceil(seq_lens / page_block_size).int()
        kv_indptr = torch.cat(
            [torch.tensor([0]), torch.cumsum(seq_lens_blocks, 0)], dim=0
        )
        kv_indptr = kv_indptr.int()
        last_page_len = seq_lens - (seq_lens_blocks - 1) * page_block_size
        last_page_len = last_page_len.int()
        num_blocks = int(kv_indptr[-1].item())

        # 创建Query和KV缓存数据
        q = torch.rand(
            batch_size, num_qo_heads, head_dim, dtype=q_dtype, device="cuda:0"
        )
        kv_data = torch.randn(
            num_blocks, 2, page_block_size, num_kv_heads, head_dim, device="cuda:0"
        ).to(kv_dtype)

        # Plan阶段
        wrapper.plan(
            kv_indptr.to(0),
            torch.arange(num_blocks).int().to(0),
            last_page_len.to(0),
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_block_size,
            data_type=kv_dtype,
            q_data_type=q_dtype,
        )

        # 性能测试
        measurements = bench_gpu_time(lambda: wrapper.run(q, kv_data))
        ms = np.median(measurements)

        # 计算IO量（bytes）
        io_bytes = (
            q.numel() * q.element_size() + kv_data.numel() * kv_data.element_size()
        )
        # 转换为GB/s
        bandwidth_gbps = io_bytes / (ms * 1e-3) / (1024**3)

        return ms, bandwidth_gbps, "success"

    except Exception as e:
        return None, None, f"error: {str(e)}"


def bench_config_group(
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_block_size,
    use_tensor_cores,
    batch_sizes,
    seq_lens,
    q_dtype,
    kv_dtype,
):
    """
    为一个head配置+tensor_cores设置创建wrapper，然后跑全部batch+seq组合

    Args:
        num_qo_heads: Query/Output头数量
        num_kv_heads: Key/Value头数量
        head_dim: 每个头的维度
        page_block_size: 页块大小
        use_tensor_cores: 是否使用Tensor Cores
        batch_sizes: 批次大小列表
        seq_lens: 序列长度列表
        q_dtype: Query数据类型
        kv_dtype: Key/Value数据类型

    Returns:
        list: 结果列表
    """
    results = []

    # 创建workspace buffer和wrapper（复用）
    workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"
    )

    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD", use_tensor_cores=use_tensor_cores
    )

    # 遍历所有batch和seq组合
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            time_ms, bandwidth_gbps, status = bench_single_case(
                wrapper,
                batch_size,
                seq_len,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                page_block_size,
                q_dtype,
                kv_dtype,
            )

            results.append(
                {
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "num_qo_heads": num_qo_heads,
                    "num_kv_heads": num_kv_heads,
                    "head_dim": head_dim,
                    "page_block_size": page_block_size,
                    "use_tensor_cores": use_tensor_cores,
                    "q_dtype": str(q_dtype),
                    "kv_dtype": str(kv_dtype),
                    "time_ms": time_ms,
                    "bandwidth_gbps": bandwidth_gbps,
                    "status": status,
                }
            )

    return results


def run_benchmark_suite():
    """
    运行完整的基准测试套件
    """
    # 测试配置
    test_configs = {
        "head_configs": [
            # (num_qo_heads, num_kv_heads, head_dim)
            (32, 16, 128),
            (32, 8, 128),
            (32, 4, 128),
        ],
        "batch_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        "seq_lens": [512, 1024, 2048, 4096, 8192, 16384],
        "page_block_size": 64,
        "use_tensor_cores_options": [True, False],
        "dtype": torch.bfloat16,
    }

    all_results = []
    total_configs = len(test_configs["head_configs"]) * len(
        test_configs["use_tensor_cores_options"]
    )
    tests_per_config = len(test_configs["batch_sizes"]) * len(test_configs["seq_lens"])

    print(f"开始运行基准测试...")
    print(f"配置组数: {total_configs}")
    print(f"每组测试数: {tests_per_config}")
    print(f"总测试数: {total_configs * tests_per_config}")
    print("=" * 80)

    config_count = 0
    for num_qo_heads, num_kv_heads, head_dim in test_configs["head_configs"]:
        for use_tensor_cores in test_configs["use_tensor_cores_options"]:
            config_count += 1
            tc_str = "ON" if use_tensor_cores else "OFF"
            print(
                f"\n[配置 {config_count}/{total_configs}] QH={num_qo_heads}, KVH={num_kv_heads}, "
                f"HD={head_dim}, TensorCores={tc_str}"
            )
            print(f"运行 {tests_per_config} 个测试...")

            # 为这个配置创建wrapper并运行所有batch+seq组合
            results = bench_config_group(
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                page_block_size=test_configs["page_block_size"],
                use_tensor_cores=use_tensor_cores,
                batch_sizes=test_configs["batch_sizes"],
                seq_lens=test_configs["seq_lens"],
                q_dtype=test_configs["dtype"],
                kv_dtype=test_configs["dtype"],
            )

            all_results.extend(results)

            # 统计成功率
            success_count = sum(1 for r in results if r["status"] == "success")
            print(f"完成: {success_count}/{len(results)} 测试成功")

    return all_results


def save_results(results, filename=None):
    """
    保存结果到CSV文件

    Args:
        results: 结果列表
        filename: 输出文件名，如果为None则自动生成
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"flashinfer_tensor_cores_benchmark_{timestamp}.csv"

    fieldnames = [
        "batch_size",
        "seq_len",
        "num_qo_heads",
        "num_kv_heads",
        "head_dim",
        "page_block_size",
        "use_tensor_cores",
        "q_dtype",
        "kv_dtype",
        "time_ms",
        "bandwidth_gbps",
        "status",
    ]

    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n结果已保存到: {filename}")


def save_comparison_markdown(results, filename=None):
    """
    保存对比表格到Markdown文件

    Args:
        results: 结果列表
        filename: 输出文件名，如果为None则自动生成
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"flashinfer_tensor_cores_comparison_{timestamp}.md"

    with open(filename, "w", encoding="utf-8") as f:
        f.write("# FlashInfer Tensor Cores 性能对比\n\n")

        # 按head配置分组
        head_configs = {}
        for r in results:
            if r["status"] != "success":
                continue
            head_key = (r["num_qo_heads"], r["num_kv_heads"], r["head_dim"])
            if head_key not in head_configs:
                head_configs[head_key] = {}

            test_key = (r["batch_size"], r["seq_len"], r["use_tensor_cores"])
            head_configs[head_key][test_key] = r

        all_speedups = []
        for head_key in sorted(head_configs.keys()):
            num_qo_heads, num_kv_heads, head_dim = head_key
            f.write(
                f"\n## 配置: QH={num_qo_heads}, KVH={num_kv_heads}, HD={head_dim}\n\n"
            )

            # 获取所有batch_size和seq_len组合
            batch_seq_pairs = set()
            for test_key in head_configs[head_key].keys():
                batch_size, seq_len, _ = test_key
                batch_seq_pairs.add((batch_size, seq_len))

            # Markdown表格
            f.write(
                "| Batch Size | Seq Len | TC=ON (ms) | TC=ON (GB/s) | TC=OFF (ms) | TC=OFF (GB/s) | 加速比 |\n"
            )
            f.write(
                "|------------|---------|------------|--------------|-------------|---------------|--------|\n"
            )

            config_speedups = []
            for batch_size, seq_len in sorted(batch_seq_pairs):
                key_on = (batch_size, seq_len, True)
                key_off = (batch_size, seq_len, False)

                if (
                    key_on in head_configs[head_key]
                    and key_off in head_configs[head_key]
                ):
                    r_on = head_configs[head_key][key_on]
                    r_off = head_configs[head_key][key_off]

                    time_on = r_on["time_ms"]
                    time_off = r_off["time_ms"]
                    bw_on = r_on["bandwidth_gbps"]
                    bw_off = r_off["bandwidth_gbps"]
                    speedup = time_off / time_on

                    config_speedups.append(speedup)
                    all_speedups.append(speedup)

                    f.write(
                        f"| {batch_size} | {seq_len} | {time_on:.4f} | {bw_on:.2f} | "
                        f"{time_off:.4f} | {bw_off:.2f} | {speedup:.2f}x |\n"
                    )

            # 统计信息
            if config_speedups:
                f.write(
                    f"\n**统计**: 平均={np.mean(config_speedups):.2f}x, "
                    f"中位数={np.median(config_speedups):.2f}x, "
                    f"最大={np.max(config_speedups):.2f}x, "
                    f"最小={np.min(config_speedups):.2f}x\n"
                )

        # 全局统计
        if all_speedups:
            f.write("\n## 全局统计\n\n")
            f.write(f"- 总测试数: {len(all_speedups)}\n")
            f.write(f"- 平均加速比: {np.mean(all_speedups):.2f}x\n")
            f.write(f"- 中位数加速比: {np.median(all_speedups):.2f}x\n")
            f.write(f"- 最大加速比: {np.max(all_speedups):.2f}x\n")
            f.write(f"- 最小加速比: {np.min(all_speedups):.2f}x\n")

    print(f"对比表格已保存到: {filename}")


def print_comparison_summary(results):
    """
    打印对比总结，按head配置分组，展示不同batch size下的TensorCores性能对比
    """
    print("\n" + "=" * 120)
    print("性能对比总结 (Tensor Cores ON vs OFF)")
    print("=" * 120)

    # 按head配置分组
    head_configs = {}
    for r in results:
        if r["status"] != "success":
            continue
        head_key = (r["num_qo_heads"], r["num_kv_heads"], r["head_dim"])
        if head_key not in head_configs:
            head_configs[head_key] = {}

        # 键为 (batch_size, seq_len, use_tensor_cores)
        test_key = (r["batch_size"], r["seq_len"], r["use_tensor_cores"])
        head_configs[head_key][test_key] = r

    # 为每个head配置打印表格
    all_speedups = []
    for head_key in sorted(head_configs.keys()):
        num_qo_heads, num_kv_heads, head_dim = head_key
        print(
            f"\n配置: num_qo_heads={num_qo_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}"
        )
        print("-" * 120)

        # 获取所有batch_size和seq_len组合
        batch_seq_pairs = set()
        for test_key in head_configs[head_key].keys():
            batch_size, seq_len, _ = test_key
            batch_seq_pairs.add((batch_size, seq_len))

        # 打印表头
        print(
            f"{'Batch Size':<12} {'Seq Len':<10} {'TC=ON (ms)':<15} {'TC=ON (GB/s)':<15} "
            f"{'TC=OFF (ms)':<15} {'TC=OFF (GB/s)':<15} {'加速比':<10}"
        )
        print("-" * 120)

        config_speedups = []
        for batch_size, seq_len in sorted(batch_seq_pairs):
            key_on = (batch_size, seq_len, True)
            key_off = (batch_size, seq_len, False)

            if key_on in head_configs[head_key] and key_off in head_configs[head_key]:
                r_on = head_configs[head_key][key_on]
                r_off = head_configs[head_key][key_off]

                time_on = r_on["time_ms"]
                time_off = r_off["time_ms"]
                bw_on = r_on["bandwidth_gbps"]
                bw_off = r_off["bandwidth_gbps"]
                speedup = time_off / time_on

                config_speedups.append(speedup)
                all_speedups.append(speedup)

                print(
                    f"{batch_size:<12} {seq_len:<10} {time_on:<15.4f} {bw_on:<15.2f} "
                    f"{time_off:<15.4f} {bw_off:<15.2f} {speedup:<10.2f}x"
                )

        # 打印该配置的统计信息
        if config_speedups:
            print("-" * 120)
            print(
                f"该配置统计: 平均加速比={np.mean(config_speedups):.2f}x, "
                f"中位数={np.median(config_speedups):.2f}x, "
                f"最大={np.max(config_speedups):.2f}x, "
                f"最小={np.min(config_speedups):.2f}x"
            )

    # 打印总体统计
    if all_speedups:
        print("\n" + "=" * 120)
        print("全局统计:")
        print(f"  总测试数: {len(all_speedups)}")
        print(f"  平均加速比: {np.mean(all_speedups):.2f}x")
        print(f"  中位数加速比: {np.median(all_speedups):.2f}x")
        print(f"  最大加速比: {np.max(all_speedups):.2f}x")
        print(f"  最小加速比: {np.min(all_speedups):.2f}x")
        print("=" * 120)


if __name__ == "__main__":
    print("FlashInfer Tensor Cores 性能测试")
    print("=" * 80)

    # 运行基准测试
    results = run_benchmark_suite()

    # # 保存原始结果到CSV
    # save_results(results)

    # # 保存对比表格到Markdown
    # save_comparison_markdown(results)

    # 打印对比总结
    print_comparison_summary(results)

    print("\n测试完成！")
