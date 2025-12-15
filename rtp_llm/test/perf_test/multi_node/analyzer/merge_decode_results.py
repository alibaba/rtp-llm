#!/usr/bin/env python3
"""
合并多个Decode_Result.json或Prefill_Result.json文件到一个表格

功能：
1. 扫描指定目录下的所有Result文件
2. 从目录结构中提取dp_size或其他参数（如hack_layer_num）
3. 加载每个JSON文件的metrics数据
4. 整合所有结果到一个CSV文件
5. 打印制表符格式的表格（可直接复制到Excel）

目录结构示例：
Benchmark_dir/
  ├── Task_dp_size-4/
  │   └── Decode_Result.json
  ├── Task_dp_size-8/
  │   └── Prefill_Result.json
  └── Task_dp_size-16/
      └── Decode_Result.json
"""

import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional


def extract_task_params_from_path(task_dir_name: str) -> dict:
    """
    从Task目录名中提取参数

    支持的格式：
    - Task_dp_size-8 -> {'dp_size': 8}
    - Task_hack_layer_num-20 -> {'hack_layer_num': 20}
    - 可扩展其他参数

    返回：参数字典，如果无法解析则返回 {'task_name': 原始目录名}
    """
    params = {}

    # 尝试匹配dp_size
    match = re.search(r"dp_size-(\d+)", task_dir_name)
    if match:
        params["dp_size"] = int(match.group(1))

    # 尝试匹配hack_layer_num
    match = re.search(r"hack_layer_num-(\d+)", task_dir_name)
    if match:
        params["hack_layer_num"] = int(match.group(1))

    if params:
        return params

    # 通用匹配：任何 key-value 模式
    match = re.search(r"(\w+)-(\d+)", task_dir_name)
    if match:
        key = match.group(1)
        value = int(match.group(2))
        params[key] = value
        return params

    # 无法解析，返回原始目录名
    return {"task_name": task_dir_name}


def scan_result_files(benchmark_dir: str) -> list:
    """
    扫描benchmark目录下的所有Decode_Result.json或Prefill_Result.json文件

    返回：[(task_params, json_file_path), ...]
    例如：[({'dp_size': 4}, '/path/to/Task_dp_size-4/Decode_Result.json'), ...]
    """
    benchmark_path = Path(benchmark_dir)
    result_files = []

    # 遍历所有Task_*目录
    for task_dir in benchmark_path.iterdir():
        if not task_dir.is_dir() or not task_dir.name.startswith("Task_"):
            continue

        # 提取任务参数
        task_params = extract_task_params_from_path(task_dir.name)

        # 查找Decode_Result.json
        decode_result_file = task_dir / "Decode_Result.json"
        if decode_result_file.exists():
            result_files.append((task_params, str(decode_result_file)))

        # 查找Prefill_Result.json
        prefill_result_file = task_dir / "Prefill_Result.json"
        if prefill_result_file.exists():
            result_files.append((task_params, str(prefill_result_file)))

        if not decode_result_file.exists() and not prefill_result_file.exists():
            print(f"警告: {task_dir.name} 没有Result文件")
            continue

    # 排序：优先按dp_size，然后按hack_layer_num，最后按task_name
    def sort_key(item):
        params = item[0]
        return (
            params.get("dp_size", 999999),
            params.get("hack_layer_num", 999999),
            params.get("task_name", ""),
        )

    result_files.sort(key=sort_key)

    return result_files


def load_result_json(json_file: str) -> Optional[dict]:
    """
    加载Result.json文件

    返回：{'title': str, 'metrics': [...]}
    """
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"错误: 无法加载 {json_file} - {e}")
        return None


def merge_all_results(result_files: list) -> list:
    """
    合并所有Result.json文件的数据

    返回：[(task_params, metrics), ...]
    其中metrics是一个列表，每个元素包含：
    {input_len, batch_size, success_rate, avg_wait_time, avg_prefill_time, avg_decode_time}
    """
    merged_data = []

    print(f"\n开始加载 {len(result_files)} 个Result文件...")
    print("=" * 80)

    for idx, (task_params, json_file) in enumerate(result_files, 1):
        # 构建任务描述
        task_desc = ", ".join(f"{k}={v}" for k, v in task_params.items())
        print(f"\n[{idx}/{len(result_files)}] 加载: {task_desc}")
        print(f"  文件: {Path(json_file).parent.name}/{Path(json_file).name}")

        # 加载JSON文件
        data = load_result_json(json_file)

        if data and "metrics" in data:
            metrics = data["metrics"]
            merged_data.append((task_params, metrics))
            print(f"  ✓ 成功: 包含 {len(metrics)} 条记录")
        else:
            print(f"  ✗ 失败: 无有效数据")

    print("\n" + "=" * 80)
    print(f"完成! 成功加载 {len(merged_data)}/{len(result_files)} 个文件")

    return merged_data


def save_merged_csv(merged_data: list, output_file: str):
    """
    将合并的结果保存到CSV文件

    CSV格式：
    task_param_1, task_param_2, ..., input_len, batch_size, success_rate,
    avg_wait_time, avg_prefill_time, avg_decode_time
    """
    if not merged_data:
        print("\n没有数据需要保存")
        return

    # 收集所有可能的任务参数键
    all_param_keys = set()
    for task_params, _ in merged_data:
        all_param_keys.update(task_params.keys())

    # 排序参数键（dp_size优先，然后其他参数）
    param_keys = sorted(
        all_param_keys,
        key=lambda k: (0 if k == "dp_size" else 1 if k == "hack_layer_num" else 2, k),
    )

    # 定义CSV字段
    fieldnames = param_keys + [
        "input_len",
        "batch_size",
        "success_rate",
        "avg_wait_time",
        "avg_prefill_time",
        "avg_decode_time",
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 写入所有数据
        for task_params, metrics in merged_data:
            for metric in metrics:
                row = {}
                # 添加任务参数
                for key in param_keys:
                    row[key] = task_params.get(key, "")
                # 添加metric数据
                row.update(
                    {
                        "input_len": metric.get("input_len", ""),
                        "batch_size": metric.get("batch_size", ""),
                        "success_rate": metric.get("success_rate", ""),
                        "avg_wait_time": f"{metric.get('avg_wait_time', 0):.6f}",
                        "avg_prefill_time": f"{metric.get('avg_prefill_time', 0):.6f}",
                        "avg_decode_time": f"{metric.get('avg_decode_time', 0):.6f}",
                    }
                )
                writer.writerow(row)

    file_size = Path(output_file).stat().st_size / 1024
    print(f"\n✓ 合并结果已保存到: {output_file}")
    print(f"  - 包含 {len(merged_data)} 个任务的数据")
    total_rows = sum(len(metrics) for _, metrics in merged_data)
    print(f"  - 总共 {total_rows} 行数据")
    print(f"  - 文件大小: {file_size:.2f} KB")


def print_summary_table(merged_data: list):
    """
    打印汇总表格（制表符格式，可直接复制到Excel）
    """
    if not merged_data:
        print("\n没有数据需要打印")
        return

    print("\n" + "=" * 100)
    print("合并结果汇总表（可直接复制到Excel）")
    print("=" * 100)
    print()

    # 收集所有任务参数键
    all_param_keys = set()
    for task_params, _ in merged_data:
        all_param_keys.update(task_params.keys())

    param_keys = sorted(
        all_param_keys,
        key=lambda k: (0 if k == "dp_size" else 1 if k == "hack_layer_num" else 2, k),
    )

    # 打印表头
    headers = param_keys + [
        "input_len",
        "batch_size",
        "success_rate",
        "avg_wait_time",
        "avg_prefill_time",
        "avg_decode_time",
    ]
    print("\t".join(headers))

    # 打印数据行
    for task_params, metrics in merged_data:
        for metric in metrics:
            row = []
            # 添加任务参数
            for key in param_keys:
                row.append(str(task_params.get(key, "")))
            # 添加metric数据
            row.extend(
                [
                    str(metric.get("input_len", "")),
                    str(metric.get("batch_size", "")),
                    f"{metric.get('success_rate', 0):.4f}",
                    f"{metric.get('avg_wait_time', 0):.6f}",
                    f"{metric.get('avg_prefill_time', 0):.6f}",
                    f"{metric.get('avg_decode_time', 0):.6f}",
                ]
            )
            print("\t".join(row))

    print()
    print("=" * 100)
    print("提示：选中上面的内容，复制后可直接粘贴到Excel")
    print("=" * 100)


def print_statistics_summary(merged_data: list):
    """
    打印统计摘要
    """
    if not merged_data:
        return

    print("\n" + "=" * 80)
    print("数据统计摘要")
    print("=" * 80)

    # 收集所有参数键
    all_param_keys = set()
    for task_params, _ in merged_data:
        all_param_keys.update(task_params.keys())

    param_keys = sorted(
        all_param_keys,
        key=lambda k: (0 if k == "dp_size" else 1 if k == "hack_layer_num" else 2, k),
    )

    # 打印表头
    headers = param_keys + ["测试用例数", "平均成功率"]
    print("\t".join(headers))

    # 统计每个任务
    for task_params, metrics in merged_data:
        row = []
        for key in param_keys:
            row.append(str(task_params.get(key, "")))

        # 计算统计信息
        num_cases = len(metrics)
        avg_success_rate = (
            sum(m.get("success_rate", 0) for m in metrics) / num_cases
            if num_cases > 0
            else 0
        )

        row.extend([str(num_cases), f"{avg_success_rate:.4f}"])
        print("\t".join(row))

    print()
    print("=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="合并多个Decode_Result.json或Prefill_Result.json文件到一个表格",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本合并
  python merge_decode_results.py /path/to/Benchmark_dir

  # 指定输出文件名
  python merge_decode_results.py /path/to/Benchmark_dir -o my_results.csv

  # 只生成CSV，不打印表格
  python merge_decode_results.py /path/to/Benchmark_dir --no-print

目录结构要求:
  Benchmark_dir/
    ├── Task_dp_size-4/
    │   └── Decode_Result.json
    ├── Task_dp_size-8/
    │   └── Prefill_Result.json
    └── Task_dp_size-16/
        └── Decode_Result.json
        """,
    )

    parser.add_argument("benchmark_dir", help="Benchmark目录路径（包含Task_*子目录）")

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="输出CSV文件名（默认：<benchmark_dir>_merged.csv）",
    )

    parser.add_argument("--no-print", action="store_true", help="不打印表格到终端")

    args = parser.parse_args()

    # 检查目录是否存在
    if not Path(args.benchmark_dir).exists():
        print(f"错误: 目录不存在 - {args.benchmark_dir}")
        sys.exit(1)

    # 确定输出文件名
    if args.output:
        output_file = args.output
    else:
        benchmark_name = Path(args.benchmark_dir).name
        output_file = f"{benchmark_name}_merged.csv"

    try:
        # 1. 扫描所有Result文件
        print("\n正在扫描Result文件...")
        result_files = scan_result_files(args.benchmark_dir)

        if not result_files:
            print("错误: 未找到任何Result.json文件")
            sys.exit(1)

        print(f"\n找到 {len(result_files)} 个Result文件:")
        for task_params, json_file in result_files:
            task_desc = ", ".join(f"{k}={v}" for k, v in task_params.items())
            filename = Path(json_file).name
            print(f"  - {task_desc} ({filename})")

        # 2. 合并所有数据
        merged_data = merge_all_results(result_files)

        if not merged_data:
            print("\n错误: 没有成功加载的数据")
            sys.exit(1)

        # 3. 保存CSV文件
        save_merged_csv(merged_data, output_file)

        # 4. 打印表格（如果需要）
        if not args.no_print:
            print_statistics_summary(merged_data)
            print_summary_table(merged_data)

        print(f"\n✓ 合并完成! 结果已保存到: {output_file}")

    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
