#!/usr/bin/env python3
"""
Chrome Trace (Perfetto UI) 格式的性能追踪分析脚本

功能：
1. 找出trace文件中的所有kernel
2. 分析出现次数超过20次的kernel
3. 对每个kernel，计算统计信息（出现次数、平均时间、最高最低时间、中位数、方差）
4. 按照kernel首次出现的时间顺序打印成表格
"""

import csv
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def clean_kernel_name(kernel_name: str) -> str:
    """
    清理kernel名称：
    1. 去掉开头的 "void "
    2. 去掉圆括号及其内容（函数参数签名）

    例如：
    void at::native::kernel<T>(int, float) -> at::native::kernel<T>

    注意：需要正确匹配模板参数的尖括号，然后找函数参数的圆括号
    """
    # 去掉开头的 "void "
    if kernel_name.startswith("void "):
        kernel_name = kernel_name[5:]

    # 使用括号匹配来找到函数参数的起始位置
    # 跟踪尖括号的嵌套深度，当深度为0时遇到的圆括号就是函数参数
    angle_depth = 0
    paren_pos = -1

    for i, char in enumerate(kernel_name):
        if char == "<":
            angle_depth += 1
        elif char == ">":
            angle_depth -= 1
        elif char == "(" and angle_depth == 0:
            # 找到了不在模板参数内的第一个圆括号
            paren_pos = i
            break

    if paren_pos != -1:
        kernel_name = kernel_name[:paren_pos]

    return kernel_name.strip()


def load_trace_file(filepath: str) -> dict:
    """加载trace JSON文件"""
    print(f"正在加载trace文件: {filepath}")
    with open(filepath, "r") as f:
        data = json.load(f)
    print(f"文件加载完成，包含 {len(data.get('traceEvents', []))} 个事件")
    return data


def analyze_all_kernels(
    trace_data: dict, min_occurrences: int = 20
) -> Tuple[List[Dict], float]:
    """
    分析Timeline中所有kernel的统计信息
    注意：会去掉时长最长的1%的kernel实例，并不计入total_duration

    返回: (kernel统计信息列表, timeline总时长)
    """
    # 1. 收集所有kernel事件
    raw_kernel_events = []  # list of {'name': name, 'ts': ts, 'dur': dur}

    for event in trace_data.get("traceEvents", []):
        if event.get("cat") == "kernel" and event.get("ph") == "X":

            raw_kernel_name = event.get("name", "Unknown")
            kernel_name = clean_kernel_name(raw_kernel_name)
            timestamp = event.get("ts", 0)
            duration = event.get("dur", 0)

            raw_kernel_events.append(
                {"name": kernel_name, "ts": timestamp, "dur": duration}
            )

    # 2. 去掉时长最高的1%的kernel
    if raw_kernel_events:
        # 按时长降序排序
        raw_kernel_events.sort(key=lambda x: x["dur"], reverse=True)

        total_count = len(raw_kernel_events)
        remove_count = int(total_count * 0.001)

        # 分离保留和移除的事件
        removed_events = raw_kernel_events[:remove_count]
        kept_events = raw_kernel_events[remove_count:]

        removed_duration = sum(e["dur"] for e in removed_events)

        print(f"\n[Filter] 总kernel数: {total_count}")
        print(f"[Filter] 移除前1% ({remove_count}个) 最长kernel")
        print(f"[Filter] 移除的总时长: {removed_duration:.3f} μs")
    else:
        kept_events = []
        removed_duration = 0

    # 3. 重新组织数据用于统计
    kernel_events = defaultdict(list)
    kernel_first_appearance = {}

    for event in kept_events:
        name = event["name"]
        ts = event["ts"]
        dur = event["dur"]

        kernel_events[name].append((ts, dur))

        if name not in kernel_first_appearance or ts < kernel_first_appearance[name]:
            kernel_first_appearance[name] = ts

    print(f"\n在Timeline中找到 {len(kernel_events)} 种不同的kernel (已过滤)")

    # 4. 过滤并计算统计信息
    kernel_stats = []

    for kernel_name, events in kernel_events.items():
        count = len(events)

        # 跳过出现次数少于min_occurrences的kernel
        if count < min_occurrences:
            continue

        # 提取所有持续时间
        durations = [dur for _, dur in events]

        # 计算统计信息
        stats = {
            "kernel_name": kernel_name,
            "count": count,
            "first_appearance": kernel_first_appearance[kernel_name],
            "avg_duration": statistics.mean(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "median_duration": statistics.median(durations),
            "variance": statistics.variance(durations) if count > 1 else 0.0,
            "std_dev": statistics.stdev(durations) if count > 1 else 0.0,
        }

        kernel_stats.append(stats)

    # 按照首次出现时间排序
    kernel_stats.sort(key=lambda x: x["first_appearance"])

    # 5. 计算timeline总时间 (原始wall time - 移除的duration)
    total_duration = 0
    if trace_data.get("traceEvents"):
        min_ts = float("inf")
        max_ts = float("-inf")
        for event in trace_data["traceEvents"]:
            if "ts" in event:
                ts = event["ts"]
                dur = event.get("dur", 0)
                min_ts = min(min_ts, ts)
                max_ts = max(max_ts, ts + dur)
        if min_ts != float("inf") and max_ts != float("-inf"):
            total_duration = max_ts - min_ts

    # 调整总时间
    original_duration = total_duration
    total_duration = max(0, total_duration - removed_duration)

    print(f"找到 {len(kernel_stats)} 种kernel出现次数 >= {min_occurrences}")
    print(f"Timeline原始时长: {original_duration:.3f} μs")
    print(f"Timeline调整时长: {total_duration:.3f} μs (扣除Top1% Kernel时长)")

    return kernel_stats, total_duration


def print_kernel_statistics_table(kernel_stats: List[Dict]):
    """以表格形式打印kernel统计信息"""
    if not kernel_stats:
        print("\n没有符合条件的kernel数据")
        return

    print("\n" + "=" * 180)
    print("Kernel统计分析结果 (按首次出现时间排序)")
    print("=" * 180)

    # 表头
    header = f"{'序号':<6} {'Kernel名称':<80} {'次数':<8} {'平均时间(μs)':<15} {'最小时间(μs)':<15} {'最大时间(μs)':<15} {'中位数(μs)':<15} {'方差':<15}"
    print(header)
    print("-" * 180)

    # 数据行
    for idx, stats in enumerate(kernel_stats, 1):
        # 截断过长的kernel名称
        kernel_name = stats["kernel_name"]
        if len(kernel_name) > 77:
            kernel_name = kernel_name[:74] + "..."

        row = (
            f"{idx:<6} "
            f"{kernel_name:<80} "
            f"{stats['count']:<8} "
            f"{stats['avg_duration']:<15.3f} "
            f"{stats['min_duration']:<15.3f} "
            f"{stats['max_duration']:<15.3f} "
            f"{stats['median_duration']:<15.3f} "
            f"{stats['variance']:<15.3f}"
        )
        print(row)

    print("=" * 180)

    # 打印汇总信息
    print(f"\n汇总信息:")
    print(f"  总计分析了 {len(kernel_stats)} 种kernel")
    total_calls = sum(s["count"] for s in kernel_stats)
    print(f"  总调用次数: {total_calls}")
    print(f"  平均每种kernel调用次数: {total_calls / len(kernel_stats):.1f}")


def save_kernel_statistics_to_csv(
    kernel_stats: List[Dict], input_json_path: str, results_dir: str = "results"
) -> str:
    """
    将kernel统计信息保存到CSV文件

    参数:
        kernel_stats: kernel统计信息列表
        input_json_path: 输入的JSON文件路径
        results_dir: 结果保存目录

    返回:
        CSV文件路径
    """
    if not kernel_stats:
        print("\n没有数据需要保存")
        return ""

    # 创建results目录（如果不存在）
    script_dir = Path(__file__).parent
    results_path = script_dir / results_dir
    results_path.mkdir(exist_ok=True)

    # 从输入的JSON文件名生成CSV文件名
    input_filename = Path(input_json_path).stem  # 获取不带扩展名的文件名
    csv_filename = f"{input_filename}_analysis.csv"
    csv_filepath = results_path / csv_filename

    # 写入CSV文件
    with open(csv_filepath, "w", newline="", encoding="utf-8") as csvfile:
        # 定义CSV列
        fieldnames = [
            "序号",
            "Kernel名称",
            "出现次数",
            "平均时间(μs)",
            "最小时间(μs)",
            "最大时间(μs)",
            "中位数(μs)",
            "标准差(μs)",
            "方差",
            "首次出现时间(μs)",
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 写入数据行
        for idx, stats in enumerate(kernel_stats, 1):
            writer.writerow(
                {
                    "序号": idx,
                    "Kernel名称": stats["kernel_name"],
                    "出现次数": stats["count"],
                    "平均时间(μs)": f"{stats['avg_duration']:.3f}",
                    "最小时间(μs)": f"{stats['min_duration']:.3f}",
                    "最大时间(μs)": f"{stats['max_duration']:.3f}",
                    "中位数(μs)": f"{stats['median_duration']:.3f}",
                    "标准差(μs)": f"{stats['std_dev']:.3f}",
                    "方差": f"{stats['variance']:.3f}",
                    "首次出现时间(μs)": f"{stats['first_appearance']:.3f}",
                }
            )

    print(f"\n✓ 结果已保存到: {csv_filepath}")
    print(f"  - 包含 {len(kernel_stats)} 条记录")
    print(f"  - 文件大小: {csv_filepath.stat().st_size / 1024:.2f} KB")

    return str(csv_filepath)


def print_excel_copyable_format(kernel_stats: List[Dict]):
    """
    打印可直接复制到Excel的制表符分隔格式数据

    使用制表符(\t)分隔，可以直接复制粘贴到Excel
    """
    if not kernel_stats:
        print("\n没有数据需要打印")
        return

    print("\n" + "=" * 180)
    print(
        "可直接复制到Excel的数据（按Ctrl+C复制下面的内容，然后在Excel中按Ctrl+V粘贴）"
    )
    print("=" * 180)
    print()

    # 打印表头（使用制表符分隔）
    headers = [
        "序号",
        "Kernel名称",
        "出现次数",
        "平均时间(μs)",
        "最小时间(μs)",
        "最大时间(μs)",
        "中位数(μs)",
        "标准差(μs)",
        "方差",
        "首次出现时间(μs)",
    ]
    print("\t".join(headers))

    # 打印数据行（使用制表符分隔）
    for idx, stats in enumerate(kernel_stats, 1):
        row = [
            str(idx),
            stats["kernel_name"],
            str(stats["count"]),
            f"{stats['avg_duration']:.3f}",
            f"{stats['min_duration']:.3f}",
            f"{stats['max_duration']:.3f}",
            f"{stats['median_duration']:.3f}",
            f"{stats['std_dev']:.3f}",
            f"{stats['variance']:.3f}",
            f"{stats['first_appearance']:.3f}",
        ]
        print("\t".join(row))

    print()
    print("=" * 180)
    print("提示：选中上面的内容（从表头到最后一行数据），复制后可直接粘贴到Excel")
    print("=" * 180)


def main():
    # 默认trace文件路径
    default_trace_file = "/home/wangyin.yx/workspace/misc/trace/traces/normal_profiler_wr0_b256_s2048_prefill0_6.json"

    # 从命令行获取文件路径，如果没有提供则使用默认路径
    trace_file = sys.argv[1] if len(sys.argv) > 1 else default_trace_file

    # 最小出现次数阈值
    min_occurrences = 20
    if len(sys.argv) > 2:
        min_occurrences = int(sys.argv[2])

    try:
        # 1. 加载trace文件
        trace_data = load_trace_file(trace_file)

        # 2. 分析Timeline中所有kernel
        kernel_stats, total_duration = analyze_all_kernels(trace_data, min_occurrences)

        # 3. 打印统计结果表格
        print_kernel_statistics_table(kernel_stats)

        # 4. 保存结果到CSV文件
        if kernel_stats:
            save_kernel_statistics_to_csv(kernel_stats, trace_file)

        # 5. 打印可直接复制到Excel的格式
        if kernel_stats:
            print_excel_copyable_format(kernel_stats)

        print("\n分析完成!")

    except FileNotFoundError:
        print(f"错误: 找不到文件 '{trace_file}'")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"错误: JSON解析失败 - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
