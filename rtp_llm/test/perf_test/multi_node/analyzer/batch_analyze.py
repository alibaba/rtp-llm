#!/usr/bin/env python3
"""
批量分析多个trace文件并整合结果

功能：
1. 扫描指定目录下的所有trace文件
2. 从目录结构中提取dp_size
3. 从文件名中提取batch_size和seq_len
4. 对每个(dp_size, batch_size, seq_len)组合，选取第一个trace文件（wr0）
    5. 调用trace_analyze.py分析每个文件（会自动过滤掉时长最长的1% Kernel实例，并相应调整总Timeline时长）
    6. 整合所有结果到一个CSV文件
7. 打印制表符格式的表格
"""

import csv
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# Add the directory containing this script to sys.path to allow importing modules from it
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

# Try to import ModuleConf and TestSetUp from configs.module_base
# Assuming configs directory is in the same directory as this script
try:
    # Check if configs/module_base.py exists relative to script location
    module_base_path = script_dir / "configs" / "module_base.py"
    if module_base_path.exists():
        # Adjust sys.path to include configs directory
        sys.path.append(str(script_dir / "configs"))
        from module_base import ModuleConf, TestSetUp
    else:
        # Fallback or try local import if module_base is in same dir
        from module_base import ModuleConf, TestSetUp
except ImportError:
    # Define dummy classes if import fails, to avoid crashing before usage
    print(
        "Warning: Could not import ModuleConf/TestSetUp from module_base. using dummy classes."
    )

    class TestSetUp:
        def __init__(self, dp_size=1, tp_size=1, batch_size=1, seq_len=1):
            self.dp_size = dp_size
            self.tp_size = tp_size
            self.batch_size = batch_size
            self.seq_len = seq_len

    class ModuleConf:
        def __init__(
            self, name_pattern=None, num_flop_calc_func=None, mem_io_calc_func=None
        ):
            self.name_pattern = name_pattern
            self.num_flop_calc_func = num_flop_calc_func
            self.mem_io_calc_func = mem_io_calc_func


def load_module_config(config_path: str) -> dict:
    """
    从指定文件加载module_config
    支持 {name: pattern} 格式 和 {name: ModuleConf} 格式
    """
    if not config_path:
        return {}

    path_obj = Path(config_path)
    if not path_obj.exists():
        print(f"警告: 配置文件不存在 {config_path}")
        return {}

    try:
        import importlib.util

        # Need to add the config file's directory to sys.path so it can import module_base
        config_dir = path_obj.parent
        if str(config_dir) not in sys.path:
            sys.path.append(str(config_dir))

        spec = importlib.util.spec_from_file_location("module_conf", path_obj)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "module_config"):
                config = module.module_config
                # Standardize config to {name: ModuleConf}
                standardized_config = {}
                for name, val in config.items():
                    if isinstance(val, str):
                        # Convert string pattern to ModuleConf
                        standardized_config[name] = ModuleConf(name_pattern=val)
                    else:
                        # Assume it's already ModuleConf (or duck-typed)
                        standardized_config[name] = val
                return standardized_config
            else:
                print(f"警告: {config_path} 中未找到 module_config 变量")
                return {}
    except Exception as e:
        print(f"警告: 加载配置文件失败 {config_path} - {e}")
        import traceback

        traceback.print_exc()
        return {}
    return {}


def extract_dp_size_from_path(task_dir_name: str) -> int:
    """
    从Task目录名中提取dp_size
    例如：Task_dp_size-8 -> 8
    """
    match = re.search(r"dp_size-(\d+)", task_dir_name)
    if match:
        return int(match.group(1))
    return None


def extract_params_from_filename(filename: str) -> dict:
    """
    从trace文件名中提取参数
    例如：normal_profiler_wr0_b512_s64_prefill0_6.json
    返回：{'wr': 0, 'batch_size': 512, 'seq_len': 64, 'prefill': 0, 'id': 6}
    """
    pattern = r"normal_profiler_wr(\d+)_b(\d+)_s(\d+)_prefill(\d+)_(\d+)\.json"
    match = re.search(pattern, filename)

    if match:
        return {
            "wr": int(match.group(1)),
            "batch_size": int(match.group(2)),
            "seq_len": int(match.group(3)),
            "prefill": int(match.group(4)),
            "id": int(match.group(5)),
        }
    return None


def scan_trace_files(benchmark_dir: str) -> list:
    """
    扫描benchmark目录下的所有trace文件

    返回：[(dp_size, batch_size, seq_len, trace_file_path), ...]
    """
    benchmark_path = Path(benchmark_dir)
    trace_configs = []

    # 遍历所有Task_dp_size-*目录
    for task_dir in benchmark_path.iterdir():
        if not task_dir.is_dir() or not task_dir.name.startswith("Task_dp_size-"):
            continue

        dp_size = extract_dp_size_from_path(task_dir.name)
        if dp_size is None:
            continue

        # 查找trace_files目录
        trace_files_dir = task_dir / "trace_files"
        if not trace_files_dir.exists():
            print(f"警告: {task_dir.name} 没有trace_files目录")
            continue

        # 收集该dp_size下的所有trace文件
        dp_traces = defaultdict(list)  # (batch_size, seq_len) -> [trace_files]

        for trace_file in trace_files_dir.glob("*.json"):
            params = extract_params_from_filename(trace_file.name)
            if params is None:
                continue

            # 只选择wr0的文件（第一个trace）
            if params["wr"] == 0:
                key = (params["batch_size"], params["seq_len"])
                dp_traces[key].append(trace_file)

        # 对每个(batch_size, seq_len)组合，选择第一个文件
        for (batch_size, seq_len), files in dp_traces.items():
            if files:
                # 按文件名排序，选择第一个
                first_file = sorted(files)[0]
                trace_configs.append((dp_size, batch_size, seq_len, str(first_file)))

    # 按dp_size, batch_size, seq_len排序
    trace_configs.sort(key=lambda x: (x[0], x[1], x[2]))

    return trace_configs


def analyze_single_trace(trace_file: str, min_occurrences: int = 20) -> dict:
    """
    调用trace_analyze.py分析单个trace文件

    返回：kernel统计信息的字典
    """
    # 获取trace_analyze.py的路径
    script_dir = Path(__file__).parent
    analyze_script = script_dir / "trace_analyze.py"

    if not analyze_script.exists():
        raise FileNotFoundError(f"找不到trace_analyze.py: {analyze_script}")

    print(f"  正在分析: {Path(trace_file).name}")

    # 使用Python直接导入并调用（避免subprocess开销）
    try:
        # 导入trace_analyze模块
        import importlib.util

        spec = importlib.util.spec_from_file_location("trace_analyze", analyze_script)
        trace_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trace_module)

        # 加载trace文件
        trace_data = trace_module.load_trace_file(trace_file)

        # 分析Timeline中所有kernel
        kernel_stats, total_duration = trace_module.analyze_all_kernels(
            trace_data, min_occurrences
        )

        # 计算总调用次数 (仅统计符合条件的kernel)
        total_kernels = sum(k["count"] for k in kernel_stats)

        return {
            "pid": -1,  # 不再特定于某个track
            "tid": -1,
            "total_kernels": total_kernels,
            "kernel_stats": kernel_stats,
            "total_duration": total_duration,
        }

    except Exception as e:
        print(f"  错误: 分析失败 - {e}")
        return None


def aggregate_results(trace_configs: list, min_occurrences: int = 20) -> list:
    """
    批量分析所有trace文件并整合结果

    返回：[(dp_size, batch_size, seq_len, kernel_stats), ...]
    """
    results = []

    print(f"\n开始批量分析 {len(trace_configs)} 个trace文件...")
    print("=" * 80)

    for idx, (dp_size, batch_size, seq_len, trace_file) in enumerate(trace_configs, 1):
        print(
            f"\n[{idx}/{len(trace_configs)}] dp_size={dp_size}, batch={batch_size}, seq={seq_len}"
        )

        analysis = analyze_single_trace(trace_file, min_occurrences)

        if analysis and analysis["kernel_stats"]:
            results.append(
                {
                    "dp_size": dp_size,
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "trace_file": trace_file,
                    "pid": analysis["pid"],
                    "tid": analysis["tid"],
                    "total_kernels": analysis["total_kernels"],
                    "kernel_stats": analysis["kernel_stats"],
                    "total_duration": analysis.get("total_duration", 0),
                }
            )
            print(
                f"  ✓ 成功: 找到 {len(analysis['kernel_stats'])} 种kernel (出现次数>={min_occurrences})"
            )
        else:
            print(f"  ✗ 失败: 无有效数据")

    print("\n" + "=" * 80)
    print(f"完成! 成功分析 {len(results)}/{len(trace_configs)} 个文件")

    return results


def save_aggregated_csv(results: list, output_file: str):
    """
    将整合的结果保存到CSV文件

    CSV格式：
    dp_size, batch_size, seq_len, kernel_name, count, avg_time, min_time, max_time, median_time, std_dev, variance
    """
    if not results:
        print("\n没有结果需要保存")
        return

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "dp_size",
            "batch_size",
            "seq_len",
            "kernel_name",
            "count",
            "avg_time_us",
            "min_time_us",
            "max_time_us",
            "median_time_us",
            "std_dev_us",
            "variance",
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 写入所有结果
        for result in results:
            dp_size = result["dp_size"]
            batch_size = result["batch_size"]
            seq_len = result["seq_len"]

            for kernel_stat in result["kernel_stats"]:
                writer.writerow(
                    {
                        "dp_size": dp_size,
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "kernel_name": kernel_stat["kernel_name"],
                        "count": kernel_stat["count"],
                        "avg_time_us": f"{kernel_stat['avg_duration']:.3f}",
                        "min_time_us": f"{kernel_stat['min_duration']:.3f}",
                        "max_time_us": f"{kernel_stat['max_duration']:.3f}",
                        "median_time_us": f"{kernel_stat['median_duration']:.3f}",
                        "std_dev_us": f"{kernel_stat['std_dev']:.3f}",
                        "variance": f"{kernel_stat['variance']:.3f}",
                    }
                )

    file_size = Path(output_file).stat().st_size / 1024
    print(f"\n✓ 整合结果已保存到: {output_file}")
    print(f"  - 包含 {len(results)} 个配置的数据")
    print(f"  - 文件大小: {file_size:.2f} KB")


def print_summary_table(results: list):
    """
    打印汇总表格（制表符格式，可直接复制到Excel）
    """
    if not results:
        print("\n没有结果需要打印")
        return

    print("\n" + "=" * 100)
    print("整合分析汇总表（可直接复制到Excel）")
    print("=" * 100)
    print()

    # 打印表头
    headers = [
        "dp_size",
        "batch_size",
        "seq_len",
        "kernel种类数",
        "总kernel调用次数",
        "最常见kernel",
        "最常见kernel调用次数",
    ]
    print("\t".join(headers))

    # 打印数据行
    for result in results:
        kernel_stats = result["kernel_stats"]

        # 找到调用次数最多的kernel
        if kernel_stats:
            most_common = max(kernel_stats, key=lambda x: x["count"])
            most_common_name = most_common["kernel_name"]
            most_common_count = most_common["count"]

            # 截断过长的kernel名称
            if len(most_common_name) > 60:
                most_common_name = most_common_name[:57] + "..."
        else:
            most_common_name = "N/A"
            most_common_count = 0

        total_calls = sum(k["count"] for k in kernel_stats)

        row = [
            str(result["dp_size"]),
            str(result["batch_size"]),
            str(result["seq_len"]),
            str(len(kernel_stats)),
            str(total_calls),
            most_common_name,
            str(most_common_count),
        ]
        print("\t".join(row))

    print()
    print("=" * 100)
    print("提示：选中上面的内容，复制后可直接粘贴到Excel")
    print("=" * 100)


def print_detailed_kernel_table(results: list, top_n: int = 10):
    """
    打印详细的kernel统计表格（制表符格式）
    """
    if not results:
        return

    print("\n" + "=" * 120)
    print(f"Top {top_n} Kernels 详细统计（按平均时间排序）")
    print("=" * 120)
    print()

    # 收集所有kernel及其平均时间
    all_kernels = []
    for result in results:
        for kernel_stat in result["kernel_stats"]:
            all_kernels.append(
                {
                    "dp_size": result["dp_size"],
                    "batch_size": result["batch_size"],
                    "seq_len": result["seq_len"],
                    **kernel_stat,
                }
            )

    # 按平均时间降序排序
    all_kernels.sort(key=lambda x: x["avg_duration"], reverse=True)

    # 打印表头
    headers = [
        "dp_size",
        "batch_size",
        "seq_len",
        "kernel_name",
        "count",
        "avg_time(μs)",
        "median(μs)",
    ]
    print("\t".join(headers))

    # 打印前N个
    for kernel in all_kernels[:top_n]:
        kernel_name = kernel["kernel_name"]
        if len(kernel_name) > 80:
            kernel_name = kernel_name[:77] + "..."

        row = [
            str(kernel["dp_size"]),
            str(kernel["batch_size"]),
            str(kernel["seq_len"]),
            kernel_name,
            str(kernel["count"]),
            f"{kernel['avg_duration']:.3f}",
            f"{kernel['median_duration']:.3f}",
        ]
        print("\t".join(row))

    print()
    print("=" * 120)


def filter_kernels_by_module(results: list, module_name: str, pattern: str) -> list:
    """
    根据module配置筛选匹配的kernel
    如果一个trace中有多个kernel匹配同一个pattern，则将它们的时间累加

    参数:
        results: 分析结果列表
        module_name: module名称
        pattern: kernel名称匹配pattern

    返回:
        匹配的kernel列表，每个元素包含：
        {dp_size, batch_size, seq_len, kernel_name, avg_duration, median_duration, count}
    """
    matched_kernels = []

    for result in results:
        dp_size = result["dp_size"]
        batch_size = result["batch_size"]
        seq_len = result["seq_len"]

        # 找出当前trace中所有匹配的kernel
        trace_matches = []
        for kernel_stat in result["kernel_stats"]:
            kernel_name = kernel_stat["kernel_name"]
            if pattern in kernel_name:
                trace_matches.append(kernel_stat)

        if not trace_matches:
            continue

        if len(trace_matches) == 1:
            # 只有一个匹配，直接使用
            k = trace_matches[0]
            matched_kernels.append(
                {
                    "dp_size": dp_size,
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "kernel_name": k["kernel_name"],
                    "count": k["count"],
                    "avg_duration": k["avg_duration"],
                    "median_duration": k["median_duration"],
                    "min_duration": k["min_duration"],
                    "max_duration": k["max_duration"],
                    "std_dev": k["std_dev"],
                    "variance": k["variance"],
                }
            )
        else:
            # 多个匹配，聚合数据
            # 时间相加 (Avg, Median, Min, Max)
            total_avg = sum(k["avg_duration"] for k in trace_matches)
            total_median = sum(k["median_duration"] for k in trace_matches)
            total_min = sum(k["min_duration"] for k in trace_matches)
            total_max = sum(k["max_duration"] for k in trace_matches)

            # count取最大值 (假设它们是同一逻辑操作的组成部分)
            max_count = max(k["count"] for k in trace_matches)

            # 构造聚合后的名称
            # 拼接所有匹配的kernel名称
            combined_name = " + ".join(k["kernel_name"] for k in trace_matches)

            matched_kernels.append(
                {
                    "dp_size": dp_size,
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "kernel_name": combined_name,
                    "count": max_count,
                    "avg_duration": total_avg,
                    "median_duration": total_median,
                    "min_duration": total_min,
                    "max_duration": total_max,
                    "std_dev": 0.0,  # 聚合后难以计算
                    "variance": 0.0,  # 聚合后难以计算
                }
            )

    return matched_kernels


def save_module_results_to_csv(
    module_name: str, kernels: list, output_dir: str = "module_results"
):
    """
    将特定module的结果保存到单独的CSV文件

    参数:
        module_name: module名称
        kernels: 匹配的kernel列表
        output_dir: 输出目录
    """
    if not kernels:
        print(f"  {module_name}: 没有匹配的kernel")
        return

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 生成CSV文件名
    csv_filename = f"{module_name}_analysis.csv"
    csv_filepath = output_path / csv_filename

    # 写入CSV文件
    with open(csv_filepath, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "kernel_name",
            "dp_size",
            "batch_size",
            "seq_len",
            "avg_time_us",
            "median_time_us",
            "min_time_us",
            "max_time_us",
            "count",
            "std_dev_us",
            "variance",
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 按dp_size, batch_size, seq_len排序
        sorted_kernels = sorted(
            kernels, key=lambda x: (x["dp_size"], x["batch_size"], x["seq_len"])
        )

        for kernel in sorted_kernels:
            writer.writerow(
                {
                    "kernel_name": kernel["kernel_name"],
                    "dp_size": kernel["dp_size"],
                    "batch_size": kernel["batch_size"],
                    "seq_len": kernel["seq_len"],
                    "avg_time_us": f"{kernel['avg_duration']:.3f}",
                    "median_time_us": f"{kernel['median_duration']:.3f}",
                    "min_time_us": f"{kernel['min_duration']:.3f}",
                    "max_time_us": f"{kernel['max_duration']:.3f}",
                    "count": kernel["count"],
                    "std_dev_us": f"{kernel['std_dev']:.3f}",
                    "variance": f"{kernel['variance']:.3f}",
                }
            )

    file_size = csv_filepath.stat().st_size / 1024
    print(
        f"  ✓ {module_name}: {len(kernels)} 条记录 -> {csv_filepath.name} ({file_size:.2f} KB)"
    )


def print_module_table(module_name: str, kernels: list):
    """
    打印特定module的表格（制表符格式，可直接复制到Excel）

    参数:
        module_name: module名称
        kernels: 匹配的kernel列表
    """
    if not kernels:
        return

    print(f"\n{'='*120}")
    print(f"{module_name.upper()} Module 详细统计")
    print(f"{'='*120}")
    print()

    # 打印表头
    headers = [
        "kernel_name",
        "dp_size",
        "batch_size",
        "seq_len",
        "avg_time(μs)",
        "median(μs)",
    ]
    print("\t".join(headers))

    # 按dp_size, batch_size, seq_len排序
    sorted_kernels = sorted(
        kernels, key=lambda x: (x["dp_size"], x["batch_size"], x["seq_len"])
    )

    # 打印数据行
    for kernel in sorted_kernels:
        kernel_name = kernel["kernel_name"]
        # 不截断kernel名称，保持完整
        row = [
            kernel_name,
            str(kernel["dp_size"]),
            str(kernel["batch_size"]),
            str(kernel["seq_len"]),
            f"{kernel['avg_duration']:.3f}",
            f"{kernel['median_duration']:.3f}",
        ]
        print("\t".join(row))

    print()
    print(f"{'='*120}")
    print(f"提示：选中上面的 {module_name} 数据，复制后可直接粘贴到Excel")
    print(f"{'='*120}")


def generate_combined_summary_table(
    results: list, module_config: dict, output_dir: str = "module_results"
):
    """
    生成所有测试的所有kernel整合表格
    列：dp_size, batch_size, seq_len, module_name, kernel_name, avg_time, median_time, timeline_percentage,
        FLOPs(G), IO(MB), TFLOPS, Bandwidth(GB/s)

    参数:
        results: 分析结果列表
        module_config: module配置字典
        output_dir: 输出目录
    """
    if not module_config:
        return

    print("\n" + "=" * 80)
    print("生成整合汇总表格")
    print("=" * 80)

    summary_data = []

    # 遍历每个测试结果
    for result in results:
        dp_size = result["dp_size"]
        batch_size = result["batch_size"]
        seq_len = result["seq_len"]
        total_duration = result.get("total_duration", 0)

        # Construct TestSetUp for calculations
        test_setup = TestSetUp(
            dp_size=dp_size,
            batch_size=batch_size,
            seq_len=seq_len,
            tp_size=1,  # Default or extract if available
        )

        if total_duration <= 0:
            print(
                f"警告: {dp_size}-{batch_size}-{seq_len} 的timeline总时长无效 ({total_duration})，跳过计算比例"
            )
            # continue # Don't continue, we can still calculate FLOPs/IO but percentage/bandwidth will be wrong

        # 遍历每个module
        for module_name, module_conf in module_config.items():
            pattern = module_conf.name_pattern
            if not pattern:
                continue

            # 筛选匹配的kernel (复用filter_kernels_by_module逻辑)
            # 手动筛选当前result匹配的kernel
            trace_matches = []
            for kernel_stat in result["kernel_stats"]:
                if pattern in kernel_stat["kernel_name"]:
                    trace_matches.append(kernel_stat)

            if not trace_matches:
                continue

            kernel_name_display = ""
            avg_duration = 0.0
            median_duration = 0.0
            count = 0

            if len(trace_matches) == 1:
                k = trace_matches[0]
                kernel_name_display = k["kernel_name"]
                avg_duration = k["avg_duration"]
                median_duration = k["median_duration"]
                count = k["count"]
            else:
                # 聚合
                kernel_name_display = " + ".join(
                    k["kernel_name"] for k in trace_matches
                )
                avg_duration = sum(k["avg_duration"] for k in trace_matches)
                median_duration = sum(k["median_duration"] for k in trace_matches)
                count = max(k["count"] for k in trace_matches)

            total_kernel_time = avg_duration * count
            percentage = (
                (total_kernel_time / total_duration) * 100 if total_duration > 0 else 0
            )

            # Calculate FLOPs and IO
            flops = 0
            io_bytes = 0

            if module_conf.num_flop_calc_func:
                try:
                    flops = module_conf.num_flop_calc_func(test_setup)
                except Exception as e:
                    pass

            if module_conf.mem_io_calc_func:
                try:
                    io_bytes = module_conf.mem_io_calc_func(test_setup)
                except Exception as e:
                    pass

            # Calculate Performance Metrics
            # avg_duration is in microseconds (us)
            # TFLOPS = (FLOPs / (time_us * 1e-6)) / 1e12
            # Bandwidth (GB/s) = (IO_bytes / (time_us * 1e-6)) / 1e9

            tflops = 0.0
            bandwidth_gbs = 0.0

            if avg_duration > 0:
                time_sec = avg_duration * 1e-6
                if flops > 0:
                    tflops = (flops / time_sec) / 1e12
                if io_bytes > 0:
                    bandwidth_gbs = (io_bytes / time_sec) / 1e9

            summary_data.append(
                {
                    "dp_size": dp_size,
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "module_name": module_name,
                    "kernel_name": kernel_name_display,
                    "avg_time": avg_duration,
                    "median_time": median_duration,
                    "percentage": percentage,
                    "flops": flops,
                    "io_bytes": io_bytes,
                    "tflops": tflops,
                    "bandwidth": bandwidth_gbs,
                }
            )

    if not summary_data:
        print("没有生成汇总数据")
        return

    # 保存CSV
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_file = output_path / "combined_module_summary.csv"

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "dp_size",
            "batch_size",
            "seq_len",
            "module_name",
            "kernel_name",
            "avg_time_us",
            "median_time_us",
            "timeline_percentage",
            "FLOPs",
            "IO_bytes",
            "TFLOPS",
            "Bandwidth_GBs",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_data:
            writer.writerow(
                {
                    "dp_size": row["dp_size"],
                    "batch_size": row["batch_size"],
                    "seq_len": row["seq_len"],
                    "module_name": row["module_name"],
                    "kernel_name": row["kernel_name"],
                    "avg_time_us": f"{row['avg_time']:.3f}",
                    "median_time_us": f"{row['median_time']:.3f}",
                    "timeline_percentage": f"{row['percentage']:.2f}%",
                    "FLOPs": row["flops"],
                    "IO_bytes": row["io_bytes"],
                    "TFLOPS": f"{row['tflops']:.3f}",
                    "Bandwidth_GBs": f"{row['bandwidth']:.3f}",
                }
            )

    print(f"✓ 整合汇总表格已保存: {csv_file}")

    # 打印表格
    print("\n" + "=" * 160)
    print("Combined Module Summary (Tab-separated for Excel)")
    print("=" * 160)
    header = [
        "dp_size",
        "batch_size",
        "seq_len",
        "module_name",
        "kernel_name",
        "avg_time(μs)",
        "median_time(μs)",
        "percentage(%)",
        "FLOPs",
        "IO(Bytes)",
        "TFLOPS",
        "BW(GB/s)",
    ]
    print("\t".join(header))

    for row in summary_data:
        line = [
            str(row["dp_size"]),
            str(row["batch_size"]),
            str(row["seq_len"]),
            row["module_name"],
            row["kernel_name"],
            f"{row['avg_time']:.3f}",
            f"{row['median_time']:.3f}",
            f"{row['percentage']:.2f}",
            f"{row['flops']:.2e}",
            f"{row['io_bytes']:.2e}",
            f"{row['tflops']:.2f}",
            f"{row['bandwidth']:.2f}",
        ]
        print("\t".join(line))
    print("=" * 160 + "\n")


def analyze_by_modules(
    results: list, module_config: dict, output_dir: str = "module_results"
):
    """
    按module分析并保存结果

    参数:
        results: 分析结果列表
        module_config: module配置字典
        output_dir: 输出目录
    """
    if not module_config:
        print("\n跳过module分析（未提供有效配置）")
        return

    print("\n" + "=" * 80)
    print("开始Module维度分析")
    print("=" * 80)

    for module_name, module_conf in module_config.items():
        pattern = module_conf.name_pattern
        if not pattern:
            continue

        print(f"\n处理 {module_name} module (pattern: {pattern[:50]}...)")

        # 筛选匹配的kernel
        matched_kernels = filter_kernels_by_module(results, module_name, pattern)

        if matched_kernels:
            # 保存到CSV
            save_module_results_to_csv(module_name, matched_kernels, output_dir)

            # 打印表格
            print_module_table(module_name, matched_kernels)
        else:
            print(f"  ⚠ {module_name}: 未找到匹配的kernel")

    print("\n" + "=" * 80)
    print(f"Module分析完成! 结果保存在 {output_dir}/ 目录")
    print("=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="批量分析多个trace文件并整合结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本分析（输出到 Benchmark_dir_analyzed/）
  python batch_analyze.py /path/to/Benchmark_dir

  # 自定义阈值
  python batch_analyze.py /path/to/Benchmark_dir -m 30

  # 显示更多详细统计
  python batch_analyze.py /path/to/Benchmark_dir --top-n 30
        """,
    )

    parser.add_argument(
        "benchmark_dir", help="Benchmark目录路径（包含Task_dp_size-*子目录）"
    )

    parser.add_argument(
        "-m",
        "--min-occurrences",
        type=int,
        default=20,
        help="kernel最小出现次数阈值（默认：20）",
    )

    parser.add_argument(
        "--top-n", type=int, default=20, help="显示前N个最耗时的kernel（默认：20）"
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--module-conf",
        type=str,
        default=None,
        help="module配置文件路径（例如：module_conf.py）",
    )

    args = parser.parse_args()

    # 检查目录是否存在
    if not Path(args.benchmark_dir).exists():
        print(f"错误: 目录不存在 - {args.benchmark_dir}")
        sys.exit(1)

    # 从输入路径提取目录名，创建输出目录
    benchmark_path = Path(args.benchmark_dir)
    benchmark_name = benchmark_path.name  # 获取最后一级目录名
    if args.output_dir:
        output_path = Path(args.output_dir)
    else:
        output_path = benchmark_path / f"{benchmark_name}_analyzed"
    output_path.mkdir(exist_ok=True)

    print(f"\n输出目录: {output_path}/")
    print(f"  - 整合CSV: {output_path}/batch_analysis.csv")
    print(f"  - Module结果: {output_path}/module_results/")

    try:
        # 1. 扫描所有trace文件
        print("\n正在扫描trace文件...")
        trace_configs = scan_trace_files(args.benchmark_dir)

        if not trace_configs:
            print("错误: 未找到任何符合条件的trace文件")
            sys.exit(1)

        print(f"\n找到 {len(trace_configs)} 个trace文件配置:")
        for dp_size, batch_size, seq_len, _ in trace_configs:
            print(f"  - dp_size={dp_size}, batch={batch_size}, seq={seq_len}")

        # 2. 批量分析
        results = aggregate_results(trace_configs, args.min_occurrences)

        if not results:
            print("\n错误: 没有成功分析的结果")
            sys.exit(1)

        # 3. 保存整合的CSV
        batch_csv_path = output_path / "batch_analysis.csv"
        save_aggregated_csv(results, str(batch_csv_path))

        # 4. 打印汇总表格
        print_summary_table(results)

        # 5. 打印详细kernel表格
        print_detailed_kernel_table(results, args.top_n)

        # 6. Module维度分析
        module_output_dir = output_path / "module_results"

        # 加载module配置
        module_config = {}
        if args.module_conf:
            module_config = load_module_config(args.module_conf)
        else:
            # 尝试默认位置
            default_conf = Path(__file__).parent / "module_conf.py"
            if default_conf.exists():
                print(f"提示: 使用默认配置文件 {default_conf.name}")
                module_config = load_module_config(str(default_conf))

        analyze_by_modules(results, module_config, str(module_output_dir))

        # 7. 生成整合汇总表格
        generate_combined_summary_table(results, module_config, str(module_output_dir))

        print(f"\n分析完成!")
        print(f"\n所有结果已保存到: {output_path}/")
        print(f"  - 查看整合结果: {batch_csv_path}")
        print(f"  - 查看Module结果: {module_output_dir}/")

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
