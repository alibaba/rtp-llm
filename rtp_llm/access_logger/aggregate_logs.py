#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志聚合工具 - 将多进程分离的日志文件合并查看

Usage:
    python aggregate_logs.py [options]

Examples:
    # 查看所有进程的access日志
    python aggregate_logs.py --log-type access

    # 查看所有进程的query_access日志
    python aggregate_logs.py --log-type query_access

    # 实时跟踪日志
    python aggregate_logs.py --log-type access --follow

    # 按时间排序显示所有日志
    python aggregate_logs.py --log-type access --sort-by-time

    # 只查看特定进程的日志
    python aggregate_logs.py --log-type access --rank-id 0 --server-id 1
"""

import argparse
import glob
import json
import re
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Optional


def get_log_files(log_path: str, log_type: str, rank_id: Optional[int] = None, server_id: Optional[int] = None) -> List[str]:
    """获取匹配的日志文件列表"""
    if rank_id is not None and server_id is not None:
        # 查找特定进程的日志文件
        pattern = f"{log_path}/{log_type}_r{rank_id}_s{server_id}.log*"
    else:
        # 查找所有进程的日志文件
        pattern = f"{log_path}/{log_type}_r*_s*.log*"

    files = glob.glob(pattern)
    return sorted(files)


def extract_process_info(filename: str) -> Dict[str, int]:
    """从文件名中提取进程信息"""
    # 匹配模式: access_r0_s1.log 或 query_access_r0_s1.log.1
    match = re.search(r'_r(\d+)_s(\d+)\.log', filename)
    if match:
        return {
            'rank_id': int(match.group(1)),
            'server_id': int(match.group(2))
        }
    return {'rank_id': 0, 'server_id': 0}


def parse_log_entry(line: str, process_info: Dict[str, int]) -> Dict[str, Any]:
    """解析日志条目"""
    try:
        # 尝试解析JSON格式的日志
        log_entry = json.loads(line.strip())
        log_entry['_process_info'] = process_info
        log_entry['_timestamp'] = datetime.now()  # 默认时间戳

        # 尝试解析日志中的时间戳
        if 'request' in log_entry and 'timestamp' in log_entry['request']:
            try:
                log_entry['_timestamp'] = datetime.fromisoformat(
                    log_entry['request']['timestamp'].replace('Z', '+00:00')
                )
            except:
                pass

        return log_entry
    except json.JSONDecodeError:
        # 如果不是JSON格式，作为普通文本处理
        return {
            'message': line.strip(),
            '_process_info': process_info,
            '_timestamp': datetime.now()
        }


def format_log_entry(entry: Dict[str, Any], show_process_info: bool = True) -> str:
    """格式化日志条目输出"""
    process_info = entry.get('_process_info', {})
    rank_id = process_info.get('rank_id', 0)
    server_id = process_info.get('server_id', 0)

    prefix = f"[R{rank_id}S{server_id}] " if show_process_info else ""

    if 'message' in entry:
        # 普通文本日志
        return f"{prefix}{entry['message']}"
    else:
        # JSON日志
        timestamp = entry.get('_timestamp', datetime.now())
        formatted_time = timestamp.strftime('%Y-%m-%d %H:%M:%S')

        # 提取关键信息
        request_id = entry.get('id', 'unknown')
        if 'request' in entry:
            method = entry['request'].get('method', '')
            path = entry['request'].get('path', '')
            summary = f"{method} {path}" if method and path else str(entry['request'])
        else:
            summary = "N/A"

        return f"{prefix}[{formatted_time}] {request_id} {summary}"


def read_logs(files: List[str], sort_by_time: bool = False) -> List[Dict[str, Any]]:
    """读取并解析所有日志文件"""
    all_entries = []

    for file_path in files:
        process_info = extract_process_info(file_path)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = parse_log_entry(line, process_info)
                        all_entries.append(entry)
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}", file=sys.stderr)

    if sort_by_time:
        all_entries.sort(key=lambda x: x.get('_timestamp', datetime.min))

    return all_entries


def follow_logs(files: List[str]):
    """实时跟踪日志文件"""
    file_positions = {}

    print(f"正在跟踪 {len(files)} 个日志文件...")
    print("按 Ctrl+C 退出\n")

    try:
        while True:
            for file_path in files:
                process_info = extract_process_info(file_path)

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # 移动到上次读取的位置
                        if file_path in file_positions:
                            f.seek(file_positions[file_path])
                        else:
                            f.seek(0, 2)  # 移动到文件末尾

                        # 读取新内容
                        new_lines = f.readlines()
                        file_positions[file_path] = f.tell()

                        # 输出新行
                        for line in new_lines:
                            if line.strip():
                                entry = parse_log_entry(line, process_info)
                                print(format_log_entry(entry))

                except FileNotFoundError:
                    pass  # 文件可能还不存在
                except Exception as e:
                    print(f"读取文件 {file_path} 时出错: {e}", file=sys.stderr)

            time.sleep(1)  # 等待1秒后再次检查

    except KeyboardInterrupt:
        print("\n停止跟踪日志文件")


def main():
    parser = argparse.ArgumentParser(
        description="聚合查看多进程日志文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('\n\n')[1]  # 显示使用示例
    )

    parser.add_argument(
        '--log-path',
        default='.',
        help='日志文件目录 (default: .)'
    )

    parser.add_argument(
        '--log-type',
        choices=['access', 'query_access'],
        default='access',
        help='日志类型 (default: access)'
    )

    parser.add_argument(
        '--rank-id',
        type=int,
        help='指定rank ID (查看特定进程)'
    )

    parser.add_argument(
        '--server-id',
        type=int,
        help='指定server ID (查看特定进程，需要和--rank-id一起使用)'
    )

    parser.add_argument(
        '--sort-by-time',
        action='store_true',
        help='按时间排序显示日志'
    )

    parser.add_argument(
        '--follow',
        '-f',
        action='store_true',
        help='实时跟踪日志文件'
    )

    parser.add_argument(
        '--no-process-info',
        action='store_true',
        help='不显示进程信息前缀'
    )

    args = parser.parse_args()

    # 检查参数
    if args.server_id is not None and args.rank_id is None:
        print("错误: --server-id 必须和 --rank-id 一起使用", file=sys.stderr)
        sys.exit(1)

    # 获取日志文件列表
    log_files = get_log_files(args.log_path, args.log_type, args.rank_id, args.server_id)

    if not log_files:
        print(f"在 {args.log_path} 中没有找到匹配的 {args.log_type} 日志文件", file=sys.stderr)
        sys.exit(1)

    print(f"找到 {len(log_files)} 个日志文件:")
    for f in log_files:
        print(f"  {f}")
    print()

    if args.follow:
        follow_logs(log_files)
    else:
        # 读取并显示日志
        entries = read_logs(log_files, args.sort_by_time)

        for entry in entries:
            print(format_log_entry(entry, not args.no_process_info))


if __name__ == '__main__':
    main()