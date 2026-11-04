#!/usr/bin/env python3
"""Command line client for the NaviSched replay-control HTTP service."""

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, Mapping, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _url(args: argparse.Namespace) -> str:
    if args.url:
        return args.url.rstrip("/")
    return "http://{}:{}/replay-control".format(args.host, args.port)


def _post(url: str, payload: Mapping[str, Any], timeout: float) -> Dict[str, Any]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError("HTTP {} 返回错误: {}".format(error.code, detail)) from error
    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError("请求 {} 失败: {}".format(url, error)) from error
    except json.JSONDecodeError as error:
        raise RuntimeError("Master 返回的内容不是合法 JSON: {}".format(error)) from error
    if not isinstance(value, dict):
        raise RuntimeError("Master 返回的 JSON 不是对象")
    return value


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("不能小于 0")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的有限数")
    return parsed


def _add_control_arguments(
    parser: argparse.ArgumentParser, *, require_mode: bool
) -> None:
    parser.add_argument(
        "--mode",
        required=require_mode,
        choices=("fixed_qps", "realtime"),
        help=(
            "回放模式：fixed_qps 使用固定总 QPS；realtime 按历史时间轴发送。"
            "update 只调整 QPS 时可省略"
        ),
    )
    parser.add_argument(
        "--history-mode",
        default="time_range" if require_mode else None,
        choices=("time_range", "recent_count"),
        help="历史选择：time_range 按时间段；recent_count 取最近 N 条",
    )
    parser.add_argument(
        "--start-time-us",
        type=_positive_int,
        help="time_range 的历史区间起始 event time，单位微秒",
    )
    parser.add_argument(
        "--recent-query-count",
        type=_positive_int,
        help="recent_count 模式加载的最近查询条数",
    )
    parser.add_argument(
        "--end-time-us",
        type=_positive_int,
        help="历史区间结束 event time，单位微秒；未指定时由 duration-ms 推导",
    )
    parser.add_argument(
        "--duration-ms",
        type=_positive_int,
        default=0,
        help="最长回放时长，单位毫秒；也可在未给 end-time-us 时确定历史区间",
    )
    parser.add_argument(
        "--total-qps",
        type=_positive_float,
        help="fixed_qps 模式的集群总 QPS，Master 自动按可参与 client 数均分",
    )
    parser.add_argument(
        "--loop-count",
        type=_nonnegative_int,
        default=0,
        help="历史区间循环次数；0 表示仅由 duration-ms 控制结束",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="navi_sched_replay",
        description="查询和控制 NaviSched 内置历史流量回放。",
    )
    parser.add_argument(
        "--url",
        help="完整接口地址，默认由 host/port 组成 http://HOST:PORT/replay-control",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get(
            "NAVI_SCHED_MASTER_HTTP_HOST",
            os.environ.get("NAVI_MASTER_HTTP_HOST", "127.0.0.1"),
        ),
        help="Master HTTP 地址，默认自动使用本容器 127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(
            os.environ.get(
                "NAVI_SCHED_MASTER_HTTP_PORT",
                os.environ.get(
                    "NAVI_MASTER_HTTP_PORT", os.environ.get("START_PORT", "7001")
                ),
            )
        ),
        help="Master HTTP 端口，默认读取 NAVI_SCHED_MASTER_HTTP_PORT/START_PORT",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=15.0,
        help="HTTP 请求超时秒数，默认 15",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("query", help="查询当前回放状态和各 client 历史范围")

    start = subparsers.add_parser("start", help="启动一项新的回放任务")
    _add_control_arguments(start, require_mode=True)
    update = subparsers.add_parser(
        "update", help="更新回放；只传 --total-qps 可连续调整当前固定 QPS"
    )
    _add_control_arguments(update, require_mode=False)
    subparsers.add_parser("stop", help="立即停止回放并清理 client 压测状态")
    return parser


def _control_payload(args: argparse.Namespace) -> Dict[str, Any]:
    if args.command == "update" and args.mode is None:
        if args.total_qps is None:
            raise ValueError("只调整 QPS 时必须指定 --total-qps")
        if (
            args.history_mode is not None
            or args.start_time_us is not None
            or args.end_time_us is not None
            or args.recent_query_count is not None
            or args.duration_ms != 0
            or args.loop_count != 0
        ):
            raise ValueError("只调整 QPS 时不能同时修改历史区间、循环或时长")
        return {"action": "update", "totalQps": args.total_qps}

    history_mode = args.history_mode or "time_range"
    if history_mode == "time_range":
        if args.start_time_us is None:
            raise ValueError("time_range 模式必须指定 --start-time-us")
        if args.recent_query_count is not None:
            raise ValueError("time_range 模式不能指定 --recent-query-count")
        if args.end_time_us is None and args.duration_ms == 0:
            raise ValueError("必须指定 --end-time-us 或正数 --duration-ms")
        if args.end_time_us is not None and args.end_time_us <= args.start_time_us:
            raise ValueError("--end-time-us 必须大于 --start-time-us")
    else:
        if args.recent_query_count is None:
            raise ValueError("recent_count 模式必须指定 --recent-query-count")
        if args.start_time_us is not None or args.end_time_us is not None:
            raise ValueError("recent_count 模式不能指定时间区间")
    if args.loop_count == 0 and args.duration_ms == 0:
        raise ValueError("必须指定正数 --loop-count 或 --duration-ms")
    if args.mode == "fixed_qps" and args.total_qps is None:
        raise ValueError("fixed_qps 模式必须指定 --total-qps")
    if args.mode == "realtime" and args.total_qps is not None:
        raise ValueError("realtime 模式不能指定 --total-qps")
    return {
        "action": args.command,
        "mode": args.mode,
        "historyMode": history_mode,
        "startTimeUs": args.start_time_us or 0,
        "endTimeUs": args.end_time_us or 0,
        "durationMs": args.duration_ms,
        "totalQps": args.total_qps or 0.0,
        "loopCount": args.loop_count,
        "recentQueryCount": args.recent_query_count or 0,
    }


def _run(argv: Optional[Sequence[str]]) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.error("必须指定操作：query、start、update 或 stop")
    if args.command in ("start", "update"):
        payload = _control_payload(args)
    else:
        payload = {"action": args.command}
    response = _post(_url(args), payload, args.timeout)
    print(json.dumps(response, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if response.get("accepted") is True else 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        return _run(argv)
    except (ValueError, RuntimeError) as error:
        print("navi_sched_replay: {}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
