#!/usr/bin/env python3
"""Inspect and update NaviSched scheduler configuration.

The master exposes SchedulerConfigService through http_arpc.  Agents do not
expose that service; their live configuration is changed only by the master's
collector graph, while independent agent changes require a restart/startup
configuration update.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MASTER_FIELDS = {
    'overload_control_strategy': str,
    'execution_utilization_global_scaling': bool,
    'execution_utilization_target': float,
    'execution_utilization_kp': float,
    'execution_utilization_ki_per_second': float,
    "lambda": float,
    "alpha": float,
    "alpha_decay": float,
    "min_alpha": float,
    "max_loop_count": int,
    "min_loop_count": int,
    "optimization_time_budget_us": int,
    "peer_cache_enabled": bool,
    "peer_cache_ready_agents": list,
    "parallelism": int,
    "latency_weight_concavity": float,
    "latency_token_normalization_power": float,
    "initial_weight_seed": int,
    "discrete_check_interval": int,
    "discrete_gap_threshold": float,
    "discrete_repair_alpha_decay": float,
    "max_wait_schedule_count": int,
    "max_wait_schedule_time_ms": int,
    "queue_overload_start_cycle_multiplier": int,
    "queue_overload_full_cycle_multiplier": int,
    "admission_timeout_ms": int,
    "batch_timeout_ms": int,
    "closed_batch_keep_ms": int,
    "latency_fit_batch_size": int,
    "latency_fit_interval": int,
    "latency_learning_rate": float,
    "latency_online_learning_rate_scale": float,
    "latency_current_sample_weight": float,
    "latency_error_clip_ms": float,
    "latency_parameter_learning_rate_scales": list,
    "latency_nonlinear_scale": float,
    "queue_learning_rate": float,
    "latency_check_gradient": bool,
    "latency_cold_start_fit_epochs": int,
    "latency_history_limit": int,
    "latency_history_bucket_capacity": int,
    "latency_persist_interval": int,
    "latency_persist_interval_ms": int,
    "latency_warmup_sample_count": int,
    "latency_model_trust_min_revision": int,
    "latency_zero_anchor_max_ms": float,
    "latency_zero_anchor_min_samples": int,
    "latency_validation_enabled": bool,
    "residual_learning_rate": float,
    "residual_update_error_floor_ms": float,
    "residual_update_error_ratio": float,
    "residual_idle_decay_start_ms": int,
    "residual_idle_decay_end_ms": int,
    "time_compression": float,
}

AGENT_FIELDS = {
    name: value_type
    for name, value_type in MASTER_FIELDS.items()
    if name
    not in {
        'overload_control_strategy',
        'execution_utilization_global_scaling',
        'execution_utilization_target',
        'execution_utilization_kp',
        'execution_utilization_ki_per_second',
        "lambda",
        "alpha",
        "alpha_decay",
        "min_alpha",
        "max_loop_count",
        "min_loop_count",
        "optimization_time_budget_us",
        "peer_cache_enabled",
        "peer_cache_ready_agents",
        "parallelism",
        "latency_weight_concavity",
        "latency_token_normalization_power",
        "initial_weight_seed",
        "discrete_check_interval",
        "discrete_gap_threshold",
        "discrete_repair_alpha_decay",
        "max_wait_schedule_count",
        "max_wait_schedule_time_ms",
        "queue_overload_start_cycle_multiplier",
        "queue_overload_full_cycle_multiplier",
        "admission_timeout_ms",
    }
}


def _json_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(
            f"value must be valid JSON (for strings use a quoted JSON string): {error}"
        ) from error


def _typed_value(name: str, value: str) -> Any:
    parser = MASTER_FIELDS[name]
    if parser in (dict, list):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"{name} must be valid JSON") from error
        if not isinstance(parsed, parser):
            raise ValueError(f"{name} must be a JSON {parser.__name__}")
        return parsed
    if parser is bool:
        lowered = value.lower()
        if lowered not in {"true", "false"}:
            raise ValueError(f"{name} must be true or false")
        return lowered == "true"
    try:
        parsed = parser(value)
    except ValueError as error:
        raise ValueError(f"invalid value for {name}: {value}") from error
    return parsed


def _read_json_arg(value: str) -> Any:
    if value.startswith("@"):
        return json.loads(Path(value[1:]).read_text(encoding="utf-8"))
    return json.loads(value)


def _add_field_options(
    parser: argparse.ArgumentParser, fields: Mapping[str, Any] = MASTER_FIELDS
) -> None:
    for name, value_type in fields.items():
        option = "--" + name.replace("_", "-")
        if value_type is bool:
            parser.add_argument(
                option,
                dest=name,
                choices=("true", "false"),
                default=argparse.SUPPRESS,
                help=f"set {name}",
            )
        else:
            parser.add_argument(
                option,
                dest=name,
                type=str,
                default=argparse.SUPPRESS,
                metavar="VALUE",
                help=f"set {name}",
            )


def _parse_set(
    values: Iterable[str], fields: Mapping[str, Any] = MASTER_FIELDS
) -> Dict[str, Any]:
    patch: Dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"--set requires FIELD=VALUE: {item}")
        name, raw = item.split("=", 1)
        if name not in fields:
            raise ValueError(f"unsupported scheduler config field: {name}")
        patch[name] = _typed_value(name, raw)
    return patch


def _field_patch(
    namespace: argparse.Namespace, fields: Mapping[str, Any] = MASTER_FIELDS
) -> Dict[str, Any]:
    patch: Dict[str, Any] = {}
    for name, value_type in fields.items():
        if not hasattr(namespace, name):
            continue
        value = getattr(namespace, name)
        patch[name] = _typed_value(name, value)
    return patch


def _merge_patch_inputs(
    namespace: argparse.Namespace, fields: Mapping[str, Any] = MASTER_FIELDS
) -> Dict[str, Any]:
    patch: Dict[str, Any] = {}
    if namespace.patch is not None:
        value = _read_json_arg(namespace.patch)
        if not isinstance(value, dict):
            raise ValueError("--patch must contain a JSON object")
        for name in value:
            if name not in fields:
                raise ValueError(f"unsupported scheduler config field: {name}")
        patch.update(value)
    if namespace.patch_file is not None:
        value = json.loads(Path(namespace.patch_file).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("--patch-file must contain a JSON object")
        for name in value:
            if name not in fields:
                raise ValueError(f"unsupported scheduler config field: {name}")
        patch.update(value)
    patch.update(_parse_set(namespace.set_values, fields))
    patch.update(_field_patch(namespace, fields))
    if not patch:
        raise ValueError("update requires at least one field or JSON patch")
    return patch


def _url(args: argparse.Namespace) -> str:
    if args.url:
        return args.url.rstrip("/")
    if not args.host or args.port is None:
        raise ValueError("provide --url or both --host and --port")
    return f"http://{args.host}:{args.port}/scheduler-config"


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
        raise RuntimeError(f"HTTP {error.code} from {url}: {detail}") from error
    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError(f"request to {url} failed: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError("scheduler-config response is not a JSON object")
    return value


def _apply_rfc6902(
    document: MutableMapping[str, Any], operations: Sequence[Any]
) -> None:
    """Apply root-level RFC 6902 add/replace operations to the flat config."""
    for operation in operations:
        if not isinstance(operation, dict):
            raise ValueError("each JSON Patch operation must be an object")
        action = operation.get("op")
        path = operation.get("path")
        if action not in {"add", "replace"} or not isinstance(path, str):
            raise ValueError(
                "only root-level add/replace JSON Patch operations are supported"
            )
        if not path.startswith("/") or "/" in path[1:]:
            raise ValueError("JSON Patch paths must be root-level scheduler fields")
        name = path[1:].replace("~1", "/").replace("~0", "~")
        if name not in MASTER_FIELDS or "value" not in operation:
            raise ValueError(f"unsupported JSON Patch path or missing value: {path}")
        document[name] = operation["value"]


def _update(args: argparse.Namespace) -> int:
    url = _url(args)
    if args.rfc6902 is not None:
        current_payload = {"hardware": args.hardware} if args.hardware else {}
        current = _post(url, current_payload, args.timeout)
        try:
            document = json.loads(current["configJson"])
            operations = _read_json_arg(args.rfc6902)
        except (KeyError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"invalid get response for --rfc6902: {error}"
            ) from error
        if not isinstance(document, dict) or not isinstance(operations, list):
            raise ValueError("--rfc6902 must contain a JSON Patch array")
        _apply_rfc6902(document, operations)
        patch = document
    else:
        patch = _merge_patch_inputs(args)
    payload = {
        "update": True,
        "configJson": json.dumps(patch, separators=(",", ":")),
    }
    if args.hardware:
        payload["hardware"] = args.hardware
    response = _post(url, payload, args.timeout)
    print(json.dumps(response, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if response.get("accepted") is True else 2


def _agent_render(args: argparse.Namespace) -> int:
    if args.input_file:
        value = json.loads(Path(args.input_file).read_text(encoding="utf-8"))
    elif args.config_json:
        value = json.loads(args.config_json)
    else:
        raise ValueError("agent-render requires --input-file or --config-json")
    if not isinstance(value, dict):
        raise ValueError("agent config must be a JSON object")
    patch = _merge_patch_inputs(args, AGENT_FIELDS)
    value.update(patch)
    output = json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if args.output_file:
        Path(args.output_file).write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query/update NaviSched master scheduler-config HTTP RPC."
    )
    parser.add_argument("--url", help="full HTTP URL, normally .../scheduler-config")
    parser.add_argument(
        "--host",
        default=os.environ.get("NAVI_MASTER_HTTP_HOST", "127.0.0.1"),
        help="master HTTP RPC host, default 127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("NAVI_MASTER_HTTP_PORT", "7001")),
        help="master HTTP RPC port, default 7001",
    )
    parser.add_argument(
        "--timeout", type=float, default=15.0, help="HTTP timeout in seconds"
    )
    subparsers = parser.add_subparsers(dest="command")

    get = subparsers.add_parser("get", help="query effective master configuration")
    get.add_argument("--hardware", help="echo/select a hardware-specific view")

    update = subparsers.add_parser("update", help="patch master and sync agent fields")
    update.add_argument(
        "--hardware", help="only update agents whose fingerprint hardware matches"
    )
    update.add_argument(
        "--patch", help="flat JSON object or @FILE; explicit options override it"
    )
    update.add_argument("--patch-file", help="file containing a flat JSON object")
    update.add_argument(
        "--rfc6902",
        help="root-level add/replace JSON Patch array or @FILE; reads current config first",
    )
    update.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="repeatable typed field assignment",
    )
    _add_field_options(update)

    render = subparsers.add_parser(
        "agent-render",
        help="render an independent agent startup config; does not update a running agent",
    )
    render.add_argument("--input-file")
    render.add_argument("--config-json")
    render.add_argument("--output-file")
    render.add_argument("--patch")
    render.add_argument("--patch-file")
    render.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
    )
    _add_field_options(render, AGENT_FIELDS)
    return parser


def main(argv: Sequence[str] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.error("a command is required: get, update, or agent-render")
    if args.command == "get":
        payload = {"hardware": args.hardware} if args.hardware else {}
        response = _post(_url(args), payload, args.timeout)
        print(json.dumps(response, ensure_ascii=True, indent=2, sort_keys=True))
        return 0 if response.get("accepted") is True else 2
    if args.command == "update":
        if args.rfc6902 is not None and any(
            (args.patch, args.patch_file, args.set_values)
        ):
            parser.error(
                "--rfc6902 cannot be combined with --patch, --patch-file or --set"
            )
        return _update(args)
    if args.command == "agent-render":
        return _agent_render(args)
    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as error:
        print(f"navi_sched_config: {error}", file=sys.stderr)
        raise SystemExit(1)
