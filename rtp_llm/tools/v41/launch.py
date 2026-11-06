"""CP8 or DP8 Prefill / DP8 Decode presets for the existing native server."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

from rtp_llm.config.v41_runtime import PPU_STRATEGY, V41RuntimeConfig, env_bool
from rtp_llm.tools.v41.preflight import collect_report

DEFAULT_PREFILL_WAIT_TIMEOUT_MS = 1800000
MAX_PREFILL_WAIT_TIMEOUT_MS = 86400000


def bounded_prefill_wait_timeout_ms(value):
    if type(value) not in (int, str):
        raise ValueError("prefill wait timeout must be integer milliseconds")
    try:
        result = int(value)
    except (ValueError, OverflowError) as error:
        raise ValueError("prefill wait timeout must be integer milliseconds") from error
    if not 1 <= result <= MAX_PREFILL_WAIT_TIMEOUT_MS:
        raise ValueError("prefill wait timeout must be in [1, 86400000] milliseconds")
    return result


def preset_env(
    role,
    checkpoint,
    *,
    rdma=False,
    parent=None,
    prefill_address=None,
    decode_address=None,
    start_port=None,
    prefill_parallel="cp",
    ced_mode="off",
    encoder_swa_replay=False,
    max_seq_len=None,
    prefill_wait_timeout_ms=DEFAULT_PREFILL_WAIT_TIMEOUT_MS,
    dspark_gamma=0,
    dspark_confidence="off",
    dspark_verify_mode="dense",
    dspark_verify_budget=-1,
    dspark_budget_mode="fixed",
    fake_balance_expert=None,
):
    if role not in ("prefill", "decode"):
        raise ValueError("role must be prefill or decode")
    if fake_balance_expert is None:
        fake_balance_expert = env_bool(
            os.environ if parent is None else parent, "FAKE_BALANCE_EXPERT"
        )
    if type(fake_balance_expert) is not bool:
        raise ValueError("fake_balance_expert must be an explicit boolean")
    if fake_balance_expert and dspark_gamma:
        raise ValueError("PPU fake balance is qualified only without DSpark")
    prefill_wait_timeout_ms = bounded_prefill_wait_timeout_ms(prefill_wait_timeout_ms)
    if prefill_parallel not in ("cp", "dp"):
        raise ValueError("prefill_parallel must be cp or dp")
    if ced_mode not in ("off", "exact", "bounded"):
        raise ValueError("ced_mode must be off, exact or bounded")
    if encoder_swa_replay and ced_mode != "bounded":
        raise ValueError("encoder SWA recovery requires --ced-mode bounded")
    if dspark_gamma < 0:
        raise ValueError("DSpark gamma must be nonnegative (zero disables it)")
    if dspark_confidence not in ("off", "observe", "bounded"):
        raise ValueError("invalid DSpark confidence mode")
    if dspark_verify_mode not in ("dense", "packed") or dspark_budget_mode not in (
        "fixed",
        "adaptive",
    ):
        raise ValueError("invalid DSpark verification/budget mode")
    if not -1 <= dspark_verify_budget <= dspark_gamma:
        raise ValueError("DSpark budget must be -1 or between 0 and gamma")
    if (
        dspark_verify_budget != -1
        or dspark_verify_mode == "packed"
        or dspark_budget_mode == "adaptive"
    ) and dspark_confidence != "bounded":
        raise ValueError(
            "DSpark budget/packed verification requires bounded confidence"
        )
    if dspark_verify_mode == "packed" and dspark_verify_budget < 0:
        raise ValueError("packed verification requires an explicit budget")
    if dspark_budget_mode == "adaptive" and dspark_verify_mode != "packed":
        raise ValueError("adaptive budget requires packed verification")
    if not dspark_gamma and (dspark_confidence != "off" or dspark_verify_budget != -1):
        raise ValueError("DSpark options require positive gamma")
    cp_prefill = prefill_parallel == "cp"
    env = dict(os.environ if parent is None else parent)
    env.update(
        {
            "RTP_V41_BACKEND": "ppu",
            "RTP_V41_ENP_SIZE": "8",
            "RTP_V41_COORDINATED_STOP": "1",
            "RTP_V41_PPU_STRATEGY": PPU_STRATEGY,
            "MODEL_TYPE": "deepseek_v41",
            "CHECKPOINT_PATH": str(checkpoint),
            "TOKENIZER_PATH": str(checkpoint),
            "ROLE_TYPE": role.upper(),
            "WORLD_SIZE": "8",
            "LOCAL_WORLD_SIZE": "8",
            "WORLD_RANK": "0",
            "WORKER_INFO_PORT_NUM": "9",
            "TP_SIZE": "8" if role == "prefill" and cp_prefill else "1",
            "DP_SIZE": "1" if role == "prefill" and cp_prefill else "8",
            "EP_SIZE": "8",
            "CP_ROTATE_METHOD": (
                ("ALL_GATHER" if role == "prefill" else "PREFILL_CP")
                if cp_prefill
                else "DISABLED"
            ),
            "PREFILL_CP_SIZE": "8" if cp_prefill else "1",
            "PREFILL_CP_KV_CACHE_SHARDED": "0",
            "MAX_SEQ_LEN": str(
                max_seq_len
                if max_seq_len is not None
                else env.get("MAX_SEQ_LEN", "32768")
            ),
            "LOAD_PYTHON_MODEL": "1",
            "ACT_TYPE": "bf16",
            "ENABLE_CUDA_GRAPH": "0",
            "SP_TYPE": "",
            "SP_MODEL_TYPE": "",
            "SP_CHECKPOINT_PATH": "",
            "GEN_NUM_PER_CIRCLE": "0",
            "SP_DSPARK_CONFIDENCE_MODE": "off",
            "SP_DSPARK_VERIFY_MODE": "dense",
            "SP_DSPARK_VERIFY_BUDGET": "-1",
            "SP_DSPARK_VERIFY_BUDGET_MODE": "fixed",
            "EPLB_MODE": "NONE",
            "REUSE_CACHE": "1" if encoder_swa_replay else "0",
            "DSV41_CED": "0" if ced_mode == "off" else "1",
            "DSV41_SWA_BOUNDED_REPLAY": "1" if ced_mode == "bounded" else "0",
            "DSV41_ENCODER_SWA_REPLAY": "1" if encoder_swa_replay else "0",
            "USE_DEEPEP_MOE": "1",
            "USE_DEEPEP_LOW_LATENCY": "0",
            "USE_DEEPEP_INTERNODE": "0",
            "USE_DEEPEP_P2P_LOW_LATENCY": "0",
            "USE_ALL_GATHER": "0",
            "USE_UB_COMM": "0",
            "FT_DISABLE_CUSTOM_AR": "1",
            "FFN_SP_SIZE": "1",
            "ENABLE_FFN_DISAGGREGATE": "0",
            "USE_MORI_EP": "0",
            "MOE_STRATEGY": "auto",
            "FAKE_BALANCE_EXPERT": str(int(fake_balance_expert)),
            "REDUNDANT_EXPERT": "0",
            "CACHE_STORE_RDMA_MODE": "1" if rdma else "0",
            "START_PORT": str(start_port or (8090 if role == "prefill" else 9090)),
            "WARM_UP": "0",
            "MODEL_WARM_UP": "0",
            "FRONTEND_SERVER_COUNT": "1",
            "KV_CACHE_MEM_MB": "8192",
            "MAX_CONTEXT_BATCH_SIZE": "1",
            "PREFILL_CHUNK_SIZE": "0",
            "LOAD_CACHE_TIMEOUT_MS": "600000",
            "PREFILL_MAX_WAIT_TIMEOUT_MS": str(prefill_wait_timeout_ms),
            "DIST_COMM_TIMEOUT": "1800",
            "DECODE_RETRY_TIMES": "0",
            "DECODE_ENTRANCE": "0",
            # RPCPool's 5 s watchdog needs ACK polling while both PD peers
            # wait for Prefill/KV work outside their synchronous gRPC calls.
            "GRPC_CLIENT_CHANNEL_BACKUP_POLL_INTERVAL_MS": "500",
        }
    )
    if dspark_gamma:
        env.update(
            SP_TYPE="dspark",
            SP_MODEL_TYPE="deepseek_v41_dspark",
            SP_CHECKPOINT_PATH=str(checkpoint),
            GEN_NUM_PER_CIRCLE=str(dspark_gamma),
            # P runs commit-only and does not rank proposals or verify rows.
            SP_DSPARK_CONFIDENCE_MODE=dspark_confidence if role == "decode" else "off",
            SP_DSPARK_VERIFY_MODE=dspark_verify_mode if role == "decode" else "dense",
            SP_DSPARK_VERIFY_BUDGET=(
                str(dspark_verify_budget) if role == "decode" else "-1"
            ),
            SP_DSPARK_VERIFY_BUDGET_MODE=(
                dspark_budget_mode if role == "decode" else "fixed"
            ),
        )
    if bool(prefill_address) != bool(decode_address):
        raise ValueError("both prefill and decode addresses are required")
    if prefill_address:
        addresses = {"prefill": prefill_address, "decode": decode_address}
        parsed = {}
        for name, address in addresses.items():
            host, separator, port = address.rpartition(":")
            if (
                not separator
                or not host
                or not port.isdecimal()
                or not 1024 <= int(port) <= 65400
            ):
                raise ValueError(
                    f"{name} address must be host:port with port in [1024, 65400]"
                )
            parsed[name] = (host, int(port))
        port = parsed[role][1]
        if start_port is not None and int(start_port) != port:
            raise ValueError("start_port must match this role's HTTP endpoint")
        peer = "decode" if role == "prefill" else "prefill"
        env["START_PORT"] = str(port)
        env["REMOTE_RPC_SERVER_IP"] = parsed[peer][0]
        env["REMOTE_SERVER_PORT"] = str(parsed[peer][1])
        route_addresses = dict(addresses)
        for name in ("prefill", "decode"):
            host, base_port = parsed[name]
            owners = 1 if name == "prefill" and cp_prefill else 8
            route_addresses[name] = ",".join(
                f"{host}:{base_port + rank * 9}" for rank in range(owners)
            )
        env["MODEL_SERVICE_CONFIG"] = json.dumps(
            {
                "service_id": "v41_ppu_pd",
                "use_local": True,
                "role_endpoints": [
                    {
                        "group": "default",
                        **{
                            f"{name}_endpoint": {
                                "type": "Vipserver",
                                "address": address,
                                "protocol": "http",
                                "path": "/",
                            }
                            for name, address in route_addresses.items()
                        },
                    }
                ],
            },
            separators=(",", ":"),
        )
    V41RuntimeConfig.from_env(env)
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("prefill", "decode"))
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument(
        "--prefill-address", help="P HTTP endpoint, e.g. 192.0.2.10:8090"
    )
    parser.add_argument(
        "--decode-address", help="D HTTP endpoint, e.g. 192.0.2.20:19090"
    )
    parser.add_argument("--start-port", type=int)
    parser.add_argument(
        "--prefill-wait-timeout-ms",
        type=bounded_prefill_wait_timeout_ms,
        default=DEFAULT_PREFILL_WAIT_TIMEOUT_MS,
        help="Prefill queue wait limit in [1, 86400000] ms; default30 minutes",
    )
    parser.add_argument("--dspark-gamma", type=int, default=0)
    parser.add_argument(
        "--fake-balance-expert",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Performance-only balanced expert routes; default off, no DSpark",
    )
    parser.add_argument(
        "--dspark-confidence", choices=("off", "observe", "bounded"), default="off"
    )
    parser.add_argument(
        "--dspark-verify-mode", choices=("dense", "packed"), default="dense"
    )
    parser.add_argument("--dspark-verify-budget", type=int, default=-1)
    parser.add_argument(
        "--dspark-budget-mode", choices=("fixed", "adaptive"), default="fixed"
    )
    parser.add_argument(
        "--ced-mode",
        choices=("off", "exact", "bounded"),
        default="off",
        help="PPU consumer prefill mode; use the same cache mode on P and D",
    )
    parser.add_argument(
        "--encoder-swa-replay",
        action="store_true",
        help="Reuse Global KV prefixes and rebuild private encoder SWA from full history",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=os.environ.get("MAX_SEQ_LEN", "32768"),
        help="Per-request input + output capacity; defaults to MAX_SEQ_LEN or 32768",
    )
    parser.add_argument(
        "--prefill-parallel",
        choices=("cp", "dp"),
        default="cp",
        help="P attention topology: CP8 (default) or DP8; set identically on P and D",
    )
    parser.add_argument(
        "--token-map-cache",
        type=Path,
        help="shared tokenizer cache under the run evidence directory",
    )
    parser.add_argument(
        "--rdma", action="store_true", help="explicit RDMA, no TCP fallback"
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--print-command", action="store_true")
    action.add_argument("--check-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probe-allocations", action="store_true")
    args = parser.parse_args()
    options = dict(
        rdma=args.rdma,
        prefill_address=args.prefill_address,
        decode_address=args.decode_address,
        start_port=args.start_port,
        prefill_parallel=args.prefill_parallel,
        ced_mode=args.ced_mode,
        encoder_swa_replay=args.encoder_swa_replay,
        max_seq_len=args.max_seq_len,
        prefill_wait_timeout_ms=args.prefill_wait_timeout_ms,
        dspark_gamma=args.dspark_gamma,
        dspark_confidence=args.dspark_confidence,
        dspark_verify_mode=args.dspark_verify_mode,
        dspark_verify_budget=args.dspark_verify_budget,
        dspark_budget_mode=args.dspark_budget_mode,
        fake_balance_expert=args.fake_balance_expert,
    )
    env = preset_env(args.role, args.checkpoint_path, **options)
    if args.token_map_cache:
        env["RTP_V41_TOKEN_MAP_CACHE"] = str(args.token_map_cache)
    command = [sys.executable, "-m", "rtp_llm.start_server"]
    if args.print_command:
        # Only print preset-owned values; never expose inherited secrets.
        owned = preset_env(
            args.role,
            args.checkpoint_path,
            parent={},
            **{**options, "fake_balance_expert": env_bool(env, "FAKE_BALANCE_EXPERT")},
        )
        if args.token_map_cache:
            owned["RTP_V41_TOKEN_MAP_CACHE"] = str(args.token_map_cache)
        print(
            "env "
            + " ".join(
                f"{key}={shlex.quote(value)}" for key, value in sorted(owned.items())
            )
            + " "
            + shlex.join(command)
        )
        return 0
    # Model providers and device detection see the same fixed env as the child.
    os.environ.update(env)
    report = collect_report(
        require_backend=not args.check_only, probe_allocations=args.probe_allocations
    )
    value = json.dumps(report, indent=2, ensure_ascii=False)
    print(value, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(value + "\n")
    if not report["prerequisites_ok"]:
        return 1
    if args.check_only:
        return 0
    if not args.prefill_address:
        parser.error(
            "native PD serving requires --prefill-address and --decode-address"
        )
    # Build once before forking eight model ranks; workers reuse the checked
    # tokenizer cache rather than rebuilding the 131k-token map independently.
    if args.token_map_cache:
        from rtp_llm.models_py.modules.dsv41.ppu.tools.driver_prefill import (
            _load_token_map,
        )

        _load_token_map(
            str(args.checkpoint_path / "tokenizer.json"), str(args.token_map_cache)
        )
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    raise SystemExit(main())
