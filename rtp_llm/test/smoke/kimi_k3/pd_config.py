"""Configuration and GPU process parsing for the paired Bazel actions."""

import csv
import ipaddress
import math
from pathlib import Path

# CP-sharded cache keys span all Prefill ranks, independently of Decode DP.
PREFILL_TP_SIZE = 8
CACHE_PAGE_TOKENS = 4096
COMPACT_CACHE_PAGE_TOKENS = 1024


def validate_config(config, *, local_role):
    if config.get("model_type", "kimi_k3") not in ("kimi_k3", "kimi_k31"):
        raise ValueError("PD smoke requires kimi_k3 or kimi_k31")
    if type(config.get("compact_smoke", False)) is not bool:
        raise ValueError("compact_smoke must be boolean")
    strategies = {"mega_moe", "mega_moe_se"}
    if local_role not in ("prefill", "decode"):
        raise ValueError("local_role must be Prefill or Decode")
    for role in ("prefill", "decode"):
        host = config[role]
        if "moe_strategy" in host and host["moe_strategy"] not in strategies:
            raise ValueError(f"Unsupported {role} MoE executor")
        if any(
            key in host
            for key in (
                "host",
                "runtime",
                "user",
                "container",
                "repo",
                "worker",
                "guard",
                "allow_hf3fs_root",
            )
        ):
            raise ValueError(
                f"Native {role} must not specify SSH/container/guard fields"
            )
        address = ipaddress.IPv4Address(host["ip"])
        if address.is_loopback or address.is_unspecified or address.is_multicast:
            raise ValueError(f"{role} needs a routable peer IPv4 address")
        if type(host.get("allow_shared_accuracy", False)) is not bool:
            raise ValueError(f"{role}.allow_shared_accuracy must be boolean")
        paths = ("checkpoint", "draft_checkpoint", "allow_checkpoint_root")
        # Each action owns only its own Bazel output directory.
        if role == local_role or "artifact_root" in host:
            paths += ("artifact_root",)
        for key in paths:
            if not Path(host[key]).is_absolute():
                raise ValueError(f"{role}.{key} must be absolute")
        if "expected_draft_shards" in host and (
            type(host["expected_draft_shards"]) is not int
            or host["expected_draft_shards"] <= 0
        ):
            raise ValueError(f"{role}.expected_draft_shards must be a positive integer")
        port = host["port"]
        if type(port) is not int or not 1024 <= port <= 65535 - 8 * 9:
            raise ValueError(f"Invalid {role} port block")
        for key in (
            "reserve_runtime_mem_mb",
            "memory_cache_size_mb",
            "kv_cache_mem_mb",
            "min_free_gib",
        ):
            if key in host:
                value = host[key]
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(f"{role}.{key} must be positive and finite")
                if key != "min_free_gib" and type(value) is not int:
                    raise ValueError(f"{role}.{key} must be an integer")
    if config["prefill"]["ip"] == config["decode"]["ip"]:
        raise ValueError("Prefill and Decode must be on distinct machines")
    if type(config.get("decode_dp", 1)) is not int or config.get(
        "decode_dp", 1
    ) not in (1, 2):
        raise ValueError("The smoke supports Decode DP1 or DP2")
    for key in ("fp8_gemm", "fp8_kv_cache", "warm_up", "model_warm_up"):
        if key in config and type(config[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if "quantization" in config:
        quantization = config["quantization"]
        supported = ("", "FP8_PER_BLOCK") + (
            ("KIMI_K31_NVFP4",) if config.get("model_type") == "kimi_k31" else ()
        )
        if not isinstance(quantization, str) or quantization.upper() not in supported:
            raise ValueError("Unsupported projection quantization")
    if "kda_nvfp4" in config or "dsa_nvfp4" in config:
        raise ValueError(
            "Use quantization=KIMI_K31_NVFP4 instead of kda_nvfp4/dsa_nvfp4"
        )
    if "moe_strategy" in config and config["moe_strategy"] not in strategies:
        raise ValueError("Unsupported MoE executor")
    for key in ("startup_timeout_s", "request_timeout_s"):
        value = config.get(key, 3600 if key == "startup_timeout_s" else 7200)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive number of seconds")
    return config


def gpu_contexts(text):
    contexts = set()
    for row in csv.reader(text.splitlines(), skipinitialspace=True):
        if not row:
            continue
        if len(row) < 2 or not row[0].strip() or not row[1].strip().isdigit():
            raise RuntimeError(f"Invalid GPU process evidence: {row}")
        contexts.add((row[0].strip(), row[1].strip()))
    return contexts
