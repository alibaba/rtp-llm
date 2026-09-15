"""Startup kernel and real-request warmup orchestration."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rtp_llm.config.py_config_modules import PyEnvConfigs

STARTUP_REAL_WARMUP_ENV = "STARTUP_REAL_WARMUP"
STARTUP_REAL_WARMUP_TIMEOUT_ENV = "STARTUP_REAL_WARMUP_TIMEOUT_S"
STARTUP_WARMUP_HEALTH_GATE_FILE_ENV = "RTP_LLM_STARTUP_WARMUP_HEALTH_GATE_FILE"

STARTUP_REAL_WARMUP_MIN_TOKEN_LEN = 2
STARTUP_REAL_WARMUP_TIMEOUT_S = 600.0
STARTUP_REAL_WARMUP_MAX_NEW_TOKENS = 1
STARTUP_REAL_WARMUP_TOKEN_ID = 100

_AUTO_WARMUP_MODELS = frozenset(("deepseek_v4", "kimi_k3"))


class _StartupRealWarmupAddressResolutionError(RuntimeError):
    pass


def startup_real_warmup_enabled(model_type: str) -> bool:
    """Resolve the shared switch used by kernel and real-request warmup."""

    flag = os.environ.get(STARTUP_REAL_WARMUP_ENV, "auto")
    normalized = flag.strip().lower()
    if normalized in ("0", "false", "off", "no"):
        return False
    if normalized in ("1", "true", "on", "yes", "force"):
        return True
    return model_type in _AUTO_WARMUP_MODELS


def _role_is_prefill(py_env_configs: PyEnvConfigs) -> bool:
    from rtp_llm.ops import RoleType

    role_type = py_env_configs.role_config.role_type
    role_value = getattr(role_type, "value", role_type)
    prefill_value = getattr(RoleType.PREFILL, "value", RoleType.PREFILL)
    role_is_prefill = role_type == RoleType.PREFILL or str(role_type).endswith(
        "PREFILL"
    )
    try:
        role_is_prefill = role_is_prefill or int(role_value) == int(prefill_value)
    except Exception:
        pass
    return role_is_prefill


def _is_startup_real_warmup_entry_rank(py_env_configs: PyEnvConfigs) -> bool:
    parallelism_config = py_env_configs.parallelism_config
    world_rank = int(parallelism_config.world_rank)
    world_size = int(parallelism_config.world_size)
    tp_size = int(parallelism_config.tp_size)
    if world_size <= 1:
        return True
    if tp_size <= 0:
        raise ValueError(
            f"parallelism_config.tp_size should be positive, got {tp_size}"
        )
    return world_rank % tp_size == 0


def _should_run_startup_real_warmup(py_env_configs: PyEnvConfigs) -> bool:
    model_type = getattr(py_env_configs.model_args, "model_type", "")
    if not startup_real_warmup_enabled(model_type):
        return False
    if not _role_is_prefill(py_env_configs):
        return False
    if not _is_startup_real_warmup_entry_rank(py_env_configs):
        parallelism_config = py_env_configs.parallelism_config
        logging.info(
            "skip startup real warmup on non-entry rank, model_type=%s, "
            "world_rank=%s, tp_size=%s, world_size=%s",
            model_type,
            parallelism_config.world_rank,
            parallelism_config.tp_size,
            parallelism_config.world_size,
        )
        return False
    return True


def setup_startup_warmup_health_gate(
    py_env_configs: PyEnvConfigs,
) -> str | None:
    if not _should_run_startup_real_warmup(py_env_configs):
        os.environ.pop(STARTUP_WARMUP_HEALTH_GATE_FILE_ENV, None)
        return None

    gate_file = os.path.join(
        "/tmp",
        "rtp_llm_startup_warmup_ready_"
        f"{os.getpid()}_{int(py_env_configs.server_config.start_port)}",
    )
    try:
        os.remove(gate_file)
    except FileNotFoundError:
        pass
    os.environ[STARTUP_WARMUP_HEALTH_GATE_FILE_ENV] = gate_file
    logging.info("startup warmup health gate enabled, gate_file=%s", gate_file)
    return gate_file


def mark_startup_warmup_health_gate_ready(gate_file: str | None) -> None:
    if not gate_file:
        return
    try:
        with open(gate_file, "w") as f:
            f.write("ready\n")
        logging.info("startup warmup health gate marked ready, gate_file=%s", gate_file)
    except Exception:
        logging.error(
            "failed to mark startup warmup health gate ready, gate_file=%s, trace=%s",
            gate_file,
            traceback.format_exc(),
        )
        raise


def _get_startup_real_warmup_pow2_lens(max_len: int) -> list[int]:
    if max_len < STARTUP_REAL_WARMUP_MIN_TOKEN_LEN:
        raise ValueError(
            "model_args.max_seq_len should be at least "
            f"{STARTUP_REAL_WARMUP_MIN_TOKEN_LEN}, got {max_len}"
        )
    lens = []
    value = STARTUP_REAL_WARMUP_MIN_TOKEN_LEN
    while value <= max_len:
        lens.append(value)
        value *= 2
    if lens[-1] != max_len:
        lens.append(max_len)
    return lens


def _get_startup_real_warmup_max_len(py_env_configs: PyEnvConfigs) -> int:
    model_max_len = int(getattr(py_env_configs.model_args, "max_seq_len", None) or 0)
    if model_max_len <= 0:
        raise ValueError(
            f"model_args.max_seq_len should be positive, got {model_max_len}"
        )
    logging.info(
        "startup real warmup max len = model max_seq_len = %d",
        model_max_len,
    )
    return model_max_len


def _get_startup_real_warmup_token_lens(
    py_env_configs: PyEnvConfigs,
) -> list[int]:
    max_len = _get_startup_real_warmup_max_len(py_env_configs)
    token_lens = _get_startup_real_warmup_pow2_lens(max_len)
    logging.info(
        "startup real warmup uses fixed pow2 token lens through max_seq_len=%d: %s",
        max_len,
        token_lens,
    )
    return token_lens


def _get_startup_real_warmup_grpc_addresses(
    py_env_configs: PyEnvConfigs,
) -> list[str]:
    from rtp_llm.distribute.distributed_server import (
        get_dp_addrs_from_world_info,
        get_world_info,
    )

    parallelism_config = py_env_configs.parallelism_config
    resolve_trace = None
    try:
        world_info = get_world_info(
            server_config=py_env_configs.server_config,
            distribute_config=py_env_configs.distribute_config,
            parallelism_config=parallelism_config,
        )
        addrs = get_dp_addrs_from_world_info(world_info, parallelism_config)
        if addrs:
            return addrs
    except Exception:
        resolve_trace = traceback.format_exc()

    world_size = int(parallelism_config.world_size)
    if world_size > 1:
        if resolve_trace:
            logging.warning(
                "failed to resolve startup real warmup grpc addrs from world info, "
                "trace=%s",
                resolve_trace,
            )
        raise _StartupRealWarmupAddressResolutionError(
            "failed to resolve startup real warmup grpc entry address "
            "in multi-rank mode; refusing to fallback to local rpc_server_port"
        )
    if resolve_trace:
        logging.warning(
            "failed to resolve startup real warmup grpc addrs from world info, "
            "fallback to local rpc_server_port, trace=%s",
            resolve_trace,
        )
    return [f"127.0.0.1:{int(py_env_configs.server_config.rpc_server_port)}"]


def _new_startup_real_warmup_request_id(index: int) -> int:
    return (int(time.time() * 1000000) + index) & 0x7FFFFFFFFFFFFFFF


def _get_startup_real_warmup_speculative_reserve_step(
    py_env_configs: PyEnvConfigs,
) -> int:
    from rtp_llm.ops import SpeculativeType

    sp_config = getattr(py_env_configs, "sp_config", None)
    if sp_config is None:
        return 0
    sp_type = getattr(sp_config, "type", SpeculativeType.NONE)
    if sp_type in (None, "", SpeculativeType.NONE):
        return 0
    return int(getattr(sp_config, "gen_num_per_cycle", 0) or 0) + 1


def _get_startup_real_warmup_request_token_len(
    token_len: int, max_len: int, reserve_step: int = 0
) -> int:
    max_request_token_len = max_len - STARTUP_REAL_WARMUP_MAX_NEW_TOKENS
    if reserve_step > 0:
        if max_len <= reserve_step:
            raise ValueError(
                "model_args.max_seq_len should be greater than speculative "
                f"reserve_step, got max_seq_len={max_len}, reserve_step={reserve_step}"
            )
        max_request_token_len = min(max_request_token_len, max_len - reserve_step)
    if max_request_token_len <= 0:
        raise ValueError(
            "startup real warmup request token len should be positive, got "
            f"max_seq_len={max_len}, reserve_step={reserve_step}, "
            f"max_new_tokens={STARTUP_REAL_WARMUP_MAX_NEW_TOKENS}"
        )
    return min(token_len, max_request_token_len)


def _get_startup_real_warmup_timeout_s() -> float:
    timeout_s = float(
        os.environ.get(
            STARTUP_REAL_WARMUP_TIMEOUT_ENV,
            STARTUP_REAL_WARMUP_TIMEOUT_S,
        )
    )
    if timeout_s <= 0:
        raise ValueError(
            f"{STARTUP_REAL_WARMUP_TIMEOUT_ENV} should be positive, got {timeout_s}"
        )
    return timeout_s


def _run_startup_real_warmup_async(coroutine):
    if hasattr(asyncio, "run"):
        return asyncio.run(coroutine)
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coroutine)


async def _run_startup_real_warmup_grpc(
    py_env_configs: PyEnvConfigs,
) -> None:
    import torch

    from rtp_llm.config.generate_config import GenerateConfig
    from rtp_llm.cpp.model_rpc.model_rpc_client import ModelRpcClient
    from rtp_llm.utils.base_model_datatypes import GenerateInput

    token_lens = _get_startup_real_warmup_token_lens(py_env_configs)
    max_len = _get_startup_real_warmup_max_len(py_env_configs)
    reserve_step = _get_startup_real_warmup_speculative_reserve_step(py_env_configs)
    addresses = _get_startup_real_warmup_grpc_addresses(py_env_configs)
    timeout_s = _get_startup_real_warmup_timeout_s()
    timeout_ms = int(timeout_s * 1000)

    client_config = (
        py_env_configs.grpc_config.get_client_config()
        if py_env_configs.grpc_config is not None
        else {}
    )
    logging.info(
        "running startup real warmup via backend grpc, model_type=%s, addrs=%s, "
        "token_lens=%s, token_id=%d, max_new_tokens=%d, reserve_step=%d, timeout=%.1fs",
        getattr(py_env_configs.model_args, "model_type", ""),
        addresses,
        token_lens,
        STARTUP_REAL_WARMUP_TOKEN_ID,
        STARTUP_REAL_WARMUP_MAX_NEW_TOKENS,
        reserve_step,
        timeout_s,
    )

    begin_all = time.time()
    total_requests = 0
    for addr_idx, addr in enumerate(addresses):
        client = ModelRpcClient(
            addresses=[addr],
            client_config=client_config,
            max_rpc_timeout_ms=timeout_ms,
        )
        try:
            for len_idx, token_len in enumerate(token_lens):
                total_requests += 1
                request_id = _new_startup_real_warmup_request_id(
                    addr_idx * len(token_lens) + len_idx
                )
                request_token_len = _get_startup_real_warmup_request_token_len(
                    token_len, max_len, reserve_step
                )
                generate_config = GenerateConfig(
                    max_new_tokens=STARTUP_REAL_WARMUP_MAX_NEW_TOKENS,
                    top_k=1,
                    top_p=1.0,
                    temperature=0.0,
                    do_sample=False,
                    can_use_pd_separation=False,
                    reuse_cache=False,
                    enable_device_cache=False,
                    enable_memory_cache=False,
                    enable_remote_cache=False,
                    aux_info=True,
                    timeout_ms=timeout_ms,
                )
                generate_input = GenerateInput(
                    request_id=request_id,
                    token_ids=torch.full(
                        (request_token_len,),
                        STARTUP_REAL_WARMUP_TOKEN_ID,
                        dtype=torch.int32,
                    ),
                    mm_inputs=[],
                    generate_config=generate_config,
                )

                begin = time.time()
                last_aux = None
                chunk_count = 0
                logging.info(
                    "startup grpc warmup request begin, "
                    "addr=%s, request_id=%d, target_token_len=%d, "
                    "request_token_len=%d, max_new_tokens=%d, reserve_step=%d",
                    addr,
                    request_id,
                    token_len,
                    request_token_len,
                    STARTUP_REAL_WARMUP_MAX_NEW_TOKENS,
                    reserve_step,
                )
                async for outputs in client.enqueue(generate_input):
                    chunk_count += 1
                    if outputs.generate_outputs:
                        last_aux = outputs.generate_outputs[0].aux_info
                if last_aux is not None:
                    logging.info(
                        "startup grpc warmup request finished, addr=%s, request_id=%d, "
                        "target_token_len=%d, request_token_len=%d, max_new_tokens=%d, "
                        "chunks=%d, input_len=%s, reuse_len=%s, output_len=%s, "
                        "cost=%.2fs",
                        addr,
                        request_id,
                        token_len,
                        request_token_len,
                        STARTUP_REAL_WARMUP_MAX_NEW_TOKENS,
                        chunk_count,
                        getattr(last_aux, "input_len", None),
                        getattr(last_aux, "reuse_len", None),
                        getattr(last_aux, "output_len", None),
                        time.time() - begin,
                    )
                else:
                    logging.info(
                        "startup grpc warmup request finished, addr=%s, request_id=%d, "
                        "target_token_len=%d, request_token_len=%d, max_new_tokens=%d, "
                        "chunks=%d, aux_info=None, cost=%.2fs",
                        addr,
                        request_id,
                        token_len,
                        request_token_len,
                        STARTUP_REAL_WARMUP_MAX_NEW_TOKENS,
                        chunk_count,
                        time.time() - begin,
                    )
        finally:
            try:
                await client.close()
            except Exception:
                logging.warning(
                    "failed to close startup grpc warmup client, addr=%s, trace=%s",
                    addr,
                    traceback.format_exc(),
                )

    logging.info(
        "startup grpc warmup finished, requests=%d, addrs=%d, token_lens=%d, cost=%.2fs",
        total_requests,
        len(addresses),
        len(token_lens),
        time.time() - begin_all,
    )


def maybe_run_startup_real_warmup(py_env_configs: PyEnvConfigs) -> bool:
    if not _should_run_startup_real_warmup(py_env_configs):
        return False

    try:
        _run_startup_real_warmup_async(_run_startup_real_warmup_grpc(py_env_configs))
        return True
    except _StartupRealWarmupAddressResolutionError:
        logging.error(
            "startup real warmup address resolution failed, trace: %s",
            traceback.format_exc(),
        )
        raise
    except Exception:
        logging.error(
            "startup real warmup failed, trace: %s",
            traceback.format_exc(),
        )
        return False


def start_post_startup_jit_cache_writer(
    py_env_configs: PyEnvConfigs,
    startup_warmup_succeeded: bool,
) -> None:
    remote_write_dir = (
        py_env_configs.jit_config.warm_up_jit_and_write_remote or ""
    ).strip()
    if not remote_write_dir:
        return

    def _write_remote_jit_cache():
        try:
            from rtp_llm.config.server_config_setup import (
                maybe_write_jit_cache_to_remote,
            )

            maybe_write_jit_cache_to_remote(
                py_env_configs,
                startup_warmup_succeeded,
            )
        except Exception:
            logging.error(
                "post-startup remote JIT cache publishing failed, trace=%s",
                traceback.format_exc(),
            )

    writer = threading.Thread(
        target=_write_remote_jit_cache,
        name="post_startup_jit_cache_writer",
        daemon=True,
    )
    writer.start()
    logging.info(
        "post-startup remote JIT cache writer started for "
        "WARM_UP_JIT_AND_WRITE_REMOTE=%s",
        remote_write_dir,
    )


__all__ = [
    "STARTUP_REAL_WARMUP_ENV",
    "STARTUP_REAL_WARMUP_TIMEOUT_ENV",
    "mark_startup_warmup_health_gate_ready",
    "maybe_run_startup_real_warmup",
    "setup_startup_warmup_health_gate",
    "start_post_startup_jit_cache_writer",
    "startup_real_warmup_enabled",
]
