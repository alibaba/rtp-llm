"""Native multi-rank RTP process ownership and versioned token RPCs."""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time
import uuid

import psutil
import ray
from verl.workers.rollout.replica import TokenOutput
from rtp_topology import ReplicaTopology, required_source_names


def _source():
    source = os.environ.get("RTP_SOURCE")
    if not source:
        raise RuntimeError("RTP_SOURCE is required")
    if source not in sys.path:
        sys.path.insert(0, source)


def _audit_root() -> Path:
    result = Path(os.environ["RLLM_RTP_ACCEPTANCE_DIR"]).resolve()
    result.mkdir(parents=True, exist_ok=True)
    return result


def _deployment_root() -> Path:
    configured = os.environ.get("RLLM_RTP_DEPLOY_ROOT") or os.environ.get(
        "RTP_DEPLOY_ROOT"
    )
    if configured:
        return Path(configured).resolve()
    audit = _audit_root().resolve()
    for parent in [audit, *audit.parents]:
        if parent.name == "acceptance":
            return parent.parent
    return Path(os.environ["RTP_SOURCE"]).resolve().parent


def _base_port(replica: int) -> int:
    return int(os.environ.get("RLLM_RTP_START_PORT", "24000")) + replica * int(
        os.environ.get("RLLM_RTP_REPLICA_PORT_STRIDE", "1000")
    )


def _check_native_http_health(host, port, tp_rank, timeout=2.0):
    """Require the rank's own HTTP listener, not only its earlier RPC service."""
    import http.client

    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request("GET", "/health")
        response = connection.getresponse()
        response.read(65536)
        if response.status != 200:
            raise RuntimeError(
                f"RTP rank {tp_rank} HTTP /health returned {response.status} on port {port}"
            )
        return {
            "tp_rank": tp_rank,
            "url": f"http://{host}:{port}/health",
            "status": response.status,
        }
    finally:
        connection.close()


def _topology(replica: int) -> ReplicaTopology:
    return ReplicaTopology.from_dict(
        json.loads((_audit_root() / f"rtp_topology_{replica}.json").read_text())
    )


def _address(replica: int) -> str:
    return _topology(replica).addresses[0]


def _write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


async def _new_control(replica: int):
    _source()
    from rtp_llm.RL.replica_rpc_client import ReplicaRpcClient

    topo = _topology(replica)
    client = ReplicaRpcClient(
        topo.addresses,
        timeout=float(os.environ.get("RLLM_RTP_CONTROL_TIMEOUT", "300")),
        topology={
            "world_size": topo.tp_size,
            "tp_size": topo.tp_size,
            "dp_size": 1,
            "ep_size": topo.ep_size,
        },
    )
    try:
        await client.status()
    except BaseException:
        await client.close()
        raise
    return client


async def _sleep(replica: int):
    """Pause the replica with weights and KV cache kept resident (CUDA Graph safe)."""
    control = await _new_control(replica)
    try:
        reply = (await control.status())[0]
        if reply.phase == "running":
            reply = (await control.pause())[0]
        if reply.phase != "paused":
            raise RuntimeError(f"RTP cannot pause from phase {reply.phase}")
    finally:
        await control.close()


async def _wake(replica: int):
    control = await _new_control(replica)
    try:
        reply = (await control.status())[0]
        if reply.phase == "running":
            return
        if reply.phase != "paused":
            raise RuntimeError(f"RTP cannot resume from phase {reply.phase}")
        await control.resume()
    finally:
        await control.close()


class RTPServer:
    """A Ray controller owns one native RTP process group and token client."""

    def __init__(self, replica_rank: int, model_path: str, cuda_device: str, tp_size: int):
        self.replica_rank = replica_rank
        self.model_path = model_path
        self.cuda_device = cuda_device
        self.tp_size = int(tp_size)
        self.process = None
        self.client = None
        self.tokenizer = None
        self.log = None
        self.poisoned = False
        self.requests = 0
        self.process_identity = None
        self._owned_processes = {}
        self._cleanup_receipt = None
        atexit.register(self._terminate)

    def _capture_process_tree(self):
        for pid, created in list(self._owned_processes.items()):
            try:
                process = psutil.Process(pid)
                if process.create_time() != created:
                    continue
                for child in process.children(recursive=True):
                    self._owned_processes[child.pid] = child.create_time()
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        if self.process_identity is not None:
            self.process_identity["descendants"] = [
                {"pid": pid, "create_time": created}
                for pid, created in sorted(self._owned_processes.items())
                if pid != self.process_identity["pid"]
            ]
            _write_json(
                _audit_root() / f"rtp_process_{self.replica_rank}.json",
                self.process_identity,
            )

    @staticmethod
    def _running(process):
        try:
            return process.is_running() and process.status() not in (
                psutil.STATUS_ZOMBIE,
                psutil.STATUS_DEAD,
            )
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            return False

    def _terminate(self):
        if self.process is None or self._cleanup_receipt is not None:
            return
        identity = self.process_identity
        if identity is None:
            raise RuntimeError(
                "RTP process identity was not captured; refusing unverified termination"
            )
        if self.process.pid != identity["pid"]:
            raise RuntimeError(
                "RTP launcher PID differs from its captured original process identity"
            )
        self._capture_process_tree()
        try:
            leader = psutil.Process(identity["pid"])
            if leader.create_time() != identity["create_time"]:
                raise RuntimeError(
                    "RTP leader PID was reused; refusing to signal unrelated process"
                )
            if self._running(leader) and os.getpgid(leader.pid) != identity["pgid"]:
                raise RuntimeError(
                    "RTP leader changed process group; refusing unverified termination"
                )
        except (psutil.NoSuchProcess, psutil.ZombieProcess, ProcessLookupError):
            pass
        targets = []
        for pid, created in self._owned_processes.items():
            try:
                process = psutil.Process(pid)
                if process.create_time() == created and self._running(process):
                    targets.append(process)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        for process in targets:
            try:
                process.terminate()
            except psutil.NoSuchProcess:
                pass
        exited, alive = psutil.wait_procs(targets, timeout=20)
        for process in alive:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        killed_exited, survivors = psutil.wait_procs(alive, timeout=10)
        survivors = [p.pid for p in survivors if self._running(p)]
        if survivors:
            raise RuntimeError(f"owned RTP readers did not terminate: {survivors}")
        for observed in exited + killed_exited:
            if (
                observed.pid == self.process.pid
                and getattr(observed, "returncode", None) is not None
            ):
                self.process.returncode = observed.returncode
        self.process.wait(timeout=10)
        self._cleanup_receipt = {
            "process_tree_terminated": True,
            "leader_actual_returncode": self.process.returncode,
            "pid": self.process.pid,
            "owned_pids": sorted(self._owned_processes),
            "pgid": identity["pgid"],
            "create_time": identity["create_time"],
            "finished_at_unix": time.time(),
        }
        _write_json(
            _audit_root() / f"rtp_shutdown_{self.replica_rank}.json",
            self._cleanup_receipt,
        )

    def _native_available(self):
        return self.process is not None and self.process.poll() is None

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        _source()
        from rtp_llm.RL.token_rpc_client import TokenSamplingParams

        if self.poisoned or not self._native_available():
            raise RuntimeError("RTP replica is unavailable")
        if any(kwargs.get(key) for key in ("image_data", "video_data", "audio_data")):
            raise NotImplementedError("this RTP RL adapter is text-only")
        params = dict(sampling_params)
        temperature = float(params.pop("temperature", 1.0))
        top_p = float(params.pop("top_p", 1.0))
        top_k = int(params.pop("top_k", 0))
        budget = int(params.pop("max_tokens", params.pop("max_new_tokens", 2048)))
        requested_logprobs = params.pop("logprobs", 1)
        if requested_logprobs not in (1, True):
            raise ValueError("RTP training requires selected-token logprobs")
        seed = params.pop("seed", params.pop("random_seed", None))
        reuse_cache = bool(
            params.pop("reuse_cache", os.environ.get("RLLM_RTP_REUSE_CACHE", "1") != "0")
        )
        return_routed_experts = bool(params.pop("return_routed_experts", False))
        if return_routed_experts:
            raise NotImplementedError(
                "live RTP return_routed_experts is not yet implemented on this build"
            )
        if params:
            raise ValueError(f"unsupported RTP sampling parameters: {sorted(params)}")
        native_params = TokenSamplingParams(
            max_new_tokens=budget,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            logprobs=True,
            logprobs_mode="raw_logprobs",
            require_raw_logprobs=True,
            reuse_cache=reuse_cache,
            enable_device_cache=reuse_cache,
            stop_token_ids=tuple(sorted(self.client.eos_token_ids)),
        )
        control = await _new_control(self.replica_rank)
        try:
            status = (await control.status())[0]
            if status.phase != "running":
                raise RuntimeError(
                    f"generation requires running RTP, got {status.phase}"
                )
            version, step = status.model_version, status.global_step
        finally:
            await control.close()
        native_request_id = uuid.uuid4().int & ((1 << 63) - 1)
        result = await self.client.generate(
            native_request_id,
            prompt_ids,
            native_params,
            expected_version=version,
        )
        if (
            result.global_step != step
            or result.model_version != version
            or not result.log_probs
            or not result.has_raw_logprobs
        ):
            raise RuntimeError(
                "RTP returned inconsistent policy-step or native FP32 raw_logprobs metadata"
            )
        self.requests += 1
        receipt = {
            "engine": "rtp-llm",
            "request_id": str(request_id),
            "replica_rank": self.replica_rank,
            "model_version": result.model_version,
            "native_request_id": native_request_id,
            "native_incarnation": getattr(self, "incarnation", None),
            "native_boot": getattr(self, "boot", None),
            "global_step": result.global_step,
            "prompt_tokens": len(prompt_ids),
            "response_tokens": len(result.token_ids),
            "logprobs": len(result.log_probs),
            "total_reuse_len": result.total_reuse_len,
            "local_reuse_len": result.local_reuse_len,
            "logprobs_mode": result.logprobs_mode,
            "finished_at_unix": time.time(),
        }
        with (_audit_root() / f"rtp_generations_{self.replica_rank}.jsonl").open(
            "a"
        ) as stream:
            stream.write(json.dumps(receipt) + "\n")
        return TokenOutput(
            token_ids=result.token_ids,
            log_probs=result.log_probs,
            stop_reason=result.finish_reason,
            extra_fields=receipt,
        )

    async def wake_up(self, **kwargs):
        await _wake(self.replica_rank)

    async def sleep(self):
        await _sleep(self.replica_rank)

    async def start_profile(self, **kwargs):
        return None

    async def stop_profile(self):
        return None

    async def score_tokens(self, prompt_ids, response_ids, expected_version=None):
        if self.poisoned or self.client is None:
            raise RuntimeError("RTP replica is unavailable")
        control = await _new_control(self.replica_rank)
        try:
            status = (await control.status())[0]
            version = (
                expected_version
                if expected_version is not None
                else status.model_version
            )
            if status.model_version != version:
                raise RuntimeError("score request policy version is stale")
        finally:
            await control.close()
        values = await self.client.score_tokens(
            uuid.uuid4().int & ((1 << 63) - 1),
            prompt_ids,
            response_ids,
            expected_version=version,
        )
        return {
            "log_probs": values,
            "model_version": version,
            "global_step": status.global_step,
        }

    async def probe_synced_policy(self, expected_step, expected_version):
        """Score a fixed continuation in the updated native model."""
        prompt_ids = self.tokenizer.encode(
            "The value of 37 * 43 is", add_special_tokens=False
        )
        response_ids = self.tokenizer.encode(" 1591.", add_special_tokens=False)
        scored = await self.score_tokens(prompt_ids, response_ids, expected_version)
        if scored["global_step"] != expected_step:
            raise RuntimeError("post-sync RTP probe reported a stale training step")
        sync = json.loads(
            (
                _audit_root()
                / f"rtp_sync_rank{self.replica_rank}_step{expected_step}.json"
            ).read_text()
        )
        if sync["model_version"] != scored["model_version"]:
            raise RuntimeError(
                "post-sync probe version does not match the committed HF weight stream"
            )
        values = scored["log_probs"]
        bits = struct.pack(f"<{len(values)}f", *values)
        record = {
            "engine": "rtp-llm",
            "replica_rank": self.replica_rank,
            "global_step": expected_step,
            "model_version": scored["model_version"],
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
            "raw_log_probs": values,
            "raw_fp32_bits_hex": bits.hex(),
            "score_sha256": hashlib.sha256(bits).hexdigest(),
            "full_source_sha256": sync["full_source_sha256"],
            "tp_size": self.tp_size,
            "transport": "native_GenerateStreamCall_prompt_target_scores",
        }
        _write_json(
            _audit_root()
            / f"rtp_probe_rank{self.replica_rank}_step{expected_step}.json",
            record,
        )
        return record

    async def shutdown(self):
        self.poisoned = True
        try:
            if self.client is not None:
                await self.client.close()
        finally:
            try:
                self._terminate()
            finally:
                if self.log is not None:
                    self.log.close()
        return {
            "replica_rank": self.replica_rank,
            "exit_code": self.process.returncode if self.process else None,
            **(self._cleanup_receipt or {}),
        }
