"""Native multi-rank RTP process ownership and versioned token RPCs."""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import time
import uuid

import psutil
import ray
import torch
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
    configured = os.environ.get("RLLM_RTP_DEPLOY_ROOT") or os.environ.get("RTP_DEPLOY_ROOT")
    if configured:
        return Path(configured).resolve()
    audit = _audit_root().resolve()
    for parent in [audit, *audit.parents]:
        if parent.name == "acceptance":
            return parent.parent
    return Path(os.environ["RTP_SOURCE"]).resolve().parent


def _base_port(replica: int) -> int:
    return int(os.environ.get("RLLM_RTP_START_PORT", "24000")) + replica * int(os.environ.get("RLLM_RTP_REPLICA_PORT_STRIDE", "1000"))


def _check_native_http_health(host, port, tp_rank, timeout=2.0):
    """Require the rank's own HTTP listener, not only its earlier RPC service."""
    import http.client

    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request('GET', '/health')
        response = connection.getresponse()
        response.read(65536)
        if response.status != 200:
            raise RuntimeError(f'RTP rank {tp_rank} HTTP /health returned {response.status} on port {port}')
        return {'tp_rank': tp_rank, 'url': f'http://{host}:{port}/health', 'status': response.status}
    finally:
        connection.close()


def _topology(replica: int) -> ReplicaTopology:
    return ReplicaTopology.from_dict(json.loads((_audit_root() / f"rtp_topology_{replica}.json").read_text()))


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

    client = ReplicaRpcClient(_topology(replica).addresses, timeout=float(os.environ.get("RLLM_RTP_CONTROL_TIMEOUT", "300")))
    try:
        await client.status()
    except BaseException:
        await client.close()
        raise
    return client


async def _sleep(replica: int):
    control = await _new_control(replica)
    try:
        reply = (await control.status())[0]
        if reply.phase == "running":
            reply = (await control.pause())[0]
        if reply.phase != "paused":
            raise RuntimeError(f"RTP cannot sleep from phase {reply.phase}")
        if reply.kv_cache_ready and os.environ.get("RLLM_RTP_KEEP_KV_CACHE", "1") != "1":
            await control.release_cache()
    finally:
        await control.close()


async def _wake(replica: int):
    control = await _new_control(replica)
    try:
        reply = (await control.status())[0]
        if reply.phase == "running":
            return
        if reply.phase != "paused":
            raise RuntimeError(f"RTP cannot wake from phase {reply.phase}")
        if not reply.kv_cache_ready:
            await control.restore_cache()
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
        self.log = None
        self.poisoned = False
        self.requests = 0
        self.process_identity = None
        self._owned_processes = {}
        self._cleanup_receipt = None
        atexit.register(self._terminate)

    def _capture_process_tree(self):
        """Snapshot only descendants of processes whose creation time matches."""
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
            self.process_identity['descendants'] = [
                {'pid': pid, 'create_time': created}
                for pid, created in sorted(self._owned_processes.items())
                if pid != self.process_identity['pid']
            ]
            _write_json(_audit_root() / f'rtp_process_{self.replica_rank}.json', self.process_identity)

    @staticmethod
    def _running(process):
        try:
            return process.is_running() and process.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            return False

    def _terminate(self):
        if self.process is None or self._cleanup_receipt is not None:
            return
        identity = self.process_identity
        if identity is None:
            raise RuntimeError("RTP process identity was not captured; refusing unverified termination")
        if self.process.pid != identity["pid"]:
            raise RuntimeError("RTP launcher PID differs from its captured original process identity")
        self._capture_process_tree()
        leader_valid = False
        try:
            leader = psutil.Process(identity["pid"])
            if leader.create_time() != identity["create_time"]:
                raise RuntimeError("RTP leader PID was reused; refusing to signal the unrelated process")
            leader_valid = self._running(leader)
            if leader_valid and os.getpgid(leader.pid) != identity["pgid"]:
                raise RuntimeError("RTP leader changed process group; refusing unverified group termination")
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
        # Native children can have separate process groups; identity-checked
        # descendants are signalled individually. Avoid signalling a numeric
        # PGID after its original leader exits and that identifier can be reused.
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
        survivors = [process.pid for process in survivors if self._running(process)]
        if survivors:
            raise RuntimeError(f"owned RTP readers did not terminate: {survivors}")
        # An early-exited leader cannot authenticate new, previously unseen
        # group members. Observe them, never signal them speculatively.
        unknown_group_members = []
        for process in psutil.process_iter(["pid", "status"]):
            try:
                if process.info["status"] not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD) and os.getpgid(process.pid) == identity["pgid"]:
                    unknown_group_members.append(process.pid)
            except (psutil.NoSuchProcess, psutil.ZombieProcess, ProcessLookupError):
                continue
        if unknown_group_members:
            raise RuntimeError(f"unacknowledged processes remain in the original RTP group; refusing unsafe IPC release: {unknown_group_members}")
        # psutil may already have reaped our direct Popen child. Preserve its
        # observed returncode before subprocess treats ECHILD as status zero.
        for observed in exited + killed_exited:
            if observed.pid == self.process.pid and getattr(observed, "returncode", None) is not None:
                self.process.returncode = observed.returncode
        self.process.wait(timeout=10)
        self._cleanup_receipt = {
            "process_tree_terminated": True,
            "leader_actual_returncode": self.process.returncode,
            "pid": self.process.pid,
            "owned_pids": sorted(self._owned_processes),
            "pgid": identity["pgid"], "create_time": identity["create_time"],
            "finished_at_unix": time.time(),
        }
        _write_json(_audit_root() / f'rtp_shutdown_{self.replica_rank}.json', self._cleanup_receipt)

    async def launch_server(self):
        if self.poisoned:
            raise RuntimeError('RTP launch refused after shutdown')
        _source()
        from transformers import AutoConfig, AutoTokenizer
        from rtp_llm.RL.token_rpc_client import TokenProtocolError, TokenRpcClient

        class StrictTokenRpcClient(TokenRpcClient):
            async def _request(self, request_id, ids, config):
                response = await super()._request(request_id, ids, config)
                if config.return_softmax_probs:
                    infos = response.flatten_output.aux_info
                    if len(infos) != 1 or "raw_logprobs" not in infos[0].DESCRIPTOR.fields_by_name or not infos[0].HasField("raw_logprobs"):
                        raise TokenProtocolError("strict RTP training requires native FP32 raw_logprobs; probability fallback is forbidden")
                return response

        root = _audit_root()
        path = root / f"rtp_replica_{self.replica_rank}.log"
        topology = _topology(self.replica_rank)
        for port in range(topology.start_port, topology.start_port + topology.tp_size * topology.rank_port_stride + 12):
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", port))
        environment = dict(os.environ)
        # RANK/WORLD_SIZE inherited from a Ray/FSDP worker must not turn each
        # separately owned RTP process into a training distributed rank.
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            environment.pop(key, None)
        environment.update(
            CUDA_VISIBLE_DEVICES=self.cuda_device,
            RTP_MODEL_PATH=self.model_path,
            RTP_START_PORT=str(_base_port(self.replica_rank)),
            RTP_REPLICA_RANK=str(self.replica_rank),
            RTP_LLM_RL_MODE="1",
            RTP_TP_SIZE=str(self.tp_size),
            LOCAL_WORLD_SIZE=str(self.tp_size),
            RTP_WEIGHT_MANIFEST_DIR=str(root / f"rtp_weight_manifest_{self.replica_rank}"),
        )
        self.log = path.open("a")
        self.process = subprocess.Popen(
            ["/bin/bash", os.environ["RTP_SERVER_LAUNCHER"]],
            env=environment,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        leader = psutil.Process(self.process.pid)
        self.process_identity = {
            "pid": self.process.pid, "replica_rank": self.replica_rank,
            "pgid": os.getpgid(self.process.pid), "create_time": leader.create_time(),
            "root": str(_deployment_root()), "rtp_source": str(Path(os.environ["RTP_SOURCE"]).resolve()),
            "cuda_visible_devices": self.cuda_device, "model_path": self.model_path,
            "rpc_address": _address(self.replica_rank), "control_addresses": topology.addresses,
            "tp_size": self.tp_size, "gpu_ids": list(topology.gpu_ids), "started": time.time(),
            "launcher": os.environ["RTP_SERVER_LAUNCHER"],
            "audit_dir": str(root.resolve()),
        }
        if ray.is_initialized():
            self.process_identity['ray_actor'] = {
                'name': f'rtp_server_{self.replica_rank}_0',
                'namespace': ray.get_runtime_context().namespace,
            }
        self._owned_processes[self.process.pid] = self.process_identity["create_time"]
        _write_json(root / f"rtp_process_{self.replica_rank}.json", self.process_identity)
        deadline = time.monotonic() + int(os.environ.get("RLLM_RTP_STARTUP_TIMEOUT", "5400"))
        error = None
        while time.monotonic() < deadline:
            if self.poisoned:
                raise RuntimeError('RTP startup interrupted by shutdown')
            self._capture_process_tree()
            if self.process.poll() is not None:
                raise RuntimeError(f"RTP replica {self.replica_rank} exited {self.process.returncode}; {path}")
            try:
                control = await _new_control(self.replica_rank)
                try:
                    replies = await control.status()
                    if len(replies) != self.tp_size:
                        raise RuntimeError("RTP did not report the configured TP rank count")
                    if sorted(reply.tp_rank for reply in replies) != list(range(self.tp_size)) or any(
                            reply.tp_size != self.tp_size for reply in replies):
                        raise RuntimeError('RTP RPC statuses do not identify the complete configured TP group')
                finally:
                    await control.close()
                # RPC services bind before each native HTTP server. A failed
                # late HTTP bind must never be reported as a ready replica.
                health = await asyncio.gather(*[
                    asyncio.to_thread(_check_native_http_health, topology.host,
                        topology.start_port + rank * topology.rank_port_stride + 5,
                        rank, max(.1, min(2.0, deadline - time.monotonic())))
                    for rank in range(self.tp_size)
                ])
                if self.process.poll() is not None:
                    raise RuntimeError('RTP native leader exited before HTTP readiness completed')
                leader = psutil.Process(self.process.pid)
                if leader.create_time() != self.process_identity["create_time"]:
                    raise RuntimeError("RTP leader identity changed during startup")
                command = leader.cmdline()
                self._capture_process_tree()
                manifest_dir = root / f'rtp_weight_manifest_{self.replica_rank}'
                names = required_source_names(manifest_dir, self.tp_size)
                manifest_pids = []
                for rank in range(self.tp_size):
                    manifest = json.loads((manifest_dir / f'rank-{rank:05d}.json').read_text())
                    pid = int(manifest['pid'])
                    process = psutil.Process(pid)
                    if self._owned_processes.get(pid) != process.create_time() or not self._running(process):
                        raise RuntimeError(f'RTP rank {rank} manifest does not name a live owned native process')
                    manifest_pids.append(pid)
                if len(set(manifest_pids)) != self.tp_size:
                    raise RuntimeError('RTP manifests do not identify distinct native TP processes')
                _write_json(root / f"rtp_initial_status_{self.replica_rank}.json", {
                    "model_version": replies[0].model_version, "global_step": replies[0].global_step,
                    "phase": replies[0].phase, "protocol_version": replies[0].protocol_version,
                    "tp_size": self.tp_size,
                    "ranks": [{"tp_rank": r.tp_rank, "tp_size": r.tp_size,
                               "phase": r.phase, "model_version": r.model_version} for r in replies],
                })
                _write_json(root / f'rtp_startup_readiness_{self.replica_rank}.json', {
                    'passed': True, 'tp_size': self.tp_size, 'http_health': health,
                    'native_rank_pids': manifest_pids, 'required_source_count': len(names),
                    'leader_pid': self.process.pid, 'leader_create_time': leader.create_time(),
                    'checked_at_unix': time.time(),
                })
                self.process_identity.update(
                    ready_at_unix=time.time(), cmdline=command,
                    cmdline_sha256=hashlib.sha256(json.dumps(command).encode()).hexdigest(),
                    descendants=[{"pid": pid, "create_time": created}
                                 for pid, created in sorted(self._owned_processes.items())
                                 if pid != self.process.pid],
                )
                _write_json(root / f"rtp_process_{self.replica_rank}.json", self.process_identity)
                break
            except Exception as exc:
                error = repr(exc)
                await asyncio.sleep(1)
        else:
            self._terminate()
            raise TimeoutError(f"RTP startup timed out: {error}; {path}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.tokenizer = tokenizer
        model_config = AutoConfig.from_pretrained(self.model_path, local_files_only=True)
        eos_ids = set()
        generation_path = Path(self.model_path) / "generation_config.json"
        generation = json.loads(generation_path.read_text()) if generation_path.exists() else {}
        for value in (tokenizer.eos_token_id, generation.get("eos_token_id")):
            eos_ids.update(value if isinstance(value, list) else ([] if value is None else [value]))
        if not eos_ids:
            raise RuntimeError("Qwen3 checkpoint did not define any EOS tokens")
        self.client = StrictTokenRpcClient(
            _address(self.replica_rank), vocab_size=getattr(model_config, "text_config", model_config).vocab_size,
            eos_token_ids=sorted(eos_ids), timeout=float(os.environ.get("RLLM_RTP_TOKEN_TIMEOUT", "600")),
        )
        return _address(self.replica_rank)

    def _native_available(self):
        return self.process is not None and self.process.poll() is None

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        _source()
        from rtp_llm.RL.token_rpc_client import TokenSamplingParams

        if self.poisoned or not self._native_available():
            raise RuntimeError("RTP replica is unavailable")
        if any(kwargs.get(key) for key in ("image_data", "video_data", "audio_data")):
            raise NotImplementedError("this Qwen3 acceptance adapter is text-only")
        params = dict(sampling_params)
        temperature = float(params.pop("temperature", 1.0))
        top_p = float(params.pop("top_p", 1.0))
        top_k = int(params.pop("top_k", 0))
        if temperature not in (0.0, 1.0) or top_p != 1.0 or top_k not in (-1, 0):
            raise ValueError("strict RTP acceptance requires temperature=0/1, top_p=1, top_k disabled")
        budget = int(params.pop("max_tokens", params.pop("max_new_tokens", 2048)))
        requested_logprobs = params.pop("logprobs", 1)
        if requested_logprobs not in (1, True):
            raise ValueError("RTP training requires selected-token logprobs")
        seed = params.pop("seed", params.pop("random_seed", None))
        if params:
            raise ValueError(f"unsupported RTP sampling parameters: {sorted(params)}")
        native_params = TokenSamplingParams(
            max_new_tokens=budget, temperature=temperature, top_p=1.0,
            top_k=0, seed=seed, logprobs=True, logprobs_mode="raw_logprobs",
            stop_token_ids=tuple(sorted(self.client.eos_token_ids)),
        )
        control = await _new_control(self.replica_rank)
        try:
            status = (await control.status())[0]
            if status.phase != "running":
                raise RuntimeError(f"generation requires running RTP, got {status.phase}")
            version, step = status.model_version, status.global_step
        finally:
            await control.close()
        native_request_id = uuid.uuid4().int & ((1 << 63) - 1)
        result = await self.client.generate(
            native_request_id, prompt_ids, native_params,
            expected_version=version,
        )
        if result.global_step != step or result.model_version != version or not result.log_probs:
            raise RuntimeError("RTP returned inconsistent policy-step or probability metadata")
        self.requests += 1
        receipt = {
            "engine": "rtp-llm", "request_id": str(request_id),
            "replica_rank": self.replica_rank, "model_version": result.model_version,
            "native_request_id": native_request_id,
            "native_incarnation": getattr(self,"incarnation",None),
            "native_boot": getattr(self,"boot",None),
            "global_step": result.global_step, "prompt_tokens": len(prompt_ids),
            "response_tokens": len(result.token_ids), "logprobs": len(result.log_probs),
            "prompt_sha256": hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest(),
            "response_sha256": hashlib.sha256(json.dumps(result.token_ids).encode()).hexdigest(),
            "raw_logprobs_fp32_sha256": hashlib.sha256(struct.pack(f"<{len(result.log_probs)}f", *result.log_probs)).hexdigest(),
            "logprobs_mode": result.logprobs_mode,
            "finished_at_unix": time.time(),
        }
        with (_audit_root() / f"rtp_generations_{self.replica_rank}.jsonl").open("a") as stream:
            stream.write(json.dumps(receipt) + "\n")
        return TokenOutput(
            token_ids=result.token_ids, log_probs=result.log_probs,
            stop_reason=result.finish_reason, extra_fields=receipt,
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
            version = expected_version if expected_version is not None else status.model_version
            if status.model_version != version:
                raise RuntimeError("score request policy version is stale")
        finally:
            await control.close()
        values = await self.client.score_tokens(
            uuid.uuid4().int & ((1 << 63) - 1), prompt_ids, response_ids,
            expected_version=version,
        )
        return {"log_probs": values, "model_version": version, "global_step": status.global_step}

    async def probe_synced_policy(self, expected_step, expected_version):
        """Score a fixed continuation in the actual updated native model."""
        prompt_ids = self.tokenizer.encode("The value of 37 * 43 is", add_special_tokens=False)
        response_ids = self.tokenizer.encode(" 1591.", add_special_tokens=False)
        scored = await self.score_tokens(prompt_ids, response_ids, expected_version)
        if scored["global_step"] != expected_step:
            raise RuntimeError("post-sync RTP probe reported a stale training step")
        sync = json.loads((_audit_root() / f"rtp_sync_rank{self.replica_rank}_step{expected_step}.json").read_text())
        if sync["model_version"] != scored["model_version"]:
            raise RuntimeError("post-sync probe version does not match the committed HF weight stream")
        values = scored["log_probs"]
        bits = struct.pack(f"<{len(values)}f", *values)
        record = {
            "engine": "rtp-llm", "replica_rank": self.replica_rank,
            "global_step": expected_step, "model_version": scored["model_version"],
            "prompt_ids": prompt_ids, "response_ids": response_ids,
            "raw_log_probs": values, "raw_fp32_bits_hex": bits.hex(),
            "score_sha256": hashlib.sha256(bits).hexdigest(),
            "full_source_sha256": sync["full_source_sha256"],
            "tp_size": self.tp_size,
            "transport": "native_GenerateStreamCall_prompt_target_scores",
        }
        if expected_step > 0:
            baseline = json.loads((_audit_root() / f"rtp_probe_rank{self.replica_rank}_step0.json").read_text())
            if (baseline["prompt_ids"], baseline["response_ids"]) != (prompt_ids, response_ids):
                raise RuntimeError("post-sync probe tokenization differs from baseline")
            if record["model_version"] == baseline["model_version"]:
                raise RuntimeError("post-sync probe still uses the initial engine version")
            if record["full_source_sha256"] == baseline["full_source_sha256"]:
                raise RuntimeError("the trained policy weights did not change before RTP sync")
            if record["score_sha256"] == baseline["score_sha256"]:
                raise RuntimeError("RTP inference scores did not change after the real parameter update")
            record["changed_from_step0"] = True
        _write_json(_audit_root() / f"rtp_probe_rank{self.replica_rank}_step{expected_step}.json", record)
        return record

    async def cold_long_decode_smoke(self, expected_version):
        """Explicit optional cold-only diagnostic; uses this controller's gated client."""
        from rtp_cold_long_decode import run
        return await run(self, expected_version)

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
        return {"replica_rank": self.replica_rank,
                "exit_code": self.process.returncode if self.process else None,
                **(self._cleanup_receipt or {})}

    async def shutdown_if_owned(self, expected_pid, expected_create_time, expected_audit_dir):
        """Reject stale actor names before a driver's emergency shutdown."""
        identity = self.process_identity
        if (identity is None or identity['pid'] != expected_pid
                or identity['create_time'] != expected_create_time
                or identity.get('audit_dir') != str(Path(expected_audit_dir).resolve())):
            raise RuntimeError('Driver cleanup does not own this RTP actor process')
        return await self.shutdown()
