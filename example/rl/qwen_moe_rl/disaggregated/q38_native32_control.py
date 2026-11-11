"""Admission and refit control for one resident TP8/DP4/EP32 native policy."""

import asyncio
from dataclasses import dataclass
import hashlib
import json
import re

from q38_native32_topology import CONTROL_TOPOLOGY, DP, WORLD


@dataclass(frozen=True)
class SourceLease:
    """Verified receive buffers retained through the matching native ACK.

    The transport owns buffers until it yields this lease. Once yielded, the
    controller calls the awaitable release only after all 32 native reads have
    completed. A failed or cancelled write leaves the lease held for cleanup.
    """

    name: str
    ipc_by_rank: tuple
    sha256_by_rank: tuple
    release: object


class Native32Policy:
    def __init__(self, topology, incarnation, required_sources, control, token_clients,
                 *, max_pending_bytes=6*1024**3, drain_timeout=1200):
        self.topology = topology
        self.incarnation = incarnation
        self.sources = frozenset(required_sources)
        if len(self.sources) != 1590 or any(not name or name.startswith('mtp.') for name in self.sources):
            raise ValueError('native control requires the exact pinned 1590 main-policy names')
        if len(token_clients) != DP or len(topology.control_addresses) != WORLD:
            raise ValueError('all native control and DP generation endpoints must be bound')
        if not 4*1024**3 < max_pending_bytes <= 8*1024**3 or not 0 < drain_timeout <= 3600:
            raise ValueError('rank-local refit memory and drain time must be bounded')
        self.control = control
        self.clients = tuple(token_clients)
        self.max_pending_bytes = max_pending_bytes
        self.drain_timeout = drain_timeout
        self._lifecycle = asyncio.Lock()
        self._inflight = asyncio.Condition()
        self._active = 0
        self._next_dp = 0
        self._version = None
        self._step = -1
        self.poisoned = False
        self.held_lease = None

    @classmethod
    def from_runtime(cls, topology, incarnation, model_path, shape_manifest, eos_token_ids,
                     *, control_timeout=600, token_timeout=900, request_journals=None):
        from rtp_llm.RL.replica_rpc_client import ReplicaRpcClient
        from rtp_llm.RL.token_rpc_client import TokenProtocolError, TokenRpcClient
        from rtp_llm.models.qwen3_next.qwen38_bootstrap_contract import load_bootstrap_contract

        class RawTokenClient(TokenRpcClient):
            async def _request(self, request_id, ids, config):
                response = await super()._request(request_id, ids, config)
                if config.return_softmax_probs:
                    infos = response.flatten_output.aux_info
                    if (len(infos) != 1 or 'raw_logprobs' not in infos[0].DESCRIPTOR.fields_by_name
                            or not infos[0].HasField('raw_logprobs')):
                        raise TokenProtocolError('native raw FP32 scores are required for rollout')
                return response

        from rtp_request_journal import JournaledTokenRpcMixin
        class JournaledRawTokenClient(JournaledTokenRpcMixin, RawTokenClient):
            pass

        if request_journals is None or len(request_journals) != DP:
            raise ValueError('retain one native RPC journal per attention DP group')
        clients = [JournaledRawTokenClient(address, vocab_size=248320, eos_token_ids=eos_token_ids,
                                          timeout=token_timeout, request_journal=journal)
                   for address,journal in zip(topology.generation_addresses,request_journals)]
        control = ReplicaRpcClient(topology.control_addresses, timeout=control_timeout, topology=CONTROL_TOPOLOGY)
        sources = load_bootstrap_contract(model_path, shape_manifest).source_shapes
        return cls(topology, incarnation, sources, control, clients)

    def _verify(self, rows, phase, version, step, *, committed):
        if (len(rows) != WORLD or {r.world_rank for r in rows} != set(range(WORLD))
                or any(r.phase != phase or r.model_version != version or r.global_step != step
                       or r.update_id or r.quiescent != (phase == 'paused')
                       or r.received_weights != (len(self.sources) if committed else 0)
                       or (r.applied_weights <= 0 if committed else r.applied_weights != 0)
                       for r in rows)):
            raise RuntimeError('incomplete native world, weight or version agreement')
        return rows

    async def bootstrap_status(self):
        return self._verify(await self.control.status(), 'paused', self.incarnation+':0', 0, committed=False)

    async def _wait_idle(self):
        async with self._inflight:
            await self._inflight.wait_for(lambda: self._active == 0)

    async def refit(self, step, stream):
        """Copy the full next policy and return its all-rank ACK receipt."""
        async with self._lifecycle:
            if self.poisoned or type(step) is not int or step != self._step+1:
                raise RuntimeError('invalid next policy step or held native incarnation')
            previous = self._version
            async with self._inflight:
                self._version = None
            try:
                await asyncio.wait_for(self._wait_idle(), timeout=self.drain_timeout)
                if previous is None:
                    await self.bootstrap_status()
                else:
                    self._verify(await self.control.status(), 'running', previous, self._step, committed=True)
                    self._verify(await self.control.pause(), 'paused', previous, self._step, committed=True)
                await self.control.begin_update(global_step=step, max_buffer_bytes=self.max_pending_bytes)
                transaction = self.control.update_id
                if transaction != f'{self.incarnation}:{step+1}':
                    raise RuntimeError('native transaction ID is not the expected next version')
                received = {}
                async for lease in stream:
                    self.held_lease = lease
                    if (not isinstance(lease, SourceLease) or lease.name not in self.sources
                            or lease.name in received or len(lease.ipc_by_rank) != WORLD
                            or len(lease.sha256_by_rank) != WORLD
                            or any(not isinstance(x, str) or not re.fullmatch('[0-9a-f]{64}', x)
                                   for x in lease.sha256_by_rank)):
                        raise ValueError('unverified, duplicate or incomplete checkpoint source lease')
                    if not lease.name.endswith(('.mlp.experts.gate_up_proj', '.mlp.experts.down_proj')):
                        if len(set(lease.sha256_by_rank)) != 1:
                            raise ValueError('shared HF source differs across native ranks')
                    await self.control.push_weight(lease.name, lease.ipc_by_rank)
                    received[lease.name] = lease.sha256_by_rank
                    await lease.release()
                    self.held_lease = None
                if set(received) != self.sources:
                    raise RuntimeError('incomplete native policy stream: commit is forbidden')
                statuses = self._verify(await self.control.commit_update(), 'paused', transaction,
                                        step, committed=True)
                if not all(row.kv_cache_ready for row in statuses):
                    self._verify(await self.control.restore_cache(), 'paused', transaction, step, committed=True)
                self._verify(await self.control.resume(), 'running', transaction, step, committed=True)
                payload = json.dumps(received, sort_keys=True, separators=(',', ':')).encode()
                receipt = dict(native_world=32, logical_policy_instances=1, native_ranks=list(range(WORLD)),
                               global_step=step, model_version=transaction, received_sources=len(received),
                               rank_source_sha256=hashlib.sha256(payload).hexdigest(), source_digests=received)
                async with self._inflight:
                    self._step, self._version = step, transaction
                return receipt
            except BaseException:
                self.poisoned = True
                raise

    async def generate(self, prompt_ids, sampling, *, request_id):
        if len(prompt_ids)+sampling.max_new_tokens > 8192 or not sampling.logprobs or sampling.logprobs_mode != 'raw_logprobs':
            raise ValueError('rollout requires the bounded context and original FP32 scores')
        async with self._inflight:
            if self.poisoned or self._version is None:
                raise RuntimeError('rollout admission is closed pending the complete native refit')
            version, step = self._version, self._step
            client = self.clients[self._next_dp]
            self._next_dp = (self._next_dp+1) % DP
            self._active += 1
        try:
            result = await client.generate(prompt_ids, sampling, request_id=request_id)
            if (result.model_version != version or result.global_step != step
                    or result.log_probs is None or len(result.log_probs) != len(result.token_ids)):
                self.poisoned = True
                raise RuntimeError('native generation returned stale policy or incomplete scores')
            return result
        finally:
            async with self._inflight:
                self._active -= 1
                self._inflight.notify_all()
