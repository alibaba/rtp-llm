"""Candidate full-HF standalone RTP adapter for the pinned verl CE API.

Requires a CE iterator implementing confirm_native_ack(). This is a baseline
full-source transfer, not a sharded expert redistribution implementation.
TP4/EP1, one rollout replica only; actual GPU qualification is mandatory.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import ray
import torch
import torch.distributed as dist

from rtp_verl_backend import RTPReplica, RTPServerAdapter
from rtp_native_server import _audit_root, _new_control, _source, _write_json
from rtp_topology import preserve_fp32_source
from verl.workers.rollout.base import BaseRollout


class StandaloneRTPReplica(RTPReplica):
    async def _control(self, operation):
        control = await _new_control(self.replica_rank)
        try:
            states = await control.status()
            phase, ready = states[0].phase, states[0].kv_cache_ready
            if operation == 'pause':
                if phase == 'running': await control.pause()
                elif phase != 'paused': raise RuntimeError('native world is not quiescent')
            elif operation == 'release':
                if phase != 'paused': raise RuntimeError('native cache release requires pause')
                if ready: await control.release_cache()
            elif operation == 'restore':
                if phase != 'paused': raise RuntimeError('native cache restore requires pause')
                if not ready: await control.restore_cache()
            elif operation == 'resume':
                if phase == 'paused' and ready: await control.resume()
                elif phase != 'running': raise RuntimeError('native world is not ready to resume')
            else:
                raise ValueError('unsupported native control operation')
        finally:
            await control.close()

    async def abort_all_requests(self):
        # The real controller closes Python admission before native PAUSE and
        # verifies all native ranks reached the same quiescent phase.
        proof = await self._server_handle.pause_for_training.remote()
        statuses = proof.get('statuses', [])
        if (not proof.get('paused') or len(statuses) != self.world_size
            or {row.get('tp_rank') for row in statuses} != set(range(self.world_size))
            or any(row.get('tp_size') != self.world_size or row.get('phase') != 'paused' for row in statuses)
            or len({row.get('model_version') for row in statuses}) != 1):
            raise RuntimeError('controller did not acknowledge a complete native pause')

    async def release_kv_cache(self):
        await self._control('release')

    async def resume_kv_cache(self):
        await self._control('restore')

    async def resume_generation(self):
        try:
            await self._control('resume')
            control = await _new_control(self.replica_rank)
            try:
                replies = await control.status()
                if (len(replies) != self.world_size or
                    any(row.phase != 'running' or row.model_version != replies[0].model_version
                        or row.global_step != replies[0].global_step for row in replies)):
                    raise RuntimeError('native ranks disagree after policy resume')
                step, version = replies[0].global_step, replies[0].model_version
            finally:
                await control.close()
            # The real controller only opens generation after this score probe.
            proof = await self._server_handle.probe_synced_policy.remote(step, version)
            if proof.get('global_step') != step or proof.get('model_version') != version:
                raise RuntimeError('controller probe did not confirm the committed policy')
        except BaseException:
            try:
                await self._server_handle.pause_for_training.remote()
            except BaseException:
                pass  # admission closure was attempted; keep the original failure
            raise


class StandaloneRTPAdapter(RTPServerAdapter):
    def __init__(self, config, model_config, device_mesh, replica_rank=-1):
        BaseRollout.__init__(self, config, model_config, device_mesh)
        if config.checkpoint_engine.backend == 'naive':
            raise ValueError('standalone RTP requires a non-naive checkpoint engine')
        if replica_rank != 0 or int(config.tensor_model_parallel_size) != 4:
            raise ValueError('candidate requires one TP4 rollout replica')
        self.replica_rank = replica_rank
        self.tp_size = int(config.tensor_model_parallel_size)
        self._last_sync_step = None
        self._last_sync_version = None
        self._unacknowledged_ipc = []
        self._poisoned = False
        checkpoint = Path(model_config.path)
        self.hf_all_sources = set(json.loads((checkpoint / 'model.safetensors.index.json').read_text())['weight_map'])
        cfg = json.loads((checkpoint / 'config.json').read_text())
        self.tied_embeddings = bool(cfg.get('tie_word_embeddings', cfg.get('text_config', {}).get('tie_word_embeddings', False)))
        self.hf_sources = None

    def _setup_group(self):
        if not dist.is_initialized() or dist.get_world_size() != self.tp_size:
            raise RuntimeError('standalone replica requires its initialized TP4 process group')
        self.distributed = True
        self.global_rank = dist.get_rank()
        self.tp_rank = self.global_rank
        self.group_ranks = list(range(self.tp_size))
        self.tp_group = None  # standalone CheckpointEngineWorker uses CPU:gloo
        self.is_leader = self.tp_rank == 0

    def _any_global_failure(self, error):
        flag = torch.tensor(int(error is not None), dtype=torch.int32, device='cpu')
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    async def update_weights(self, weights, global_steps=None, **kwargs):
        if self._poisoned: raise RuntimeError('uncertain native transaction; restart replica')
        if kwargs.get('peft_config') is not None: raise ValueError('LoRA is not supported')
        if type(global_steps) is not int or global_steps < 0: raise ValueError('invalid training step')
        if not callable(getattr(weights, 'confirm_native_ack', None)):
            raise TypeError('checkpoint iterator must enforce native ACK before next tensor')
        self._setup_group()
        _source()
        from rtp_llm.utils.cuda_ipc import CudaIpcV2
        control = None
        workspace = None
        source_digest = hashlib.sha256()
        records = []
        try:
            from q36_refit_bounded_digest_candidate import CudaDigestWorkspace
            workspace = CudaDigestWorkspace(torch.device('cuda', torch.cuda.current_device()))
            restart = None
            if self.is_leader:
                try:
                    server = ray.get_actor(f'rtp_server_{self.replica_rank}_0')
                    restart = await server.ensure_native_running.remote(global_steps, self._last_sync_version)
                    restart['error'] = None
                except BaseException as error:
                    restart = {'error': repr(error)}
            restart = self._broadcast(restart)
            if restart['error'] is not None: raise RuntimeError(restart['error'])
            boot = restart.get('native_boot')
            incarnation = restart.get('native_incarnation')
            if type(boot) is not int or boot < 0 or not isinstance(incarnation, str) or not incarnation:
                raise RuntimeError('native boot and incarnation are required before weight transfer')
            local_error = None
            try:
                self.hf_sources = self._required_sources()
                binding = self._local_ipc_binding(global_steps)
            except BaseException as error:
                local_error, binding = repr(error), None
            checks = self._gather({'error': local_error, 'binding': binding})
            if any(row['error'] for row in checks): raise RuntimeError('native IPC binding failed')
            if any(str(row['binding'].get('native_incarnation')) != str(restart['native_incarnation']) for row in checks):
                raise RuntimeError('native IPC binding names a different incarnation')

            setup = None
            if self.is_leader:
                try:
                    control = await _new_control(self.replica_rank)
                    before = (await control.status())[0]
                    if before.phase == 'running': await control.pause()
                    elif before.phase != 'paused': raise RuntimeError('native is not paused')
                    responses = await control.begin_update(global_step=global_steps,
                        max_buffer_bytes=int(os.environ.get('RLLM_RTP_UPDATE_BUFFER_BYTES', str(16 * 1024**3))))
                    setup = {'error': None, 'before_version': before.model_version,
                             'update_id': responses[0].update_id}
                    if not setup['update_id']: raise RuntimeError('missing native update id')
                except BaseException as error:
                    setup = {'error': repr(error)}
            setup = self._broadcast(setup)
            if setup['error']: raise RuntimeError(setup['error'])
            seen = set()
            async for name, tensor in weights:
                item = {'name': name, 'error': None}
                value = None
                try:
                    if name not in self.hf_sources or name in seen:
                        raise RuntimeError('unexpected or duplicate HF source: ' + str(name))
                    if tensor.device != torch.device('cuda', torch.cuda.current_device()):
                        raise RuntimeError('received source must belong to the local rollout GPU')
                    target_dtype = torch.float32 if preserve_fp32_source(name) else torch.bfloat16
                    if tensor.dtype not in (torch.float32, torch.bfloat16):
                        raise RuntimeError('unsupported checkpoint source dtype')
                    value = tensor.detach().to(dtype=target_dtype).contiguous()
                    self._unacknowledged_ipc = [value]
                    digest, source_bytes = workspace.hash(value, name, source_digest)
                    item.update(shape=list(value.shape), dtype=str(value.dtype),
                                sha256=digest, source_bytes=source_bytes,
                                metadata=CudaIpcV2.export_tensor(value))
                except BaseException as error:
                    item['error'] = repr(error)
                gathered = self._gather(item)
                outcome = None
                if self.is_leader:
                    try:
                        if any(row['error'] for row in gathered): raise RuntimeError('source preparation failed')
                        if any(any(row[key] != gathered[0][key] for key in ('name', 'shape', 'dtype', 'sha256')) for row in gathered):
                            raise RuntimeError('rollout TP ranks disagree on received source')
                        replies = await control.push_weight(name, [row['metadata'] for row in gathered])
                        if (len(replies) != self.tp_size or any(reply.tp_rank != rank or
                            reply.update_id != setup['update_id'] or reply.received_weights != len(seen) + 1
                            for rank, reply in enumerate(replies))):
                            raise RuntimeError('native all-rank source ACK mismatch')
                        outcome = {'error': None, 'proof': {'name': name, 'global_step': global_steps,
                            'native_update_id': setup['update_id'],
                            'native_incarnation': str(restart['native_incarnation']),
                            'all_rank_acked': True, 'replica_world_size': self.tp_size,
                            'acknowledged_ranks': list(range(self.tp_size))}}
                    except BaseException as error:
                        outcome = {'error': repr(error)}
                outcome = self._broadcast(outcome)
                if outcome['error']: raise RuntimeError(outcome['error'])
                weights.confirm_native_ack(name, outcome['proof'])
                self._unacknowledged_ipc = []
                value = None
                seen.add(name)
                records.append({key: item[key] for key in ('name', 'shape', 'dtype', 'sha256', 'source_bytes')})
                # Do not keep the CE receive buffer alive while it allocates
                # the next source on this GPU.
                del tensor
            if seen != self.hf_sources: raise RuntimeError('incomplete native source coverage')
            finish = None
            if self.is_leader:
                try:
                    replies = await control.commit_update()
                    if (len(replies) != self.tp_size or any(row.global_step != global_steps or
                        row.model_version == setup['before_version'] or row.received_weights != len(seen)
                        or row.applied_weights <= 0 for row in replies)):
                        raise RuntimeError('native commit verification failed')
                    finish = {'error': None, 'version': replies[0].model_version,
                              'received_weights': replies[0].received_weights,
                              'applied_weights': replies[0].applied_weights,
                              'rank_status': [{'tp_rank': row.tp_rank, 'tp_size': row.tp_size,
                                               'model_version': row.model_version,
                                               'global_step': row.global_step, 'phase': row.phase}
                                              for row in replies]}
                except BaseException as error:
                    finish = {'error': repr(error)}
            finish = self._broadcast(finish)
            if finish['error']: raise RuntimeError(finish['error'])
            self._last_sync_step, self._last_sync_version = global_steps, finish['version']
            if self.is_leader:
                _write_json(_audit_root() / f'rtp_sync_rank{self.replica_rank}_step{global_steps}.json',
                    {'step': global_steps, 'global_step': global_steps, 'model_version': finish['version'], 'source_count': len(seen),
                     'replica_rank': self.replica_rank, 'native_boot': restart['native_boot'],
                     'rank_status': finish['rank_status'], 'hf_tensor_count': len(seen),
                     'native_received_weights': finish['received_weights'],
                     'native_applied_weights': finish['applied_weights'], 'finished_unix': time.time(),
                     'full_source_sha256': source_digest.hexdigest(), 'sources': records,
                     'native_update_id': setup['update_id'], 'native_incarnation': str(restart['native_incarnation']),
                     'ipc_bindings': [row['binding'] for row in checks]})
        except BaseException:
            self._poisoned = True
            # The owner must tear down the whole replica before reclaiming any
            # unacknowledged IPC allocation. Preserve the original exception.
            raise
        finally:
            original_error = sys.exc_info()[0] is not None
            close_error = None
            try:
                if control is not None: await control.close()
            except BaseException as error:
                close_error = error
            try:
                if workspace is not None: workspace.close()
            except BaseException as error:
                close_error = close_error or error
            if close_error is not None and not original_error: raise close_error
