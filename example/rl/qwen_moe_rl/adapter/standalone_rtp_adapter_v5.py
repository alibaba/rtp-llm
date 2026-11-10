"""Q36 standalone opt-in resident-KV profile; native allocation stays live.

The native RL server only permits requests with prefix reuse disabled, and
acknowledges PAUSE only after scheduler drain and executor quiescence. This
profile must still be qualified with real hybrid KV and GPU memory evidence.
"""
import hashlib
import importlib
import importlib.machinery
import json
import os
from pathlib import Path
import sys

V4_SHA256 = '25eb970ccc5eb3dc591d99e5f4ac616c8fa9c29b3a64e116c4d1724cd81308d4'
MODEL_CONFIG_SHA256 = '93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99'
root = Path(os.environ['Q36_ADAPTER_DIR']).resolve(strict=True)
source = root / 'standalone_rtp_adapter_v4.py'
spec = importlib.machinery.PathFinder.find_spec('standalone_rtp_adapter_v4', sys.path)
loaded = sys.modules.get('standalone_rtp_adapter_v4')
if (source.is_symlink() or not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != V4_SHA256
    or spec is None or not spec.origin or Path(spec.origin).resolve() != source
    or loaded is not None and Path(getattr(loaded, '__file__', '')).resolve() != source):
    raise RuntimeError('Q36 standalone v4 adapter source identity mismatch')
base = importlib.import_module('standalone_rtp_adapter_v4')
from rtp_native_server import _new_control
StandaloneRTPAdapter = base.StandaloneRTPAdapter


def _require_resident_profile(replica):
    required = {'Q36_ONPOLICY_NO_CONCURRENT_ROLLOUT':'1', 'RLLM_RTP_KEEP_KV_CACHE':'1',
                'RTP_LLM_RL_MODE':'1', 'RTP_RL_SHARED_BACKING':'0',
                'RTP_TP_SIZE':'4', 'RTP_EP_SIZE':'1', 'RLLM_RTP_GPUS_PER_NODE':'4',
                'RLLM_RTP_EXPECTED_REPLICAS':'1'}
    if any(os.environ.get(key) != value for key, value in required.items()) or replica.world_size != 4:
        raise RuntimeError('resident KV requires the explicit Q36 independent TP4/EP1 profile')
    model = Path(os.environ['RTP_MODEL_PATH']) / 'config.json'
    if not model.is_file() or hashlib.sha256(model.read_bytes()).hexdigest() != MODEL_CONFIG_SHA256:
        raise RuntimeError('resident KV is limited to the pinned Q36 model config')


def _rank_proof(rows, world_size):
    if len(rows) != world_size or {row.tp_rank for row in rows} != set(range(world_size)):
        raise RuntimeError('resident KV requires every TP rank exactly once')
    if any(row.protocol_version != 2 or row.tp_size != world_size or row.phase != 'paused'
           or not row.quiescent or not row.kv_cache_ready or row.kv_cache_bytes <= 0
           or not row.model_version or row.global_step < 0 or row.update_id for row in rows):
        raise RuntimeError('resident KV requires paused, quiescent, resident, initialized ranks')
    if len({row.model_version for row in rows}) != 1 or len({row.global_step for row in rows}) != 1:
        raise RuntimeError('resident KV ranks disagree about policy')
    return {'model_version':rows[0].model_version, 'global_step':rows[0].global_step,
            # Hybrid backingBytes currently observes only the primary pool;
            # these values are telemetry and not a complete capacity estimate.
            'ranks':[{'tp_rank':row.tp_rank,'kv_cache_bytes_reported':row.kv_cache_bytes,
                      'torch_allocated_bytes':row.torch_allocated_bytes,
                      'torch_reserved_bytes':row.torch_reserved_bytes} for row in rows]}


class StandaloneRTPReplica(base.StandaloneRTPReplica):
    async def _retain_resident(self, stage):
        _require_resident_profile(self)
        if stage not in ('release', 'restore'):
            raise ValueError('invalid resident KV stage')
        if stage == 'release' and getattr(self, '_resident_kv_lease', None) is not None:
            raise RuntimeError('resident KV lease is already active')
        if stage == 'restore' and getattr(self, '_resident_kv_lease', None) is None:
            raise RuntimeError('resident KV lease is missing')
        # Existing controller closes Python admission, pauses native and binds
        # the proof to its actual boot/incarnation; this may not restart it.
        proof = await self._server_handle.pause_for_training.remote()
        boot = proof.get('native_boot')
        incarnation = proof.get('native_incarnation')
        if (not proof.get('paused') or not proof.get('kv_pool_retained') or
            proof.get('native_processes_released') is not False or type(boot) is not int or boot < 0 or
            not isinstance(incarnation, str) or not incarnation):
            raise RuntimeError('controller did not prove a live resident paused replica')
        control = await _new_control(self.replica_rank)
        try:
            rows = await control.status()
            observed = _rank_proof(rows, self.world_size)
        finally:
            await control.close()
        observed.update(native_boot=proof['native_boot'], native_incarnation=proof['native_incarnation'])
        if stage == 'release':
            self._resident_kv_lease = observed
        else:
            prior = self._resident_kv_lease
            if (observed['native_boot'] != prior['native_boot'] or
                observed['native_incarnation'] != prior['native_incarnation'] or
                observed['model_version'] == prior['model_version'] or
                observed['global_step'] < prior['global_step']):
                raise RuntimeError('resident KV restore did not follow an update in the same native incarnation')
            self._resident_kv_lease = None
        print('Q36_RESIDENT_KV_' + stage.upper() + ' ' + json.dumps(observed, sort_keys=True), flush=True)

    async def release_kv_cache(self):
        mode = os.environ.get('Q36_DISAGG_RESIDENT_KV', '0')
        if mode not in ('0', '1'):
            raise ValueError('Q36_DISAGG_RESIDENT_KV must be 0 or 1')
        if mode == '1':
            return await self._retain_resident('release')
        return await super().release_kv_cache()

    async def resume_kv_cache(self):
        mode = os.environ.get('Q36_DISAGG_RESIDENT_KV', '0')
        if mode not in ('0', '1'):
            raise ValueError('Q36_DISAGG_RESIDENT_KV must be 0 or 1')
        if mode == '1':
            return await self._retain_resident('restore')
        return await super().resume_kv_cache()
