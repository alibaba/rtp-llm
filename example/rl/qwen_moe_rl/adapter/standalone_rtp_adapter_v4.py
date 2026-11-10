"""Pinned v3 RTP adapter with a fail-closed verl abort API compatibility shim.

The controller rejects arrivals while paused and does not park requests. The
default ``reject_request=False`` is allowed only for an explicitly selected
on-policy run with no concurrent rollout requests.
"""
import hashlib
import importlib
import importlib.machinery
import os
from pathlib import Path
import sys

V3_SHA256 = '60c7fc2daeb73a85353b1ae4d03bf50dfd48ce82cd22c6fec0498df013453006'
root = Path(os.environ['Q36_ADAPTER_DIR']).resolve(strict=True)
source = root / 'standalone_rtp_adapter_v3.py'
spec = importlib.machinery.PathFinder.find_spec('standalone_rtp_adapter_v3', sys.path)
loaded = sys.modules.get('standalone_rtp_adapter_v3')
if (source.is_symlink() or not source.is_file() or
    hashlib.sha256(source.read_bytes()).hexdigest() != V3_SHA256 or
    spec is None or not spec.origin or Path(spec.origin).resolve() != source or
    loaded is not None and Path(getattr(loaded, '__file__', '')).resolve() != source):
    raise RuntimeError('Q36 standalone v3 adapter source identity mismatch')
base = importlib.import_module('standalone_rtp_adapter_v3')

# Deliberately the original class: its __module__ stays
# standalone_rtp_adapter_v3, and the native ACK/lease behavior is unchanged.
StandaloneRTPAdapter = base.StandaloneRTPAdapter


class StandaloneRTPReplica(base.StandaloneRTPReplica):
    async def init_standalone(self, resource_pool=None):
        if os.environ.get('Q36_STRICT_4PLUS4_CLAIM') == '1':
            from onpolicy_standalone_candidate import _bind_claimed_pool
            if resource_pool is None:
                from verl.single_controller.ray.base import RayResourcePool
                if self.nnodes != 1 or self.gpus_per_replica_node != 4:
                    raise ValueError('strict 4+4 rollout requires one four-GPU replica')
                resource_pool = RayResourcePool(process_on_nodes=[4], use_gpu=True,
                    max_colocate_count=2, name_prefix=f'q36_rollout_{self.replica_rank}')
            _bind_claimed_pool(resource_pool, 'rollout')
        return await super().init_standalone(resource_pool=resource_pool)

    async def launch_servers(self):
        if os.environ.get('Q36_STRICT_4PLUS4_CLAIM') == '1':
            import asyncio
            from onpolicy_standalone_candidate import _worker_gpu_identity
            from q36_4plus4_claim_loader import load_guard
            rows = await asyncio.gather(*[worker.__ray_call__.remote(_worker_gpu_identity) for worker in self.workers])
            guard = load_guard()
            guard.validate_role_ownership(rows, 'rollout', guard.load_claim())
        return await super().launch_servers()

    async def abort_all_requests(self, reject_request: bool = False):
        if type(reject_request) is not bool:
            raise TypeError('reject_request must be a boolean')
        if not reject_request and os.environ.get('Q36_ONPOLICY_NO_CONCURRENT_ROLLOUT') != '1':
            raise RuntimeError('default pause requires the explicit no-concurrent-rollout on-policy profile')
        return await super().abort_all_requests()
