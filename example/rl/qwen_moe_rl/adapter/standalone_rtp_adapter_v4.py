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

V3_SHA256 = 'b849f17ab606e0554db3802eb43979dd10cedd64361cc8082f716c173cf64f5c'
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
    async def abort_all_requests(self, reject_request: bool = False):
        if type(reject_request) is not bool:
            raise TypeError('reject_request must be a boolean')
        if not reject_request and os.environ.get('Q36_ONPOLICY_NO_CONCURRENT_ROLLOUT') != '1':
            raise RuntimeError('default pause requires the explicit no-concurrent-rollout on-policy profile')
        return await super().abort_all_requests()
