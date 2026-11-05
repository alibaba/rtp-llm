"""Opt-in native entrypoint; top-level hooks also execute in every spawn child."""
import json
import os
from pathlib import Path
import runpy
import socket
import sys
import time

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
for directory in [HERE,PROJECT/'rtp',PROJECT/'adapter']:
    sys.path.insert(0,str(directory))

from rtp_arm_bf16_import_hook import install as install_arm, install_canonical_fusion_guard, verify_rtp_decode_dispatch

arm_provenance=install_arm()
from qwen36_overlay import install, preflight_required_kernels

install()
from rtp_pure_tp_eplb import install as install_pure_tp_eplb
pure_tp_eplb_proof=install_pure_tp_eplb()
arm_fusion_guard=install_canonical_fusion_guard()
arm_decode_dispatch=verify_rtp_decode_dispatch()

if arm_provenance is not None:
    arm_proof=dict(arm_provenance,fusion=arm_fusion_guard,decode_dispatch=arm_decode_dispatch,
        pid=os.getpid(),hostname=socket.gethostname(),world_rank_env=os.getenv('WORLD_RANK'),
        local_node_world_rank=os.getenv('RTP_WORLD_RANK'),replica_rank=os.getenv('RTP_REPLICA_RANK'),
        cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),installed_unix=time.time())
    if os.getenv('RLLM_RTP_ACCEPTANCE_DIR'):
        evidence=Path(os.environ['RLLM_RTP_ACCEPTANCE_DIR'])/'arm-bf16-backend'
        evidence.mkdir(parents=True,exist_ok=True)
        path=evidence/f"backend-replica{os.getenv('RTP_REPLICA_RANK','standalone')}-pid{os.getpid()}.json"
        temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(arm_proof,indent=2)+'\n')
        temporary.replace(path)

if __name__=='__main__':
    if arm_provenance is not None:
        print('RTP_ARM_BF16_ROPE_BACKEND '+json.dumps(arm_proof),flush=True)
    print('RTP_RUNTIME_PREFLIGHT '+json.dumps(preflight_required_kernels()),flush=True)
    runpy.run_module('rtp_llm.start_server',run_name='__main__')
