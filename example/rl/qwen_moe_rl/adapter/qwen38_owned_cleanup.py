"""Driver-side identity-bound cleanup of the one actual world72 RTP controller.

No local cluster-file discovery assumptions, signal operations, Torch or CUDA.
The existing owned controller terminates only its current 18 native child trees.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import time


def require(value,message):
    if not value:raise RuntimeError(message)


def validate_identity(identity,root,deployment,namespace):
    from qwen38_world_contract import World72Topology
    require(identity.get('replica_rank')==0 and identity.get('world_size')==72
            and identity.get('node_owner_count')==18 and len(identity.get('node_owners',[]))==18,
            'World72 cleanup must discover exactly one controller with18 owners and72 native ranks')
    require(Path(identity['audit_dir']).resolve()==root and Path(identity['root']).resolve()==deployment
            and identity.get('ray_actor')==dict(name='rtp_server_0_0',namespace=namespace),
            'Discovered controller belongs to a different deployment, run or Ray namespace')
    topo=World72Topology.from_dict(identity['topology'])
    require({row['node_index'] for row in identity['node_owners']}==set(range(18))
            and all(row['owner_ip']==topo.node_groups[row['node_index']]['host']
                    for row in identity['node_owners']),
            'Discovered controller does not cover the exact current18 physical node owners')
    require(type(identity.get('pid')) is int and identity['pid']>0
            and isinstance(identity.get('create_time'),(float,int)) and identity['create_time']>0,
            'Controller process identity is incomplete')
    return topo


def validate_shutdown(proof):
    require(proof.get('world_size')==72 and proof.get('all_owned_native_processes_exited') is True
            and proof.get('process_tree_terminated') is True and proof.get('remaining_owned_gpu_contexts')==[],
            'The owned native controller did not acknowledge real complete-world process/context cleanup')
    if proof.get('native_never_started') is not True:
        require(len(proof.get('checked_native_ranks',[]))==72
                and set(proof.get('checked_native_ranks',[]))==set(range(72))
                and len(proof.get('nodes',[]))==18,
                'The native shutdown receipt omits one or more physical ranks/nodes')
    return proof


def cleanup_owned_world72(audit_dir):
    import ray
    root=Path(audit_dir).resolve()
    require(root==Path(os.environ['RLLM_RTP_ACCEPTANCE_DIR']).resolve(),
            'World72 cleanup must use only the current run audit directory')
    deployment=Path(os.environ['RLLM_RTP_DEPLOY_ROOT']).resolve()
    require(root!=deployment and root.is_relative_to(deployment.parent),
            'World72 run audit directory is outside the current deployment allocation root')
    root.mkdir(parents=True,exist_ok=True)
    result=dict(schema='q38-world72-owned-driver-cleanup-v1',audit_dir=str(root),
                started_at_unix=time.time(),passed=False,replicas=[])
    try:
        require(ray.is_initialized(),'Live current-job Ray is required for owned world72 identity discovery')
        namespace=ray.get_runtime_context().namespace
        actor=ray.get_actor('rtp_server_0_0',namespace=namespace)
        timeout=float(os.environ.get('RLLM_RTP_CLEANUP_RPC_TIMEOUT','180'))
        identity=ray.get(actor.owned_identity.remote(),timeout=timeout)
        validate_identity(identity,root,deployment,namespace)
        proof=ray.get(actor.shutdown_if_owned.remote(identity['pid'],identity['create_time'],str(root)),timeout=timeout)
        validate_shutdown(proof)
        result['replicas']=[dict(replica_rank=0,method='verified_world72_controller_rpc',identity=identity,
                                rpc_receipt=proof,passed=True)]
        result['passed']=True
    except BaseException as error:
        result['error']=repr(error)
        raise
    finally:
        result['finished_at_unix']=time.time()
        path=root/'driver-cleanup.json';temporary=path.with_suffix('.writing')
        temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(path)
    return result
