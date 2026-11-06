"""Freeze current18-container allocation and previously verified72 UUIDs.

Historical process IDs are deliberately discarded. Native process ownership is
established anew by the real current controller during this separate preflight.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import sys


def build(cluster_path,collective_path,targets_path,model_path,start_port=28000):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'adapter'))
    from qwen38_world_contract import World72Topology,canonical_uuid
    paths=(cluster_path,collective_path,targets_path)
    cluster,collective,targets=[json.loads(path.read_text()) for path in paths]
    if not (collective.get('passed') is True and collective.get('ranks')==72
            and collective.get('UUIDs_match_user_allocation') is True):
        raise ValueError('Require the real current72-rank hardware UUID/collective acceptance')
    nodes=sorted(cluster['nodes'],key=lambda row:row['node_rank'])
    if len(nodes)!=18 or [row['node_rank'] for row in nodes]!=list(range(18)):
        raise ValueError('Require the complete ordered current18-node Ray allocation')
    by_target={row['target']:row for row in collective['node_results']}
    frozen=[];all_uuids=[]
    for node in nodes:
        target=targets['targets'][node['target']]
        if (node['container_ip']!=target['container_ip'] or node['gpu_container']!=target['gpu_container']
                or node['gpu_count']!=4):raise ValueError('The cluster plan predates the current container allocation')
        identities=sorted((record['identity'] for record in by_target[node['target']]['ranks']),key=lambda row:row['rank'])
        base=node['node_rank']*4
        if ([row['rank'] for row in identities]!=list(range(base,base+4))
                or [row['local_rank'] for row in identities]!=list(range(4))
                or any(row['capability']!=[10,0] for row in identities)):
            raise ValueError('The accepted actual four-GPU identities differ from the current SM100 node rank mapping')
        uuids=[canonical_uuid(row['gpu_uuid']) for row in identities];all_uuids.extend(uuids)
        frozen.append(dict(node,target_service=target['node'],physical_ip=target['physical_ip'],gpu_uuids=uuids))
    if len(set(all_uuids))!=72:raise ValueError('Every current physical GPU UUID must occur exactly once')
    if not Path(model_path).is_absolute():raise ValueError('Require an explicit absolute remote model mount')
    World72Topology(gpu_ids=tuple('GPU-'+value for value in all_uuids),gpu_uuids=tuple(all_uuids),
        rank_hosts=tuple(node['container_ip'] for node in frozen for _ in range(4)),start_port=start_port)
    return dict(schema='q38-native-metadata72-frozen-inputs-v1',address=cluster['address'],nodes=frozen,
        rank_gpu_uuids=all_uuids,start_port=start_port,model_path=str(model_path),
        source_files=[dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths],
        historical_PIDs_copied=False,allowed_native_transactions=['STATUS','PAUSE','identity-bound native teardown'],
        forbidden=['BEGIN_UPDATE','PUSH_WEIGHT','COMMIT','RESUME','generation','HF model creation','training'],
        full_model_training_accepted=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cluster-plan',type=Path,required=True);parser.add_argument('--collective-proof',type=Path,required=True)
    parser.add_argument('--targets',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model-path',type=Path,required=True,help='Existing remote official BF16 model mount; not downloaded by this tool')
    parser.add_argument('--start-port',type=int,default=28000)
    args=parser.parse_args();value=build(args.cluster_plan,args.collective_proof,args.targets,args.model_path,args.start_port)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),nodes=18,GPUs=72,started=False)))
