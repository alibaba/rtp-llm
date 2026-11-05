"""CPU-only validation of 397B distributed reference and optimizer storage records."""
from hashlib import sha256
import json
import math
import re

from checkpoint_contract import identity


def require(condition, message):
    if not condition:
        raise ValueError(message)


def gpu_uuid(value):
    require(isinstance(value, str), 'Missing physical GPU UUID')
    bare = value.removeprefix('GPU-').lower()
    require(re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', bare) is not None,
            'Malformed physical GPU UUID')
    return bare


def hf_residency(reference, checkpoint, training_uuids):
    require(reference.get('schema') == 'qwen35-397b-hf-cold-reference-v1', 'Wrong independent HF reference schema')
    require(all(reference.get(key) == value for key, value in identity().items()), 'Wrong model/revision in HF reference')
    require(reference.get('dtype') == 'bfloat16' and reference.get('mtp') is False and
            reference.get('device') == 'layer_sharded_cuda' and reference.get('input_device') == 'cuda:0',
            'HF reference must use the complete BF16 checkpoint across CUDA layers')
    require(reference.get('model_class') == 'Qwen3_5MoeForConditionalGeneration', 'Wrong full HF model class')
    expected = checkpoint.text | checkpoint.visual
    placements = reference.get('parameter_placements', {})
    require(set(placements) == set(expected), 'HF placement record does not cover the exact full model')
    require(reference.get('parameter_device_sha256') == sha256(json.dumps(placements, sort_keys=True).encode()).hexdigest(),
            'HF placement hash differs from its recorded parameters')
    counts = {f'cuda:{i}': 0 for i in range(4)}
    for name, device in placements.items():
        require(device in counts, f'HF parameter is not on one of the four GPUs: {name}')
        counts[device] += math.prod(expected[name])
    devices = reference.get('devices', [])
    require(len(devices) == 4 and {d.get('index') for d in devices} == set(range(4)), 'Missing HF CUDA device records')
    uuids = set()
    for row in devices:
        device = f'cuda:{row["index"]}'
        require(row.get('device') == device and row.get('cuda_numel') == counts[device] and counts[device] > 0,
                'HF actual per-device parameter counts disagree with the full checkpoint')
        require(row.get('peak_allocated_bytes', 0) >= row.get('allocated_bytes', -1) >= counts[device]*2,
                'HF full BF16 parameter residency is absent on a recorded CUDA device')
        require(row.get('peak_reserved_bytes', 0) >= row.get('peak_allocated_bytes', 0), 'Impossible HF allocator counters')
        uuids.add(gpu_uuid(row.get('gpu_uuid')))
    require(len(uuids) == 4 and uuids <= set(training_uuids), 'Independent HF used repeated GPUs or GPUs outside the actual formal training set')
    total = checkpoint.text_numel + checkpoint.visual_numel
    require(reference.get('cuda_numel') == sum(counts.values()) == total, 'HF CUDA parameters do not reconstruct the full model')
    dtype_counts = reference.get('parameter_dtype_numel', {})
    require(set(dtype_counts) <= {'torch.bfloat16', 'torch.float32'} and
            sum(dtype_counts.values()) == total and dtype_counts.get('torch.bfloat16', 0) >= .99*total,
            'HF actual parameter dtypes differ from the BF16 reference path')
    require(reference.get('peak_allocated_bytes') == sum(row['peak_allocated_bytes'] for row in devices),
            'HF aggregate allocation is not the sum of recorded CUDA device peaks')
    return dict(cuda_numel=total, devices=sorted(uuids), allocation_bytes=reference['peak_allocated_bytes'])


def reference_gpu_membership(hf_uuids, cold_native_uuids, training_uuids):
    """Independent HF and cold native may run concurrently on disjoint nodes."""
    hf = {gpu_uuid(value) for value in hf_uuids}
    cold = {gpu_uuid(value) for value in cold_native_uuids}
    training = {gpu_uuid(value) for value in training_uuids}
    require(len(training)==32 and len(hf)==4 and len(cold)==16,
            'Reference/training physical GPU counts differ from HF4, cold TP16 and FSDP32')
    require(hf<=training, 'Independent HF used a GPU outside the actual formal FSDP32 run')
    require(cold<=training, 'Cold native used a GPU outside the actual formal FSDP32 run')
    return dict(hf_gpus=sorted(hf),cold_native_gpus=sorted(cold),training_gpu_count=len(training),
                hf_native_overlap_count=len(hf&cold))


def native_boot(boot, tp_size=16):
    """Inspect original per-node process and SMI evidence, keyed by node and PID."""
    require(boot.get('passed') is True and boot.get('manifest_count') == tp_size,
            'Native boot has no complete original process evidence')
    incarnation=boot.get('native_incarnation')
    require(isinstance(incarnation,str) and re.fullmatch(r'[0-9a-f]{32}',incarnation), 'Invalid native incarnation')
    nodes=boot.get('nodes',[])
    require(len(nodes)==4 and {n.get('node_index') for n in nodes}=={0,1,2,3}, 'All four native owners are required')
    require(len({n.get('hostname') for n in nodes})==4 and len({n.get('node_ip') for n in nodes})==4,
            'Native owners do not identify four distinct nodes')
    by_rank={}
    all_statuses=None
    for node in nodes:
        index=node['node_index']
        ranks=list(range(index*4,(index+1)*4))
        require(node.get('local_ranks')==ranks and node.get('boot_id')==boot.get('native_boot') and
                node.get('incarnation')==incarnation, 'Native owner belongs to another boot or rank group')
        process=node.get('process_identity',{})
        require(process.get('owner_hostname')==node['hostname'] and process.get('owner_ip')==node['node_ip'] and
                process.get('node_index')==index and process.get('owned_tp_ranks')==ranks and
                process.get('incarnation')==incarnation and process.get('boot_id')==boot['native_boot'],
                'Original native process owner is inconsistent')
        require(all(type(process.get(key)) in (int,float) and process[key]>0 for key in ('pid','pgid','create_time')),
                'Original PID/process-group/creation-time identity is absent')
        owned={process['pid']}|{r['pid'] for r in process.get('descendants',[])}
        inventory=node.get('inventory',{})
        require(inventory.get('hostname')==node['hostname'], 'Native SMI inventory is from another node')
        devices={r.get('index'):gpu_uuid(r.get('uuid')) for r in inventory.get('devices',[])}
        require(set(devices)==set(range(4)), 'Native owner lacks its four physical device records')
        resident={r.get('pid'):r for r in inventory.get('processes',[])}
        manifests=node.get('rank_manifests',[])
        require(len(manifests)==4 and {r.get('tp_rank') for r in manifests}==set(ranks), 'Original local TP ranks are incomplete')
        local_pids=set()
        for row in manifests:
            rank=row['tp_rank']
            local=rank%4
            pid=row.get('pid')
            require(row.get('owner_hostname')==node['hostname'] and row.get('owner_ip')==node['node_ip'] and
                    row.get('owner_node_index')==index and row.get('boot_id')==boot['native_boot'] and
                    row.get('incarnation')==incarnation, 'Native rank manifest owner or incarnation is wrong')
            require(row.get('tp_size')==tp_size and row.get('world_rank')==rank and row.get('dp_size')==row.get('ep_size')==1,
                    'Native TP/DP/EP rank layout differs from pure TP16')
            require(type(pid) is int and pid>0 and pid in owned and pid not in local_pids,
                    'Native rank PID is not a distinct member of its owned local process tree')
            local_pids.add(pid)
            require(gpu_uuid(row.get('gpu_uuid'))==devices[local], 'Native rank did not use its assigned physical GPU')
            require(pid in resident and gpu_uuid(resident[pid].get('uuid'))==devices[local] and
                    float(resident[pid].get('used_mib','0'))>0, 'Native rank GPU residency is absent in original SMI evidence')
            by_rank[rank]=row
        statuses={s.get('tp_rank'):s for s in node.get('statuses',[])}
        require(set(statuses)==set(range(tp_size)) and all(s.get('tp_size')==tp_size and
                s.get('phase')=='running' for s in statuses.values()), 'All-rank native running statuses are absent')
        if all_statuses is None:all_statuses=statuses
        else:require(statuses==all_statuses, 'Native owners observed inconsistent RPC states')
    require(set(by_rank)==set(range(tp_size)) and len({gpu_uuid(r['gpu_uuid']) for r in by_rank.values()})==tp_size,
            'Native cold/boot world did not use sixteen distinct physical GPUs')
    require(len({(r['model_version'],r['global_step']) for r in all_statuses.values()})==1,
            'Native TP ranks disagree on the baseline checkpoint version')
    return dict(manifests=[by_rank[i] for i in range(tp_size)],statuses=all_statuses,
                uuids={gpu_uuid(r['gpu_uuid']) for r in by_rank.values()})


def native_shutdown(boot, shutdown):
    require(shutdown.get('process_tree_terminated') is True and shutdown.get('native_boot')==boot.get('native_boot') and
            shutdown.get('native_incarnation')==boot.get('native_incarnation'),
            'Native shutdown does not identify the boot it released')
    nodes={n['node_index']:n for n in boot.get('nodes',[])}
    stops=shutdown.get('nodes',[])
    require(len(stops)==4 and {n.get('node_index') for n in stops}==set(nodes), 'All four original native owners must be shut down')
    for stop in stops:
        original=nodes[stop['node_index']]
        process=original['process_identity']
        require(stop.get('owner_hostname')==original['hostname'] and stop.get('owner_ip')==original['node_ip'] and
                stop.get('boot_id')==boot['native_boot'], 'Native shutdown owner identity differs from its original boot')
        require(stop.get('process_tree_terminated') is True and stop.get('owned_gpu_processes_after')==[],
                'Native owner still holds a GPU process or has no termination acknowledgment')
        for field in ('pid','pgid','create_time'):
            require(stop.get(field)==process.get(field), 'Native shutdown process identity differs: '+field)
        required={process['pid']}|{r['pid'] for r in process.get('descendants',[])}|{r['pid'] for r in original['rank_manifests']}
        require(required <= set(stop.get('owned_pids',[])), 'Native owner cleanup omitted an original TP rank or subprocess')
    require(isinstance(shutdown.get('finished_at_unix'),(int,float)) and shutdown['finished_at_unix']>0,
            'Native group cleanup completion is not recorded')
    return dict(boot=boot['native_boot'],nodes=len(stops),released_ranks=sum(len(n['local_ranks']) for n in nodes.values()))
