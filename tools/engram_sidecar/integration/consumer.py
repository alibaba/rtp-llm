#!/usr/bin/env python3
"""Two-layer FP8 Engram integration; real RTP lookup and pinned CUDA Graph.

Requires a live fixture owner and a private capabilities.json transferred via
an authenticated channel. This harness does not implement production RPC.
"""
import argparse
import base64
import gc
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'native'))
import torch
from load_native import load_native
from fixture import LAYERS, DIM, expected_layout, raw_rows, expected_rows
from rtp_llm.models_py.modules.dsv41.engram_storage import FabricEngramStorage, CapturedEngramGraph, StorageBusyError
from rtp_llm.models_py.modules.dsv41.engram import HostEngramEmbedding


def emit(stage, **values):
    print(json.dumps(dict(stage=stage, **values)), flush=True)


def exercise_hash_pipeline(storage, embeddings, device, replays, label):
    """Actual CUDA hash -> external FP8 lookup, independently checked on CPU.

    Explicit token windows include sequence starts, different request histories,
    a changed speculative candidate and a blocked image-token position. This is
    the Engram hash/lookup component, not a model scheduler or PD runtime test.
    """
    from rtp_llm.models_py.modules.dsv41.engram import EngramLayout, NgramHashState
    config = dict(engram_layer_ids=[layer for layer, _ in LAYERS],
                  engram_num_embeddings=[rows for _, rows in LAYERS],
                  engram_max_ngram_size=4, engram_n_heads=2, engram_head_dim=DIM,
                  engram_compressed_vocab_size=32, engram_pad_token_id=2,
                  engram_vocab_size=13)
    layout = EngramLayout(config)
    oracle_state = NgramHashState(layout, token_map=list(range(32)))
    state = NgramHashState(layout, token_map=list(range(32)), device=torch.device('cuda', device))
    windows = torch.zeros((6, 4), dtype=torch.int32, device=device)
    dead = torch.zeros_like(windows, dtype=torch.bool)

    def forward():
        hashes = state.hash_token_windows(windows, dead_mask=dead)
        rows = tuple(embeddings[layer](hashes[:, index].contiguous(), torch.device('cuda', device))
                     for index, (layer, _) in enumerate(LAYERS))
        return (hashes, *rows)

    def inputs(step):
        host = torch.tensor([[3 + step % 29, -1, -1, -1], [7, 6, 5, 4],
                             [13, 12, 11, 10], [20 + step % 12, 7, 6, 5],
                             [8, 7, 6, 5], [-1, -1, -1, -1]], dtype=torch.int32)
        blocked = torch.zeros_like(host, dtype=torch.bool)
        blocked[4, 1] = True
        return host, blocked

    compared = 0
    def check(outputs, host, blocked):
        nonlocal compared
        hashes = oracle_state.hash_token_windows(host, dead_mask=blocked)
        torch.testing.assert_close(outputs[0].cpu(), hashes, rtol=0, atol=0)
        for index, (_, count) in enumerate(LAYERS):
            expected = expected_rows(hashes[:, index], count)
            torch.testing.assert_close(outputs[index + 1].cpu(), expected, rtol=0, atol=0)
            compared += expected.numel()

    with torch.inference_mode():
        for step in (0, 1, 28, 29):
            host, blocked = inputs(step)
            windows.copy_(host); dead.copy_(blocked)
            check(forward(), host, blocked)
        graph = CapturedEngramGraph.capture(forward, [storage])
        try:
            for step in range(replays):
                host, blocked = inputs(step)
                windows.copy_(host); dead.copy_(blocked)
                check(graph.replay(), host, blocked)
        finally:
            graph.close()
    emit('hash_lookup_graph_pass', label=label, eager_cases=4, graph_replays=replays,
         compared_values=compared, hash_exact_match=True, max_abs_error=0,
         explicit_boundary_and_dead_token_cases=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--label', required=True)
    p.add_argument('--graph-replays', type=int, default=20)
    p.add_argument('--exercise-device-guard',action='store_true')
    p.add_argument('--exercise-hash-pipeline',action='store_true')
    args=p.parse_args()
    if args.graph_replays <= 0:
        p.error('--graph-replays must be positive; zero replays cannot pass Graph validation')
    directory=args.directory
    torch.cuda.set_device(args.device)
    torch.cuda.init()
    native=load_native()
    assert not any(native.resource_counters().values())
    manifest_bytes=(directory/'manifest.json').read_bytes()
    document=json.loads(manifest_bytes)
    capabilities=json.loads((directory/'capabilities.json').read_bytes())
    handles={aid:base64.b64decode(encoded, validate=True) for aid,encoded in capabilities.items()}
    del capabilities
    torch.cuda.synchronize()
    baseline_bytes=torch.cuda.memory_allocated()
    storage=FabricEngramStorage.attach(
        manifest_bytes,expected_layout=expected_layout(),dataset_id=document['dataset_id'],
        generation=document['generation'],handles=handles,device=args.device,native=native)
    del handles
    attach_hbm_delta=torch.cuda.memory_allocated()-baseline_bytes
    assert attach_hbm_delta==0, 'attach allocated a private PyTorch HBM tensor'
    assert storage.info()['state']=='MAPPED'
    try:
        storage.table_views(LAYERS[0][0])
    except RuntimeError:
        pass
    else:
        raise AssertionError('unverified storage was usable')
    provenance=[]
    for t in storage.validated.manifest.tensors:
        part=t.shards[0]
        provenance.append(dict(tensor=t.name,allocation_id=part.allocation_id,offset=part.byte_offset))
    def verify(views, validated):
        for layer, rows in LAYERS:
            ids=torch.tensor([0,1,rows//2,rows-1],dtype=torch.int64)
            oracle=raw_rows(ids)
            gpu_ids=ids.to(device=args.device)
            for role,expected in zip(('weight','scale'),oracle):
                name=f'layers.{layer}.engram.embed.{role}'
                view=views[name]
                tensor=next(t for t in validated.manifest.tensors if t.name==name)
                part=tensor.shards[0]
                base=storage.info()['allocations'][part.allocation_id]['base_address']
                assert view.data_ptr()==base+part.byte_offset
                torch.testing.assert_close(view.index_select(0,gpu_ids).cpu(),expected,rtol=0,atol=0)
        torch.cuda.synchronize()
        return True
    storage.verify(verify)
    emit('verified',label=args.label,generation=document['generation'],
         dataset_id=document['dataset_id'],physical_bytes=storage.info()['physical_bytes'],
         allocation_count=storage.info()['allocation_count'],attach_hbm_bytes=attach_hbm_delta,
         pointers_match_native=True,tensors=provenance,counters=native.resource_counters())
    embeddings={layer:HostEngramEmbedding.from_storage(storage,layer) for layer,_ in LAYERS}
    assert all(e.weight is None and e.scales is None for e in embeddings.values())
    if args.exercise_hash_pipeline:
        exercise_hash_pipeline(storage, embeddings, args.device, args.graph_replays, args.label)
    eager_cases=0
    for layer,rows in LAYERS:
        indices=torch.tensor([[0,rows-1,-1,37],[1,1,rows,rows+17]],dtype=torch.int64)
        for ids in (indices,indices.T,indices.to(torch.int32),torch.empty((0,4),dtype=torch.int64)):
            result=embeddings[layer](ids.to(device=args.device),torch.device('cuda',args.device))
            torch.testing.assert_close(result.cpu(),expected_rows(ids,rows),rtol=0,atol=0)
            eager_cases+=1
    if args.exercise_device_guard:
        assert torch.cuda.device_count()==2, 'guard test requires two authorized idle GPUs'
        ids=torch.tensor([[0,17,-1,4096]],dtype=torch.int64,device=args.device)
        other=1 if args.device==0 else 0
        torch.cuda.set_device(other)
        try:
            result=embeddings[LAYERS[0][0]](ids,torch.device('cuda',args.device))
            torch.testing.assert_close(result.cpu(),expected_rows(ids.cpu(),LAYERS[0][1]),rtol=0,atol=0)
            assert torch.cuda.current_device()==other, 'device guard failed to restore caller device'
        finally:
            torch.cuda.set_device(args.device)
        emit('alternate_current_device_pass',label=args.label,data_device=args.device,
             caller_device=other,restored=True,max_abs_error=0)
    # Capturing a failed forward must drop pins only after native CUDA work drains.
    def bad_forward():
        raise ValueError('injected integration capture input failure')
    try:
        CapturedEngramGraph.capture(bad_forward,[storage])
    except ValueError:
        pass
    else:
        raise AssertionError('capture input failure was swallowed')
    assert storage.info()['graph_pins']==0
    static={layer:torch.zeros((2,4),dtype=torch.int64,device=args.device) for layer,_ in LAYERS}
    graph=CapturedEngramGraph.capture(
        lambda:tuple(embeddings[layer](static[layer],torch.device('cuda',args.device)) for layer,_ in LAYERS),[storage])
    try:
        storage.close()
    except StorageBusyError:
        pass
    else:
        raise AssertionError('graph pin failed to block storage close')
    assert storage.info()['state']=='READY'
    # Drop model-side views: storage + graph lease must still retain the addresses.
    for embedding in embeddings.values():
        embedding.release_external_views()
    compared_values=0
    for iteration in range(args.graph_replays):
        expected=[]
        for layer,rows in LAYERS:
            ids=torch.tensor([[iteration%rows,rows-1,-1,37],
                              [1,1,(iteration*17)%rows,rows]],dtype=torch.int64)
            static[layer].copy_(ids)
            expected.append(expected_rows(ids,rows))
        outputs=graph.replay()
        for output,oracle in zip(outputs,expected):
            torch.testing.assert_close(output.cpu(),oracle,rtol=0,atol=0)
            compared_values+=output.numel()
    emit('lookup_graph_pass',label=args.label,eager_cases=eager_cases,
         graph_replays=args.graph_replays,graph_compared_values=compared_values,
         max_abs_error=0,model_views_released_before_replay=True)
    graph.close()
    storage.close()
    assert not any(native.resource_counters().values()), native.resource_counters()
    emit('consumer_complete',label=args.label,passed=True,state=storage.info()['state'],
         generation=document['generation'],counters=native.resource_counters())


if __name__=='__main__':
    main()
