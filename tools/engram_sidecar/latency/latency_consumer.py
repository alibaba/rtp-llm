"""Four-path same-GPU comparison; baseline copies exist only for this experiment."""
import argparse
import base64
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from uuid import uuid4

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE.parent / 'native'), str(HERE / 'quant_probe')]
import torch
from load_native import load_native
from build_quant_probe import load_probe, default_build_directory
from test_quant_probe import packed_scales, unpack_scale_bytes
from latency_fixture import raw_rows, local_raw_table
from canonical_preprocess import canonicalize_chunks
from canonical_gather import (CanonicalMetadata, CANONICAL_REPRESENTATION,
                              LEGACY_QUANT_ABI, V2_QUANT_ABI,
                              allocate_outputs, gather_into)
from rtp_llm.models_py.modules.dsv41.engram import _PinnedSharedTable
from rtp_llm.models_py.modules.dsv41._engram_triton import lookup_host_rows
from bench_consumer import run_bench


COMPARISONS = {
    'raw_remote_minus_raw_local': ('remote_raw', 'local_raw'),
    'canonical_remote_minus_canonical_local': ('remote_canonical', 'local_canonical'),
    'canonical_remote_minus_raw_local': ('remote_canonical', 'local_raw'),
}
# An isolated failed worker may retain resources until OS/context teardown.
# Never turn a failed drain into an early unmap/unregister or a successful run.
_FAILED_RESOURCES = []


def emit(stage, **fields):
    print(json.dumps(dict(stage=stage, **fields)), flush=True)


def numa_affinity(node=0, slot=0):
    selected = set()
    for part in Path(f'/sys/devices/system/node/node{node}/cpulist').read_text().strip().split(','):
        low, _, high = part.partition('-')
        selected.update(range(int(low), int(high or low) + 1))
    allowed = os.sched_getaffinity(0)
    choices = sorted(selected & allowed)
    if type(slot) is not int or slot < 0 or slot >= len(choices):
        raise RuntimeError(f'NUMA{node} has insufficient authorized CPUs for slot {slot}')
    os.sched_setaffinity(0, {choices[slot]})
    return choices[slot]


def wait_start_barrier(directory, name, tag, *, generation, cpu_slot):
    """One fresh single-case barrier; files are retained as coordination evidence."""
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', name) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,128}', tag):
        raise ValueError('invalid barrier name or tag')
    ready = directory / f'{name}.ready.{tag}'
    go = directory / f'{name}.GO'
    if go.exists() or go.is_symlink():
        raise FileExistsError('barrier GO already exists; use a fresh barrier name')
    record = dict(tag=tag, generation=generation, cpu_slot=cpu_slot,
                  pid=os.getpid(), host=os.uname().nodename,
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  ready_monotonic_ns=time.monotonic_ns())
    fd = os.open(ready, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as output:
        json.dump(record, output, sort_keys=True)
        output.write('\n')
        output.flush()
        os.fsync(output.fileno())
    emit('barrier_ready', barrier=name, ready_file=str(ready), **record)
    deadline = time.monotonic() + 60.0
    while True:
        if go.is_symlink():
            raise ValueError('barrier GO must not be a symlink')
        if go.is_file():
            record['go_observed_monotonic_ns'] = time.monotonic_ns()
            emit('barrier_released', barrier=name, tag=tag,
                 go_observed_monotonic_ns=record['go_observed_monotonic_ns'])
            return record
        if time.monotonic() >= deadline:
            raise TimeoutError('barrier GO was not observed within 60 seconds')
        time.sleep(0.05)


def gpu_host_numa(device=0):
    from cuda.bindings import driver
    status, = driver.cuInit(0)
    if int(status):
        raise RuntimeError(f'cuInit failed: {status}')
    status, gpu = driver.cuDeviceGet(device)
    if int(status):
        raise RuntimeError(f'cuDeviceGet failed: {status}')
    status, node = driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, gpu)
    if int(status) or int(node) < 0:
        raise RuntimeError(f'GPU HOST_NUMA_ID unavailable: status={status}, node={node}')
    return int(node)


def verify_pinned_numa(controls, expected_node):
    lines = Path('/proc/self/numa_maps').read_text().splitlines()
    records = []
    for control in controls:
        address = control.storage.data_ptr()
        matches = [line for line in lines
                   if line.split() and int(line.split()[0], 16) == address
                   and 'rtp_llm_engram_' in line]
        if len(matches) != 1:
            raise RuntimeError('cannot uniquely locate this pinned table in numa_maps')
        pages = {int(node): int(count) for node, count in
                 re.findall(r'\bN(\d+)=(\d+)\b', matches[0]) if int(count) > 0}
        if not pages or set(pages) != {expected_node}:
            raise RuntimeError(f'pinned table not exclusively GPU-local NUMA{expected_node}: {pages}')
        records.append(dict(node_pages=pages, expected_gpu_host_numa=expected_node,
                            numa_maps_line=matches[0]))
    return records


def check_outputs(outputs, layer_count, groups, expected=None):
    references = []
    for index in range(layer_count):
        baseline_q, baseline_s = outputs['local_raw'][index]
        baseline_q = baseline_q.view(torch.uint8).cpu()
        baseline_s = unpack_scale_bytes(baseline_s, groups)
        if expected is not None:
            torch.testing.assert_close(baseline_q, expected[index][0], rtol=0, atol=0)
            torch.testing.assert_close(baseline_s, expected[index][1], rtol=0, atol=0)
        for q_list in outputs.values():
            q, s = q_list[index]
            torch.testing.assert_close(q.view(torch.uint8).cpu(), baseline_q, rtol=0, atol=0)
            torch.testing.assert_close(unpack_scale_bytes(s, groups), baseline_s, rtol=0, atol=0)
        references.append((baseline_q, baseline_s))
    return references


def retain_until_exit(imports, tables, pinned):
    # weakref.finalize would otherwise run at interpreter atexit even while
    # these strong references are live. OS teardown is the boundary on failure.
    for control in pinned:
        control._registration_finalizer.atexit = False
        control._lease_finalizer.atexit = False
    _FAILED_RESOURCES.append((imports, tables, pinned))


def cleanup_resources(imports, tables, pinned, native, *, graph_state_uncertain):
    try:
        torch.cuda.synchronize()
    except BaseException as error:
        retain_until_exit(imports, tables, pinned)
        emit('cleanup_deferred_to_process_exit', reason='GPU drain failed', error_type=type(error).__name__)
        raise
    if graph_state_uncertain:
        # A failed benchmark may retain graph objects in its traceback. Do not
        # free their backing even if a subsequent synchronize happens to pass.
        retain_until_exit(imports, tables, pinned)
        emit('cleanup_deferred_to_process_exit', reason='benchmark did not confirm Graph reset')
        raise RuntimeError('Graph teardown not confirmed; resources retained until process exit')
    tables.clear()
    gc.collect()
    errors = []
    for importer in imports:
        try:
            importer.close()
        except BaseException as error:
            errors.append(('native_import_close', error))
    for index, control in enumerate(pinned):
        try:
            finalizer = control._registration_finalizer
            if not finalizer.alive:
                raise RuntimeError('pinned registration finalizer was unexpectedly consumed')
            # Main's _unregister logs failures and returns None. Execute the
            # same API with an observable result while owner/storage stay live,
            # then detach the finalizer only after successful unregister.
            with torch.cuda.device(control.device):
                status, = control.runtime.cudaHostUnregister(control.storage.data_ptr())
            emit('pinned_unregister', table_index=index, cuda_status=int(status), success=int(status) == 0)
            if int(status):
                raise RuntimeError(f'cudaHostUnregister failed: {status}')
            if finalizer.detach() is None:
                raise RuntimeError('cannot detach successfully executed unregister finalizer')
            control._lease_finalizer()
        except BaseException as error:
            errors.append(('pinned_close', error))
    if errors:
        retain_until_exit(imports, tables, pinned)
        emit('cleanup_failed', errors=[dict(stage=stage, error_type=type(error).__name__,
                                           message=str(error)) for stage, error in errors])
        raise RuntimeError('one or more benchmark resource cleanup operations failed')
    imports.clear()
    pinned.clear()
    gc.collect()
    counters = native.resource_counters()
    if any(counters.values()):
        raise RuntimeError(f'native resources remain after cleanup: {counters}')
    emit('cleanup_complete', counters=counters, pinned_unregister_confirmed=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--samples', type=int, default=1000)
    p.add_argument('--batches', default='1,8,32')
    p.add_argument('--layers', default='1,8')
    p.add_argument('--heads', type=int, default=24)
    p.add_argument('--block-packs', type=int, default=4)
    p.add_argument('--num-warps', type=int, default=4)
    p.add_argument('--with-wkv', action='store_true')
    p.add_argument('--distributions', default='hot,random')
    p.add_argument('--l2-disturb', action='store_true')
    p.add_argument('--tag', default='baseline')
    p.add_argument('--cpu-slot', type=int, choices=(0, 1), default=0)
    p.add_argument('--barrier-name')
    args = p.parse_args()
    if not re.fullmatch(r'\d+(,\d+)*', args.batches) or not re.fullmatch(r'\d+(,\d+)*', args.layers):
        p.error('batches and layers must be comma-separated positive integers')
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,128}', args.tag):
        p.error('tag must contain only letters, digits, underscore or hyphen')
    batches = [int(v) for v in args.batches.split(',')]
    layer_counts = [int(v) for v in args.layers.split(',')]
    if min(batches + layer_counts + [args.samples, args.heads]) <= 0:
        p.error('positive dimensions/samples required')
    distributions = args.distributions.split(',')
    if args.samples < 5 or args.block_packs not in (1,2,4,8) or args.num_warps not in (1,2,4,8):
        p.error('samples>=5 and supported block-packs/num-warps are required')
    if not distributions or len(set(distributions)) != len(distributions) or any(x not in ('hot','random') for x in distributions):
        p.error('distributions must be distinct hot/random values')
    if args.barrier_name is not None:
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', args.barrier_name):
            p.error('invalid barrier-name')
        if len(batches) != 1 or len(layer_counts) != 1:
            p.error('barrier mode supports exactly one batch/layer case')
    torch.cuda.set_device(0)
    baseline_node = gpu_host_numa(0)
    cpu = numa_affinity(baseline_node, args.cpu_slot)
    torch.set_num_threads(1)
    native = load_native()
    quant = load_probe(str(REPO), default_build_directory())
    document = json.loads((args.directory / 'manifest.json').read_bytes())
    if document['experiment_schema'] != 1 or document['experiment'] != 'engram-latency-canonical-v0':
        raise ValueError('only explicit experimental metadata is supported')
    rows, dim, variant = document['rows'], document['dim'], document['variant']
    if dim not in (32,64,128,256) or not 0 < rows <= (1 << 24) or variant not in ('legacy','v2'):
        raise ValueError('unexpected experimental geometry/ABI')
    metadata = CanonicalMetadata(CANONICAL_REPRESENTATION,
                                LEGACY_QUANT_ABI if variant == 'legacy' else V2_QUANT_ABI)
    handles = json.loads((args.directory / 'capabilities.json').read_bytes())
    if set(handles) != {'raw','canonical'} or set(document['tables']) != set(handles):
        raise ValueError('incomplete experimental generation')
    imports, tables, pinned = [], {}, []
    graph_state_uncertain = False
    failure = None
    try:
        attach_hbm_before = torch.cuda.memory_allocated()
        for name in ('raw','canonical'):
            part = document['tables'][name]
            if part['scale_offset'] != rows * dim:
                raise ValueError('unexpected packed owner offset')
            importer = native.Consumer(base64.b64decode(handles[name], validate=True),
                                      part['allocation_bytes'], part['granularity'], 0)
            imports.append(importer)
            tables['remote_' + name] = (importer.tensor_view([rows,dim],0),
                                        importer.tensor_view([rows,dim//32],rows*dim))
        del handles
        attach_hbm_after = torch.cuda.memory_allocated()
        emit('remote_attached', torch_hbm_delta_bytes=attach_hbm_after-attach_hbm_before,
             counters=native.resource_counters())
        # Local full tables are benchmark controls only. They never replace shared
        # backing in the remote paths, and are not part of the deployment proposal.
        raw_q, raw_s = local_raw_table(rows, dim)
        converted_q, converted_s = torch.empty_like(raw_q), torch.empty_like(raw_s)
        for chunk in canonicalize_chunks(raw_q, raw_s, variant=variant, quant_module=quant,
                                         device='cuda:0', chunk_rows=16384):
            end = chunk.row_start + len(chunk.q_bytes)
            converted_q[chunk.row_start:end] = chunk.q_bytes
            converted_s[chunk.row_start:end] = chunk.scale_bytes
        del chunk
        for name, (q, s) in [('raw',(raw_q,raw_s)),('canonical',(converted_q,converted_s))]:
            control = _PinnedSharedTable(q.view(torch.float8_e4m3fn), s,
                                        f'slice4-{document["generation"]}-{name}-{uuid4()}', 'cuda:0')
            pinned.append(control)
            tables['local_' + name] = (control.cuda_view(control.weight), control.cuda_view(control.scales))
        del q,s,control,raw_q,raw_s,converted_q,converted_s
        numa = verify_pinned_numa(pinned, baseline_node)
        properties = torch.cuda.get_device_properties(0)
        emit('prepared', rows=rows, dim=dim, heads=args.heads, variant=variant,
             generation=document['generation'], local_baseline_cpu=cpu,
             cpu_slot=args.cpu_slot, barrier_name=args.barrier_name,
             local_baseline_numa=baseline_node, gpu_host_numa_query='CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID',
             local_baseline_numa_maps=numa, capability=torch.cuda.get_device_capability(0),
             device_name=properties.name, l2_bytes=getattr(properties,'L2_cache_size',None),
             allocations=document['tables'], torch=torch.__version__)
        reports = []
        for layer_count in layer_counts:
            for batch in batches:
                if rows < layer_count:
                    raise ValueError('table cannot provide disjoint layer row regions')
                ids = [torch.empty((batch,args.heads), dtype=torch.int64, device='cuda') for _ in range(layer_count)]
                outputs = {name:[allocate_outputs(rows=batch, heads=args.heads, head_dim=dim, device=0)
                                 for _ in range(layer_count)] for name in tables}
                wkv_layers, wkv_metadata = [], []
                if args.with_wkv:
                    from latency_wkv import prepare_wkv_paths, compare_wkv_outputs, compare_wkv_cpu_reference
                    for layer in range(layer_count):
                        runners, details = prepare_wkv_paths(tables, batch=batch, k=args.heads*dim,
                                                             n=256, seed=4101610+layer)
                        wkv_layers.append(runners)
                        wkv_metadata.append(details)
                    del runners
                paths = {}
                for name in tables:
                    def forward(name=name):
                        for index in range(layer_count):
                            out_q, out_s = outputs[name][index]
                            if name.endswith('_raw'):
                                bf16 = lookup_host_rows(*tables[name], ids[index], properties.multi_processor_count)
                                quant.quant(bf16.flatten(-2), out_q, out_s, variant == 'v2')
                            else:
                                gather_into(*tables[name], ids[index], out_q, out_s, metadata=metadata,
                                            block_packs=args.block_packs, num_warps=args.num_warps)
                            if args.with_wkv:
                                wkv_layers[index][name](out_q, out_s)
                        return outputs[name]
                    paths[name] = forward
                # Pin a trace once. Updating IDs is outside the measured interval.
                generator = torch.Generator().manual_seed(4101609 + layer_count * 100 + batch)
                per_layer = rows // layer_count
                trace = torch.randint(per_layer, (args.samples + 128,layer_count,batch,args.heads),
                                      generator=generator, dtype=torch.int64).pin_memory()
                trace += torch.arange(layer_count)[None,:,None,None] * per_layer
                trace_details = dict(seed=4101609 + layer_count * 100 + batch,
                                     sampling='with replacement, independent disjoint layer row ranges',
                                     trace_rows=args.samples,
                                     logical_rows_per_random_pair=layer_count*batch*args.heads,
                                     distinct_rows_random=int(torch.unique(trace[:args.samples]).numel()),
                                     distinct_rows_hot=int(torch.unique(trace[0]).numel()),
                                     epochs='consecutive blocks from one seeded trace; not independent reruns')
                def update(sample, distribution):
                    index = 0 if distribution == 'hot' else sample % len(trace)
                    for layer in range(layer_count):
                        ids[layer].copy_(trace[index,layer], non_blocking=True)
                update(1,'random')
                for name in paths: paths[name]()
                torch.cuda.synchronize()
                check_outputs(outputs, layer_count, args.heads * dim // 32)
                if args.with_wkv:
                    for layer in range(layer_count):
                        compare_wkv_outputs(wkv_layers[layer])
                        compare_wkv_cpu_reference(wkv_layers[layer]['local_raw'], *outputs['local_raw'][layer])
                case = dict(rows=rows, dim=dim, heads=args.heads, batch=batch,
                            layer_count=layer_count, variant=variant, block_packs=args.block_packs,
                            num_warps=args.num_warps, block_rows=1, trace=trace_details,
                            base_commit='492ecf45b3ef55b22fd4882622abfdb2e2f664f2',
                            source_sha256={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest()
                                           for f in ('latency_consumer.py','latency_fixture.py',
                                                     'canonical_gather.py','canonical_preprocess.py','bench_consumer.py','latency_wkv.py')},
                            table_bytes=rows*(dim+dim//32),
                            scope=('hash IDs ready to fixed synthetic WKV ready; no gate/residual' if args.with_wkv
                                   else 'hash IDs ready to GEMM input ready; no GEMM'),
                            wkv=wkv_metadata,
                            fixed_generation=document['generation'], disjoint_layer_row_regions=True)
                barrier = None
                if args.barrier_name is not None:
                    barrier = wait_start_barrier(args.directory, args.barrier_name, args.tag,
                                                 generation=document['generation'], cpu_slot=args.cpu_slot)
                case['concurrency'] = dict(cpu_slot=args.cpu_slot, cpu=cpu, tag=args.tag,
                                           host=os.uname().nodename,
                                           cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                                           barrier_name=args.barrier_name, barrier=barrier)
                graph_state_uncertain = True
                benchmark_start_ns = time.monotonic_ns()
                emit('run_bench_start', tag=args.tag, cpu_slot=args.cpu_slot,
                     host=os.uname().nodename, monotonic_ns=benchmark_start_ns)
                try:
                    report = run_bench(paths, update, case_metadata=case, samples=args.samples,
                                       comparisons=COMPARISONS,
                                       distributions=tuple(distributions), l2_disturb=args.l2_disturb)
                finally:
                    benchmark_end_ns = time.monotonic_ns()
                    emit('run_bench_end', tag=args.tag, cpu_slot=args.cpu_slot,
                         host=os.uname().nodename, monotonic_ns=benchmark_end_ns)
                report['summary']['run_bench_window'] = dict(
                    host=os.uname().nodename, tag=args.tag, cpu_slot=args.cpu_slot,
                    clock='monotonic_ns_same_host', start_ns=benchmark_start_ns,
                    end_ns=benchmark_end_ns,
                    scope='entire run_bench call including warmup/capture; not per-sample overlap')
                graph_state_uncertain = False
                # All paths finished the same final input IDs. Check the actual
                # replay outputs before an eager call could overwrite them.
                graph_expected = check_outputs(outputs, layer_count, args.heads * dim // 32)
                if args.with_wkv:
                    for runners in wkv_layers:
                        compare_wkv_outputs(runners)
                    wkv_expected = [runners['local_raw'].output.cpu() for runners in wkv_layers]
                for name in paths: paths[name]()
                torch.cuda.synchronize()
                check_outputs(outputs, layer_count, args.heads * dim // 32, expected=graph_expected)
                if args.with_wkv:
                    for layer, runners in enumerate(wkv_layers):
                        compare_wkv_outputs(runners)
                        torch.testing.assert_close(runners['local_raw'].output.cpu(), wkv_expected[layer],rtol=0,atol=0)
                    del runners,wkv_expected
                del graph_expected
                emit('post_benchmark_numerics_pass', layer_count=layer_count, batch=batch)
                reports.append(report)
                output = args.directory / f'latency-{args.tag}-L{layer_count}-M{batch}.json'
                output.write_text(json.dumps(report,indent=2))
                emit('case_complete', case=case, report=str(output))
                paths.clear(); outputs.clear(); ids.clear(); wkv_layers.clear()
                del forward,paths,outputs,ids,trace,update
    except BaseException as error:
        failure = error
        emit('benchmark_failed', error_type=type(error).__name__, message=str(error))
        raise
    finally:
        try:
            cleanup_resources(imports, tables, pinned, native,
                              graph_state_uncertain=graph_state_uncertain)
        except BaseException as cleanup_error:
            if failure is None:
                raise
            emit('cleanup_failure_after_benchmark_error', error_type=type(cleanup_error).__name__,
                 message=str(cleanup_error))
    emit('complete', cases=len(reports), counters=native.resource_counters(),
         production_model_validated=False)


if __name__ == '__main__':
    main()
