#!/usr/bin/env python3
"""CPU-only archive/source/budget/statistical audit for completed real-table runs.

Retains every original member byte; only tar/gzip metadata is normalized.
Does not launch a process on a device or turn component timings into a model SLO.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import io
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import random
import re
import statistics
import subprocess
import tarfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    with path.open('x') as out:
        json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
        out.write('\n')


def scan(data):
    patterns = [rb'(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])', rb'e\d\d-cn-[A-Za-z0-9]+',
                rb'(?:/home/|/Users/|/workspace/|/mnt/fuse/)', rb'-----BEGIN .*PRIVATE KEY',
                rb'(?i)(?:bearer |api[_-]?key|authorization|password|fabric_handle|capabilities\.json)']
    return [pattern.decode() for pattern in patterns if re.search(pattern, data)]


def stats(values):
    ordered = sorted(values)
    n = len(values)
    result = dict(n=n, min_us=ordered[0], max_us=ordered[-1], mean_us=statistics.mean(values),
                  stdev_us=statistics.pstdev(values), threshold_us=100.0,
                  over_threshold_count=sum(x > 100 for x in values), quantile_method='nearest_rank')
    result['over_threshold_fraction'] = result['over_threshold_count'] / n
    for name, q in [('p50_us', .5), ('p95_us', .95), ('p99_us', .99), ('p99_9_us', .999)]:
        result[name] = ordered[math.ceil(n * q) - 1]
    return result


def check_stats(actual, expected):
    assert set(actual) == set(expected)
    for key, value in actual.items():
        if isinstance(value, (int, float)):
            assert math.isclose(value, expected[key], rel_tol=1e-10, abs_tol=1e-7), (key, value, expected[key])
        else:
            assert value == expected[key]


def balanced_order(pair, seed):
    names = ['local_raw', 'remote_raw']
    random.Random(seed).shuffle(names)
    cycle, rotation = divmod(pair, 2)
    if cycle % 2:
        names.reverse()
    return names[rotation:] + names[:rotation]


def compact(case, rows):
    paths = {}
    for name in ('local_raw', 'remote_raw'):
        g = next(r for r in rows if r['path'] == name and r['clock'].startswith('gpu_'))
        w = next(r for r in rows if r['path'] == name and r['clock'].startswith('cpu_'))
        paths[name] = dict(gpu_us=g['duration_us'], wall_us=w['duration_us'],
                          submit_call_us=g['cpu_submit_call_us'], completion_sync_call_us=g['cpu_completion_sync_call_us'],
                          thread_nvcsw_delta=g['thread_nvcsw_delta'], thread_nivcsw_delta=g['thread_nivcsw_delta'],
                          wall_begin_perf_counter_ns=g['wall_begin_perf_counter_ns'], wall_end_perf_counter_ns=g['wall_end_perf_counter_ns'])
    return dict(case=case, distribution=rows[0]['distribution'], pair_id=rows[0]['pair_id'], epoch=0,
                path_order=rows[0]['path_order'], paths=paths,
                gpu_delta_us=paths['remote_raw']['gpu_us']-paths['local_raw']['gpu_us'],
                wall_delta_us=paths['remote_raw']['wall_us']-paths['local_raw']['wall_us'])


def normalize(args, archive_meta):
    assert args.archive.stat().st_size == archive_meta['archive_bytes']
    assert sha(args.archive.read_bytes()) == archive_meta['archive_sha256']
    docs = {}
    with tarfile.open(args.archive, 'r:gz') as source:
        members = source.getmembers()
        assert len(members) == len(archive_meta['files']) == len({m.name for m in members})
        with (args.output / 'measurements.tar.gz').open('xb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as gz:
                with tarfile.open(fileobj=gz, mode='w', format=tarfile.USTAR_FORMAT) as dest:
                    for m in members:
                        assert m.isfile() and not PurePosixPath(m.name).is_absolute() and '..' not in PurePosixPath(m.name).parts
                        data = source.extractfile(m).read()
                        assert len(data) == archive_meta['files'][m.name]['bytes']
                        assert sha(data) == archive_meta['files'][m.name]['sha256']
                        assert not scan(data), (m.name, scan(data))
                        info = tarfile.TarInfo(m.name)
                        info.size, info.mode, info.mtime = len(data), 0o644, 0
                        info.uid = info.gid = 0
                        info.uname = info.gname = ''
                        dest.addfile(info, io.BytesIO(data))
                        if m.name.endswith('.json'):
                            docs[m.name] = json.loads(data)
    p = args.output/'measurements.tar.gz'
    write(args.output/'archive.json', dict(archive_bytes=p.stat().st_size, archive_sha256=sha(p.read_bytes()),
        original_archive_bytes=archive_meta['archive_bytes'], original_archive_sha256=archive_meta['archive_sha256'],
        files=archive_meta['files'], normalization='tar/gzip ownership, mode and time only; original member bytes unchanged'))
    return docs


def memory_evidence(docs, run):
    result = {}
    for layer in run['layers']:
        source = docs[f'source-cache-layer{layer}.json']
        alias = docs[f'baseline-alias-layer{layer}.json']
        assert alias == dict(weight_alias=True, scale_alias=True, cuda_allocated_delta_bytes=0)
        assert source['eviction_prevented'] is False
        pin = source['pinned_complete']
        assert pin['expected_node'] == 0 and pin['node_pages'] == {'0':pin['expected_pages']}
        assert pin['resident_pages'] == pin['expected_pages'] and pin['resident_bytes'] == pin['resident_pages'] * pin['page_bytes']
        assert pin['page_bytes'] == 65536
        assert source['source_before_construct'] == source['source_after_construct']
        source_bytes = 0
        for item in source['source_before_construct']:
            assert item['expected_node'] == 1 and item['node_pages'] == {'1':item['expected_pages']}
            assert item['expected_pages'] == item['resident_pages'] == math.ceil(item['file_bytes']/item['page_bytes'])
            assert item['resident_bytes'] == item['resident_pages']*item['page_bytes']
            source_bytes += item['resident_bytes']
        phases = {}
        for name in ('before_prefault','prefaulted','after_load'):
            evidence = source[name]
            assert evidence['phase'] == name and evidence['reserved_headroom_bytes'] == run['memory_headroom_bytes']
            assert evidence['transient_touch_bytes'] == 8 << 20
            table = pin['resident_bytes'] if name != 'after_load' else 0
            planned_source = source_bytes if name == 'before_prefault' else 0
            assert evidence['future_table_bytes'] == table and evidence['future_source_bytes'] == planned_source
            margin = run['memory_headroom_bytes'] + (8 << 20)
            snapshot = evidence['snapshot']
            slacks = dict(host=snapshot['host_available_bytes']-table-planned_source-margin,
                          pinned_node=snapshot['numa_free_bytes']-table-margin,
                          source_node=snapshot['source_numa_free_bytes']-planned_source-margin,
                          shm=snapshot['shm_free_bytes']-table)
            for i, cg in enumerate(snapshot['cgroups']):
                if cg['max_bytes'] is not None:
                    slacks[f'visible_cgroup_{i}'] = cg['max_bytes']-cg['current_bytes']-table-planned_source-margin
            assert all(v >= 0 for v in slacks.values()), slacks
            phases[name] = dict(verified_admission=True, slack_after_budget_bytes=slacks)
        post = docs[f'memory-layer{layer}-after.json']
        assert post['numa_free_bytes'] >= run['memory_headroom_bytes']
        assert post['host_available_bytes'] >= run['memory_headroom_bytes']
        for cg in post['cgroups']:
            assert cg['max_bytes'] is None or cg['max_bytes']-cg['current_bytes'] >= run['memory_headroom_bytes']
        result[str(layer)] = dict(complete_pinned=pin, complete_source_before_and_after=source['source_before_construct'],
             alias=alias, source_cpu=source['source_cpu'], phases=phases,
             actual_source_prepare_wall_time='NOT_RECORDED; no phase timestamps', eviction_prevented=False)
    return result


def analyze(args):
    args.output.mkdir(parents=True, exist_ok=True)
    meta = json.loads(args.archive_manifest.read_text())
    docs = normalize(args, meta)
    run, complete = docs['run.json'], docs['complete.json']
    assert run['source_cache_numa'] == 1 and run['local_gpu_host_numa'] == 0 and run['discard_private_source_cache']
    assert run['layer_mode'] == complete['layer_mode']
    assert set(complete['counters']) == {'cleanup_failures','contexts','deferred_cleanups','imports','mappings','owner_allocations','reservations','view_storages'}
    assert not any(complete['counters'].values())
    assert complete['decode_step_slo'] == 'NOT_MEASURED'
    source = []
    for name, digest in run['source_sha256'].items():
        actual = sha(subprocess.check_output(['git','-C',str(args.source_repo),'show',args.commit+':'+name]))
        assert actual == digest
        source.append(dict(file=name, sha256=digest, commit_match=True))
    memory = memory_evidence(docs, run)
    case_names = sorted(name[:-len('/summary.json')] for name in docs if name.endswith('/summary.json'))
    assert sorted(complete['cases']) == case_names
    assert len(case_names) == len(run['batches'])*len(run['modes'])*len(run['layer_groups'])
    summaries, traces, all_pairs, calibration = [], {}, [], []
    raw_count = 0
    with tarfile.open(args.archive, 'r:gz') as original:
        for case in case_names:
            summary = docs[case+'/summary.json']
            cm, protocol = summary['case_metadata'], summary['protocol']
            mode, batch = cm['mode'], cm['batch']
            assert cm['identity'] == run['identity'] and protocol['seed'] == run['seed']
            assert cm['layer_mode'] == run['layer_mode'] and protocol['samples_per_distribution'] == run['samples']
            assert protocol['warmup'] == 20 and not protocol['instrumentation_subtracted']
            assert not protocol['input_preparation_timed']
            if protocol['scratch_bytes']:
                assert protocol['runtime_l2_bytes'] > 0
                assert protocol['scratch_bytes'] >= 2 * protocol['runtime_l2_bytes']
                assert protocol['cache_claim'] == 'scratch_disturbed_not_guaranteed_cold'
            else:
                assert protocol['cache_claim'] == 'uncontrolled'
            assert protocol['common_graph_guard'] == (mode=='graph') and summary['model_slo'] == 'NOT_RUN'
            layers = cm['layer_ids_measured'] if run['layer_mode']=='together' else [cm['layer_id']]
            assert layers in run['layer_groups'] and cm['layer_count_measured'] == len(layers)
            if run['layer_mode']=='together':
                assert cm['covers_all_model_engram_layers'] == (layers==cm['actual_model_engram_layers'])
                for layer in layers:
                    assert cm['trace_seed'][str(layer)] == run['seed'] + layer*100 + batch
                    assert cm['baseline_numa'][str(layer)]['node_pages'] == memory[str(layer)]['complete_pinned']['node_pages']
                    expected = cm['table_rows'][str(layer)] * (cm['dim']+cm['dim']//32)
                    assert math.ceil(expected/65536) == memory[str(layer)]['complete_pinned']['resident_pages']
                assert cm['table_bytes'] == sum(cm['table_rows'][str(layer)]*(cm['dim']+cm['dim']//32) for layer in layers)
            else:
                layer=layers[0]
                assert cm['trace_seed'] == run['seed']+layer*100+batch
                assert cm['baseline_numa']['node_pages'] == memory[str(layer)]['complete_pinned']['node_pages']
            trace_key=(tuple(layers),batch)
            assert traces.setdefault(trace_key,cm['trace_sha256']) == cm['trace_sha256']
            assert [x['trace_index'] for x in summary['correctness']] == [0,1,0,run['samples']-1]
            assert all(x['mode']==mode for x in summary['correctness'])
            values, deltas, seen = defaultdict(list), defaultdict(list), defaultdict(list)
            rows_iter=(json.loads(line) for line in original.extractfile(case+'/samples.jsonl'))
            for (distribution,pair_id), group in itertools.groupby(rows_iter,key=lambda r:(r['distribution'],r['pair_id'])):
                rows=list(group)
                assert len(rows)==4 and distribution in ('hot','random')
                seen[distribution].append(pair_id)
                assert pair_id == len(seen[distribution])-1
                order=balanced_order(pair_id,run['seed'])
                assert [r['path'] for r in rows]==[order[0]]*2+[order[1]]*2
                assert {r['clock'] for r in rows}=={f'gpu_{mode}_event_interval','cpu_submit_to_event_complete'}
                for r in rows:
                    assert r['epoch']==0 and r['mode']==mode and r['path_order']==order and r['order_index']==order.index(r['path'])
                    wall=(r['wall_end_perf_counter_ns']-r['wall_begin_perf_counter_ns'])/1000
                    assert math.isfinite(r['duration_us']) and r['duration_us']>0
                    assert math.isclose(wall,r['cpu_submit_call_us']+r['cpu_completion_sync_call_us'],abs_tol=1e-7)
                    if r['clock'].startswith('cpu_'):assert math.isclose(wall,r['duration_us'],abs_tol=1e-7)
                    values[(distribution,r['clock'],r['path'])].append(r['duration_us'])
                assert rows[0]['wall_end_perf_counter_ns'] <= rows[2]['wall_begin_perf_counter_ns']
                for path in order:
                    a,b=[r for r in rows if r['path']==path]
                    assert {k:v for k,v in a.items() if k not in ('clock','duration_us')} == {k:v for k,v in b.items() if k not in ('clock','duration_us')}
                pair=compact(case,rows);all_pairs.append(pair)
                deltas[(distribution,f'gpu_{mode}_event_interval')].append(pair['gpu_delta_us'])
                deltas[(distribution,'cpu_submit_to_event_complete')].append(pair['wall_delta_us'])
                raw_count+=4
            assert seen=={d:list(range(run['samples'])) for d in ('hot','random')}
            for item in summary['absolute']:
                computed=stats(values[(item['distribution'],item['clock'],item['path'])])
                check_stats(computed,item['stats']);check_stats(computed,item['epochs']['0'])
            for item in summary['paired_deltas']:
                computed=stats(deltas[(item['distribution'],item['clock'])])
                check_stats(computed,item['paired_delta_stats']);check_stats(computed,item['epochs']['0'])
                summaries.append(dict(case=case, mode=mode, batch=batch, layers=layers, distribution=item['distribution'],clock=item['clock'],
                                      paired=computed,local=stats(values[(item['distribution'],item['clock'],'local_raw')]),
                                      remote=stats(values[(item['distribution'],item['clock'],'remote_raw')]),
                                      threshold_observation=item['threshold_observation']))
            calibration.extend(dict(case=case,**r) for r in summary['instrumentation_control'].get('records',[]))
    assert raw_count==len(case_names)*run['samples']*2*4
    observed=[p for p in all_pairs if p['gpu_delta_us']>100 or p['wall_delta_us']>100]
    with (args.output/'all_positive_threshold_pairs.jsonl').open('x') as stream:
        for pair in observed:stream.write(json.dumps(pair,sort_keys=True,allow_nan=False)+'\n')
    peaks={}
    for mode in run['modes']:
        eligible=[p for p in all_pairs if p['case'].endswith('-'+mode)]
        peaks[mode]={key:dict(positive_top_4=sorted(eligible,key=lambda p:p[key],reverse=True)[:4],
                             negative_bottom_4=sorted(eligible,key=lambda p:p[key])[:4]) for key in ('gpu_delta_us','wall_delta_us')}
    write(args.output/'outlier_examples.json',dict(extremes=peaks,calibration_top_wall=sorted(calibration,key=lambda r:r.get('cpu_submit_to_event_complete_us',r.get('cpu_submit_to_stream_complete_us',0)),reverse=True)[:8]))
    selected = {}
    for pair in observed:
        if pair['case'].endswith('-graph') and pair['gpu_delta_us'] > 100:
            selected[(pair['case'],pair['distribution'],pair['pair_id'])] = 'graph_gpu_positive_over_100us'
    for mode in run['modes']:
        for clock in ('gpu_delta_us','wall_delta_us'):
            for direction in ('positive_top_4','negative_bottom_4'):
                pair = peaks[mode][clock][direction][0]
                selected.setdefault((pair['case'],pair['distribution'],pair['pair_id']), mode+'_'+clock+'_'+direction)
    neighborhoods = []
    for (case,distribution,pair_id),reason in selected.items():
        neighbors = [p for p in all_pairs if p['case']==case and p['distribution']==distribution and abs(p['pair_id']-pair_id)<=2]
        neighborhoods.append(dict(case=case,distribution=distribution,pair_id=pair_id,reason=reason,neighbors=neighbors))
    write(args.output/'anomaly_neighborhoods.json',dict(scope='retained paired neighborhoods; same epoch; no causal attribution',selection=neighborhoods))
    report=dict(run_status='COMPLETE_WITH_ZERO_NATIVE_COUNTERS', component_threshold_status='FAIL_OBSERVED' if observed else 'NO_OBSERVED_VIOLATION',
          full_decode_slo='NOT_MEASURED', raw_records_verified=raw_count, paired_observations=len(all_pairs),source_commit=args.commit,
          source_files=source,run=run,complete=complete,memory=memory,statistics=summaries,
          positive_threshold_unique_pairs=len(observed),trace_scope='recorded eager/Graph trace hash parity and source use of same IDs verified; trace bytes not regenerated',
          no_timing_or_outliers_subtracted=True,independent_restarts=1,
          privacy='all original payloads scanned; tar owner/time/mode metadata normalized only')
    write(args.output/'report.json',report)
    print(json.dumps(dict(run_status=report['run_status'],raw_records=raw_count,positive_pairs=len(observed),
                         statistics=[{k:r[k] for k in ('case','distribution','clock','paired')} for r in summaries]),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('archive','archive-manifest','source-repo','output'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--commit',required=True)
    analyze(p.parse_args())
