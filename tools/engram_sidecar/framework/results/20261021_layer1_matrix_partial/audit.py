#!/usr/bin/env python3
"""Recompute preserved partial-run statistics without CUDA or framework imports.

Example: audit.py --archive ORIGINAL --archive-manifest MANIFEST --source-repo REPO
--diagnose PRIVATE_DIAGNOSE --cleanup-summary PRIVATE_SUMMARY --output NEW_DIRECTORY
All raw member bytes are preserved; tar identity metadata is normalized for sharing.
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

COMMIT = '60e0dd54936d5b154102bc700b8396a46f02e642'
BASE = '492ecf45b3ef55b22fd4882622abfdb2e2f664f2'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    with path.open('x') as out:
        json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
        out.write('\n')


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


def verify_stats(actual, expected):
    assert set(actual) == set(expected), (set(actual), set(expected))
    for key, value in actual.items():
        if isinstance(value, (int, float)):
            assert math.isclose(value, expected[key], rel_tol=1e-10, abs_tol=1e-7), (key, value, expected[key])
        else:
            assert value == expected[key], (key, value, expected[key])


def order_for(pair, seed):
    names = ['local_raw', 'remote_raw']
    random.Random(seed).shuffle(names)
    cycle, rotation = divmod(pair, 2)
    if cycle % 2:
        names.reverse()
    return names[rotation:] + names[:rotation]


def private_pattern(data):
    # Published package contains only controlled benchmark/source metadata.
    # Exact JSON and member schema checks below complement this lexical scan.
    patterns = [rb'(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])', rb'e\d\d-cn-[A-Za-z0-9]+',
                rb'(?:/home/|/Users/|/workspace/|/mnt/fuse/)',
                rb'-----BEGIN .*PRIVATE KEY', rb'(?i)(?:bearer |api[_-]?key|authorization|password|fabric_handle|capabilities\.json)']
    return [p.decode() for p in patterns if re.search(p, data)]


def pack(args, manifest):
    original = args.archive.read_bytes()
    assert len(original) == manifest['archive_bytes'] and sha(original) == manifest['archive_sha256']
    files = {}
    with tarfile.open(fileobj=io.BytesIO(original), mode='r:gz') as src:
        members = src.getmembers()
        assert len(members) == len(manifest['files']) == 29
        assert len({m.name for m in members}) == len(members)
        with (args.output / 'measurements.tar.gz').open('xb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as gz:
                with tarfile.open(fileobj=gz, mode='w', format=tarfile.USTAR_FORMAT) as dst:
                    for member in members:
                        name = member.name
                        assert member.isfile() and not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts
                        assert name in manifest['files']
                        data = src.extractfile(member).read()
                        assert len(data) == manifest['files'][name]['bytes'] and sha(data) == manifest['files'][name]['sha256']
                        assert not private_pattern(data), (name, private_pattern(data))
                        info = tarfile.TarInfo(name)
                        info.size, info.mode, info.mtime = len(data), 0o644, 0
                        info.uid = info.gid = 0
                        info.uname = info.gname = ''
                        dst.addfile(info, io.BytesIO(data))
                        files[name] = data if not name.endswith('samples.jsonl') else None
    archive = args.output / 'measurements.tar.gz'
    write(args.output / 'archive.json', dict(archive_bytes=archive.stat().st_size,
          archive_sha256=sha(archive.read_bytes()), original_archive_sha256=manifest['archive_sha256'],
          original_archive_bytes=manifest['archive_bytes'], files=manifest['files'],
          normalization='Only gzip/tar owner, mode and timestamps normalized; all 29 member payload hashes unchanged.',
          payload_privacy_scan='No IPv4/SN/private-home-or-workspace-path/credential-or-capability patterns found.'))
    return files


def compact_pair(case, distribution, pair_id, rows):
    paths = {}
    for name in ('local_raw', 'remote_raw'):
        gpu = next(r for r in rows if r['path'] == name and r['clock'].startswith('gpu_'))
        wall = next(r for r in rows if r['path'] == name and r['clock'].startswith('cpu_'))
        paths[name] = dict(gpu_us=gpu['duration_us'], wall_us=wall['duration_us'],
                           submit_call_us=gpu['cpu_submit_call_us'], completion_sync_call_us=gpu['cpu_completion_sync_call_us'],
                           thread_nvcsw_delta=gpu['thread_nvcsw_delta'], thread_nivcsw_delta=gpu['thread_nivcsw_delta'],
                           wall_begin_perf_counter_ns=gpu['wall_begin_perf_counter_ns'],
                           wall_end_perf_counter_ns=gpu['wall_end_perf_counter_ns'])
    return dict(case=case, distribution=distribution, pair_id=pair_id, epoch=0, path_order=rows[0]['path_order'], paths=paths,
                gpu_delta_us=paths['remote_raw']['gpu_us'] - paths['local_raw']['gpu_us'],
                wall_delta_us=paths['remote_raw']['wall_us'] - paths['local_raw']['wall_us'])


def analyze(args):
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.archive_manifest.read_text())
    files = pack(args, manifest)
    run = json.loads(files['run.json'])
    assert not any(name.endswith('complete.json') for name in files)
    source_checks = []
    for name, expected in run['source_sha256'].items():
        actual = sha(subprocess.check_output(['git', '-C', str(args.source_repo), 'show', COMMIT + ':' + name]))
        assert actual == expected
        source_checks.append(dict(file=name, sha256=actual, commit_match=True))
    summaries = []
    all_exceed = []
    extremes = defaultdict(list)
    calibration_records = []
    graph_neighbor = []
    total_records = 0
    traces = {}
    with tarfile.open(args.archive, 'r:gz') as src:
        for mode in ('eager', 'graph'):
            for batch in (1, 2, 4, 8, 16, 32, 64):
                case = f'layer1-M{batch}-{mode}'
                s = json.loads(files[case + '/summary.json'])
                cm, protocol = s['case_metadata'], s['protocol']
                assert cm['batch'] == batch and cm['layer_id'] == 1 and cm['mode'] == mode
                assert cm['trace_seed'] == run['seed'] + 100 + batch
                assert cm['baseline_numa']['node_pages'] == {'0': 1546900}
                assert cm['table_bytes'] == 101377628352 and cm['table_rows'] == 384006168
                assert cm['identity'] == run['identity'] and protocol['independent_runs'] == 1
                assert protocol['samples_per_distribution'] == 10000 and protocol['warmup'] == 20
                assert protocol['scratch_bytes'] == 0 and not protocol['instrumentation_subtracted']
                assert s['model_slo'] == 'NOT_RUN'
                assert [x['trace_index'] for x in s['correctness']] == [0, 1, 0, 9999]
                assert all(x['mode'] == mode for x in s['correctness'])
                assert traces.setdefault(batch, cm['trace_sha256']) == cm['trace_sha256']
                values, deltas, seen = defaultdict(list), defaultdict(list), defaultdict(list)
                stream = src.extractfile(case + '/samples.jsonl')
                rows_iter = (json.loads(line) for line in stream)
                for (distribution, pair_id), group in itertools.groupby(rows_iter, key=lambda r: (r['distribution'], r['pair_id'])):
                    rows = list(group)
                    assert len(rows) == 4 and distribution in ('hot', 'random')
                    seen[distribution].append(pair_id)
                    assert pair_id == len(seen[distribution]) - 1
                    expected_order = order_for(pair_id, run['seed'])
                    assert [r['path'] for r in rows] == [expected_order[0]] * 2 + [expected_order[1]] * 2
                    assert {r['clock'] for r in rows} == {f'gpu_{mode}_event_interval', 'cpu_submit_to_event_complete'}
                    for row in rows:
                        assert row['epoch'] == 0 and row['mode'] == mode and row['path_order'] == expected_order
                        assert row['order_index'] == expected_order.index(row['path'])
                        assert math.isfinite(row['duration_us']) and row['duration_us'] > 0
                        wall = (row['wall_end_perf_counter_ns'] - row['wall_begin_perf_counter_ns']) / 1000
                        assert math.isclose(wall, row['cpu_submit_call_us'] + row['cpu_completion_sync_call_us'], abs_tol=1e-7)
                        if row['clock'].startswith('cpu_'):
                            assert math.isclose(row['duration_us'], wall, abs_tol=1e-7)
                        values[(distribution, row['clock'], row['path'])].append(row['duration_us'])
                    for path in expected_order:
                        one = [r for r in rows if r['path'] == path]
                        assert {k:v for k,v in one[0].items() if k not in ('clock','duration_us')} == {k:v for k,v in one[1].items() if k not in ('clock','duration_us')}
                    assert rows[0]['wall_end_perf_counter_ns'] <= rows[2]['wall_begin_perf_counter_ns']
                    pair = compact_pair(case, distribution, pair_id, rows)
                    for metric in ('gpu', 'wall'):
                        clock = f'gpu_{mode}_event_interval' if metric == 'gpu' else 'cpu_submit_to_event_complete'
                        delta = pair[metric + '_delta_us']
                        deltas[(distribution, clock)].append(delta)
                        extremes[mode + '_' + metric].append(pair)
                    if pair['gpu_delta_us'] > 100 or pair['wall_delta_us'] > 100:
                        all_exceed.append(pair)
                    if case == 'layer1-M2-graph' and distribution == 'random' and 1626 <= pair_id <= 1632:
                        graph_neighbor.append(pair)
                    total_records += 4
                assert seen == {d: list(range(10000)) for d in ('hot', 'random')}
                for entry in s['absolute']:
                    verify_stats(stats(values[(entry['distribution'], entry['clock'], entry['path'])]), entry['stats'])
                for entry in s['paired_deltas']:
                    recomputed = stats(deltas[(entry['distribution'], entry['clock'])])
                    verify_stats(recomputed, entry['paired_delta_stats'])
                    expected_status = 'FAIL_OBSERVED' if recomputed['over_threshold_count'] else 'NO_OBSERVED_VIOLATION'
                    assert entry['threshold_observation'] == expected_status
                    summaries.append(dict(case=case, batch=batch, mode=mode, distribution=entry['distribution'], clock=entry['clock'],
                                          paired=recomputed, local=stats(values[(entry['distribution'], entry['clock'], 'local_raw')]),
                                          remote=stats(values[(entry['distribution'], entry['clock'], 'remote_raw')]),
                                          threshold_observation=expected_status))
                for record in s['instrumentation_control'].get('records', []):
                    calibration_records.append(dict(case=case, **record))
    assert total_records == 1120000 and len(summaries) == 56
    peak_records = {}
    for name, records in extremes.items():
        key = 'gpu_delta_us' if name.endswith('_gpu') else 'wall_delta_us'
        peak_records[name] = {'positive_top_8': sorted(records, key=lambda r: r[key], reverse=True)[:8],
                              'negative_bottom_4': sorted(records, key=lambda r: r[key])[:4]}
    write(args.output / 'outlier_examples.json', dict(extremes=peak_records, graph_M2_random_neighbors=graph_neighbor,
          calibration_top_wall=sorted(calibration_records, key=lambda r: r.get('cpu_submit_to_event_complete_us', r.get('cpu_submit_to_stream_complete_us', 0)), reverse=True)[:8]))
    with (args.output / 'all_positive_threshold_pairs.jsonl').open('x') as stream:
        for pair in all_exceed:
            stream.write(json.dumps(pair, sort_keys=True, allow_nan=False) + '\n')
    diagnose = json.loads(args.diagnose.read_text())
    clean = json.loads(args.cleanup_summary.read_text())
    d = diagnose['diagnostic']
    assert sum(d['resident_numa_pages'].values()) == d['resident_pages'] == 1546943
    assert d['resident_numa_pages'] == {'0': 1287941, '1': 259002}
    assert clean['diagnostic_numa_pages'] == d['resident_numa_pages']
    assert clean['cleanup_status'] == 'CLEANED_EXACT_TWO_FILES' and clean['exact_original_process_absent']
    states = diagnose['log']['states']
    completed = [x['case'] for x in states if x['status'] == 'CASE_COMPLETE']
    assert len(completed) == 14 and states[-1]['status'] == 'FAILED'
    failure = dict(original_run_status='PARTIAL_FAILED', original_complete_json_present=False,
        completed_cases=completed, failed_layer=14,
        final_log_states=[{k:v for k,v in x.items() if k != 'paired_deltas'} for x in states[-3:]],
        retained_table_bytes=diagnose['file_identity']['table']['bytes'],
        diagnostic=dict(page_size_bytes=65536, resident_pages=d['resident_pages'], node_pages=d['resident_numa_pages'],
                        numa1_fraction=d['resident_numa_pages']['1']/d['resident_pages'],
                        mapping=d['mapping'], mapping_closed=d['mapping_closed']),
        separate_operator_cleanup=dict(status=clean['cleanup_status'], original_process_absent=clean['exact_original_process_absent'],
            shm_before_bytes=clean['shm_before_bytes'], shm_after_bytes=clean['shm_after_bytes'],
            no_compute_pids_before_or_after=not clean['gpu_compute_pids_before'] and not clean['gpu_compute_pids_after'],
            host_shm_modified=clean['host_shm_modified'], checkpoint_source_modified=clean['checkpoint_source_modified']),
        private_evidence_sha256=dict(diagnose=sha(args.diagnose.read_bytes()), cleanup_summary=sha(args.cleanup_summary.read_bytes())),
        interpretation='Post-failure diagnostic proves mixed NUMA placement of the retained layer14 baseline. Generic RuntimeError does not establish the exact throw site. Later operator cleanup is not original-run native-resource PASS.')
    write(args.output / 'failure_evidence.json', failure)
    aggregate = {}
    for mode in ('eager', 'graph'):
        for label, prefix in [('gpu','gpu_'),('wall','cpu_')]:
            selected = [r for r in summaries if r['mode'] == mode and r['clock'].startswith(prefix)]
            aggregate[mode + '_' + label] = dict(pairs=sum(r['paired']['n'] for r in selected),
                positive_over_100_count=sum(r['paired']['over_threshold_count'] for r in selected),
                largest_delta_us=max(r['paired']['max_us'] for r in selected),
                smallest_delta_us=min(r['paired']['min_us'] for r in selected))
    result = dict(run_status='PARTIAL_FAILED', overall_acceptance='NOT_PASSED', whole_decode_slo='NOT_MEASURED',
                  source_commit=COMMIT, main_base=BASE, original_run=run, raw_records_verified=total_records,
                  paired_observations=280000, complete_case_count=14, distribution_case_count=28,
                  aggregate=aggregate, statistics=summaries,
                  proof=dict(all_raw_stats_match_summaries=True, archive_and_all_member_hashes_match=True,
                  balanced_nonoverlapping_pair_order=True, matching_eager_graph_trace_hashes=True,
                  full_trace_bytes_regenerated=False, source_files=source_checks,
                  trace_limit='Verified recorded trace hashes/seeds match between modes and verified the hashed source uses the same IDs for both paths; raw trace tensor was not exported or regenerated.',
                  privacy='All raw member contents scanned; tar owner metadata normalized; diagnostic report selectively redacted; no endpoint/capability or private path published.'),
                  independent_runs=1, confidence_interval='NOT_COMPUTED; one process/epoch, no independent-restart repetitions',
                  after_failure_cleanup='Separate operator action; original complete/native-zero proof absent')
    write(args.output / 'matrix.json', result)
    print(json.dumps(dict(aggregate=aggregate, raw_records=total_records, outlier_pairs=len(all_exceed),
                          graph_M2_random_1629=next(p for p in graph_neighbor if p['pair_id']==1629)), indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('archive','archive-manifest','source-repo','diagnose','cleanup-summary','output'):
        parser.add_argument('--'+name, required=True, type=Path)
    analyze(parser.parse_args())
