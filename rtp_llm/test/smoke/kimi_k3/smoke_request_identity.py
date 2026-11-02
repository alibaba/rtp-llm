"""Resolve existing evidence without fabricating frontend marker events.

The frozen shared runner waits for a whole stage before admitting the next.
We verify that causal barrier against the complete native request inventory.
Within concurrent stages identical signatures remain explicitly set-valued.
"""
from collections import Counter, defaultdict


def resolve(stages, raw, prefill, decode, tp_size=4):
    metadata, order = {}, {}
    for e in prefill[0]:
        if e.get('event') != 'mtp_target_prefill_forward' or e.get('fake'):
            continue
        ids, qs, prefixes = e['request_ids'], e['context_lengths'], e['reuse_lengths']
        assert len(ids) == len(qs) == len(prefixes)
        for rid, q, prefix in zip(ids, qs, prefixes):
            assert rid not in metadata, 'request had multiple initial Prefill calls'
            metadata[rid] = (q+prefix, prefix)
            order[rid] = e['execution_step']
    owners, iterations = {}, Counter()
    for owner in range(8 // tp_size):
        for e in decode[owner*tp_size]:
            if e.get('event') == 'pd_cache_loaded':
                rid = e['request_id']
                assert rid not in owners or owners[rid] == owner
                owners[rid] = owner
            if e.get('event') == 'mtp_target_verify_forward' and not e.get('fake'):
                for rid in e['request_ids']:
                    assert rid not in owners or owners[rid] == owner
                    owners[rid] = owner
                    iterations[rid] += 1
    expected_names = [name for s in stages for name in s['case_names']]
    assert len(expected_names) == len(set(expected_names))
    assert set(expected_names) == set(raw), 'incomplete raw request inventory'
    explicit = {name: a['request_id'] for name,a in raw.items()
                if a.get('transport') == 'pd_group_rpc'}
    assert len(set(explicit.values())) == len(explicit)
    assert set(explicit.values()) <= metadata.keys()
    unknown = sorted(set(metadata)-set(explicit.values()), key=lambda rid:order[rid])
    assert len(unknown) == len(raw)-len(explicit), 'unexplained native traffic'
    index, last_step = 0, -1
    case_ids, methods, proofs = {}, {}, []
    for stage in stages:
        names = stage['case_names']
        modes = {name in explicit for name in names}
        assert len(modes) == 1, 'mixed transport stage needs explicit handling'
        if True in modes:
            ids = [explicit[name] for name in names]
            for name in names:
                case_ids[name] = [explicit[name]]
                methods[name] = 'explicit_group_RPC_request_id'
        else:
            ids = unknown[index:index+len(names)]
            assert len(ids) == len(names)
            index += len(names)
        steps = [order[rid] for rid in ids]
        assert min(steps) > last_step, 'stage completion/admission barrier violated'
        last_step = max(steps)
        if stage.get('passed'):
            expected, actual = {}, defaultdict(list)
            for rid in ids:
                assert rid in owners
                # iter_count includes one Prefill and all target Decode calls;
                # valid-draft verify_rounds is a different counter.
                sig = (*metadata[rid], owners[rid], iterations[rid]+1)
                actual[sig].append(rid)
            for name in names:
                row = raw[name]['result']
                assert raw[name]['passed']
                expected[name] = (row['input_len'], row['prefill_total_reuse_len'],
                                  row['decode_owner_rank'], row['iter_count'])
                if True in modes:
                    rid = explicit[name]
                    assert expected[name] == (*metadata[rid], owners[rid], iterations[rid]+1), name
            assert Counter(expected.values()) == Counter({s:len(v) for s,v in actual.items()}), (
                stage['name'], Counter(expected.values()), {s:len(v)for s,v in actual.items()})
            if True not in modes:
                for name,sig in expected.items():
                    case_ids[name] = list(actual[sig])
                    methods[name] = ('stage_barrier_and_unique_execution_signature'
                                     if len(actual[sig]) == 1 else
                                     'stage_barrier_and_complete_equivalent_signature_set')
        else:
            # Failed stages reserve their observed traffic but are never reused.
            for name in names:
                case_ids.setdefault(name, [])
                methods.setdefault(name, 'failed_not_reusable')
        proofs.append(dict(stage=stage['name'], passed=stage.get('passed'),
                           native_request_ids=ids, first_step=min(steps), last_step=max(steps),
                           signature_multiset_checked=bool(stage.get('passed'))))
    assert index == len(unknown)
    return case_ids, methods, proofs
