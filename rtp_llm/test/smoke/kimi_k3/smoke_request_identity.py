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
    owners, verifies, terminal = {}, defaultdict(list), defaultdict(list)
    for owner in range(8 // tp_size):
        for e in decode[owner*tp_size]:
            if e.get('event') == 'pd_cache_loaded':
                rid = e['request_id']
                assert rid not in owners or owners[rid] == owner
                owners[rid] = owner
            if e.get('event') == 'terminal_generate_output':
                terminal[e['request_id']].append((owner, e))
            if e.get('event') == 'mtp_target_verify_forward' and not e.get('fake'):
                for slot, rid in enumerate(e['request_ids']):
                    assert rid not in owners or owners[rid] == owner
                    owners[rid] = owner
                    verifies[rid].append((e, slot))

    def response_signature(rid, trailing_steps):
        events = verifies[rid]
        if rid not in terminal:
            return len(events) + 1, None
        outputs = sorted(terminal[rid], key=lambda pair: pair[1]['time_ns'])
        assert all(owner == owners[rid] for owner, _ in outputs), (
            'terminal output came from the wrong Decode owner')
        output = outputs[0][1]
        trailing = [(event, slot) for event, slot in events if event['time_ns'] > output['time_ns']]
        # Only an already scheduled round at the final context is allowed
        # after the terminal snapshot. All forwards still undergo the all-rank
        # and Graph audits, including this round.
        if trailing:
            assert len(trailing) == 1, 'multiple verifies after terminal output'
            event, slot = trailing[0]
            assert event['context_lengths'][slot] == output['output_len'], (
                'post-terminal verify did not start at the final context')
            trailing_steps[rid] = event['execution_step']
        # The stream iteration is incremented when the next round is
        # scheduled. The first terminal snapshot can therefore include that
        # round even though its forward begins after the snapshot.
        total = len(events) + 1
        assert output['iter_count'] in (total - len(trailing), total), (
            'terminal iteration count differs from native forwards')
        if len(outputs) > 1:
            assert len(outputs) == 2 and trailing, 'unexplained duplicate terminal snapshot'
            later = outputs[1][1]
            assert (later['time_ns'] > trailing[0][0]['time_ns']
                    and later['iter_count'] == total
                    and later['output_len'] >= output['output_len']), (
                'duplicate terminal snapshot did not come from the trailing verify')
        return output['iter_count'], output['output_len']
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
        trailing_steps = {}
        if stage.get('passed'):
            expected, actual = {}, defaultdict(list)
            native_signatures = {}
            verify_output_length = all(rid in terminal for rid in ids)
            for rid in ids:
                assert rid in owners
                count, output_len = response_signature(rid, trailing_steps)
                sig = (*metadata[rid], owners[rid], count)
                if verify_output_length:
                    sig += (output_len,)
                native_signatures[rid] = sig
                actual[sig].append(rid)
            for name in names:
                row = raw[name]['result']
                assert raw[name]['passed']
                expected[name] = (row['input_len'], row['prefill_total_reuse_len'],
                                  row['decode_owner_rank'], row['iter_count'])
                if verify_output_length:
                    expected[name] += (row['output_len'],)
                if True in modes:
                    rid = explicit[name]
                    assert expected[name] == native_signatures[rid], name
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
                           signature_multiset_checked=bool(stage.get('passed')),
                           post_terminal_verify_steps=trailing_steps))
    assert index == len(unknown)
    return case_ids, methods, proofs
