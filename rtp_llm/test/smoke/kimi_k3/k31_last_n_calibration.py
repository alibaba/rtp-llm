"""Six existing Last-5 fixtures measured with retention disabled.

This is precision calibration, not a complete smoke. It records two cold
references, an unbounded C10 hit and the unbounded C7 positive control before
the 304-request candidate run. The candidate keeps its original request count.
"""
from dataclasses import replace
import json
from pathlib import Path
import sys

if __package__:
    from .k31_full_smoke import K31FullRunner, legacy, parse_args
else:
    from k31_full_smoke import K31FullRunner, legacy, parse_args


class CalibrationRunner(K31FullRunner):
    def run_calibration(self):
        tag = f'{self.args.namespace}-tier-0'
        prompt = self.cache_seed_prompt(tag, 'MEM-0')
        seed = self._orthogonal_answer('tier-memory-seed-0', prompt, 'MEM-0', 'miss',
            preparation_only=True, expected_input_len=self.cache_probe_tokens, expected_reuse_len=0)
        self._memory_candidates = [(tag, seed, 'MEM-0')]
        for name in ('tier-memory-seed-1', 'tier-witness-seed'):
            cold = self._precision_reference(replace(seed, name=name))
            self._required_stage(name, [cold])
        self._required_stage(seed.name, [seed])
        probe = replace(self.cache_probe_case(seed), name='tier-device-probe-0', reuse='hit',
                        expected_cache_tier='device')
        self._required_stage(probe.name, [probe])
        a, b = self.linear_c7_cases()
        self._required_stage('prefix_A_seed', [a])
        self._required_stage('prefix_B_partial', [replace(b, reuse='partial',
                                                        expected_reuse_len=7*self.reuse_unit_tokens)])

    def save(self, passed, error=None):
        complete = len(self.records) == 6 and self.submitted_requests == 6
        if passed and not complete:
            passed, error = False, 'unbounded Last-N calibration requires exactly six requests'
        legacy.Runner.save(self, passed, error)
        path = self.args.output
        result = json.loads(path.read_text())
        result.update(source_commit=self.args.source_commit, geometry=self.args.geometry_value,
                      target_only=False, native_K31_MTP=self.args.draft_contract is not None,
                      draft_contract=self.args.draft_contract, last_n_calibration=True,
                      validation_profile='last-n-unbounded-calibration', complete_smoke_passed=False,
                      execution_audit_pending=True, request_budget=dict(total=6),
                      actual_request_counts=dict(total=len(self.records)),
                      last_n_c7_fixture=getattr(self, '_last_n_c7_fixture', None))
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')


def audit_calibration(result, prefill, decode, output, *, prefill_chunk_tokens=65536):
    if __package__:
        from .audit_k31_full_smoke import group_evidence, resolve_request_identity
        from .k31_last_n_precision import check_response, read_capture, lock_thresholds
        from . import result_check
    else:
        from audit_k31_full_smoke import group_evidence, resolve_request_identity
        from k31_last_n_precision import check_response, read_capture, lock_thresholds
        import result_check
    names = ('tier-memory-seed-1', 'tier-witness-seed', 'tier-memory-seed-0',
             'tier-device-probe-0', 'prefix-A-seed', 'prefix-B-partial')
    rows = {row['name']: row for row in result['cases']}
    if (not result.get('passed') or set(rows) != set(names)
            or result.get('validation_profile') != 'last-n-unbounded-calibration'
            or result.get('native_K31_MTP') is not True
            or result.get('target_only') is not False
            or any(not stage.get('passed') or stage.get('skipped') for stage in result['stages'])):
        raise ValueError('unbounded calibration is incomplete or lacks native MTP')
    g = result['geometry']
    page = g.get('page_size')
    if page not in (64, 128) or any(g.get(key) != value for key,value in dict(
            runtime_verified=True, runtime_layers=93, kernel_page_size=page, source_shards=8,
            checkpoint_span=page * 8, reuse_cache_tail_blocks=2147483647,
            last_n_calibration=True).items()):
        raise ValueError('calibration must use the original unbounded path at the candidate CP geometry')
    span = g['checkpoint_span']
    for row in rows.values():
        errors = result_check.independent_answer_check(row) + result_check._cache_errors(row, page)
        if errors:
            raise ValueError('; '.join(errors))
    raw = {name: dict(result=row, passed=True, transport='http') for name,row in rows.items()}
    ids, _, barriers = resolve_request_identity(result['stages'], raw, prefill, decode, tp_size=8)
    if any(len(value) != 1 for value in ids.values()):
        raise ValueError('calibration requires unique actual native request identities')
    forwards = [row for row in prefill[0] if row.get('event') == 'mtp_target_prefill_forward'
                and not row.get('fake')]
    for forward in forwards:
        group_evidence(prefill, forward, 8)
        if any(length > prefill_chunk_tokens for length in forward['context_lengths']):
            raise ValueError('calibration exceeded the configured Prefill chunk capacity')
    for name, checkpoint in (('tier-device-probe-0', 10 * span), ('prefix-B-partial', 7 * span)):
        rid = ids[name][0]
        calls = [row for row in forwards if rid in row['request_ids']]
        if not calls or calls[0]['reuse_lengths'] != [checkpoint]:
            raise ValueError(f'{name}: unbounded control did not actually resume at C{checkpoint//span}')
        # NormalEngine schedules and owns the cache tree on TP0. Other ranks
        # execute the broadcast model inputs; group_evidence checks all eight.
        if not any(row.get('event') == 'linear_cache_matched' and row.get('request_id') == rid
                   and row.get('group_tag') == 'linear' and row.get('checkpoint_tokens') == checkpoint
                   and row.get('source_tier') == 'DEVICE' and row.get('source_block', 0) > 0
                   and row['time_ns'] <= calls[0]['time_ns'] for row in prefill[0]):
            raise ValueError(f'{name}: actual unbounded LINEAR checkpoint match is absent on TP0')
    output = Path(output)
    roots = [output/role/'logits' for role in ('prefill', 'decode')]
    captures = [read_capture(*roots, ids[name][0], result['source_commit'])
                for name in ('tier-witness-seed', 'tier-memory-seed-1', 'tier-device-probe-0')]
    for capture, name in zip(captures, ('tier-witness-seed', 'tier-memory-seed-1', 'tier-device-probe-0')):
        check_response(capture, rows[name])
    for name in ('tier-witness-seed', 'tier-memory-seed-1', 'tier-device-probe-0'):
        descriptor = dict(prefill_dir=str(roots[0].resolve()), decode_dir=str(roots[1].resolve()),
                          request_id=ids[name][0], source_commit=result['source_commit'],
                          greedy_output_ids=rows[name]['output_ids'][0])
        with (output/(name+'.capture.json')).open('x') as artifact:
            json.dump(descriptor, artifact, indent=2)
    contract = lock_thresholds(*captures, output/'precision-contract.json',
                               prefill_chunk_tokens=prefill_chunk_tokens)
    return dict(passed=True, complete_smoke_passed=False, request_ids=ids, stage_barriers=barriers,
                limits=contract['limits'], reference=str(output/'tier-witness-seed.capture.json'),
                precision_contract=str(output/'precision-contract.json'))


def main(argv=None):
    args = parse_args([*(sys.argv[1:] if argv is None else argv), '--last-n-checkpoints', '5'], calibration=True)
    runner = CalibrationRunner(args)
    try:
        runner.run_calibration()
        runner.save(True)
        return 0
    except Exception as error:
        runner.save(False, f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
