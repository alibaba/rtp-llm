"""Two existing C10 inputs under an isolated Last-5 tier and 2K chunks.

This supplements the 304-request smoke. It does not report a complete smoke.
Only coordinator CPU evidence is polled before probing an asynchronous
HOST/SSD write. Successful settlement follows all transfer-worker responses.
"""
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

if __package__:
    from .k31_full_smoke import K31FullRunner, legacy, parse_args
    from .audit_k31_full_smoke import read_events, resolve_request_identity, group_evidence
    from .audit_k31_last_n import audit_prefill_states, audit_checkpoint_reuse
    from .k31_last_n_precision import read_capture, read_descriptor, check_response, validate
else:
    from k31_full_smoke import K31FullRunner, legacy, parse_args
    from audit_k31_full_smoke import read_events, resolve_request_identity, group_evidence
    from audit_k31_last_n import audit_prefill_states, audit_checkpoint_reuse
    from k31_last_n_precision import read_capture, read_descriptor, check_response, validate


def completed_seed_writes(events, tier):
    """Require TP0's C10 key and successful multi-rank CRC-backed settlement.

    MultiRankTransferAsyncContext rejects missing responses and merges every
    worker's status before BlockTreeStorer/BlockTreeEvictor emit completion.
    Cache-tree metadata is centralized; it is not repeated on worker ranks.
    """
    proof = {}
    for rank in (0,):
        seeds = [row for row in events[rank]
                 if row.get('event') == 'linear_cache_publish_candidates'
                 and row.get('group_tag') == 'linear'
                 and row.get('prompt_tokens') == 12416
                 and row.get('target_tier') == tier
                 and [item['tokens'] for item in row['checkpoints']]
                 == list(range(8192, 12289, 1024))]
        if len(seeds) != 1:
            return None
        seed = seeds[0]
        key = next(item['cache_key'] for item in seed['checkpoints'] if item['tokens'] == 10240)
        writes = [row for row in events[rank]
                  if row.get('event') == 'linear_cache_transfer_completed'
                  and row.get('group_tag') == 'linear' and row.get('cache_key') == key
                  and row.get('target_tier') == tier and row.get('crc_enabled') is True
                  and row.get('payload_bytes', 0) > 0 and row['time_ns'] >= seed['time_ns']]
        if not writes:
            return None
        proof[str(rank)] = dict(cache_key=key, target_block=writes[0]['target_block'])
    return proof


class TierProbeRunner(K31FullRunner):
    def wait_seed_write(self, tier):
        if tier == 'DEVICE':
            return
        engine = self.args.prefill_engine_log
        deadline = time.monotonic() + self.args.case_deadline_s
        while time.monotonic() < deadline:
            try:
                proof = completed_seed_writes(read_events(engine.parent, engine), tier)
            except (FileNotFoundError, json.JSONDecodeError):
                proof = None  # A rank may still be appending its final record.
            if proof:
                self._completed_seed_writes = proof
                return
            time.sleep(0.5)
        raise legacy.SmokeFailure(f'no completed C10 {tier} multi-rank write on TP0')

    def run_probe(self, tier):
        prompt = self.cache_seed_prompt(f'{self.args.namespace}-tier-0', 'MEM-0')
        seed = self._orthogonal_answer('tier-memory-seed-0', prompt, 'MEM-0', 'miss',
            preparation_only=True, expected_input_len=12416, expected_reuse_len=0)
        self._required_stage(seed.name, [seed])
        self.wait_seed_write(tier)
        probe = replace(self.cache_probe_case(seed), name='tier-device-probe-0', reuse='hit',
                        preparation_only=False,
                        expected_cache_tier={'DEVICE': 'device', 'HOST': 'memory', 'DISK': 'disk'}[tier])
        self._required_stage(probe.name, [probe])

    def save(self, passed, error=None):
        if passed and (len(self.records) != 2 or self.submitted_requests != 2):
            passed, error = False, 'isolated tier probe requires exactly two requests'
        legacy.Runner.save(self, passed, error)
        result = json.loads(self.args.output.read_text())
        result.update(source_commit=self.args.source_commit, geometry=self.args.geometry_value,
                      native_K31_MTP=self.args.draft_contract is not None, target_only=False,
                      validation_profile='last5-isolated-tier-2k', complete_smoke_passed=False,
                      completed_seed_writes=getattr(self, '_completed_seed_writes', None))
        self.args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')


def audit_probe(result, prefill, decode, output, config):
    tier = config['last_n_tier_probe']
    names = ('tier-memory-seed-0', 'tier-device-probe-0')
    rows = {row['name']: row for row in result['cases']}
    if (not result.get('passed') or set(rows) != set(names)
            or result.get('validation_profile') != 'last5-isolated-tier-2k'
            or result.get('native_K31_MTP') is not True or result.get('target_only') is not False
            or any(not stage.get('passed') or stage.get('skipped') for stage in result['stages'])):
        raise ValueError('isolated tier probe is incomplete')
    raw = {name: dict(result=row, passed=True, transport='http') for name,row in rows.items()}
    ids, _, barriers = resolve_request_identity(result['stages'], raw, prefill, decode, tp_size=8)
    if any(len(ids[name]) != 1 for name in names):
        raise ValueError('isolated tier probe needs unique native request identities')
    for forward in prefill[0]:
        if forward.get('event') == 'mtp_target_prefill_forward' and not forward.get('fake'):
            group_evidence(prefill, forward, 8)
    lifecycle = {0: audit_prefill_states(prefill, max_chunk_tokens=2048,
                         require_one_million=False, root_rank=0)}
    for proof in lifecycle.values():
        seed = [row for row in proof['requests'] if row['request_id'] == ids[names[0]][0]]
        if (len(seed) != 1 or seed[0]['rounds'] != 7 or seed[0]['aliased_rounds'] < 1
                or seed[0]['new_state_blocks'] != 6 or seed[0]['completed_tokens'] != 12416):
            raise ValueError('seed did not execute seven real chunks with six bounded state blocks')
    reuse = {0: audit_checkpoint_reuse(prefill, ids[names[0]], ids[names[1]], tier)}
    writes = completed_seed_writes(prefill, tier) if tier != 'DEVICE' else None
    if tier != 'DEVICE' and not writes:
        raise ValueError('isolated lower-tier probe lacks completed multi-rank CRC-backed C10 writes')
    output = Path(output)
    actual = read_capture(output/'prefill/logits', output/'decode/logits',
                          ids[names[1]][0], result['source_commit'])
    check_response(actual, rows[names[1]])
    contract = json.loads(Path(config['precision_contract']).read_text())
    if contract.get('prefill_chunk_tokens') != 2048:
        raise ValueError('isolated probe requires a prelocked 2K calibration')
    precision = validate(read_descriptor(config['precision_reference']), actual, contract)
    return dict(passed=True, complete_smoke_passed=False, source_tier=tier,
                request_ids=ids, stage_barriers=barriers, lifecycle=lifecycle,
                checkpoint_reuse=reuse, completed_seed_writes=writes, precision=precision)


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    # The tier is an explicit controller option, not a request-side override.
    index = arguments.index('--isolated-tier')
    tier = arguments[index+1]
    del arguments[index:index+2]
    if tier not in ('DEVICE', 'HOST', 'DISK'):
        raise ValueError('invalid isolated tier')
    args = parse_args(arguments)
    runner = TierProbeRunner(args)
    try:
        runner.run_probe(tier)
        runner.save(True)
        return 0
    except Exception as error:
        runner.save(False, f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
