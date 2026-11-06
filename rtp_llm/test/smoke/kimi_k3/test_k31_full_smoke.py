"""The full K3.1 entry keeps MTP active and cannot pass incomplete coverage."""
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from k31_full_smoke import (K31FullRunner, legacy, parse_args, pending_host_load,
                           decode_batch_sizes, request_budget)


class FullK31SmokeTest(unittest.TestCase):
    def test_bs63_shares_one_barrier_across_both_decode_owners(self):
        runner = object.__new__(K31FullRunner)
        runner.args = SimpleNamespace(output=Path("/tmp/smoke/RESULT.json"))
        runner.decode_role_addrs = [
            dict(ip="10.0.0.2", grpc_port=20001), dict(ip="10.0.0.2", grpc_port=20037),
        ]
        cases = [object() for _ in range(63)]
        relay = MagicMock()
        relay.return_value.__enter__.return_value.ports = [10001, 10002]
        module = SimpleNamespace(PdGroupRelay=relay)
        with patch.dict(sys.modules, {"pd_group_relay": module}), \
             patch.object(legacy.Runner, "_request_cases_group", create=True) as submit:
            self.assertEqual(runner._request_cases_group(cases), submit.return_value)
            self.assertEqual(relay.call_args.args[:2], (["10.0.0.2:20001", "10.0.0.2:20037"], 63))
            submit.assert_called_once_with(cases, decode_relay_ports=[10001, 10002])

    def test_group_failure_cancels_before_waiting_for_earlier_stream(self):
        cancelled = threading.Event()
        calls = []
        def fetch(request_id):
            if request_id == 11:
                if not cancelled.wait(5):
                    raise AssertionError('earlier stream blocked failure propagation')
                return request_id
            raise ValueError('later request failed')
        def cancel(request_id):
            calls.append(request_id)
            cancelled.set()
        with self.assertRaisesRegex(ValueError, 'later request failed'):
            legacy.collect_grouped_responses(fetch, [11, 12], cancel)
        self.assertEqual(calls, [11, 12])

    def test_group_response_keeps_input_order_and_never_cancels_success(self):
        later_finished = threading.Event()
        def fetch(request_id):
            if request_id == 11:
                self.assertTrue(later_finished.wait(5))
            else:
                later_finished.set()
            return request_id
        calls = []
        self.assertEqual(legacy.collect_grouped_responses(fetch, [11, 12], calls.append), [11, 12])
        self.assertEqual(calls, [])

    def test_grouped_stages_reuse_one_pinned_tokenizer_without_changing_tokens(self):
        from unittest.mock import Mock
        runner = object.__new__(K31FullRunner)
        runner.args = SimpleNamespace(long_prefix_checkpoint='/checkpoint')
        factory = Mock()
        tokenizer = factory.from_pretrained.return_value
        tokenizer.encode.return_value = [1, 2, 3]
        first = runner.group_tokenizer(factory)
        second = runner.group_tokenizer(factory)
        self.assertIs(first, second)
        self.assertEqual(first.encode('input'), second.encode('input'))
        factory.from_pretrained.assert_called_once_with(
            '/checkpoint', trust_remote_code=True, local_files_only=True)
        runner.args.long_prefix_checkpoint = '/different'
        with self.assertRaisesRegex(legacy.SmokeFailure, 'checkpoint changed'):
            runner.group_tokenizer(factory)

    def test_cancel_observation_requires_actual_unfinished_host_load(self):
        def event(name, rid, stamp):
            return '[K3_SMOKE_EVENT] ' + json.dumps(dict(
                event=name, request_id=rid, time_ns=stamp)) + '\n'
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'engine.log'
            for data, expected in (
                ('', False),
                (event('host_cache_load_started', 8, 10), False),
                (event('host_cache_load_started', 7, 10), True),
                (event('host_cache_load_started', 7, 10)
                 + event('host_cache_load_done', 7, 11), False),
            ):
                path.write_text(data)
                with self.subTest(data=data):
                    result = pending_host_load(path, 7, timeout=0)
                    self.assertEqual(result['pending_observed'], expected)

    def test_cancel_observation_handles_async_partial_log_line(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'engine.log'
            line = '[K3_SMOKE_EVENT] ' + json.dumps(dict(
                event='host_cache_load_started', request_id=7, time_ns=10)) + '\n'
            path.write_text(line[:30])
            def append(_):
                with path.open('a') as log:
                    log.write(line[30:])
            result = pending_host_load(path, 7, sleep=append)
            self.assertTrue(result['pending_observed'])

    def test_three_cancel_seeds_keep_reference_count_and_real_64k_load(self):
        runner = object.__new__(K31FullRunner)
        runner.args = SimpleNamespace(namespace='cancel-fixture')
        lengths, cases = [], []
        def fit(head, tail, length, **kwargs):
            lengths.append(length)
            return head+tail, []
        runner.fit_prompt = fit
        runner._orthogonal_answer = lambda name, prompt, value, reuse: legacy.Case(
            name, prompt, '.', reuse)
        runner._required_stage = lambda name, group: cases.extend(group)
        self.assertEqual(runner._seed_cancel_candidates(), cases)
        self.assertEqual(lengths, [65536]*3)

    def test_http_validation_keeps_shared_verified_input_length(self):
        runner = object.__new__(K31FullRunner)
        case = legacy.Case('http-replay', 'question', '.', 'miss')
        ids = [1, 2, 3]
        response = {'debug_info': {'input_ids': ids}}
        runner.tokenize = lambda prompt: ids
        with patch.object(legacy.Runner, 'validate', return_value={'input_len': 3, 'content': '1'}) as shared:
            row = runner.validate(case, response, 0.1, 128, verified_input_len=3)
        self.assertEqual(shared.call_args.kwargs, {'verified_input_len': 3})
        self.assertEqual(row['input_tokens'], ids)

    def test_pressure_evicts_witness_and_cancel_seeds_from_95_page_device_pool(self):
        runner = object.__new__(K31FullRunner)
        runner.args = SimpleNamespace(namespace='pressure-fixture', block_size=1024,
                                      reuse_unit_tokens=8192)
        runner._memory_candidates = [0, 1, 2]
        runner._cancel_host_candidates = [0, 1, 2]
        runner._orthogonal_device_blocks = lambda: 95
        runner.fit_prompt = lambda head, tail, length: (str(length), [])
        probes = []
        runner._required_stage = lambda name, cases: probes.extend(cases)
        runner._orthogonal_bounded_cache_pressure()
        self.assertEqual(len(probes), 15)
        device = ['memory0', 'memory1', 'memory2', 'witness', 'cancel0', 'cancel1', 'cancel2']
        for index, case in enumerate(probes):
            tokens = int(case.prompt)
            self.assertLessEqual(tokens, 65536)
            device.extend([f'pressure{index}'] * ((tokens - 1) // 8192))
            device = device[-95:]
        self.assertNotIn('witness', device)
        for index in range(3):
            self.assertNotIn(f'cancel{index}', device)

    def test_pressure_scales_to_a_larger_prefill_pool(self):
        runner = object.__new__(K31FullRunner)
        runner.args = SimpleNamespace(namespace='large-pool', block_size=1024,
                                      reuse_unit_tokens=8192)
        runner._memory_candidates = [0, 1, 2]
        runner._cancel_host_candidates = [0, 1, 2]
        runner._orthogonal_device_blocks = lambda: 216
        runner.fit_prompt = lambda head, tail, length: (str(length), [])
        probes = []
        runner._required_stage = lambda name, cases: probes.extend(cases)
        runner._orthogonal_bounded_cache_pressure()
        self.assertEqual(len(probes), 31)
        self.assertTrue(all(case.preparation_only for case in probes))
        self.assertTrue(all(int(case.prompt) <= 65536 for case in probes))
        self.assertGreater(sum((int(case.prompt) - 1) // 8192 for case in probes), 216)

    def test_prefill_admission_checks_linear_pool_independently(self):
        runner = object.__new__(K31FullRunner)
        runner.args = SimpleNamespace(decode_batch_profile='smoke', reuse_unit_tokens=8192)
        pools = {'full': 104, 'linear': 81}
        runner._orthogonal_device_blocks = lambda tag='full': pools[tag]
        with self.assertRaisesRegex(legacy.SmokeFailure, 'LINEAR 81/82, FULL 104/82'):
            runner.validate_prefill_capacity()
        pools['linear'] = 104
        runner.validate_prefill_capacity()
        runner.args.decode_batch_profile = 'capacity'
        pools['full'] = 64
        runner.validate_prefill_capacity()

    def test_default_decode_inventory_uses_short_63_and_both_owners(self):
        runner = object.__new__(K31FullRunner)
        runner.args = SimpleNamespace(decode_batch_profile='smoke')
        runner.decode_role_addrs = ['owner0', 'owner1']
        runner.record_case = lambda name, owner, **kw: legacy.Case(
            name, 'question', '.', 'miss', decode_owner_rank=owner)
        stages = []
        runner._required_stage = lambda name, cases, **kw: stages.append((name, cases, kw))
        runner._orthogonal_decode_batches()
        self.assertEqual([len(cases) for _, cases, _ in stages], [1, 4, 8, 63])
        self.assertEqual(sum(len(cases) for _, cases, _ in stages), 76)
        cases63 = stages[-1][1]
        self.assertEqual([sum(c.decode_owner_rank == owner for c in cases63)
                          for owner in range(2)], [32, 31])
        self.assertTrue(all(c.require_mtp_draft for _, cases, _ in stages for c in cases))
        self.assertTrue(all(c.max_tokens == 64 for _, cases, _ in stages for c in cases))
        self.assertTrue(all(options['grouped_pd'] for _, cases, options in stages[1:]))
        self.assertEqual(runner.max_grouped_prefill_requests(), 63)
        shared = object.__new__(legacy.Runner)
        self.assertEqual(shared.max_grouped_prefill_requests(), 16)

    def test_large_single_owner_cases_can_overlap_after_rdma_load(self):
        runner = object.__new__(K31FullRunner)
        runner.args = SimpleNamespace(decode_batch_profile='smoke', namespace='rdma', max_tokens=128)
        runner.decode_role_addrs = ['owner0']
        stages = []
        runner._required_stage = lambda name, cases, **kw: stages.append(cases)
        runner._orthogonal_decode_batches()
        self.assertTrue(all(len(c.expected_json['value'].split()) == 1
                            for cases in stages[:-1] for c in cases))
        self.assertEqual(len(stages[-1]), 63)
        self.assertTrue(all(len(c.expected_json['value'].split()) == 8
                            and c.max_tokens == 160 for c in stages[-1]))

    def test_capacity_suite_only_runs_64_and_cannot_claim_complete_smoke(self):
        runner = object.__new__(K31FullRunner)
        runner.args = SimpleNamespace(decode_batch_profile='capacity')
        runner.decode_role_addrs = ['owner0', 'owner1']
        runner.record_case = lambda name, owner, **kw: legacy.Case(
            name, 'question', '.', 'miss', decode_owner_rank=owner)
        stages = []
        runner._required_stage = lambda name, cases, **kw: stages.append((name, cases))
        runner.run_main_text = lambda: self.fail('capacity must not rerun the service suite')
        runner.run_k3_full()
        self.assertEqual([len(cases) for _, cases in stages], [64])
        self.assertEqual([sum(c.decode_owner_rank == owner for c in stages[0][1])
                          for owner in range(2)], [32, 32])
        self.assertEqual(request_budget('smoke'), dict(total=304, formal=267, preparation=37))
        self.assertEqual(request_budget('capacity'), dict(total=64, formal=64, preparation=0))
        self.assertEqual(decode_batch_sizes('capacity'), (64,))
        with self.assertRaises(ValueError):
            request_budget('unknown')

    def test_structural_batch_answers_are_short_and_request_specific(self):
        runner = object.__new__(K31FullRunner)
        runner.args = SimpleNamespace(namespace='short-batch', max_tokens=128)
        first = runner.record_case('batch-shape-0-0', 0, words=8)
        other = runner.record_case('batch-shape-0-1', 0, words=8)
        self.assertEqual(first.max_tokens, 64)
        self.assertEqual(len(first.expected_json['value'].split()), 1)
        self.assertNotEqual(first.expected_json, other.expected_json)
        rolling = runner.record_case('rolling-4', 0, words=5)
        self.assertEqual(len(rolling.expected_json['value'].split()), 5)

    def test_case_retains_required_draft_execution(self):
        case = legacy.Case('boundary-plus', 'question', '.', 'miss',
                           require_mtp=True, require_mtp_draft=True)
        runner = object.__new__(K31FullRunner)
        prepared = runner.prepare_case(case)
        self.assertTrue(prepared.require_mtp)
        self.assertTrue(prepared.require_mtp_draft)
        config = SimpleNamespace(force_sp_accept=True)
        self.assertFalse(runner.prepare_group_config(config, prepared).force_sp_accept)
        payload = runner.prepare_http_payload({}, prepared)
        self.assertFalse(payload['extra_configs']['force_sp_accept'])

    def test_missing_requests_cannot_produce_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'accuracy.json'
            runner = object.__new__(K31FullRunner)
            runner.args = SimpleNamespace(output=path, source_commit='a'*40,
                geometry_value={}, synthetic_draft={'trained':False})
            runner.records, runner.submitted_requests = [], 0
            def save(passed, error):
                path.write_text(json.dumps({'passed':passed,'error':error}))
            with patch.object(legacy.Runner, 'save', side_effect=save):
                with self.assertRaisesRegex(legacy.SmokeFailure, 'count mismatch'):
                    runner.save(True)
            result = json.loads(path.read_text())
            self.assertFalse(result['passed'])
            self.assertFalse(result['complete_smoke_passed'])
            self.assertEqual(result['request_budget']['total'],304)
            self.assertEqual(result['legacy_reference_request_budget']['total'],368)

    def test_entry_requires_explicit_untrained_fixture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            geometry = root/'geometry.json'
            geometry.write_text(json.dumps(dict(runtime_verified=True,runtime_layers=93,
                page_size=1024,kernel_page_size=128,source_shards=8,
                checkpoint_span=8192,cache_key_stride=8192)))
            draft = root/'draft';draft.mkdir()
            (draft/'config.json').write_text(json.dumps(dict(model_type='kimi_k31',
                text_config={'num_nextn_predict_layers':1},rtp_llm_fixture=dict(
                    kind='synthetic_dsa_mtp',synthetic=True,native=False,trained=False))))
            args = parse_args(['--base-url','http://127.0.0.1:1',
                '--decode-health-url','http://127.0.0.1:2/health','--namespace','full',
                '--source-commit','a'*40,'--output',str(root/'result/accuracy.json'),
                '--geometry',str(geometry),'--prefill-engine-log',str(root/'engine.log'),
                '--prefill-rpc-runfiles',str(root),'--checkpoint',str(root/'target'),
                '--synthetic-draft-checkpoint',str(draft),'--prefill-grpc-port','12345',
                '--decode-role-addr','192.0.2.2:22100:22101'])
            self.assertTrue(args.require_mtp)
            self.assertEqual(args.runtime_layers,93)
            self.assertEqual(args.synthetic_draft['kind'],'synthetic_untrained')


    def test_native_draft_entry_preserves_the_existing_case_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            geometry = root/'geometry.json'
            geometry.write_text(json.dumps(dict(runtime_verified=True,runtime_layers=93,
                page_size=1024,kernel_page_size=128,source_shards=8,
                checkpoint_span=8192,cache_key_stride=8192)))
            target = root/'target';target.mkdir()
            (target/'config.json').write_text(json.dumps({'model_type':'kimi_k31'}))
            contract = {'kind':'native_k3_mla_mtp','synthetic':False,'trained':True}
            with patch('k31_full_smoke.native_mla_draft', return_value=contract) as validate:
                args = parse_args(['--base-url','http://127.0.0.1:1',
                    '--decode-health-url','http://127.0.0.1:2/health','--namespace','full',
                    '--source-commit','a'*40,'--output',str(root/'result/accuracy.json'),
                    '--geometry',str(geometry),'--prefill-engine-log',str(root/'engine.log'),
                    '--prefill-rpc-runfiles',str(root),'--checkpoint',str(target),
                    '--draft-checkpoint',str(root/'draft'),'--prefill-grpc-port','12345',
                    '--decode-role-addr','192.0.2.2:22100:22101'])
            validate.assert_called_once_with(root/'draft', {'model_type':'kimi_k31'})
            self.assertIs(args.draft_contract, contract)
            self.assertIsNone(args.synthetic_draft)
            self.assertEqual(args.run_kind, 'real93-native-mla-mtp-smoke')
            self.assertEqual(args.suite, 'main-text-64k-capped')
            self.assertEqual(args.orthogonal_phases, ('cache','cancel','page','chunk','decode'))
            self.assertTrue(args.require_mtp)
            self.assertEqual(request_budget(args.decode_batch_profile),
                             dict(total=304, formal=267, preparation=37))


if __name__ == '__main__':
    unittest.main()
