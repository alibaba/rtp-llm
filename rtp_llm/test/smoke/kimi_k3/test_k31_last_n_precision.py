"""Independent arithmetic and corrupt-evidence checks for P/D logits."""
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

from k31_last_n_precision import audit_cases, compare, error_metrics, lock_thresholds, read_capture, validate


class LastNPrecisionTest(unittest.TestCase):
    def test_precision_is_bound_to_the_same_five_real_case_responses(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = self.read(self.fixture(directory, 'cold'))
            repeated = self.read(self.fixture(directory, 'repeat', delta=0.005))
            baseline = self.read(self.fixture(directory, 'unbounded', delta=0.02))
            contract = lock_thresholds(reference, repeated, baseline, Path(directory)/'contract.json')
            names = ('tier-device-probe-0', 'tier-memory-seed-1', 'tier-witness-seed',
                     'tier-mixed-0-memory', 'tier-memory-probe')
            for index, name in enumerate(names):
                roots = self.fixture(directory, f'case{index}', request_id=10+index, delta=0.01,
                                     capture_time=contract['created_ns']+1)
                shared = [Path(directory)/'all'/phase for phase in ('prefill', 'decode')]
                for src, dst in zip(roots, shared):
                    dst.mkdir(parents=True, exist_ok=True)
                    for path in src.iterdir():
                        (dst/path.name).write_bytes(path.read_bytes())
            ids = {name: [10+i] for i,name in enumerate(names)}
            result = dict(source_commit='a'*40, cases=[dict(name=name, input_len=2,
                input_tokens=[1, 2], output_ids=[[3, 4, 5, 6, 7]]) for name in names])
            self.assertEqual(set(audit_cases(result, ids, *shared, reference, contract)), set(names))
            result['cases'][3]['input_tokens'] = [1, 7]
            with self.assertRaisesRegex(ValueError, 'actual response'):
                audit_cases(result, ids, *shared, reference, contract)
            result['cases'][3]['input_tokens'] = [1, 2]
            ids[names[3]].append(99)
            with self.assertRaisesRegex(ValueError, 'one actual native'):
                audit_cases(result, ids, *shared, reference, contract)

    def fixture(self, directory, name, request_id=10, delta=0.0, capture_time=1):
        roots = [Path(directory)/name/phase for phase in ('prefill', 'decode')]
        for phase, root in zip(('prefill', 'decode'), roots):
            root.mkdir(parents=True)
            (root/f'{request_id}.input.i32').write_bytes(struct.pack('<2i', 1, 2))
            frames = [(0, [2], [3], [0])] if phase == 'prefill' else [
                (1, [3, 4], [4, 5], [1]), (3, [5], [6], [3])]
            for begin, inputs, tokens, positions in frames:
                stem = f'{request_id}.{phase}.{begin}'
                values = []
                for position in positions:
                    row = [delta, *([0.0]*7)]
                    row[position+3] = 10.0
                    values.extend(row)
                (root/f'{stem}.f32').write_bytes(struct.pack('<'+'f'*len(values), *values))
                metadata = dict(schema=1, source_commit='a'*40, request_id=request_id, phase=phase,
                    begin_output_position=begin, prompt_tokens=2, input_file=f'{request_id}.input.i32',
                    row_input_tokens=inputs, accepted_tokens=tokens, positions=positions, vocab_size=8,
                    dtype='float32_le', data_file=f'{stem}.f32', time_ns=capture_time)
                (root/f'{stem}.json').write_text(json.dumps(metadata))
        return roots

    def read(self, roots, request_id=10):
        capture = read_capture(*roots, request_id, 'a'*40)
        capture['greedy_output_ids'] = [3, 4, 5, 6, 7]
        return capture

    def test_later_greedy_difference_cannot_hide_behind_matching_fixed_positions(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = self.read(self.fixture(directory, 'reference'))
            actual = self.read(self.fixture(directory, 'actual'))
            actual['greedy_output_ids'][-1] = 0
            with self.assertRaisesRegex(ValueError, 'greedy_output_ids'):
                compare(reference, actual)
            actual.pop('greedy_output_ids')
            with self.assertRaisesRegex(ValueError, 'full greedy output'):
                compare(reference, actual)

    def test_independent_maximum_absolute_and_relative_l2(self):
        metrics = error_metrics([1.0, 2.0], [1.0, 3.0])
        self.assertEqual(metrics['max_abs'], 1.0)
        self.assertAlmostEqual(metrics['relative_l2'], 1/math.sqrt(5))
        self.assertEqual(error_metrics([0], [0])['relative_l2'], 0)
        self.assertEqual(error_metrics([0], [1])['relative_l2'], math.inf)
        with self.assertRaises(ValueError):
            error_metrics([0], [math.nan])

    def test_actual_accepted_history_and_fixed_p_d_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = self.read(self.fixture(directory, 'cold'))
            self.assertEqual(capture['output_ids'], [3, 4, 5, 6])
            self.assertEqual(sorted(capture['logits']), [0, 1, 3])
            self.assertEqual(capture['logits'][3][6], 10.0)

    def test_thresholds_are_locked_before_candidate_and_recomputed_from_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = self.read(self.fixture(directory, 'cold'))
            repeated = self.read(self.fixture(directory, 'repeat', delta=0.005))
            baseline = self.read(self.fixture(directory, 'unbounded', delta=0.02))
            destination = Path(directory)/'precision-contract.json'
            contract = lock_thresholds(reference, repeated, baseline, destination)
            contract = json.loads(destination.read_text())
            actual = self.read(self.fixture(directory, 'candidate', delta=0.015,
                                            capture_time=contract['created_ns']+1))
            self.assertTrue(validate(reference, actual, contract)['passed'])
            bad = self.read(self.fixture(directory, 'bad', delta=0.04,
                                         capture_time=contract['created_ns']+1))
            with self.assertRaisesRegex(ValueError, 'exceeds locked bound'):
                validate(reference, bad, contract)
            actual['first_capture_ns'] = contract['created_ns']
            with self.assertRaisesRegex(ValueError, 'before the candidate'):
                validate(reference, actual, contract)
            with self.assertRaises(FileExistsError):
                lock_thresholds(reference, repeated, baseline, destination)

    def test_corrupt_bytes_or_unaccepted_history_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = self.fixture(directory, 'short')
            (roots[1]/'10.decode.1.f32').write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError, 'byte count'):
                self.read(roots)
            roots = self.fixture(directory, 'history')
            path = roots[1]/'10.decode.1.json'
            row = json.loads(path.read_text()); row['row_input_tokens'][1] = 99
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, 'accepted tokens'):
                self.read(roots)
            roots = self.fixture(directory, 'detached')
            path = roots[1]/'10.decode.3.json'
            row = json.loads(path.read_text()); row['row_input_tokens'][0] = 99
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, 'detached'):
                self.read(roots)

    def test_missing_decode_position_or_different_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = self.fixture(directory, 'missing')
            (roots[1]/'10.decode.3.json').unlink()
            with self.assertRaisesRegex(ValueError, 'coverage'):
                self.read(roots)
            roots = self.fixture(directory, 'sha')
            path = roots[1]/'10.decode.1.json'
            row = json.loads(path.read_text()); row['source_commit'] = 'b'*40
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, 'provenance'):
                self.read(roots)
            cold = self.read(self.fixture(directory, 'cold'))
            actual = self.read(self.fixture(directory, 'other'))
            actual['input_sha256'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'input_sha256'):
                compare(cold, actual)


if __name__ == '__main__':
    unittest.main()
