"""Compare actual accepted target logits at fixed P/D token positions."""
from array import array
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time


POSITIONS = (0, 1, 3)


def _integers(values):
    return isinstance(values, list) and all(type(value) is int for value in values)


def _binary(path, kind, count):
    content = path.read_bytes()
    values = array(kind)
    if values.itemsize != 4 or len(content) != count*4:
        raise ValueError(f"{path.name}: tensor byte count differs from its shape")
    values.frombytes(content)
    if sys.byteorder != 'little':
        values.byteswap()
    return values, hashlib.sha256(content).hexdigest()


def _sibling(root, name):
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError('capture tensor must be a sibling artifact')
    path = root/name
    if path.is_symlink():
        raise ValueError('capture artifact cannot be a symlink')
    return path


def read_capture(prefill_dir, decode_dir, request_id, source_commit):
    if type(request_id) is not int or not re.fullmatch('[0-9a-f]{40}', source_commit):
        raise ValueError('capture needs a numeric request identity and exact source SHA')
    prompt, logits, accepted, frames, hashes, timestamps = None, {}, {}, [], {}, []
    for phase, directory in (('prefill', prefill_dir), ('decode', decode_dir)):
        root = Path(directory)
        paths = sorted(root.glob(f'{request_id}.{phase}.*.json'))
        if not paths:
            raise ValueError(f'{phase}: actual target logits capture is missing')
        for path in paths:
            raw = path.read_bytes()
            row = json.loads(raw)
            begin = row.get('begin_output_position')
            if (row.get('schema') != 1 or row.get('source_commit') != source_commit
                    or row.get('request_id') != request_id or row.get('phase') != phase
                    or row.get('dtype') != 'float32_le' or type(begin) is not int
                    or (begin != 0 and phase == 'prefill') or (begin < 1 and phase == 'decode')
                    or type(row.get('time_ns')) is not int or row['time_ns'] <= 0
                    or type(row.get('prompt_tokens')) is not int or row['prompt_tokens'] <= 0
                    or type(row.get('vocab_size')) is not int or row['vocab_size'] <= 0):
                raise ValueError(f'{path.name}: capture provenance or geometry differs')
            inputs, outputs, positions = (row.get(key) for key in
                                          ('row_input_tokens', 'accepted_tokens', 'positions'))
            if (not all(_integers(value) for value in (inputs, outputs, positions))
                    or not outputs or len(inputs) != len(outputs)
                    or any(not 0 <= token < row['vocab_size'] for token in outputs)
                    or positions != [begin+i for i in range(len(outputs)) if begin+i in POSITIONS]
                    or any(inputs[i] != outputs[i-1] for i in range(1, len(outputs)))):
                raise ValueError(f'{path.name}: target row does not follow accepted tokens')
            input_path = _sibling(root, row['input_file'])
            data_path = _sibling(root, row['data_file'])
            if (input_path.name != f'{request_id}.input.i32'
                    or data_path.name != path.stem+'.f32'):
                raise ValueError('capture file identity differs from its request/frame')
            tokens, input_sha = _binary(input_path, 'i', row['prompt_tokens'])
            if prompt is not None and tokens != prompt:
                raise ValueError('Prefill and Decode input token IDs differ')
            prompt = tokens
            values, data_sha = _binary(data_path, 'f', len(positions)*row['vocab_size'])
            if any(not math.isfinite(value) for value in values):
                raise ValueError('non-finite target logits')
            for i, position in enumerate(positions):
                if position in logits:
                    raise ValueError('duplicated target logits position')
                logits[position] = values[i*row['vocab_size']:(i+1)*row['vocab_size']]
                actual_token = outputs[position-begin]
                if logits[position][actual_token] != max(logits[position]):
                    raise ValueError('captured output token is not a greedy target maximum')
            for i, token in enumerate(outputs):
                position = begin+i
                if position in accepted:
                    raise ValueError('overlapping accepted target positions')
                accepted[position] = token
            frames.append((begin, inputs))
            timestamps.append(row['time_ns'])
            hashes[f'{phase}/{path.name}'] = hashlib.sha256(raw).hexdigest()
            hashes[f'{phase}/{input_path.name}'] = input_sha
            hashes[f'{phase}/{data_path.name}'] = data_sha
    if set(logits) != set(POSITIONS) or not set(range(4)) <= set(accepted):
        raise ValueError('P last / D first / fixed later target logits coverage is incomplete')
    if len({len(value) for value in logits.values()}) != 1:
        raise ValueError('target vocabulary changed between captured positions')
    for begin, inputs in frames:
        expected = prompt[-1] if begin == 0 else accepted.get(begin-1)
        if inputs[0] != expected:
            raise ValueError('target verify frame is detached from the actual accepted history')
    return dict(source_commit=source_commit, input_sha256=input_sha,
                input_tokens=len(prompt), output_ids=[accepted[i] for i in range(4)],
                logits=logits, artifact_sha256=hashes,
                first_capture_ns=min(timestamps), last_capture_ns=max(timestamps))


def error_metrics(reference, actual):
    if len(reference) != len(actual) or not reference:
        raise ValueError('logits vectors have different or empty shapes')
    if any(not math.isfinite(value) for value in (*reference, *actual)):
        raise ValueError('non-finite logits cannot establish a precision threshold')
    square_error = math.fsum((float(a)-float(b))**2 for a, b in zip(reference, actual))
    square_reference = math.fsum(float(value)**2 for value in reference)
    relative = math.sqrt(square_error/square_reference) if square_reference else (0.0 if square_error == 0 else math.inf)
    return dict(max_abs=max(abs(float(a)-float(b)) for a, b in zip(reference, actual)), relative_l2=relative)


def compare(reference, actual):
    for capture in (reference, actual):
        if not _integers(capture.get('greedy_output_ids')) or len(capture['greedy_output_ids']) < 4:
            raise ValueError('full greedy output token IDs are missing')
    for field in ('source_commit', 'input_sha256', 'input_tokens', 'output_ids', 'greedy_output_ids'):
        if reference[field] != actual[field]:
            raise ValueError(f'precision captures differ in {field}')
    if set(reference['logits']) != set(POSITIONS) or set(actual['logits']) != set(POSITIONS):
        raise ValueError('fixed target positions are missing')
    return {position: error_metrics(reference['logits'][position], actual['logits'][position])
            for position in POSITIONS}


def lock_thresholds(reference, cold_repeat, unbounded_hit, destination, *, headroom=1.25,
                    prefill_chunk_tokens=None):
    if not math.isfinite(headroom) or not 1 <= headroom <= 2:
        raise ValueError('calibration headroom must be fixed in [1,2] before the candidate run')
    samples = [compare(reference, cold_repeat), compare(reference, unbounded_hit)]
    bounds = {position: {metric: max(sample[position][metric] for sample in samples)*headroom
                         for metric in ('max_abs', 'relative_l2')} for position in POSITIONS}
    if any(not math.isfinite(value) for row in bounds.values() for value in row.values()):
        raise ValueError('non-finite calibration threshold')
    created = time.time_ns()
    if max(capture['last_capture_ns'] for capture in (reference, cold_repeat, unbounded_hit)) > created:
        raise ValueError('calibration capture timestamp is in the future')
    contract = dict(schema=1, source_commit=reference['source_commit'], input_sha256=reference['input_sha256'],
                    input_tokens=reference['input_tokens'], output_ids=reference['output_ids'],
                    greedy_output_ids=reference['greedy_output_ids'],
                    created_ns=created, headroom=headroom, limits=bounds,
                    calibration=[capture['artifact_sha256'] for capture in
                                 (reference, cold_repeat, unbounded_hit)],
                    observed=samples)
    if prefill_chunk_tokens is not None:
        if prefill_chunk_tokens not in (2048, 65536):
            raise ValueError('calibration chunk capacity must be 2048 or 65536')
        contract['prefill_chunk_tokens'] = prefill_chunk_tokens
    with Path(destination).open('x') as output:
        json.dump(contract, output, indent=2)
        output.write('\n')
    return contract


def validate(reference, actual, contract):
    metrics = compare(reference, actual)
    if (contract.get('schema') != 1 or contract.get('source_commit') != reference['source_commit']
            or contract.get('input_sha256') != reference['input_sha256']
            or contract.get('input_tokens') != reference['input_tokens']
            or contract.get('output_ids') != reference['output_ids']
            or contract.get('greedy_output_ids') != reference['greedy_output_ids']
            or reference['artifact_sha256'] not in contract.get('calibration', [])
            or type(contract.get('created_ns')) is not int
            or contract['created_ns'] >= actual['first_capture_ns']):
        raise ValueError('precision contract was not locked against this reference before the candidate run')
    for position, row in metrics.items():
        limits = contract['limits'].get(str(position), contract['limits'].get(position))
        if limits is None:
            raise ValueError('precision contract lacks a fixed target position')
        for metric, value in row.items():
            bound = limits.get(metric)
            if type(bound) not in (int, float) or not math.isfinite(bound) or bound < 0 or value > bound:
                raise ValueError(f'position{position}: {metric}={value} exceeds locked bound {bound}')
    return dict(passed=True, input_sha256=actual['input_sha256'], output_ids=actual['output_ids'],
                greedy_output_ids=actual['greedy_output_ids'],
                metrics=metrics, capture_artifacts=actual['artifact_sha256'])


def read_descriptor(path):
    """Resolve a calibration capture without moving or rewriting its tensors."""
    path = Path(path)
    descriptor = json.loads(path.read_text())
    roots = [Path(descriptor[key]) for key in ('prefill_dir', 'decode_dir')]
    roots = [root if root.is_absolute() else path.parent/root for root in roots]
    capture = read_capture(*roots, descriptor['request_id'], descriptor['source_commit'])
    greedy = descriptor.get('greedy_output_ids')
    if not _integers(greedy) or greedy[:4] != capture['output_ids']:
        raise ValueError('reference descriptor lacks the matching full greedy output')
    capture['greedy_output_ids'] = greedy
    return capture


def audit_cases(result, request_ids, prefill_dir, decode_dir, reference, contract):
    names = ('tier-device-probe-0', 'tier-memory-seed-1', 'tier-witness-seed',
             'tier-mixed-0-memory', 'tier-memory-probe')
    rows = {row['name']: row for row in result['cases']}
    proof = {}
    for name in names:
        ids = request_ids.get(name, [])
        if len(ids) != 1 or name not in rows:
            raise ValueError(f'{name}: precision case needs one actual native request identity')
        actual = read_capture(prefill_dir, decode_dir, ids[0], result['source_commit'])
        check_response(actual, rows[name])
        proof[name] = dict(request_id=ids[0], **validate(reference, actual, contract))
    return proof


def check_response(capture, row):
    output_ids, actual_inputs = row.get('output_ids'), row.get('input_tokens')
    if not _integers(actual_inputs):
        raise ValueError(f"{row['name']}: actual response input token IDs are missing")
    wire_inputs = array('i', actual_inputs)
    if sys.byteorder != 'little':
        wire_inputs.byteswap()
    if (row['input_len'] != capture['input_tokens'] or not isinstance(output_ids, list)
            or hashlib.sha256(wire_inputs.tobytes()).hexdigest() != capture['input_sha256']
            or len(output_ids) != 1 or not _integers(output_ids[0])
            or output_ids[0][:4] != capture['output_ids']):
        raise ValueError(f"{row['name']}: captured logits do not belong to the actual response token IDs")
    capture['greedy_output_ids'] = list(output_ids[0])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    # All three descriptors must refer to measurements collected before the
    # candidate. validate() enforces that time boundary again during auditing.
    for key in ('reference', 'cold-repeat', 'unbounded-hit', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--headroom', type=float, default=1.25)
    args = parser.parse_args(argv)
    captures = [read_descriptor(path) for path in (args.reference, args.cold_repeat, args.unbounded_hit)]
    lock_thresholds(*captures, args.output, headroom=args.headroom)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
