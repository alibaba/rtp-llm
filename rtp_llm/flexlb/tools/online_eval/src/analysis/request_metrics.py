"""Executable request-window calculators; provenance follows the selected operation."""

import bisect
import math
import re
from dataclasses import dataclass
from typing import Callable

from analysis.statistics import percentile_nr
from input_contract import finite_number, mapping_fields


@dataclass(frozen=True)
class Calculator:
    evaluate: Callable
    unit: str
    fields: frozenset = frozenset()
    percentile: bool = False


def _value(row, field):
    if field == 'tpot_ms':
        count = _value(row, 'observed_output_tokens')
        if count <= 1:
            return None
        return (_value(row, 'total_ms') - _value(row, 'ttft_ms')) / (count - 1)
    value = row.get(field)
    if not finite_number(value) or value < 0:
        raise ValueError('request calculation requires finite nonnegative ' + field)
    return value


def _token_throughput(bucket, spec):
    return sum(_value(row, spec['token_field']) for row in bucket.rows) / bucket.duration_s


def _request_rate(bucket, spec):
    return len(bucket.rows) / bucket.duration_s


def _mean(bucket, spec):
    return (sum(_value(row, spec['field']) for row in bucket.rows) / len(bucket.rows)
            if bucket.rows else None)


def _quantile(bucket, spec):
    values = [_value(row, spec['field']) for row in bucket.rows]
    return percentile_nr([value for value in values if value is not None], spec['percentile'])


def _success_share(bucket, spec):
    return (sum(row['status'] == 'ok' for row in bucket.rows) / len(bucket.rows)
            if bucket.rows else None)


def _inflight(bucket, spec):
    return bucket.ledger.inflight(bucket.end_ms)


CALCULATORS = {
    'token_throughput': Calculator(_token_throughput, 'tokens/s', frozenset({'token_field'})),
    'request_rate': Calculator(_request_rate, 'requests/s'),
    'mean': Calculator(_mean, 'tokens', frozenset({'field'})),
    'quantile': Calculator(_quantile, 'ms', frozenset({'field'}), percentile=True),
    'success_share': Calculator(_success_share, 'ratio'),
    'inflight': Calculator(_inflight, 'requests'),
}


def describe_calculation(spec, *, windows):
    """Validate data-only arguments and derive the contract from a real calculator."""
    if not isinstance(spec, dict) or type(spec.get('calculator')) is not str:
        raise ValueError('calculation requires a registered calculator')
    calculator = CALCULATORS.get(spec['calculator'])
    if calculator is None:
        raise ValueError('unknown request calculator: ' + spec['calculator'])
    required = {'calculator', 'window', 'selection', 'bucket_s'} | set(calculator.fields)
    if calculator.percentile:
        required.add('percentile')
    mapping_fields(spec, required, 'calculation', required=required)
    window = spec['window']
    if type(window) is not str or not re.fullmatch(r'[a-z][a-z0-9_]*', window) or window not in windows:
        raise ValueError('unknown calculation window: ' + str(window))
    if not finite_number(spec['bucket_s']) or spec['bucket_s'] <= 0:
        raise ValueError('calculation.bucket_s must be finite and positive')
    selection = mapping_fields(spec['selection'], {'time_basis', 'status'},
                               'calculation.selection', required={'time_basis', 'status'})
    if (type(selection['time_basis']) is not str or selection['time_basis'] not in {'arrival', 'completion', 'lifetimes'}
            or type(selection['status']) is not str or selection['status'] not in {'all', 'ok', 'non_ok'}):
        raise ValueError('invalid request population selection')
    if spec['calculator'] == 'inflight':
        if selection != {'time_basis': 'lifetimes', 'status': 'all'}:
            raise ValueError('inflight requires all request lifetimes')
    elif selection['time_basis'] == 'lifetimes':
        raise ValueError('request buckets require arrival or completion time')
    if spec['calculator'] == 'success_share' and selection['status'] != 'all':
        raise ValueError('success_share requires all requests as denominator')
    if 'token_field' in spec and spec['token_field'] not in ('input_len', 'observed_output_tokens'):
        raise ValueError('unknown request token field')
    if 'field' in spec:
        fields = ('ttft_ms', 'total_ms', 'tpot_ms') if calculator.percentile else ('input_len', 'observed_output_tokens')
        if spec['field'] not in fields:
            raise ValueError('unknown request calculation field')
    if calculator.percentile and (not finite_number(spec['percentile']) or not 0 < spec['percentile'] <= 1):
        raise ValueError('calculation.percentile must be in (0,1]')
    return dict(source_type='client_journal', unit=calculator.unit, value_kind='gauge', labels=[],
                measurement=dict(method=spec['calculator'],
                    population=window + ':' + selection['time_basis'] + ':' + selection['status'],
                    accuracy='request_ledger', requires_request_identity=True))


@dataclass(frozen=True)
class RequestBucket:
    rows: list
    duration_s: float
    end_ms: float
    ledger: object


class RequestLedger:
    """Index complete request records once; selectors retain identity and exact time."""

    def __init__(self, records):
        self.records = list(records)
        seen = set()
        for row in self.records:
            identity = row.get('rid')
            if type(identity) is not str or not identity or identity in seen:
                raise ValueError('request ledger requires unique request identities')
            seen.add(identity)
            if (not finite_number(row.get('send_start_epoch_ms')) or row['send_start_epoch_ms'] <= 0
                    or not finite_number(row.get('total_ms')) or row['total_ms'] < 0
                    or type(row.get('status')) is not str or row['status'] in {'', 'scheduled', 'unknown'}):
                raise ValueError('request ledger requires issue time and full terminal status')
            if not finite_number(self.completion(row)):
                raise ValueError('request completion timestamp is non-finite')
        self.sends = sorted(row['send_start_epoch_ms'] for row in self.records)
        self.ends = sorted(self.completion(row) for row in self.records)
        self._indexes = {}

    @staticmethod
    def completion(row):
        return row['send_start_epoch_ms'] + row['total_ms']

    def inflight(self, timestamp_ms):
        return bisect.bisect_left(self.sends, timestamp_ms) - bisect.bisect_left(self.ends, timestamp_ms)

    def _selected(self, selection):
        key = (selection['time_basis'], selection['status'])
        if key not in self._indexes:
            status = selection['status']
            rows = [row for row in self.records if status == 'all'
                    or (row['status'] == 'ok') == (status == 'ok')]
            time = self.completion if key[0] == 'completion' else lambda row: row['send_start_epoch_ms']
            rows = sorted(rows, key=time)
            self._indexes[key] = rows, [time(row) for row in rows]
        return self._indexes[key]

    def series(self, spec, windows):
        describe_calculation(spec, windows=windows)
        lo, hi = windows[spec['window']]
        if not finite_number(lo) or not finite_number(hi) or lo >= hi:
            raise ValueError('request calculation requires increasing finite window bounds')
        width_ms = spec['bucket_s'] * 1000
        if not finite_number(width_ms) or not finite_number(hi - lo):
            raise ValueError('request calculation window or bucket is non-finite')
        rows, stamps = self._selected(spec['selection'])
        evaluate = CALCULATORS[spec['calculator']].evaluate
        points = []
        for i in range(math.ceil((hi - lo) / width_ms)):
            start, end = lo + i * width_ms, min(hi, lo + (i + 1) * width_ms)
            selected = rows[bisect.bisect_left(stamps, start):bisect.bisect_left(stamps, end)]
            bucket = RequestBucket(selected, (end - start) / 1000, end, self)
            points.append([start / 1000, evaluate(bucket, spec)])
        return points
