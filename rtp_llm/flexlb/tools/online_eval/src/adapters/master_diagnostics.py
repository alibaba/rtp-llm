"""Read-only Master diagnostics v2; incomplete coverage never means an empty fleet."""

import json
import math
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass

from adapters.debug_client import DebugUnavailable

MAX_BYTES = 1_048_576


def _integer(value):
    return type(value) is int and value >= 0


def _decimal(value):
    return isinstance(value, str) and value.isascii() and value.isdigit()


def validate_snapshot(payload):
    try:
        if type(payload['schema_version']) is not int or payload['schema_version'] != 2:
            raise ValueError('unsupported diagnostic schema')
        for key in ('instance_id', 'snapshot_id'):
            if not isinstance(payload[key], str) or not payload[key]:
                raise ValueError('missing identity')
        if payload['status'] not in ('ok', 'partial'):
            raise ValueError('unavailable capture')
        pages = payload['components']
        if not isinstance(pages, dict) or not pages:
            raise ValueError('missing components')
        for page in [payload, *pages.values()]:
            left, right = page['capture_started_at_ms'], page['capture_finished_at_ms']
            if not _integer(left) or not _integer(right) or right < left:
                raise ValueError('invalid capture clock')
        for name, page in pages.items():
            if name not in ('workers', 'scheduler') or page['status'] not in ('ok', 'partial', 'unavailable', 'busy'):
                raise ValueError('unsupported component')
            if page['status'] not in ('ok', 'partial'):
                continue
            capability = payload['capabilities'][name]
            if type(capability['version']) is not int or capability['version'] != 1 or not isinstance(capability['features'], list):
                raise ValueError('unsupported component version')
            if name == 'scheduler':
                if not all(_integer(page['data'][k]) for k in ('queued_requests', 'inflight_requests')):
                    raise ValueError('invalid scheduler counts')
                continue
            coverage, rows = page['coverage'], page['rows']
            if (coverage['scope'] != 'prefill_decode_endpoint_registry'
                    or type(coverage['complete']) is not bool or type(coverage['truncated']) is not bool
                    or not isinstance(rows, list) or not _integer(coverage['returned_count'])
                    or coverage['returned_count'] != len(rows)):
                raise ValueError('invalid coverage')
            identities = set()
            for row in rows:
                if (row['role'] not in ('prefill', 'decode') or not isinstance(row['address'], str) or not row['address']
                        or row['worker_id'] != row['role'] + ':' + row['address'] or row['worker_id'] in identities
                        or not _decimal(row['generation']) or int(row['generation']) <= 0):
                    raise ValueError('invalid or duplicate Worker identity')
                identities.add(row['worker_id'])
                if row['lifecycle_state'] not in ('active', 'retiring'):
                    raise ValueError('invalid lifecycle state')
                if row['last_reported_alive'] is not None and type(row['last_reported_alive']) is not bool:
                    raise ValueError('invalid health flag')
                for field in ('poll_response_age_ms', 'applied_observation_age_ms'):
                    value = row[field]
                    if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                        raise ValueError('invalid age')
                if row['applied_status_version'] is not None and not _decimal(row['applied_status_version']):
                    raise ValueError('invalid status version')
                if row['role'] == 'prefill':
                    if type(row['index_initialized']) is not bool:
                        raise ValueError('invalid cache state')
                    for field in ('received_cache_version', 'indexed_cache_version'):
                        if row[field] is not None and not _decimal(row[field]):
                            raise ValueError('invalid cache version')
                    if row['cached_key_count'] is not None and not _integer(row['cached_key_count']):
                        raise ValueError('invalid cache size')
        if payload['status'] == 'ok' and any(p['status'] != 'ok' for p in pages.values()):
            raise ValueError('inconsistent capture status')
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise DebugUnavailable(f'invalid diagnostic response: {error}') from error
    return payload


@dataclass(frozen=True)
class DiagnosticCapture:
    payload: dict
    started_mono: float
    finished_mono: float

    def component(self, name):
        page = self.payload['components'].get(name)
        if page is None or page['status'] != 'ok':
            raise DebugUnavailable(f'diagnostic component {name!r} unavailable')
        if name == 'workers' and (not page['coverage']['complete'] or page['coverage']['truncated']):
            raise DebugUnavailable('incomplete Worker directory')
        return page


class MasterDiagnosticsClient:
    def __init__(self, base_url, timeout_s=5.0):
        if timeout_s <= 0:
            raise ValueError('timeout must be positive')
        self.base_url, self.timeout_s = base_url.rstrip('/'), timeout_s

    def snapshot(self, *, include=('workers', 'scheduler'), limit=1024, budget_ms=100):
        if (not include or len(set(include)) != len(include) or not set(include) <= {'workers', 'scheduler'}
                or type(limit) is not int or not 0 < limit <= 2048
                or type(budget_ms) is not int or not 0 < budget_ms <= 250):
            raise ValueError('invalid diagnostic query')
        params = urllib.parse.urlencode(dict(include=','.join(include), limit=limit, budget_ms=budget_ms))
        started = time.monotonic()
        try:
            with urllib.request.urlopen(f'{self.base_url}/rtp_llm/debug/snapshot?{params}', timeout=self.timeout_s) as response:
                body = response.read(MAX_BYTES + 1)
            if len(body) > MAX_BYTES:
                raise DebugUnavailable('diagnostic response exceeds byte budget')
            payload = validate_snapshot(json.loads(body))
            if not set(include) <= payload['components'].keys():
                raise DebugUnavailable('requested component missing')
        except (OSError, ValueError) as error:
            raise DebugUnavailable(f'diagnostic capture failed: {error}') from error
        return DiagnosticCapture(payload, started, time.monotonic())
