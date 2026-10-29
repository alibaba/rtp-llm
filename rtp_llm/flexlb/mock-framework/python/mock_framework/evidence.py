"""Streaming reader for Client-owned facts; an unsealed stream is never complete evidence."""
import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import struct

from google.protobuf.message import DecodeError
import zstandard

from . import client_evidence_pb2 as pb
from .execution.transport import ExecutionError

MAX_FRAME = 2 * 1024 * 1024
OUTCOMES = {'success', 'error', 'timeout', 'cancelled'}


@dataclass(frozen=True)
class Attempt:
    number: int
    master: str
    started_at_ms: int
    finished_at_ms: int
    grpc_code: str
    error_code: int
    message: str
    response_success: bool | None


@dataclass(frozen=True)
class RequestRecord:
    client: str
    playback: int
    request_id: int
    sent_at_ms: int
    finished_at_ms: int
    ttft_nanos: int | None
    elapsed_nanos: int
    output_tokens: int
    outcome: str
    grpc_code: str
    error_code: int
    message: str
    attempts: list[Attempt]


def _batch(encoded):
    content_size = zstandard.frame_content_size(encoded)
    if content_size not in (zstandard.CONTENTSIZE_UNKNOWN, zstandard.CONTENTSIZE_ERROR) and content_size > MAX_FRAME:
        raise ValueError('Oversized decompressed evidence frame')
    raw = zstandard.ZstdDecompressor().decompress(encoded, max_output_size=MAX_FRAME, allow_extra_data=False)
    size = 0
    for i, value in enumerate(raw[:5]):
        size |= (value & 127) << (7 * i)
        if value < 128:
            if size != len(raw) - i - 1:
                raise ValueError('Invalid length-delimited EventBatch')
            return pb.EventBatch.FromString(raw[i + 1:])
    raise ValueError('Invalid EventBatch length')


def read_requests(directory: Path, cancelled=None):
    """Yield completed requests with bounded memory; consume through EOF to validate the seal."""
    path = directory / 'requests.binpb.zst'
    active, outcomes = {}, Counter()
    frame = sequence = issued = finished = 0
    sealed = False
    identity = None
    try:
        plan = json.loads((directory / 'plan.json').read_text())
        with path.open('rb') as stream:
            while header := stream.read(4):
                if cancelled is not None and cancelled.is_set():
                    raise ExecutionError('Client evidence scan cancelled')
                if sealed or len(header) != 4:
                    raise ValueError('Trailing or truncated evidence frame')
                size, = struct.unpack('>I', header)
                if not 0 < size <= MAX_FRAME:
                    raise ValueError('Invalid compressed frame size')
                encoded = stream.read(size)
                if len(encoded) != size:
                    raise ValueError('Truncated evidence frame')
                batch = _batch(encoded)
                frame += 1
                current = (batch.version, batch.client, batch.playback, batch.first_request_id,
                           batch.first_index, batch.stride, batch.planned, batch.concurrency)
                if identity is None:
                    identity = current
                    expected = (1, directory.parent.parent.name, plan['sequence'], plan['first_request_id'],
                                plan['first_index'], plan['stride'], plan['count'], plan['concurrency'])
                    if current != expected or not (batch.first_request_id > 0 and batch.stride > 0
                            and batch.first_index < batch.stride and batch.planned > 0 and batch.concurrency > 0):
                        raise ValueError('Evidence identity does not match playback plan')
                if current != identity or batch.frame != frame or not (batch.events or batch.HasField('seal')):
                    raise ValueError('Evidence frame identity/sequence mismatch')
                for event in batch.events:
                    sequence += 1
                    if event.sequence != sequence or event.epoch_ms <= 0 or event.elapsed_nanos < 0:
                        raise ValueError('Invalid event sequence or timestamp')
                    request_id = event.request_id
                    if event.kind == pb.Event.ISSUED:
                        expected_id = batch.first_request_id + batch.first_index + issued * batch.stride
                        if request_id != expected_id or issued >= batch.planned or len(active) >= batch.concurrency:
                            raise ValueError('Duplicate, out-of-order or out-of-plan request')
                        issued += 1
                        active[request_id] = {'sent': event.epoch_ms, 'attempts': [], 'pending': None, 'first': None,
                                              'elapsed': event.elapsed_nanos}
                        continue
                    state = active.get(request_id)
                    if state is None or event.elapsed_nanos < state['elapsed']:
                        raise ValueError('Event without active request or decreasing elapsed time')
                    state['elapsed'] = event.elapsed_nanos
                    if event.kind == pb.Event.SCHEDULE_START:
                        if state['pending'] or event.attempt != len(state['attempts']) + 1 or event.attempt > 2 or not event.master:
                            raise ValueError('Invalid Schedule attempt')
                        state['pending'] = event
                    elif event.kind == pb.Event.SCHEDULE_RESULT:
                        start = state['pending']
                        if start is None or event.attempt != start.attempt or event.master != start.master or not event.grpc_code:
                            raise ValueError('Schedule result without matching attempt')
                        state['attempts'].append(Attempt(event.attempt, event.master, start.epoch_ms, event.epoch_ms,
                                                         event.grpc_code, event.error_code, event.message,
                                                         event.response_success if event.HasField('response_success') else None))
                        state['pending'] = None
                    elif event.kind == pb.Event.FIRST_TOKEN:
                        if state['first'] is not None or state['pending'] or not state['attempts']:
                            raise ValueError('Invalid first token event')
                        state['first'] = event.elapsed_nanos
                    elif event.kind == pb.Event.TERMINAL:
                        if state['pending'] or event.outcome not in OUTCOMES or not event.grpc_code:
                            raise ValueError('Invalid terminal event')
                        if event.HasField('ttft_nanos') != (state['first'] is not None) or (
                                event.HasField('ttft_nanos') and not 0 <= event.ttft_nanos <= event.elapsed_nanos):
                            raise ValueError('Inconsistent first token and terminal result')
                        if (event.output_tokens > 0) != (state['first'] is not None):
                            raise ValueError('Inconsistent output token count')
                        finished += 1
                        outcomes[event.outcome] += 1
                        del active[request_id]
                        yield RequestRecord(batch.client, batch.playback, request_id, state['sent'], event.epoch_ms,
                                            event.ttft_nanos if event.HasField('ttft_nanos') else None,
                                            event.elapsed_nanos, event.output_tokens, event.outcome, event.grpc_code,
                                            event.error_code, event.message, state['attempts'])
                    else:
                        raise ValueError('Unknown event kind')
                if batch.HasField('seal'):
                    if active or issued != finished or batch.seal.issued != issued or batch.seal.finished != finished:
                        raise ValueError('Evidence seal does not reconcile requests')
                    sealed = True
        if not sealed:
            raise ValueError('Missing evidence seal (playback still running, writer failed or JVM exited)')
        result_path = directory / 'result.json'
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if (not result['done'] or result['issued'] != issued or result['finished'] != finished
                    or result['unsent'] != plan['count'] - issued
                    or result['outcomes'] != {key: outcomes[key] for key in OUTCOMES}):
                raise ValueError('Evidence does not reconcile Client result')
    except (OSError, ValueError, KeyError, TypeError, DecodeError, zstandard.ZstdError) as error:
        raise ExecutionError(f'Invalid Client evidence at {path}: {error}') from error


def playback_directories(root, topology, targets=None, playback=None):
    clients = {node['rpc']: node for node in topology if node['role'] == 'client'}
    chosen = list(clients) if targets is None else targets
    if not chosen or len(set(chosen)) != len(chosen) or set(chosen) - clients.keys():
        raise ExecutionError('Expected nonempty, unique Client addresses')
    if playback is None:
        rounds = root / 'components' / 'client-0' / 'coordinator'
        available = [int(path.name) for path in rounds.iterdir() if path.is_dir() and path.name.isdecimal()] if rounds.exists() else []
        if not available:
            raise ExecutionError('No Client playback exists')
        playback = max(available)
    return [root / 'components' / clients[target]['id'] / 'playback' / str(playback) for target in chosen]


def verify_playback(root, topology, results, cancelled=None):
    for directory in playback_directories(root, topology):
        if not (directory / 'result.json').is_file():
            raise ExecutionError(f'Missing Client result at {directory}')
        outcome = Counter()
        count = 0
        for record in read_requests(directory, cancelled):
            count += 1
            outcome[record.outcome] += 1
        node = next(node for node in topology if node['id'] == directory.parent.parent.name)
        expected = results[node['control']]
        if count != expected['issued'] or expected['outcomes'] != {key: outcome[key] for key in OUTCOMES}:
            raise ExecutionError('Client control result and evidence disagree')


def main():
    parser = argparse.ArgumentParser(description='Stream Client request records as JSON; nonzero exit means incomplete/invalid evidence')
    parser.add_argument('output', type=Path, help='Case output directory')
    parser.add_argument('--playback', type=int)
    parser.add_argument('--target', action='append', dest='targets', help='Client rpc address; defaults to all')
    parser.add_argument('--request-id', type=int, action='append')
    parser.add_argument('--outcome', choices=sorted(OUTCOMES))
    args = parser.parse_args()
    try:
        topology = json.loads((args.output / 'topology-history.json').read_text())
        for directory in playback_directories(args.output, topology, args.targets, args.playback):
            for record in read_requests(directory):
                if (args.request_id is None or record.request_id in args.request_id) and (args.outcome is None or record.outcome == args.outcome):
                    print(json.dumps(asdict(record), ensure_ascii=False))
    except (ExecutionError, OSError, ValueError) as error:
        parser.exit(2, f'{error}\nPreviously emitted records are only a recovered prefix, not complete evidence.\n')


if __name__ == '__main__':
    main()
