"""Exercise release HTTP frontend -> Master -> Mock P -> ViT -> Mock D.

Standard library only; pass the dedicated frontend and Mock control addresses.
"""
import argparse
import base64
import json
from pathlib import Path
import struct
import time
import urllib.error
import urllib.request
import zlib


def png(width, height, color):
    def chunk(name, value):
        return struct.pack('>I', len(value)) + name + value + struct.pack('>I', zlib.crc32(name + value))
    raw = b''.join(b'\0' + bytes(color) * width for _ in range(height))
    data = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
    data += chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b'')
    return 'data:image/png;base64,' + base64.b64encode(data).decode()


def http(url, body=None):
    request = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        text = error.read(8192).decode(errors='replace')
        try:
            text = json.loads(text)
        except ValueError:
            pass
        return error.code, text


def run(frontend, control, expected_workers=200):
    records = []
    small, large = png(224, 224, (13, 70, 90)), png(1024, 768, (81, 12, 31))
    cases = [('text', []), ('small', [small]), ('large', [large]),
             ('multi', [small, large]), ('small-repeat', [small]), ('single-output-token', [small])]
    for name, images in cases:
        content = [{'type': 'text', 'text': 'Describe these inputs briefly.'}]
        content += [{'type': 'image_url', 'image_url': {'url': image}} for image in images]
        body = {'model': 'test', 'messages': [{'role': 'user', 'content': content}],
                'max_tokens': 1 if name == 'single-output-token' else 4, 'stream': False, 'temperature': 0}
        status, rendered = http(frontend + '/v1/chat/render', body)
        assert status == 200 and isinstance(rendered.get('input_ids'), list), (name, status, rendered)
        expected = len(rendered['input_ids'])
        status, response = http(frontend + '/v1/chat/completions', body)
        record = {'name': name, 'status': status, 'rendered_tokens': expected,
                  'usage': response.get('usage') if isinstance(response, dict) else None}
        records.append(record)
        assert status == 200, (name, status, response)
        assert response['usage']['prompt_tokens'] == expected, record
        assert response['usage']['completion_tokens'] == body['max_tokens'], record
    bad = {'model': 'test', 'messages': [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,bm90LWFuLWltYWdl'}}]}],
        'max_tokens': 4, 'stream': False}
    status, _ = http(frontend + '/v1/chat/completions', bad)
    assert status >= 400, status
    records.append({'name': 'invalid-image', 'status': status})
    status, response = http(frontend + '/v1/chat/completions', {
        'model': 'test', 'messages': [{'role': 'user', 'content': 'Recovery check'}],
        'max_tokens': 4, 'stream': False})
    assert status == 200, (status, response)
    recovery_status = status
    stream_body = {'model': 'test', 'messages': [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'Describe these inputs briefly.'},
        {'type': 'image_url', 'image_url': {'url': small}}]}], 'max_tokens': 4, 'stream': True}
    request = urllib.request.Request(frontend + '/v1/chat/completions',
        data=json.dumps(stream_body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=90) as response:
        wire = response.read().decode()
        assert response.status == 200
    events = [json.loads(line[6:]) for line in wire.splitlines()
              if line.startswith('data: ') and line[6:] != '[DONE]']
    assert '[DONE]' in wire and events
    usage = [event['usage'] for event in events if event.get('usage')][-1]
    small_tokens = next(case['rendered_tokens'] for case in records if case['name'] == 'small')
    assert usage['prompt_tokens'] == small_tokens and usage['completion_tokens'] == 4, usage
    addresses = next(event['aux_info']['role_addrs'] for event in reversed(events)
                     if event.get('aux_info', {}).get('role_addrs'))
    assert {'PREFILL', 'DECODE', 'VIT'} <= {address['role'] for address in addresses}
    records.append({'name': 'stream-image', 'status': 200, 'usage': usage,
                    'events': len(events), 'selected_roles': addresses})
    deadline = time.monotonic() + 30
    while True:
        status, snapshot = http(control + '/snapshot')
        assert status == 200 and len(snapshot['engines']) == expected_workers
        totals = {key: sum(engine[key] for engine in snapshot['engines'])
                  for key in ('running', 'waiting', 'held_blocks', 'referenced_blocks')}
        if not any(totals.values()):
            break
        assert time.monotonic() < deadline, totals
        time.sleep(.2)
    return {'status': 'PASS', 'cases': records, 'recovery_status': recovery_status, 'drain': totals}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frontend', required=True)
    parser.add_argument('--control', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--expected-workers', type=int, default=200)
    args = parser.parse_args()
    result = run(args.frontend.rstrip('/'), args.control.rstrip('/'), args.expected_workers)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
