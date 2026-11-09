"""Build test playback files with the production dataset writer."""
import json
import subprocess
import sys

from . import FRAMEWORK


def write_inputs(path, requests, *, media=()):
    source = path.with_suffix('.source.jsonl')
    count = 0
    with source.open('x') as stream:
        for row in media:
            stream.write(json.dumps(['media', *row]) + '\n')
        for count, row in enumerate(requests, 1):
            stream.write(json.dumps(['tokens', *row]) + '\n')
        stream.write(json.dumps(['end', count]) + '\n')
    subprocess.run([sys.executable, str(FRAMEWORK / 'scripts/build-indexed-traffic.py'), str(source), str(path)], check=True)
