"""Materialize the qualified Q36 provider/DDP parameter schema, without weights."""
import base64
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys

ARCHIVE_SHA = '1a8fa0e0155008901a1867d058158ba9d83a8fed5af5ae644fb84ceea34bd5ab'
MANIFEST_SHA = 'd824d3ac60b76ac7699dfe36f3935580fdf4033780e3ebb9f8b311e10ac0d4e0'
SOURCE = Path(__file__).parent / 'data/q36_provider_ddp_sr_manifest.json.gz.b64'
ENCODED_BYTES = 286146
ARCHIVE_BYTES = 212483
MANIFEST_BYTES = 17672741


def decode_manifest(encoded):
    if len(encoded) != ENCODED_BYTES:
        raise ValueError('Q36 SR schema encoded length mismatch')
    archive = base64.b64decode(b''.join(encoded.split()), validate=True)
    if len(archive) != ARCHIVE_BYTES or hashlib.sha256(archive).hexdigest() != ARCHIVE_SHA:
        raise ValueError('Q36 SR schema archive SHA256 mismatch')
    with gzip.GzipFile(fileobj=io.BytesIO(archive), mode='rb') as source:
        raw = source.read(MANIFEST_BYTES + 1)
    if len(raw) != MANIFEST_BYTES or hashlib.sha256(raw).hexdigest() != MANIFEST_SHA:
        raise ValueError('Q36 SR schema SHA256 mismatch')
    manifest = json.loads(raw)
    if (manifest.get('binding_scope') != 'full-provider-ddp-live-source-verified' or
            manifest.get('parameter_count') != 20943 or
            manifest.get('distinct_hf_source_count') != 693 or
            manifest.get('training_authorized') is not False):
        raise ValueError('unexpected Q36 SR schema coverage')
    return raw


def materialize(encoded, output):
    raw = decode_manifest(encoded)
    destination = Path(output)
    stream = destination.open('xb')
    identity = os.fstat(stream.fileno())
    try:
        with stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            current = destination.lstat()
            if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
                destination.unlink()
        except OSError:
            pass
        raise


def main():
    if len(sys.argv) != 2:
        raise SystemExit('usage: prepare_sr_manifest.py OUTPUT')
    with SOURCE.open('rb') as stream:
        encoded = stream.read(ENCODED_BYTES + 1)
    materialize(encoded, sys.argv[1])
    print(MANIFEST_SHA)


if __name__ == '__main__':
    main()
