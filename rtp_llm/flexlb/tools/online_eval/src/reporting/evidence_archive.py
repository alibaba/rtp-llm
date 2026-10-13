"""Lossless shared series archives, linked into portable report bundles."""
from file_digest import sha256_file
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import gzip
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import tempfile

_FIELDS = ('series', 'statistic_sources')
_REFERENCE = 'series_evidence'
_ARCHIVES = ContextVar('report_series_archives', default=None)


@dataclass(frozen=True)
class EvidenceReference:
    path: str
    sha256: str
    fields: tuple[str, ...]

    @classmethod
    def parse(cls, value, analysis):
        if not isinstance(value, dict) or set(value) != {'path', 'sha256', 'fields'}:
            raise ValueError('invalid series evidence reference')
        name, digest, fields = value['path'], value['sha256'], value['fields']
        if (type(digest) is not str or re.fullmatch(r'[0-9a-f]{64}', digest) is None
                or name != 'series-' + digest + '.json.gz'
                or not isinstance(fields, list) or not fields
                or any(type(field) is not str or field not in _FIELDS or field in analysis for field in fields)
                or len(set(fields)) != len(fields)):
            raise ValueError('invalid series evidence reference')
        return cls(name, digest, tuple(fields))

    def to_dict(self):
        return dict(path=self.path, sha256=self.sha256, fields=list(self.fields))


@contextmanager
def archive_session():
    token = _ARCHIVES.set({})
    try:
        yield
    finally:
        _ARCHIVES.reset(token)


def _archive(root, values):
    pool = Path(root) / 'reports' / 'evidence'
    pool.mkdir(parents=True, exist_ok=True)
    archive_key = (str(pool.resolve()), tuple((name, id(value)) for name, value in values.items()))
    cache = _ARCHIVES.get()
    if cache is not None and archive_key in cache:
        return cache[archive_key][1]
    # Encode individual inventory entries, rather than another full JSON string.
    with tempfile.NamedTemporaryFile(dir=pool, suffix='.tmp', delete=False) as raw:
        temporary = Path(raw.name)
        try:
            with gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=1, mtime=0, filename='') as compressed:
                # Series are independent streams. Encode one stream at a time with
                # the C JSON encoder; token-by-token Python encoding is much slower.
                compressed.write(b'{"series_evidence_schema_version":1,"values":{')
                for field_index, (field, inventory) in enumerate(values.items()):
                    if field_index:
                        compressed.write(b',')
                    compressed.write(json.dumps(field).encode() + b':')
                    if not isinstance(inventory, dict):
                        raise ValueError('series evidence inventories must be mappings')
                    compressed.write(b'{')
                    for index, (key, value) in enumerate(inventory.items()):
                        if type(key) is not str:
                            raise ValueError('series evidence identity must be a string')
                        if index:
                            compressed.write(b',')
                        compressed.write(json.dumps(key, ensure_ascii=False).encode('utf-8') + b':')
                        compressed.write(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                                    separators=(',', ':')).encode('utf-8'))
                    compressed.write(b'}')
                compressed.write(b'}}')
            raw.flush()
            digest = sha256_file(temporary)
            path = pool / (digest + '.json.gz')
            if path.exists():
                if sha256_file(path) != digest:
                    raise ValueError('shared series archive checksum mismatch')
                temporary.unlink()
            else:
                temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if cache is not None:
        cache[archive_key] = (values, path)  # Retain identities for this publication session only.
    return path


def project(root, bundle, analysis, files):
    result = dict(analysis)
    if _REFERENCE in result:
        raise ValueError('analysis already contains a series evidence reference')
    values = {key: result.pop(key) for key in _FIELDS if key in result}
    if values:
        archive = _archive(root, values)
        name = 'series-' + archive.name
        destination = bundle / name
        if not destination.exists():
            try:
                os.link(archive, destination)
            except OSError:
                shutil.copyfile(archive, destination)
        digest = archive.name.split('.')[0]
        if sha256_file(destination) != digest:
            raise ValueError('bundle series archive checksum mismatch')
        files[name] = dict(path=name, sha256=digest)
        result[_REFERENCE] = EvidenceReference(name, digest, tuple(values)).to_dict()
    return result


def restore(bundle, analysis):
    result = dict(analysis)
    if _REFERENCE in result:
        reference = EvidenceReference.parse(result.pop(_REFERENCE), result)
        path = Path(bundle) / reference.path
        if path.resolve().parent != Path(bundle).resolve() or sha256_file(path) != reference.sha256:
            raise ValueError('series evidence checksum or path mismatch')
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            document = json.load(stream)
        from schema_contract import matches_schema
        if (not matches_schema(document, 'series_evidence_schema_version', 1)
                or not isinstance(document.get('values'), dict)
                or set(document['values']) != set(reference.fields)
                or any(not isinstance(value, dict) for value in document['values'].values())):
            raise ValueError('invalid series evidence document')
        result.update(document['values'])
    return result


def verify_references(analysis, files):
    if not isinstance(analysis, dict):
        raise ValueError('report analysis must be a mapping')
    if _REFERENCE in analysis:
        reference = EvidenceReference.parse(analysis[_REFERENCE], analysis)
        if (reference.path not in files
                or files[reference.path].get('path') != reference.path
                or files[reference.path].get('sha256') != reference.sha256):
            raise ValueError('series evidence is not verified by the bundle manifest')
