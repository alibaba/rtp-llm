"""Typed, indexed, lossless experiment containers and offline frozen views."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
import zipfile

from artifacts.archive import PRECOMPRESSED, _secret_like
from schema_contract import matches_schema


class ArtifactRole(str, Enum):
    DEFINITION = 'definition'
    EXECUTION = 'execution'
    OBSERVATIONS = 'observations'
    GATE_INPUT = 'gate_input'
    GATE_RESULT = 'gate_result'
    REQUESTS = 'requests'
    CONFIGURATION = 'configuration'
    VIEW = 'view'
    ANALYSIS = 'analysis'
    EXTENSION = 'extension'
    RAW = 'raw'


@dataclass(frozen=True)
class ArtifactEntry:
    path: str
    blob: str
    sha256: str
    size: int
    role: ArtifactRole
    origin: str | None = None

    def to_dict(self):
        return dict(asdict(self), role=self.role.value)

    @classmethod
    def read(cls, value):
        if not isinstance(value, dict) or set(value) != {'path', 'blob', 'sha256', 'size', 'role', 'origin'}:
            raise ValueError('invalid experiment entry')
        path, digest = value['path'], value['sha256']
        if (type(path) is not str or not path or PurePosixPath(path).is_absolute()
                or '\\' in path or any(part in {'', '.', '..'} for part in path.split('/'))
                or type(digest) is not str or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)
                or value['blob'] != 'blobs/' + digest or type(value['size']) is not int or value['size'] < 0
                or value['origin'] is not None and type(value['origin']) is not str):
            raise ValueError('invalid experiment entry')
        return cls(path, value['blob'], digest, value['size'], ArtifactRole(value['role']), value['origin'])


@dataclass(frozen=True)
class MetricDescriptor:
    identity: str
    kind: str
    definition: dict

    @classmethod
    def from_definition(cls, identity, definition):
        kind = ('derived' if 'producer' in definition else
                'raw_observation' if definition.get('mode') == 'scrape' else 'query_result')
        return cls(identity, kind, definition)


@dataclass(frozen=True)
class ViewDescriptor:
    layout: str
    declaration: dict | None

    @classmethod
    def read(cls, value):
        if (not isinstance(value, dict) or set(value) != {'layout', 'declaration'}
                or type(value['layout']) is not str
                or value['declaration'] is not None and not isinstance(value['declaration'], dict)):
            raise ValueError('invalid experiment view')
        return cls(value['layout'], value['declaration'])


def _verify_gate_inputs(entries, read_json):
    for name in entries:
        if not name.endswith('-gate-manifest.json'):
            continue
        document = read_json(name)
        if not matches_schema(document, 'gate_result_schema_version', 1):
            raise ValueError('invalid frozen gate manifest')
        stem = name[:-len('-manifest.json')]
        for key in ('evidence', 'result'):
            target = stem + '-' + key + '.json'
            reference = document[key]
            if (reference['file'] != PurePosixPath(target).name or target not in entries
                    or entries[target].sha256 != reference['sha256']):
                raise ValueError('frozen gate evidence checksum mismatch')


class ExperimentWriter:
    """Atomic ZIP publication, one content-addressed blob per distinct payload."""
    def __init__(self, output, kind='experiment', deadline=None):
        self.output = Path(output)
        self.kind = kind
        self.deadline = deadline
        self.entries = {}
        self.blobs = set()
        self.views = {}
        self.omitted = []

    def check(self):
        if self.deadline is not None:
            self.deadline.check()

    @contextmanager
    def publication(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.output.parent, prefix='.experiment-') as directory:
            temporary = Path(directory) / 'artifact.zip'
            self.scratch = Path(directory)
            with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=3, allowZip64=True) as archive:
                self.archive = archive
                yield self
                self.check()
                _verify_gate_inputs(self.entries, lambda name: json.loads(
                    archive.read(self.entries[name].blob)))
                manifest = dict(experiment_schema_version=1, kind=self.kind,
                    files={name: entry.to_dict() for name, entry in self.entries.items()},
                    views={name: asdict(view) for name, view in self.views.items()}, omitted=self.omitted)
                archive.writestr(zipfile.ZipInfo('manifest.json'), json.dumps(manifest, ensure_ascii=False,
                    allow_nan=False, separators=(',', ':')), compress_type=zipfile.ZIP_DEFLATED, compresslevel=3)
            self.check()
            temporary.replace(self.output)

    def add_file(self, name, path, role):
        self.check()
        path = Path(path)
        digest = hashlib.sha256()
        size = 0
        # Snapshot and hash in one pass: the committed bytes must match the digest.
        with tempfile.TemporaryFile(dir=self.scratch) as snapshot:
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    self.check()
                    digest.update(block)
                    size += len(block)
                    snapshot.write(block)
            sha = digest.hexdigest()
            blob = 'blobs/' + sha
            entry = ArtifactEntry(name, blob, sha, size, role, str(path.resolve()))
            ArtifactEntry.read(entry.to_dict())
            if name in self.entries:
                raise ValueError('duplicate experiment path: ' + name)
            if blob not in self.blobs:
                member = blob
                if path.suffix.lower() in PRECOMPRESSED:
                    member = zipfile.ZipInfo(blob)
                    member.compress_type = zipfile.ZIP_STORED
                snapshot.seek(0)
                with self.archive.open(member, 'w', force_zip64=True) as target:
                    for block in iter(lambda: snapshot.read(1024 * 1024), b''):
                        self.check()
                        target.write(block)
                self.blobs.add(blob)
            self.entries[name] = entry
        return name

    def add_json(self, name, value, role):
        path = self.scratch / 'document.json'
        with path.open('w', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')))
        self.add_file(name, path, role)
        # Generated payloads do not reference the temporary encoder directory.
        entry = self.entries[name]
        self.entries[name] = ArtifactEntry(entry.path, entry.blob, entry.sha256, entry.size, entry.role)
        return name

    def add_view(self, name, spec, declaration=None):
        """Keep layout separate from full point arrays; repeated curves share blobs."""
        from reporting.spec import validate
        validate(spec, versioned=True)
        projected = dict(spec)
        panels = []
        for panel_index, panel in enumerate(spec.get('panels', [])):
            panel = dict(panel)
            curves = []
            for curve_index, curve in enumerate(panel.get('series', [])):
                curve = dict(curve)
                for field in ('points', 'statistics_points'):
                    if curve.get(field) is not None:
                        path = f'views/{name}/datasets/{panel_index}-{curve_index}-{field}.json'
                        self.add_json(path, curve[field], ArtifactRole.VIEW)
                        curve[field] = {'dataset': path}
                curves.append(curve)
            panel['series'] = curves
            panels.append(panel)
        projected['panels'] = panels
        path = self.add_json(f'views/{name}/layout.json', projected, ArtifactRole.VIEW)
        self.views[name] = ViewDescriptor(path, declaration)


class ExperimentReader:
    def __init__(self, path):
        self.archive = zipfile.ZipFile(path)
        try:
            names = self.archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError('duplicate experiment ZIP members')
            manifest = json.loads(self.archive.read('manifest.json'))
            if (not matches_schema(manifest, 'experiment_schema_version', 1)
                    or manifest.get('kind') not in {'experiment', 'raw'}
                    or not isinstance(manifest.get('files'), dict)
                    or not isinstance(manifest.get('views'), dict)):
                raise ValueError('invalid experiment manifest')
            self.kind = manifest['kind']
            self.entries = {name: ArtifactEntry.read(value) for name, value in manifest['files'].items()}
            self.views = {name: ViewDescriptor.read(value) for name, value in manifest['views'].items()}
            self.omitted = manifest['omitted']
            if any(name != entry.path for name, entry in self.entries.items()):
                raise ValueError('experiment path mismatch')
            expected = {'manifest.json'} | {entry.blob for entry in self.entries.values()}
            if set(names) != expected:
                raise ValueError('experiment ZIP inventory mismatch')
            if self.kind == 'experiment' and not {'experiment.json', 'execution.json', 'completeness.json'} <= self.entries.keys():
                raise ValueError('experiment lacks required facts')
            if any(view.layout not in self.entries or self.entries[view.layout].role is not ArtifactRole.VIEW
                   for view in self.views.values()):
                raise ValueError('experiment view layout is not indexed')
        except Exception:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.archive.close()

    @contextmanager
    def open(self, name):
        entry = self.entries[name]
        digest, size = hashlib.sha256(), 0
        # Verify while copying to a disk spool, rather than holding all bytes twice.
        with tempfile.TemporaryFile() as snapshot:
            with self.archive.open(entry.blob) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
                    size += len(block)
                    snapshot.write(block)
            if digest.hexdigest() != entry.sha256 or size != entry.size:
                raise ValueError('experiment checksum mismatch: ' + name)
            snapshot.seek(0)
            yield snapshot

    def read_json(self, name):
        with self.open(name) as stream:
            return json.load(stream)

    def resolve_source(self, path):
        """Resolve frozen physical provenance without reading the original host."""
        if path in self.entries:
            return path
        matches = [name for name, entry in self.entries.items() if entry.origin == str(path)]
        if not matches:
            raise KeyError('source is not in this experiment: ' + str(path))
        return matches[0]

    def verify(self):
        checked = set()
        for name, entry in self.entries.items():
            if entry.blob not in checked:
                with self.open(name):
                    pass
                checked.add(entry.blob)
        _verify_gate_inputs(self.entries, self.read_json)

    def verify_raw(self, path):
        from file_digest import sha256_file
        reference = self.read_json('raw-attachment.json')
        if Path(path).stat().st_size != reference['size'] or sha256_file(path) != reference['sha256']:
            raise ValueError('raw attachment checksum mismatch')
        with ExperimentReader(path) as raw:
            if raw.kind != 'raw':
                raise ValueError('invalid raw attachment kind')
            raw.verify()

    def view_spec(self, name):
        spec = self.read_json(self.views[name].layout)
        for panel in spec.get('panels', []):
            for curve in panel.get('series', []):
                for field in ('points', 'statistics_points'):
                    if curve.get(field) is not None:
                        reference = curve[field]
                        if not isinstance(reference, dict) or set(reference) != {'dataset'}:
                            raise ValueError('invalid view dataset reference')
                        if self.entries[reference['dataset']].role is not ArtifactRole.VIEW:
                            raise ValueError('invalid view dataset role')
                        curve[field] = self.read_json(reference['dataset'])
        from reporting.spec import validate
        validate(spec, versioned=True)
        return spec

    def render(self, name, output):
        from reporting import render
        from artifacts.file_io import write_text
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if name in self.views:
            spec = self.view_spec(name)
        elif name == 'default.yaml' and 'analysis.json' in self.entries:
            from reporting.workload import build_spec
            definition = self.read_json('experiment.json')
            from reporting.view_model import ViewDefinition
            spec = build_spec(self.analysis(), None, presentation=ViewDefinition.read(definition['views'][name]))
        else:
            raise ValueError('view was not frozen in this experiment: ' + name)
        write_text(output, render(spec))
        return output

    @property
    def available_views(self):
        names = list(self.views)
        if 'analysis.json' in self.entries and 'default.yaml' not in names:
            names.append('default.yaml')
        return names

    @property
    def preferred_view(self):
        definition = self.read_json('experiment.json')
        names = self.available_views
        for name in definition['plan'].get('reports', []):
            if name != 'default.yaml' and name in names:
                return name
        if names:
            return names[0]
        raise ValueError('this experiment has no renderable view')

    def analysis(self):
        document = self.read_json('analysis.json')
        if not matches_schema(document, 'experiment_analysis_schema_version', 1):
            raise ValueError('invalid experiment analysis')
        from monitoring.metric_store import MetricStore
        store = MetricStore(self.read_json(document['metrics']))
        result = dict(document['metadata'])
        result['series'], result['statistic_sources'], _, _ = store.series(document['anchor_epoch_s'])
        return result


def _role(path):
    name = path.name
    if name == 'metrics.json' or name == 'queries.json':
        return ArtifactRole.OBSERVATIONS
    if name.endswith('-gate-evidence.json'):
        return ArtifactRole.GATE_INPUT
    if name.endswith('-gate-result.json'):
        return ArtifactRole.GATE_RESULT
    if name.endswith('samples.jsonl'):
        return ArtifactRole.OBSERVATIONS
    if name in {'client_events.jsonl', 'client_lifecycle.jsonl'}:
        return ArtifactRole.REQUESTS
    if name in {'flow-input.json', 'environment.json', 'actual-master-config.json',
                'master_config.json', 'master-artifact.json', 'perf.json'}:
        return ArtifactRole.CONFIGURATION
    if name == 'analysis.json':
        return ArtifactRole.ANALYSIS
    return ArtifactRole.EXTENSION


def publish_run(directory, definition, execution, *, analysis=None, deadline=None):
    """Package settled per-run evidence. HTML is a regenerable cache, not input."""
    from reporting.core import read_bundle
    directory = Path(directory)
    output, raw_output = directory / 'experiment.eval.zip', directory / 'experiment.raw.zip'
    definition_path = directory / 'experiment-definition.json'
    files = sorted(directory.rglob('*'))
    primary = ExperimentWriter(output, deadline=deadline)
    with primary.publication():
        raw = ExperimentWriter(primary.scratch / 'raw.zip', kind='raw', deadline=deadline)
        with raw.publication():
            primary.add_json('experiment.json', definition, ArtifactRole.DEFINITION)
            primary.add_json('execution.json', execution, ArtifactRole.EXECUTION)
            primary.add_json('run/result.json', execution, ArtifactRole.EXECUTION)
            for path in files:
                primary.check()
                if path.is_symlink():
                    primary.omitted.append(dict(path=str(path.relative_to(directory)), reason='symbolic link'))
                    continue
                if not path.is_file() or path in {output, raw_output, definition_path, directory / 'result.json'}:
                    continue
                if path.parent == directory and re.fullmatch(r'experiment-[0-9a-f]{64}\.raw\.zip', path.name):
                    continue
                relative = path.relative_to(directory)
                if _secret_like(relative):
                    primary.omitted.append(dict(path=str(relative), reason='secret-like filename'))
                    continue
                # Report manifests belong to the materialized directory protocol;
                # this container has its own checked index and frozen view layouts.
                if relative.parts[0] == 'reports':
                    continue  # Views are projections of the canonical facts below.
                if path.suffix.lower() == '.html':
                    continue
                structured = path.suffix.lower() in {'.json', '.jsonl', '.yaml', '.yml', '.csv'}
                evidence_archive = path.name.startswith('series-') and path.name.endswith('.json.gz')
                writer = primary if structured or evidence_archive else raw
                writer.add_file('run/' + relative.as_posix(), path,
                                _role(path) if structured else ArtifactRole.OBSERVATIONS
                                if evidence_archive else ArtifactRole.RAW)
            metrics = directory / 'metrics.json'
            if metrics.is_file():
                from monitoring.metric_store import MetricStore
                store = MetricStore.read(directory)
                primary.add_json('datasets/metric-catalog.json', [asdict(MetricDescriptor.from_definition(k, v))
                    for k, v in store.document['definitions'].items()], ArtifactRole.OBSERVATIONS)
                if analysis is not None:
                    primary.add_json('analysis.json', dict(experiment_analysis_schema_version=1,
                        metadata={key: value for key, value in analysis.items()
                                  if key not in {'series', 'statistic_sources'}},
                        metrics='run/metrics.json', anchor_epoch_s=analysis['clock_anchor']['epoch_s']),
                        ArtifactRole.ANALYSIS)
            reports = execution.get('workload', {}).get('reports', {})
            view_names = {str(Path(path).parent.resolve()): name for name, path in reports.items()}
            invalid_views = []
            for manifest in sorted((directory / 'reports/run').glob('*/manifest.json')):
                try:
                    bundle = read_bundle(manifest.parent)
                    with (bundle / 'report-spec.json').open(encoding='utf-8') as stream:
                        spec = json.load(stream)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    invalid_views.append(dict(view=manifest.parent.name, error=str(exc)))
                    continue
                name = view_names.get(str(bundle.resolve()), bundle.name)
                primary.add_view(name, spec, definition.get('views', {}).get(name))
            missing = [name for name in ('metrics.json', 'workload-evidence.json')
                       if execution.get('test_kind') == 'workload' and not (directory / name).is_file()]
            missing.extend(name.replace('-result.json', '-manifest.json') for name in primary.entries
                           if name.endswith('-gate-result.json')
                           and name.replace('-result.json', '-manifest.json') not in primary.entries)
            primary.add_json('completeness.json', dict(missing=missing, omissions=primary.omitted,
                outcome=execution.get('outcome'), invalid_views=invalid_views,
                available_views=list(primary.views)), ArtifactRole.EXECUTION)
        from file_digest import sha256_file
        digest = sha256_file(raw.output)
        raw_output = directory / ('experiment-' + digest + '.raw.zip')
        # Immutable attachment names preserve an earlier committed primary when
        # republishing fails. A primary never points to an overwritten raw ZIP.
        if raw_output.exists():
            if sha256_file(raw_output) != digest:
                raise ValueError('raw attachment checksum mismatch')
        else:
            raw.output.replace(raw_output)
        primary.add_json('raw-attachment.json', dict(file=raw_output.name,
            sha256=digest, size=raw_output.stat().st_size,
            files={name: entry.to_dict() for name, entry in raw.entries.items()}), ArtifactRole.EXECUTION)
    return dict(experiment=str(output), raw=str(raw_output))


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Inspect, verify or render a single experiment file')
    parser.add_argument('artifact', type=Path)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--view', help='frozen view name from the artifact index')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if args.view and not args.out:
        parser.error('--view requires --out')
    with ExperimentReader(args.artifact) as reader:
        if args.verify:
            reader.verify()
        if args.out:
            reader.render(args.view or reader.preferred_view, args.out)
        print(json.dumps(dict(kind=reader.kind, views=reader.available_views,
                             files=len(reader.entries), omitted=reader.omitted), ensure_ascii=False))


if __name__ == '__main__':
    main()
