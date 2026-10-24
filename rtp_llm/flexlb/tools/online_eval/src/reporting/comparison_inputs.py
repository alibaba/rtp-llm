"""Normalize checked frozen bundles and experiment containers for comparison."""
from __future__ import annotations
from file_io import read_json
from dataclasses import dataclass
import hashlib
from pathlib import Path
import zipfile

from reporting.experiment import ExperimentReader
from reporting.core import load_analysis, read_bundle
from reporting.renderer import render
from reporting.comparison_model import FrozenReport


@dataclass(frozen=True)
class RunSnapshot:
    source: Path
    analysis: dict
    spec: dict
    report: FrozenReport
    digest: str
    input_kind: str
    view: str | None
    execution: dict

    def original_html(self):
        if self.input_kind == 'bundle':
            return (self.source / 'report.html').read_text(encoding='utf-8')
        # Render the frozen layout only; no current YAML or domain analyzer.
        spec = dict(self.spec, sections=[s for s in self.spec.get('sections', [])
                                       if s['type'] != 'links'])
        return render(spec)


def load_snapshot(path, *, view=None):
    path = Path(path).resolve()
    if path.is_dir():
        if view is not None:
            raise ValueError('view selection applies only to experiment ZIP inputs')
        directory = read_bundle(path)
        manifest = read_json(directory / 'manifest.json')
        if manifest.get('kind') != 'run':
            raise ValueError('comparison input must be a run bundle: ' + str(path))
        required = {'analysis.json', 'report-spec.json', 'report.html'}
        if any(manifest['files'][name]['path'] != name for name in required):
            raise ValueError('run bundle must verify canonical report files')
        spec = read_json(directory / 'report-spec.json')
        analysis = load_analysis(directory)
        digest = hashlib.sha256((directory / 'manifest.json').read_bytes()).hexdigest()
        return RunSnapshot(path, analysis, spec, FrozenReport.read(spec), digest,
                           'bundle', None, read_json(directory / 'execution.json')
                           if manifest['files'].get('execution.json', {}).get('path') == 'execution.json' else {})
    if not path.is_file() or not zipfile.is_zipfile(path):
        raise ValueError('comparison input must be a run bundle directory or experiment ZIP: ' + str(path))
    with ExperimentReader(path) as reader:
        if reader.kind != 'experiment':
            raise ValueError('raw attachment is not an experiment comparison input')
        selected = view or reader.preferred_view
        if selected not in reader.views:
            raise ValueError('comparison requires a frozen view: ' + selected)
        spec = reader.view_spec(selected)
        execution = reader.read_json('execution.json')
        analysis = reader.analysis() if 'analysis.json' in reader.entries else {'execution': execution}
        digest = hashlib.sha256(reader.archive.read('manifest.json')).hexdigest()
        return RunSnapshot(path, analysis, spec, FrozenReport.read(spec), digest,
                           'experiment', selected, execution)
