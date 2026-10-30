from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import tempfile
import os


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec='microseconds')


def write_json(path, value):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.publish-', encoding='utf-8', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, default=str, indent=2) + '\n')
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class Records:
    def __init__(self, directory, case, launch):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        write_json(self.directory / 'configuration.json', {'case': asdict(case), 'launch': launch})
        if case.view is not None:
            view = json.loads(case.view.read_text(encoding='utf-8'))
            if not isinstance(view, dict) or not isinstance(view.get('panels'), list) or not view.get('title'):
                raise ValueError('view must be a native Grafana dashboard JSON with title and panels')
            write_json(self.directory / 'view.json', view)
        source = case.source
        if source.is_file():
            (self.directory / 'case.yaml').write_bytes(source.read_bytes())
        self._streams = {name: (self.directory / f'{name}.jsonl').open('x', encoding='utf-8')
                         for name in ('events', 'gates')}

    def emit(self, stream='events', **event):
        event.setdefault('recorded_at', timestamp())
        self._streams[stream].write(json.dumps(event, ensure_ascii=False, allow_nan=False, default=str) + '\n')
        self._streams[stream].flush()

    def versions(self, directories):
        versions = []
        for directory in sorted(set(map(str, directories))):
            item = {'directory': directory}
            for key, arguments in [('commit', ['rev-parse', 'HEAD']), ('status', ['status', '--porcelain']),
                                   ('diff', ['diff', '--binary', 'HEAD'])]:
                try:
                    result = subprocess.run(['git', '-C', directory, *arguments], capture_output=True,
                                            text=True, timeout=10, check=True)
                    item[key] = result.stdout.rstrip()
                except (OSError, subprocess.SubprocessError) as error:
                    item[key] = {'unavailable': str(error)}
            versions.append(item)
        write_json(self.directory / 'code.json', versions)

    def close(self, result):
        write_json(self.directory / 'result.json', result)
        for stream in self._streams.values():
            stream.close()
