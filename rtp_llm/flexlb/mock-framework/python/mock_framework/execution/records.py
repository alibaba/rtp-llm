from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec='microseconds')


def write_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, default=str, indent=2) + '\n', encoding='utf-8')


class Records:
    def __init__(self, directory, case, launch):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        write_json(self.directory / 'configuration.json', {'case': asdict(case), 'launch': launch})
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
