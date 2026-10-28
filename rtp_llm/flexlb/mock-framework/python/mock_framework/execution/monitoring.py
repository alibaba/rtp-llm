"""One owned Prometheus per run; configuration and TSDB stay in the run directory."""
import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import urllib.parse
import urllib.request

import yaml

from ..config.values import duration, fields, text
from .records import timestamp, write_json
from .transport import ExecutionError


def prepare_prometheus(raw, base):
    fields(raw, {'binary', 'scrape_interval', 'startup_timeout'}, {'binary'}, where='launch.prometheus')
    binary = (Path(base) / text(raw['binary'], 'prometheus.binary')).resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ExecutionError('prometheus.binary must be an executable file')
    interval = duration(raw.get('scrape_interval', '1s'), positive=True) * 1000
    if interval < 1 or interval != int(interval):
        raise ExecutionError('Prometheus scrape_interval must be an integral number of milliseconds, at least 1ms')
    startup = raw.get('startup_timeout', '30s')
    duration(startup, positive=True)
    return {'binary': str(binary), 'scrape_interval': f'{int(interval)}ms', 'startup_timeout': startup}


def _get(url):
    # Only idempotent loopback reads. A cancelled await can leave at most this bounded read running.
    with urllib.request.urlopen(url, timeout=2) as response:
        return json.load(response)


def _epoch(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ExecutionError('Prometheus scrape timestamp must include timezone')
    return parsed.timestamp()


class Prometheus:
    JOB = 'mock'

    def __init__(self, owner, records, recipe):
        self.owner, self.records, self.recipe = owner, records, recipe
        self.directory = records.directory / 'prometheus'
        self.startup_timeout = float(duration(recipe['startup_timeout']))
        self.interval = float(duration(recipe['scrape_interval']))
        self.url, self.child = None, None
        self.targets, self.baselined = {}, False
        self.metadata = {'job': self.JOB, 'recipe': recipe, 'complete': False}

    def save(self):
        temporary = self.directory / 'collection.tmp'
        write_json(temporary, self.metadata)
        temporary.replace(self.directory / 'collection.json')

    async def start(self):
        self.directory.mkdir()
        write_json(self.directory / 'targets.json', [])
        config = {'global': {'scrape_interval': self.recipe['scrape_interval'], 'scrape_timeout': self.recipe['scrape_interval']},
                  'scrape_configs': [{'job_name': self.JOB, 'file_sd_configs': [
                      {'files': [str(self.directory / 'targets.json')], 'refresh_interval': '100ms'}]}]}
        (self.directory / 'prometheus.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
        command = [self.recipe['binary'], '--config.file=' + str(self.directory / 'prometheus.yaml'),
                   '--web.listen-address=127.0.0.1:0', '--storage.tsdb.path=' + str(self.directory / 'data'), '--log.format=json']
        self.metadata['command'] = command
        self.save()
        self.child = await self.owner.spawn('prometheus', command, str(self.directory), dict(os.environ), self.directory)
        while self.url is None:
            # Discover the actual bound port from this process's structured log, not a guessed free port.
            for line in (self.directory / 'stderr.log').read_text(encoding='utf-8', errors='replace').splitlines():
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue  # A final partially written log line is retried on the next read.
                if entry.get('msg') == 'Listening on' and entry.get('component') == 'web':
                    address = entry.get('address', '')
                    if address.startswith('127.0.0.1:') and address[10:].isdigit() and int(address[10:]) > 0:
                        self.url = 'http://' + address
            if self.url is None:
                await asyncio.sleep(.02)
        while True:
            try:
                # Ready is plain text; the build-info endpoint is JSON and requires a ready server.
                value = await asyncio.to_thread(_get, self.url + '/api/v1/status/buildinfo')
                if value.get('status') == 'success':
                    self.metadata.update(url=self.url, build=value['data'])
                    self.save()
                    self.records.emit(event='process_ready', node='prometheus', url=self.url)
                    return
            except (OSError, ValueError):
                pass
            await asyncio.sleep(.05)

    def discover(self, node, directory):
        try:
            endpoint = json.loads((directory / 'endpoints.json').read_text(encoding='utf-8'))
            address = urllib.parse.urlsplit(endpoint['metrics'])
            if endpoint.get('id') != node or address.scheme not in ('http', 'https') or not address.hostname \
                    or address.username or address.password or address.query or address.fragment:
                raise ValueError('Invalid metrics endpoint')
            rpc = endpoint['rpc']
            if not isinstance(rpc, str) or not rpc or any(v['labels']['instance'] == rpc for k, v in self.targets.items() if k != node):
                raise ValueError('Missing or duplicate instance address')
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ExecutionError(f'{node}: managed collection requires endpoints.json with id, unique rpc and metrics URL') from error
        self.targets[node] = {'targets': [address.netloc], 'labels': {'instance': rpc,
                              '__scheme__': address.scheme, '__metrics_path__': address.path or '/metrics'}}
        temporary = self.directory / 'targets.tmp'
        write_json(temporary, list(self.targets.values()))
        temporary.replace(self.directory / 'targets.json')
        self.metadata['targets'] = dict(self.targets)
        self.save()

    async def scrapes(self):
        result = await asyncio.to_thread(_get, self.url + '/api/v1/targets?state=active')
        if result.get('status') != 'success':
            raise ExecutionError('Prometheus target query failed')
        by_address = {value['labels']['instance']: node for node, value in self.targets.items()}
        targets = {by_address[target['labels']['instance']]: target for target in result['data']['activeTargets']
                   if target.get('labels', {}).get('job') == self.JOB and target['labels'].get('instance') in by_address}
        return targets

    async def baseline(self):
        if not self.targets:
            raise ExecutionError('Managed Prometheus has no metrics targets')
        seen = {node: set() for node in self.targets}
        while True:
            for node, target in (await self.scrapes()).items():
                if node in seen and target['health'] == 'up':
                    seen[node].add(target['lastScrape'])
            if all(len(times) >= 2 for times in seen.values()):
                self.baselined = True
                self.metadata['baseline_at'] = timestamp()
                self.save()
                self.records.emit(event='collection_ready', url=self.url, targets=sorted(self.targets))
                return
            await asyncio.sleep(.05)

    async def final_scrape(self):
        if not self.baselined:
            return  # Setup never completed; do not claim a fully collected experiment.
        marker = timestamp()
        async with asyncio.timeout(max(5, 2 * self.interval + 2)):
            while True:
                if self.child.returncode is not None:
                    raise ExecutionError('Prometheus exited before final collection')
                for node, child in self.owner.children:
                    if node in self.targets and child.returncode is not None:
                        raise ExecutionError(f'{node} exited before final collection')
                targets = await self.scrapes()
                if set(targets) == set(self.targets) and all(target['health'] == 'up' and
                        _epoch(target['lastScrape']) >= _epoch(marker) for target in targets.values()):
                    self.metadata['final_scrapes'] = {node: target['lastScrape'] for node, target in targets.items()}
                    self.records.emit(event='collection_finished', after=marker, scrapes=self.metadata['final_scrapes'])
                    self.save()
                    return
                await asyncio.sleep(.05)

    def finish(self, errors):
        if not self.directory.exists():
            return
        self.metadata.update(complete=self.baselined and 'final_scrapes' in self.metadata and not errors,
                             finished_at=timestamp(), errors=list(errors))
        self.save()
