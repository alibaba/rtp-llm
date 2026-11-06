import asyncio
import json
import os

from ...config.validation import duration
from .. import ExecutionError


class ZooKeeperRole:
    def __init__(self, recipe, supervisor, records):
        self.recipe, self.supervisor, self.records = recipe, supervisor, records

    async def start(self):
        if self.recipe is None:
            return
        recipe = self.recipe
        directory = self.records.directory / 'zookeeper'
        directory.mkdir()
        ready = directory / 'ready.json'
        await self.supervisor.spawn('zookeeper', recipe['command'], recipe['cwd'],
            {**os.environ, **recipe['env'], 'MOCK_OUTPUT_DIR': str(directory), 'MOCK_READY_FILE': str(ready)}, directory)
        async def await_ready():
            while not ready.exists():
                await asyncio.sleep(.02)
            value = json.loads(ready.read_text())
            host, port = value['connect'].split(':')
            if host != '127.0.0.1' or not 0 < int(port) < 65536:
                raise ExecutionError('Invalid ZooKeeper readiness address')
            return value['connect']
        connect = await self.supervisor.guard(await_ready(), float(duration(recipe['startup_timeout'])))
        self.records.emit(event='process_ready', node='zookeeper', connect=connect,
                          role='zookeeper', instance_id='zookeeper', generation=0)
        return connect

    async def close(self):
        return await self.supervisor.stop(['zookeeper'], 5)
