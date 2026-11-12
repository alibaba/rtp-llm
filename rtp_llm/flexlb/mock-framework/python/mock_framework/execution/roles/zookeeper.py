import asyncio
import json
import os

from ...config.validation import duration
from .. import ExecutionError
from ..processes import ProcessIdentity


class ZooKeeperRole:
    def __init__(self, launch_config, supervisor, records):
        self.launch_config, self.supervisor, self.records = (
            launch_config,
            supervisor,
            records,
        )

    async def start(self):
        if self.launch_config is None:
            return
        launch_config = self.launch_config
        directory = self.records.directory / "zookeeper"
        directory.mkdir()
        ready = directory / "ready.json"
        identity = ProcessIdentity("zookeeper")
        await self.supervisor.spawn(
            identity,
            launch_config["command"],
            launch_config["cwd"],
            {
                **os.environ,
                **launch_config["env"],
                "MOCK_OUTPUT_DIR": str(directory),
                "MOCK_READY_FILE": str(ready),
            },
            directory,
        )

        async def await_ready():
            while not ready.exists():
                await asyncio.sleep(0.02)
            value = json.loads(ready.read_text())
            host, port = value["connect"].split(":")
            if host != "127.0.0.1" or not 0 < int(port) < 65536:
                raise ExecutionError("Invalid ZooKeeper readiness address")
            return value["connect"]

        connect = await self.supervisor.guard(
            await_ready(), float(duration(launch_config["startup_timeout"]))
        )
        self.records.emit(
            event="process_ready",
            node=identity.process_name,
            connect=connect,
            **identity.event_fields(),
        )
        return connect

    async def close(self):
        return await self.supervisor.stop(["zookeeper"], 5)
