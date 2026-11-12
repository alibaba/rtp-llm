"""One owned Prometheus per run; configuration and TSDB stay in the run directory."""

import asyncio
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime

import yaml

from ...config.validation import duration
from .. import ExecutionError
from ..processes import ProcessIdentity
from ..step_trace import timestamp, write_json


def _get(url):
    # Only idempotent loopback reads. A cancelled await can leave at most this bounded read running.
    with urllib.request.urlopen(url, timeout=2) as response:
        return json.load(response)


def _epoch(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ExecutionError("Prometheus scrape timestamp must include timezone")
    return parsed.timestamp()


class PrometheusRole:
    JOB = "mock"

    def __init__(self, supervisor, records, launch_config):
        self.supervisor, self.records, self.launch_config = (
            supervisor,
            records,
            launch_config,
        )
        self.directory = records.directory / "prometheus"
        self.startup_timeout = float(duration(launch_config["startup_timeout"]))
        self.interval = float(duration(launch_config["scrape_interval"]))
        self.url, self.child = None, None
        self.targets, self.baseline_ready = {}, False
        self.metadata = {"job": self.JOB, "recipe": launch_config, "complete": False}

    def _save(self):
        write_json(self.directory / "collection.json", self.metadata)

    async def start(self):
        self.directory.mkdir()
        write_json(self.directory / "targets.json", [])
        config = {
            "global": {
                "scrape_interval": self.launch_config["scrape_interval"],
                "scrape_timeout": self.launch_config["scrape_interval"],
            },
            "scrape_configs": [
                {
                    "job_name": self.JOB,
                    "file_sd_configs": [
                        {
                            "files": [str(self.directory / "targets.json")],
                            "refresh_interval": "100ms",
                        }
                    ],
                }
            ],
        }
        (self.directory / "prometheus.yaml").write_text(
            yaml.safe_dump(config), encoding="utf-8"
        )
        command = [
            self.launch_config["binary"],
            "--config.file=" + str(self.directory / "prometheus.yaml"),
            "--web.listen-address=127.0.0.1:0",
            "--storage.tsdb.path=" + str(self.directory / "data"),
            "--log.format=json",
        ]
        self.metadata["command"] = command
        self._save()
        self.child = await self.supervisor.spawn(
            ProcessIdentity("prometheus"),
            command,
            str(self.directory),
            dict(os.environ),
            self.directory,
            zero_exit=True,
        )
        while self.url is None:
            # Discover the actual bound port from this process's structured log, not a guessed free port.
            for line in (
                (self.directory / "stderr.log")
                .read_text(encoding="utf-8", errors="replace")
                .splitlines()
            ):
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue  # A final partially written log line is retried on the next read.
                if (
                    entry.get("msg") == "Listening on"
                    and entry.get("component") == "web"
                ):
                    address = entry.get("address", "")
                    if (
                        address.startswith("127.0.0.1:")
                        and address[10:].isdigit()
                        and int(address[10:]) > 0
                    ):
                        self.url = "http://" + address
            if self.url is None:
                await asyncio.sleep(0.02)
        while True:
            try:
                # Ready is plain text; the build-info endpoint is JSON and requires a ready server.
                value = await asyncio.to_thread(
                    _get, self.url + "/api/v1/status/buildinfo"
                )
                if value.get("status") == "success":
                    self.metadata.update(url=self.url, build=value["data"])
                    self._save()
                    self.records.emit(
                        event="process_ready", node="prometheus", url=self.url
                    )
                    return
            except (OSError, ValueError):
                pass
            await asyncio.sleep(0.05)

    def register_target(self, instance_id, endpoint):
        try:
            if not isinstance(endpoint, dict):
                raise ValueError("Missing or invalid endpoints")
            backend = endpoint.get("metrics_backend", "prometheus")
            if backend != "prometheus":
                raise ExecutionError(
                    f"{instance_id}: managed Prometheus requires metrics_backend=prometheus, got {backend!r}"
                )
            address = urllib.parse.urlsplit(endpoint["metrics"])
            if (
                endpoint.get("id") != instance_id
                or address.scheme not in ("http", "https")
                or not address.hostname
                or address.username
                or address.password
                or address.query
                or address.fragment
            ):
                raise ValueError("Invalid metrics endpoint")
            rpc = endpoint["rpc"]
            if (
                not isinstance(rpc, str)
                or not rpc
                or any(
                    v["labels"]["instance"] == rpc
                    for k, v in self.targets.items()
                    if k != instance_id
                )
            ):
                raise ValueError("Missing or duplicate instance address")
        except (ValueError, KeyError, TypeError) as error:
            raise ExecutionError(
                f"{instance_id}: managed collection requires endpoints.json with id, unique rpc and metrics URL"
            ) from error
        self.targets[instance_id] = {
            "targets": [address.netloc],
            "labels": {
                "instance": rpc,
                "__scheme__": address.scheme,
                "__metrics_path__": address.path or "/metrics",
            },
        }
        self._write_targets()

    def retire(self, node):
        self.targets.pop(node)
        self.metadata.get("final_scrapes", {}).pop(node, None)
        self._write_targets()

    def _write_targets(self):
        write_json(self.directory / "targets.json", list(self.targets.values()))
        self.metadata["targets"] = dict(self.targets)
        self._save()

    async def _fetch_scrape_status(self):
        result = await asyncio.to_thread(
            _get, self.url + "/api/v1/targets?state=active"
        )
        if result.get("status") != "success":
            raise ExecutionError("Prometheus target query failed")
        by_address = {
            value["labels"]["instance"]: node for node, value in self.targets.items()
        }
        targets = {
            by_address[target["labels"]["instance"]]: target
            for target in result["data"]["activeTargets"]
            if target.get("labels", {}).get("job") == self.JOB
            and target["labels"].get("instance") in by_address
        }
        return targets

    async def wait_for_baseline(self, nodes=None):
        if not self.targets:
            raise ExecutionError("Managed Prometheus has no metrics targets")
        seen = {node: set() for node in (self.targets if nodes is None else nodes)}
        while True:
            for node, target in (await self._fetch_scrape_status()).items():
                if node in seen and target["health"] == "up":
                    seen[node].add(target["lastScrape"])
            if all(len(times) >= 2 for times in seen.values()):
                self.baseline_ready = True
                self.metadata.setdefault("baseline_at", timestamp())
                self._save()
                self.records.emit(
                    event="collection_ready", url=self.url, targets=sorted(self.targets)
                )
                return
            await asyncio.sleep(0.05)

    async def wait_for_final_scrape(self, nodes=None):
        if not self.baseline_ready:
            return  # Setup never completed; do not claim a fully collected experiment.
        selected = set(self.targets if nodes is None else nodes)
        marker = timestamp()
        async with asyncio.timeout(max(5, 2 * self.interval + 2)):
            while True:
                if self.child.returncode is not None:
                    raise ExecutionError("Prometheus exited before final collection")
                for process in self.supervisor.process_identities():
                    if (
                        process.instance_id in self.targets
                        and self.supervisor.child(process.process_name).returncode
                        is not None
                    ):
                        raise ExecutionError(
                            f"{process.process_name} exited before final collection"
                        )
                targets = await self._fetch_scrape_status()
                if selected <= targets.keys() and all(
                    targets[node]["health"] == "up"
                    and _epoch(targets[node]["lastScrape"]) >= _epoch(marker)
                    for node in selected
                ):
                    self.metadata.setdefault("final_scrapes", {}).update(
                        {node: targets[node]["lastScrape"] for node in selected}
                    )
                    self.records.emit(
                        event="collection_finished",
                        after=marker,
                        scrapes=self.metadata["final_scrapes"],
                    )
                    self._save()
                    return
                await asyncio.sleep(0.05)

    def write_collection_result(self, errors):
        if not self.directory.exists():
            return
        self.metadata.update(
            complete=self.baseline_ready
            and "final_scrapes" in self.metadata
            and not errors,
            finished_at=timestamp(),
            errors=list(errors),
        )
        self._save()

    async def close(self):
        return await self.supervisor.stop(["prometheus"], 5)
