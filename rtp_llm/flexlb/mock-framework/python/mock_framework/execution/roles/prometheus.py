"""One owned Prometheus per run; configuration and TSDB stay in the run directory."""

import asyncio
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import yaml

from ...config.validation import duration, fields, text
from .. import ExecutionError
from ..step_trace import timestamp, write_json


def prepare_prometheus(raw, base):
    fields(
        raw,
        {"binary", "scrape_interval", "startup_timeout"},
        {"binary"},
        where="launch.prometheus",
    )
    binary = (Path(base) / text(raw["binary"], "prometheus.binary")).resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ExecutionError("prometheus.binary must be an executable file")
    interval = duration(raw.get("scrape_interval", "1s"), positive=True) * 1000
    if interval < 1 or interval != int(interval):
        raise ExecutionError(
            "Prometheus scrape_interval must be an integral number of milliseconds, at least 1ms"
        )
    startup = raw.get("startup_timeout", "30s")
    duration(startup, positive=True)
    return {
        "binary": str(binary),
        "scrape_interval": f"{int(interval)}ms",
        "startup_timeout": startup,
    }


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

    def __init__(self, supervisor, records, recipe):
        self.supervisor, self.records, self.recipe = supervisor, records, recipe
        self.directory = records.directory / "prometheus"
        self.startup_timeout = float(duration(recipe["startup_timeout"]))
        self.interval = float(duration(recipe["scrape_interval"]))
        self.url, self.child = None, None
        self.targets, self.baselined = {}, False
        self.metadata = {"job": self.JOB, "recipe": recipe, "complete": False}

    def _save(self):
        write_json(self.directory / "collection.json", self.metadata)

    async def start(self):
        self.directory.mkdir()
        write_json(self.directory / "targets.json", [])
        config = {
            "global": {
                "scrape_interval": self.recipe["scrape_interval"],
                "scrape_timeout": self.recipe["scrape_interval"],
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
            self.recipe["binary"],
            "--config.file=" + str(self.directory / "prometheus.yaml"),
            "--web.listen-address=127.0.0.1:0",
            "--storage.tsdb.path=" + str(self.directory / "data"),
            "--log.format=json",
        ]
        self.metadata["command"] = command
        self._save()
        self.child = await self.supervisor.spawn(
            "prometheus",
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

    def discover(self, node, directory):
        try:
            endpoint = json.loads(
                (directory / "endpoints.json").read_text(encoding="utf-8")
            )
            backend = endpoint.get("metrics_backend", "prometheus")
            if backend != "prometheus":
                raise ExecutionError(
                    f"{node}: managed Prometheus requires metrics_backend=prometheus, got {backend!r}"
                )
            address = urllib.parse.urlsplit(endpoint["metrics"])
            if (
                endpoint.get("id") != node
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
                    if k != node
                )
            ):
                raise ValueError("Missing or duplicate instance address")
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ExecutionError(
                f"{node}: managed collection requires endpoints.json with id, unique rpc and metrics URL"
            ) from error
        self.targets[node] = {
            "targets": [address.netloc],
            "labels": {
                "instance": rpc,
                "__scheme__": address.scheme,
                "__metrics_path__": address.path or "/metrics",
            },
        }
        write_json(self.directory / "targets.json", list(self.targets.values()))
        self.metadata["targets"] = dict(self.targets)
        self._save()

    def retire(self, node):
        self.targets.pop(node)
        self.metadata.get("final_scrapes", {}).pop(node, None)
        write_json(self.directory / "targets.json", list(self.targets.values()))
        self.metadata["targets"] = dict(self.targets)
        self._save()

    async def _scrapes(self):
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

    async def baseline(self, nodes=None):
        if not self.targets:
            raise ExecutionError("Managed Prometheus has no metrics targets")
        seen = {node: set() for node in (self.targets if nodes is None else nodes)}
        while True:
            for node, target in (await self._scrapes()).items():
                if node in seen and target["health"] == "up":
                    seen[node].add(target["lastScrape"])
            if all(len(times) >= 2 for times in seen.values()):
                self.baselined = True
                self.metadata.setdefault("baseline_at", timestamp())
                self._save()
                self.records.emit(
                    event="collection_ready", url=self.url, targets=sorted(self.targets)
                )
                return
            await asyncio.sleep(0.05)

    async def final_scrape(self, nodes=None):
        if not self.baselined:
            return  # Setup never completed; do not claim a fully collected experiment.
        selected = set(self.targets if nodes is None else nodes)
        marker = timestamp()
        async with asyncio.timeout(max(5, 2 * self.interval + 2)):
            while True:
                if self.child.returncode is not None:
                    raise ExecutionError("Prometheus exited before final collection")
                for node, child in self.supervisor.children:
                    if node in self.targets and child.returncode is not None:
                        raise ExecutionError(f"{node} exited before final collection")
                targets = await self._scrapes()
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

    def finish(self, errors):
        if not self.directory.exists():
            return
        self.metadata.update(
            complete=self.baselined and "final_scrapes" in self.metadata and not errors,
            finished_at=timestamp(),
            errors=list(errors),
        )
        self._save()

    async def close(self):
        return await self.supervisor.stop(["prometheus"], 5)
