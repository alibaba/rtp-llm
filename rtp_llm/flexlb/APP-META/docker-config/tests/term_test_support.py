"""Local TERM fixtures using this branch's smoke protocol builders and Java mock."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import threading
import time
from types import SimpleNamespace
import urllib.request

import grpc

from flexlb_smoke_base import FlexLBSmokeBase
from online_eval import proto_utils

FLEXLB = Path(__file__).resolve().parents[3]
API_JAR = FLEXLB / "flexlb-api/target/flexlb-api-1.0.0-SNAPSHOT.jar"
MOCK_JAR = FLEXLB / "flexlb-mock-engine/target/flexlb-mock-engine-1.0.0-SNAPSHOT-all.jar"
LOCAL_HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
JAVA_MODULE_OPTS = ["--add-modules", "ALL-SYSTEM"] + [
    f"--add-opens={module}/{package}=ALL-UNNAMED"
    for module, package in (
        ("java.base", "java.lang"), ("java.base", "java.lang.invoke"),
        ("java.base", "java.util"), ("java.base", "java.util.concurrent"),
        ("java.base", "jdk.internal.misc"), ("java.base", "java.nio"),
        ("java.base", "sun.nio.ch"), ("java.instrument", "sun.instrument"),
    )
]


def resolve_java21():
    home = os.environ.get("JAVA_HOME")
    java = str(Path(home) / "bin/java") if home else shutil.which("java")
    if not java:
        raise RuntimeError("Set JAVA_HOME to a JDK 21 installation")
    return java


def http_get_json(url):
    try:
        with LOCAL_HTTP.open(url, timeout=2) as response:
            return json.load(response)
    except (OSError, ValueError):
        return None


def http_post_json(url, payload):
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with LOCAL_HTTP.open(request, timeout=2) as response:
            return response.status, json.load(response)
    except (OSError, ValueError):
        return None, None


def wait_for(predicate, timeout, interval=0.1):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def free_ports(start, count):
    for base in range(start, start + 1000, 10):
        sockets = []
        try:
            for port in range(base, base + count):
                sock = socket.socket()
                sockets.append(sock)
                sock.bind(("127.0.0.1", port))
            return base
        except OSError:
            pass
        finally:
            for sock in sockets:
                sock.close()
    raise RuntimeError("No free test port block")


def test_config():
    return {
        "schemaVersion": 1,
        "scheduler": {"type": "QUEUE", "queueTimeoutMs": 60000,
                      "ordering": {"type": "FIFO"}},
        "dispatcher": {"type": "BATCH", "maxRequests": 1, "maxCollectionWaitMs": 0},
        "router": {"roles": {"decode": {"availability": {"maxEngineRequests": 1}}}},
    }


class MockEngine:
    def __init__(self, root):
        self.root = root
        self.process = None

    def start(self):
        self.root.mkdir(parents=True)
        master_port = free_ports(18080, 3)
        mock_port = free_ports(55150, 3)
        endpoints = self.root / "endpoints.json"
        performance = self.root / "performance.json"
        # 24s exceeds the quiet period plus Python gRPC's idle-connection close delay.
        performance.write_text(json.dumps({"block_size": 1024, "sleep_scale": 1,
                                          "prefill": {"fixed_ms": 100},
                                          "decode": {"per_token_ms": 12}}))
        master_config = self.root / "master-config.json"
        master_config.write_text(json.dumps({"zone_process_setting": {"process_info": {
            "envs": [["FLEXLB_CONFIG", json.dumps(test_config())]]}}}))
        command = [resolve_java21(), "-Xms256m", "-Xmx512m", "-jar", str(MOCK_JAR),
                   "--n-prefill", "1", "--n-decode", "1",
                   "--base-grpc-port", str(mock_port + 1),
                   "--event-loop-threads", "2", "--completion-threads", "2",
                   "--performance", str(performance), "--master-config", str(master_config),
                   "--endpoint-file", str(endpoints)]
        (self.root / "command.json").write_text(json.dumps(command))
        with (self.root / "mock.log").open("w") as log:
            self.process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        if not wait_for(lambda: endpoints.is_file() and http_get_json(
                f"http://127.0.0.1:{mock_port}/snapshot") is not None, 30):
            raise RuntimeError(f"Mock failed to start: {self.root / 'mock.log'}")
        self.endpoints = endpoints
        return SimpleNamespace(master_http_port=master_port,
                               master_management_port=master_port + 1,
                               mock_http_port=mock_port,
                               master_http=lambda path: f"http://127.0.0.1:{master_port}{path}")

    def master_env(self):
        return dict(json.loads(self.endpoints.read_text())["env"],
                    HIPPO_ROLE="flexlb-term-test",
                    FLEXLB_GRPC_EXECUTOR_CORE_SIZE="4", FLEXLB_GRPC_EXECUTOR_MAX_SIZE="4",
                    OTEL_TRACE_SKIP_PATTERN=".*", OTEL_EXPORTER_OTLP_ENDPOINT="none")

    def teardown(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)


class StreamHandle:
    def __init__(self, call):
        self.call = call
        self.snap = SimpleNamespace(outputs=[], completed=False, error=None,
                                    stream_error_code=None, terminated=False)
        self.thread = threading.Thread(target=self.consume, daemon=True)
        self.thread.start()

    def consume(self):
        try:
            for output in self.call:
                self.snap.outputs.append(output)
                if output.HasField("error_info"):
                    self.snap.stream_error_code = output.error_info.error_code
                if any(output.flatten_output.finished):
                    self.snap.completed = True
        except Exception as error:
            self.snap.error = repr(error)
        finally:
            self.snap.terminated = True

    def wait_end(self, timeout):
        self.thread.join(timeout)
        if self.thread.is_alive():
            self.call.cancel()
            self.thread.join(5)
            return False
        return True


class EngineOps:
    def __init__(self, host, master_port, mock_port, management_port):
        proto_utils.DEFAULT_OUT_DIR = Path(os.environ["FLEXLB_EVAL_PROTO_OUT"])
        self.builder = FlexLBSmokeBase(SimpleNamespace(request_id_base=20000))
        self.channel = grpc.insecure_channel(f"{host}:{master_port + 2}")
        self.stub = self.builder.schedule_pb2_grpc.FlexlbServiceStub(self.channel)
        self.mock_url = f"http://{host}:{mock_port}"
        self.channels = []
        self.recovery_id = 19990

    def schedule(self, request_id, timeout_s=30, **kwargs):
        return self.stub.Schedule(self.builder._build_schedule_request(request_id, **kwargs),
                                  timeout=timeout_s)

    def start_stream(self, response, request_id):
        target = self.builder._role_addr(response, "PREFILL")
        if not target:
            raise RuntimeError("Missing prefill address")
        channel = grpc.insecure_channel(target)
        self.channels.append(channel)
        stub = self.builder.pb2_grpc.RpcServiceStub(channel)
        return StreamHandle(stub.FetchResponse(
            self.builder.pb2.FetchRequestPB(request_id=request_id), timeout=60))

    def verify_recovery(self):
        try:
            self.recovery_id += 1
            response = self.schedule(self.recovery_id, output_len=1, input_len=1024, timeout_s=20)
            if not response.success:
                return False, str(response)
            stream = self.start_stream(response, self.recovery_id)
            ok = stream.wait_end(5) and stream.snap.completed and not stream.snap.error
            return ok, str(stream.snap.error)
        except grpc.RpcError as error:
            return False, str(error)

    def snapshot(self):
        return http_get_json(self.mock_url + "/snapshot") or {}

    def close(self):
        self.channel.close()
        for channel in self.channels:
            channel.close()
