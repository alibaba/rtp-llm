"""Real SDK and PACE contracts without importing the model/GPU smoke runner."""

import argparse
import json
import os
import subprocess
import tempfile
import time
from decimal import Decimal
from pathlib import Path
from urllib.request import urlopen

from rtp_llm.test.smoke.pace_fixture import runfile
from rtp_llm.test.smoke.remote_kvcm_server import RemoteKVCMServer


_TRANSPORT_COUNTERS = {
    "tcp": (
        "tairmempool_mp_server_tent_tcp_read_succ_bytes",
        "tairmempool_mp_server_tent_tcp_write_succ_bytes",
    ),
    "rdma": (
        "tairmempool_mp_server_tent_rdma_read_succ_bytes",
        "tairmempool_mp_server_tent_rdma_write_succ_bytes",
    ),
    "gdr": (
        "tairmempool_mp_server_gdr_read_success_bytes",
        "tairmempool_mp_server_gdr_write_success_bytes",
    ),
}
_AFT_COUNTER_GROUPS = {
    **_TRANSPORT_COUNTERS,
    "aft": (
        "tairmempool_mp_server_read_succ_count",
        "tairmempool_mp_server_write_succ_count",
    ),
    "tent": (
        "tairmempool_mp_server_tent_read_succ_count",
        "tairmempool_mp_server_tent_write_succ_count",
    ),
}
_AFT_PROVIDER_READ_BYTES = "tairmempool_mp_server_provider_read_total_bytes"


def read_counters(url, names):
    counters = {}
    with urlopen(url, timeout=5) as response:
        for line in response.read().decode().splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[0] in names:
                counters[fields[0]] = Decimal(fields[1])
    if counters.keys() != names:
        missing = sorted(names - counters.keys())
        raise RuntimeError(f"Missing PACE transport counters at {url}: {missing}")
    return counters


def check_transport(sample_counters, before, transport, counter_groups):
    deadline = time.monotonic() + 5
    counter_mode = "aft" if transport.startswith("aft-") else transport
    expected = {counter_mode}
    if transport == "gdr":
        expected.add("rdma")
    elif transport == "aft-gdr":
        expected.add("gdr")
    while True:
        after = sample_counters()
        delta = {
            mode: tuple(after[name] - before[name] for name in names)
            for mode, names in counter_groups.items()
        }
        active = {mode for mode, counts in delta.items() if counts != (0, 0)}
        read, written = delta[counter_mode]
        path_verified = True
        evidence = ""
        if transport == "gdr":
            path_verified = delta["gdr"] == delta["rdma"]
        elif transport == "aft-rdma":
            read_bytes = after[_AFT_PROVIDER_READ_BYTES] - before[_AFT_PROVIDER_READ_BYTES]
            path_verified = read_bytes > 0
            evidence = f" provider_read_bytes={read_bytes}"
        elif transport == "aft-gdr":
            read_bytes, write_bytes = delta["gdr"]
            path_verified = read_bytes > 0 and write_bytes == 0
            evidence = f" gdr_read_bytes={read_bytes}"
        if read > 0 and written > 0 and active == expected and path_verified:
            unit = "operations" if counter_mode == "aft" else "bytes"
            print(f"PASS: {transport} payload {unit} read={read} write={written}{evidence}")
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Expected {transport} payload I/O in both directions, got {delta}{evidence}")
        time.sleep(0.25)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk", required=True)
    transports = ("local", *_TRANSPORT_COUNTERS, "aft-rdma", "aft-gdr")
    parser.add_argument("--transport", choices=transports)
    args = parser.parse_args()
    server_path = runfile(
        os.environ["TEST_WORKSPACE"],
        "3rdparty/remote_kv_cache_manager/bin/kv_cache_manager_bin",
    ).parent.parent
    output = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd()))
    with tempfile.TemporaryDirectory(prefix="pace-smoke-", dir=os.environ.get("TEST_TMPDIR")) as directory:
        server = RemoteKVCMServer(str(server_path), {
            "PACE_REQUIRED": "true",
        }, str(Path(directory) / "logs"), str(output / "kvcm_logs"))
        try:
            fixture_config = server.pace_fixture.config
            transport = fixture_config.get("transport", args.transport or "local")
            if transport not in transports:
                parser.error(f"invalid PACE fixture transport {transport!r}")
            if args.transport is not None and args.transport != transport:
                parser.error(
                    f"--transport={args.transport} conflicts with PACE fixture transport {transport!r}"
                )
            metrics_url = fixture_config.get("consumer_metrics_url") if transport != "local" else None
            if transport != "local" and not metrics_url:
                parser.error("the PACE fixture needs consumer_metrics_url for transport verification")
            counter_groups = _TRANSPORT_COUNTERS
            if transport.startswith("aft-"):
                if fixture_config.get("client_env", {}).get("TAIR_MEMPOOL_ENABLE_TENT") != "0":
                    parser.error("native AFT requires a fixture with TENT disabled")
                counter_groups = _AFT_COUNTER_GROUPS
            provider_metrics_url = fixture_config.get("provider_metrics_url") if transport == "aft-rdma" else None
            if transport == "aft-rdma" and not provider_metrics_url:
                parser.error("the AFT RDMA fixture needs provider_metrics_url for native read verification")
            counter_names = {name for pair in counter_groups.values() for name in pair}

            def sample_counters():
                counters = read_counters(metrics_url, counter_names)
                if provider_metrics_url:
                    counters.update(read_counters(provider_metrics_url, {_AFT_PROVIDER_READ_BYTES}))
                return counters

            fault_socket = fixture_config.get("fault_socket")
            scenarios = [None]
            if "fault_socket" in fixture_config:
                if not isinstance(fault_socket, str) or not fault_socket or transport != "tcp":
                    parser.error("PACE SDK fault cases require a TCP fixture with a fault socket")
                scenarios = ["slow-read", "slow-write", "packet-loss"]

            if not server.start_server():
                raise RuntimeError("KVCM manager did not start")
            env = os.environ.copy()
            env.update(server.client_env())
            instance_id = f"pace_contract_{server.rpc_port}"
            config_path = Path(directory) / "client.json"
            command = [args.sdk, str(config_path), str(server.pace_fixture.storage_type)]
            if transport == "gdr":
                command.append("--gdr")
            elif transport == "aft-gdr":
                command.append("--aft-gdr")
            for scenario in scenarios:
                fault_args = ["--fault", scenario, fault_socket] if scenario else []
                case_id = f"{instance_id}_{scenario.replace('-', '_')}" if scenario else instance_id
                config = server.pace_fixture.sdk_config(server, case_id)
                config_path.write_text(json.dumps(config), encoding="utf-8")
                before = sample_counters() if metrics_url else None
                subprocess.run(command + fault_args, env=env, check=True, timeout=120)
                if before is not None:
                    check_transport(sample_counters, before, transport, counter_groups)
        finally:
            try:
                server.close()
            finally:
                server.copy_logs()


if __name__ == "__main__":
    main()
