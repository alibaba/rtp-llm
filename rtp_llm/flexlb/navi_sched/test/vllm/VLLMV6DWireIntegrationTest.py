"""Verify native Agent serialization and V6D updates in a real local Master."""

import ctypes
import json
import os
import re
import select
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from navi_sched.adapters.vllm.VLLMBlockHash import block_hash_to_uint64
from navi_sched.adapters.vllm.VLLMCachePolicy import VLLMCachePolicy, VLLMCacheState
from navi_sched.adapters.vllm.VLLMEngineObserver import VLLMEngineObserver

from navi_sched import NaviSchedAgent, NaviSchedClient, NaviSchedMaster
from navi_sched import _navi_sched_core as core


class SnapshotChanges:
    """Wait for native snapshot writes using Linux directory notifications."""

    def __init__(self, root):
        self.root = root
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.fd = self.libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if self.fd < 0:
            raise OSError(ctypes.get_errno(), "inotify_init1")
        self.watched = set()
        self.add_directories()

    def add_directories(self):
        for path in [self.root, *self.root.rglob("*")]:
            if path.is_dir() and path not in self.watched:
                result = self.libc.inotify_add_watch(
                    self.fd, os.fsencode(path), 0x00000100 | 0x00000080 | 0x00000008
                )
                if result < 0:
                    raise OSError(ctypes.get_errno(), str(path))
                self.watched.add(path)

    def wait(self, matches, timeout=20):
        deadline = time.monotonic() + timeout
        last = ""
        while True:
            self.add_directories()
            candidates = []
            for path in self.root.rglob("*_snapshot"):
                try:
                    if not path.is_file():
                        continue
                    candidates.append((path.stat().st_mtime_ns, path))
                except FileNotFoundError:
                    pass
            for _, path in sorted(candidates, reverse=True):
                try:
                    text = path.read_text()
                except FileNotFoundError:
                    continue
                if "Master\n" not in text:
                    continue
                last = text
                if matches(text):
                    return text
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.fd], [], [], remaining)[0]:
                raise AssertionError(
                    "native Master did not receive V6D state:\n" + last[-12000:]
                )
            try:
                os.read(self.fd, 65536)
            except BlockingIOError:
                pass

    def close(self):
        os.close(self.fd)


class VLLMV6DWireIntegrationTest(unittest.TestCase):
    def test_full_prefix_uses_local_objects_and_one_remote_peer(self):
        original_dir = os.getcwd()
        with tempfile.TemporaryDirectory(prefix="navi-v6d-prefix-") as directory:
            os.chdir(directory)
            notifications = SnapshotChanges(Path(directory))
            agents = []
            master = client = None
            try:
                for index in range(3):
                    agent = NaviSchedAgent(
                        "v6d_prefix_prefill",
                        server_ip="127.0.0.1",
                        server_port=17100 + index,
                        total_blocks=100,
                        block_size=16,
                        linear_step=1,
                        hybrid_tail_required=True,
                        peer_cache_enabled=True,
                        latency_model_fingerprint="v6d-prefix-contract",
                        runtime_config={"thread_num": 2},
                        name="prefix_agent_" + str(index),
                        enable_kmon=False,
                        enable_snapshot=False,
                        wait_no_query=False,
                    ).init()
                    agent.report_cache_status(
                        -1,
                        16,
                        1,
                        True,
                        cache_key_type=1,
                        full_attention_query_drop_tokens=0,
                        mamba_replay_stop_shift=False,
                        mamba_retention_interval=-1,
                        mamba_align=False,
                    )
                    agents.append(agent)
                master = NaviSchedMaster(
                    "v6d_prefix_master",
                    runtime_config={
                        "thread_num": 2,
                        "local_nodes": [
                            dict(
                                biz_name="v6d_prefix_prefill",
                                grpc_port=a.grpc_port,
                                support_heartbeat=True,
                            )
                            for a in agents
                        ],
                    },
                    max_wait_schedule_count=1,
                    scheduler_name="max_cache_hit",
                    scheduler_config=json.dumps({"peer_cache_enabled": True}),
                    name="master_snapshot",
                    snapshot_dir=directory,
                    enable_kmon=False,
                    enable_snapshot=True,
                    wait_no_query=False,
                ).init()
                request = urllib.request.Request(
                    f"http://127.0.0.1:{master.http_port}/SchedulerConfigService/scheduler_config",
                    data=json.dumps(
                        dict(
                            update=True,
                            configJson=json.dumps({"peer_cache_enabled": True}),
                        )
                    ).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    config = json.load(response)
                self.assertTrue(config["accepted"])
                self.assertTrue(json.loads(config["configJson"])["peer_cache_enabled"])
                policy = VLLMCachePolicy(
                    block_size=16,
                    total_blocks=100,
                    linear_step=1,
                    hybrid_tail_required=True,
                    group_roles=("full", "tail"),
                    group_block_scales=(1, 1),
                    spec_names=("FullAttentionSpec", "MambaSpec"),
                )
                observers = []
                for agent in agents:
                    observer = VLLMEngineObserver.__new__(VLLMEngineObserver)
                    observer._bridge = core._VLLMAgentBridge(agent, pd_separated=True)
                    observer._cache_policy = policy
                    observer._cache_states = {1: VLLMCacheState(policy)}
                    observer._cache_lifecycle_sequences = {1: 0}
                    observer._cache_lifecycle_lock = threading.RLock()
                    observers.append(observer)
                hashes = [bytes([i]) * 32 for i in range(1, 8)]

                def publish(index, full_indexes, tail_indexes):
                    objects = [
                        dict(
                            tier="vineyard",
                            sealedResident=True,
                            meta={
                                "navi_sched_cache": json.dumps(
                                    dict(
                                        group_id=group,
                                        block_hashes=[hashes[i].hex() for i in indexes],
                                    )
                                )
                            },
                        )
                        for group, indexes in ((0, full_indexes), (1, tail_indexes))
                        if indexes
                    ]
                    observers[index].on_v6d_cache_snapshots(
                        [
                            json.dumps(
                                dict(peer_cache_read_available=True, objects=objects)
                            ).encode()
                            for _ in range(4)
                        ]
                    )

                def resident_counts(text, expected):
                    if text.count("    peer_cache_read_available: true\n") != 3:
                        return False
                    if text.count("    peer_cache_enabled: true\n") != 3:
                        return False
                    for index, (full, tail) in enumerate(expected):
                        node = next(
                            (
                                value
                                for value in re.split(r"(?m)(?=^    id: )", text)
                                if f"    server_address: 127.0.0.1:{17100 + index}\n"
                                in value
                            ),
                            "",
                        )
                        cache = re.search(r"v6d_cache:\n((?:      .*\n)+)", node)
                        if (
                            cache is None
                            or f"      full_ready_block_count: {full}\n" not in cache[1]
                            or f"      tail_ready_block_count: {tail}\n" not in cache[1]
                        ):
                            return False
                    return True

                publish(0, [0, 1], [])
                publish(1, [2, 3], [])
                publish(2, [4, 5], [5])
                notifications.wait(
                    lambda text: resident_counts(text, [(2, 0), (2, 0), (2, 1)])
                )
                client = NaviSchedClient(
                    "v6d_prefix_client",
                    master_biz_name="v6d_prefix_master",
                    runtime_config={
                        "thread_num": 2,
                        "local_nodes": [
                            dict(
                                biz_name="v6d_prefix_master",
                                grpc_port=master.grpc_port,
                                support_heartbeat=True,
                            )
                        ],
                    },
                    name="prefix_client",
                    enable_kmon=False,
                    enable_snapshot=False,
                    wait_no_query=False,
                ).init()

                def schedule(request_id):
                    return client.schedule(
                        112,
                        "v6d_prefix_prefill",
                        request_id=request_id,
                        block_size=16,
                        block_hashes=[block_hash_to_uint64(h) for h in hashes],
                        timeout_ms=5000,
                        route_only=True,
                    )

                # Each peer holds a different Full segment. No receiver can
                # reach the sole Linear checkpoint with one remote Full read.
                self.assertEqual(0, schedule(100).predicted_cache_hit_tokens)

                # Receiver A can combine its first segment with one supplier B.
                publish(1, [2, 3, 4, 5], [5])
                notifications.wait(
                    lambda text: resident_counts(text, [(2, 0), (4, 1), (2, 1)])
                )
                decision = schedule(101)
                self.assertEqual(96, decision.predicted_cache_hit_tokens)
                self.assertIn(
                    decision.target.address, ("127.0.0.1:17100", "127.0.0.1:17101")
                )

                # Full and Linear groups may still come from different peers.
                publish(0, list(range(6)), [])
                publish(1, [], [5])
                publish(2, [], [])
                notifications.wait(
                    lambda text: resident_counts(text, [(6, 0), (0, 1), (0, 0)])
                )
                self.assertEqual(96, schedule(102).predicted_cache_hit_tokens)
            finally:
                if client is not None:
                    client.stop()
                if master is not None:
                    master.stop()
                for agent in reversed(agents):
                    agent.stop()
                notifications.close()
                os.chdir(original_dir)

    def test_native_switch_residency_and_eviction_reach_master(self):
        original_dir = os.getcwd()
        with tempfile.TemporaryDirectory(prefix="navi-v6d-wire-") as directory:
            os.chdir(directory)
            root = Path(directory)
            notifications = SnapshotChanges(root)
            agents = []
            master = None
            client = None
            try:
                for enabled in (False, True):
                    agent = NaviSchedAgent(
                        "v6d_wire_prefill",
                        server_ip="127.0.0.1",
                        server_port=17000 + int(enabled),
                        total_blocks=100,
                        block_size=16,
                        linear_step=1,
                        hybrid_tail_required=True,
                        peer_cache_enabled=enabled,
                        runtime_config={"thread_num": 2},
                        name="wire_agent_" + str(int(enabled)),
                        enable_kmon=False,
                        enable_snapshot=False,
                        wait_no_query=False,
                    ).init()
                    agents.append(agent)
                master = NaviSchedMaster(
                    "v6d_wire_master",
                    runtime_config={
                        "thread_num": 2,
                        "local_nodes": [
                            dict(
                                biz_name="v6d_wire_prefill",
                                grpc_port=a.grpc_port,
                                support_heartbeat=True,
                            )
                            for a in agents
                        ],
                    },
                    max_wait_schedule_count=1,
                    scheduler_name="max_cache_hit",
                    scheduler_config=json.dumps({"peer_cache_enabled": True}),
                    name="master_snapshot",
                    snapshot_dir=directory,
                    enable_kmon=False,
                    enable_snapshot=True,
                    wait_no_query=False,
                ).init()
                agent = agents[1]
                agent.report_cache_status(
                    -1,
                    16,
                    1,
                    True,
                    cache_key_type=1,
                    full_attention_query_drop_tokens=0,
                    mamba_replay_stop_shift=False,
                    mamba_retention_interval=-1,
                    mamba_align=False,
                )
                bridge = core._VLLMAgentBridge(agent, pd_separated=True)
                policy = VLLMCachePolicy(
                    block_size=16,
                    total_blocks=100,
                    linear_step=1,
                    hybrid_tail_required=True,
                    group_roles=("full", "tail"),
                    group_block_scales=(1, 1),
                    spec_names=("FullAttentionSpec", "MambaSpec"),
                )
                observer = VLLMEngineObserver.__new__(VLLMEngineObserver)
                observer._bridge = bridge
                observer._cache_policy = policy
                observer._cache_states = {1: VLLMCacheState(policy)}
                observer._cache_lifecycle_sequences = {1: 0}
                observer._cache_lifecycle_lock = threading.RLock()
                rows = [
                    dict(
                        tier="vineyard",
                        sealedResident=True,
                        meta={
                            "navi_sched_cache": json.dumps(
                                dict(group_id=group, block_hashes=["01" * 32])
                            )
                        },
                    )
                    for group in range(2)
                ]
                snapshots = [
                    dict(peer_cache_read_available=True, objects=rows) for _ in range(4)
                ]
                observer.on_v6d_cache_snapshots(
                    [json.dumps(row).encode() for row in snapshots]
                )

                def switched_on(text):
                    flags = re.findall(
                        r"^    peer_cache_enabled: (true|false)$", text, re.M
                    )
                    return (
                        sorted(flags) == ["false", "true"]
                        and bool(
                            re.search(
                                r"v6d_cache:\n(?:      .*\n)*?      full_ready_block_count: 1\n",
                                text,
                            )
                        )
                        and "    peer_cache_read_available: true" in text
                    )

                text = notifications.wait(switched_on)
                self.assertIn("      tail_ready_block_count: 1", text)

                def resident(text, count):
                    blocks = re.split(r"(?m)(?=^    id: )", text)
                    block = next(
                        (
                            value
                            for value in blocks
                            if "    server_address: 127.0.0.1:17001\n" in value
                        ),
                        "",
                    )
                    cache = re.search(r"v6d_cache:\n((?:      .*\n)+)", block)
                    return (
                        cache is not None
                        and f"      full_ready_block_count: {count}\n" in cache.group(1)
                        and f"      tail_ready_block_count: {count}\n" in cache.group(1)
                        and "    peer_cache_read_available: false" in block
                    )

                # Local V6D remains usable when remote cache transfer is unavailable.
                snapshots = [
                    dict(peer_cache_read_available=False, objects=rows)
                    for _ in range(4)
                ]
                observer.on_v6d_cache_snapshots(
                    [json.dumps(row).encode() for row in snapshots]
                )
                notifications.wait(lambda text: resident(text, 1))
                client = NaviSchedClient(
                    "v6d_wire_client",
                    master_biz_name="v6d_wire_master",
                    runtime_config={
                        "thread_num": 2,
                        "local_nodes": [
                            dict(
                                biz_name="v6d_wire_master",
                                grpc_port=master.grpc_port,
                                support_heartbeat=True,
                            )
                        ],
                    },
                    name="wire_client",
                    enable_kmon=False,
                    enable_snapshot=False,
                    wait_no_query=False,
                ).init()
                hashes = [
                    block_hash_to_uint64(bytes.fromhex(value * 32))
                    for value in ("01", "02")
                ]
                decision = client.schedule(
                    32,
                    "v6d_wire_prefill",
                    request_id=1,
                    block_size=16,
                    block_hashes=hashes,
                    timeout_ms=5000,
                    route_only=True,
                )
                self.assertEqual("127.0.0.1:17001", decision.target.address)
                self.assertEqual(16, decision.predicted_cache_hit_tokens)
                snapshots[2] = dict(peer_cache_read_available=False, objects=[])
                observer.on_v6d_cache_snapshots(
                    [json.dumps(row).encode() for row in snapshots]
                )

                def evicted(text):
                    return resident(text, 0)

                text = notifications.wait(evicted)
                self.assertIn("      tail_ready_block_count: 0", text)
                decision = client.schedule(
                    32,
                    "v6d_wire_prefill",
                    request_id=2,
                    block_size=16,
                    block_hashes=hashes,
                    timeout_ms=5000,
                    route_only=True,
                )
                self.assertEqual(0, decision.predicted_cache_hit_tokens)
            finally:
                if client is not None:
                    client.stop()
                if master is not None:
                    master.stop()
                for agent in reversed(agents):
                    agent.stop()
                notifications.close()
                os.chdir(original_dir)


if __name__ == "__main__":
    unittest.main()
