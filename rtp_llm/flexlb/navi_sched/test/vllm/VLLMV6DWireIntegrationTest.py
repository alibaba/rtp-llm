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
from pathlib import Path
from types import SimpleNamespace

from navi_sched import NaviSchedAgent, NaviSchedMaster
from navi_sched import _navi_sched_core as core
from navi_sched.adapters.vllm.VLLMCachePolicy import VLLMCachePolicy, VLLMCacheState
from navi_sched.adapters.vllm.VLLMEngineObserver import VLLMEngineObserver


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
    def test_native_switch_residency_and_eviction_reach_master(self):
        original_dir = os.getcwd()
        with tempfile.TemporaryDirectory(prefix="navi-v6d-wire-") as directory:
            os.chdir(directory)
            root = Path(directory)
            notifications = SnapshotChanges(root)
            agents = []
            master = None
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
                observer.on_v6d_cache_snapshots(snapshots)

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
                snapshots[2] = dict(peer_cache_read_available=False, objects=[])
                observer.on_v6d_cache_snapshots(snapshots)

                def evicted(text):
                    return (
                        switched_on(text) is False
                        and bool(
                            re.search(
                                r"v6d_cache:\n(?:      .*\n)*?      full_ready_block_count: 0\n",
                                text,
                            )
                        )
                        and "    peer_cache_read_available: true" not in text
                    )

                text = notifications.wait(evicted)
                self.assertIn("      tail_ready_block_count: 0", text)
            finally:
                if master is not None:
                    master.stop()
                for agent in reversed(agents):
                    agent.stop()
                notifications.close()
                os.chdir(original_dir)


if __name__ == "__main__":
    unittest.main()
