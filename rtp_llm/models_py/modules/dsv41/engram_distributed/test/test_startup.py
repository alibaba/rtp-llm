# SPDX-License-Identifier: Apache-2.0
"""Startup topology and exchange with in-memory and real TCPStore clients."""

import base64
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from unittest import mock
from unittest.mock import patch
from uuid import UUID

from rtp_llm.models_py.modules.dsv41.engram_distributed import (
    EngramRuntime,
    EngramStartup,
)
from rtp_llm.models_py.modules.dsv41.engram_distributed import config as topology
from rtp_llm.models_py.modules.dsv41.engram_distributed import (
    exchange,
    resident,
    shard_publisher,
)
from rtp_llm.models_py.modules.dsv41.engram_distributed.placement import (
    plan_shards,
    segment_rows,
)


def fixture(n=2, p=3, ranks=2):
    group = dict(
        generation=str(UUID(int=123)), model_revision="test-revision", members=[]
    )
    for instance in range(n):
        shards = [
            dict(index=i, host_id=f"host-{i}", publisher_rank=pos)
            for pos, i in enumerate(range(instance, p, n))
        ]
        group["members"].append(
            dict(instance_id=f"i{instance}", world_size=ranks, shards=shards)
        )
    return group, publications(group)


def publications(group, memory_kind="host"):
    hosts = {s["index"]: s["host_id"] for m in group["members"] for s in m["shards"]}
    count = len(hosts)
    plan = plan_shards(count * 16384 - 7, 64, count, granularity=4096)
    locations = (0, 1, 2, 3) if memory_kind == "gpu" else (0,)
    documents = []
    for shard, host in sorted(hosts.items()):
        segments = []
        for location, start, end, padded in segment_rows(plan, shard, locations):
            segment = dict(row_begin=start, row_end=end, location=location)
            for role, width in (("weight", 64), ("scale", 2)):
                token = f"{shard}/{location}/{role}".encode().ljust(64, b"0")
                segment[role] = dict(
                    allocation_bytes=padded * width,
                    handle=base64.b64encode(token).decode(),
                )
            segments.append(segment)
        documents.append(
            dict(
                schema_version=4,
                shard=shard,
                shards=count,
                generation=group["generation"],
                model_revision=group["model_revision"],
                hash_abi="sha256:" + "b" * 64,
                host_id=host,
                memory_kind=memory_kind,
                granularity=4096,
                layers=[
                    dict(
                        layer_id=1, rows=plan.total_rows, head_dim=64, segments=segments
                    )
                ],
            )
        )
    return documents


def samples(_, manifest):
    return {
        t.name: dict(
            row_ids=sorted(
                {0, t.shape[0] // 2, t.shape[0] - 1}
                | {r for part in t.shards for r in (part.row_begin, part.row_end - 1)}
            ),
            sha256="sha256:" + "c" * 64,
        )
        for t in manifest.tensors
    }


def package(group, docs):
    return prepared_package(group, docs)[0]


def prepared_package(group, docs):
    return exchange._assemble_package(group, docs, None, sample_rows=samples)


def cohort_env(p, instances=2, index=0, timeout=3):
    return {
        "RTP_ENGRAM_STORAGE": "host_shared",
        topology.COHORT_ENV: ",".join(
            f"127.0.0.{i + 1}:{29500 + i}" for i in range(instances)
        ),
        topology.INDEX_ENV: str(index),
        topology.REVISION_ENV: "test-revision",
        topology.COUNT_ENV: str(p),
        topology.TIMEOUT_ENV: str(timeout),
        topology.CHECKPOINT_ENV: "/tmp/checkpoint",
        "RTP_ENGRAM_LOCAL_DEVICE": "0",
        "RTP_ENGRAM_LOCAL_NUMA": "0",
    }


def derived(env, world_size):
    """Descriptors whose host identity matches the group derived from env."""
    config = topology.build_group(env, world_size)
    return config, publications(config["group"])


class GroupTest(unittest.TestCase):
    def test_single_host_gpu_fabric_uses_world_store_and_all_local_devices(self):
        env = {
            "RTP_ENGRAM_STORAGE": "cuda_fabric",
            topology.REVISION_ENV: "test",
            topology.COHORT_ENV: "host-a",
        }
        config = topology.build_config(
            env,
            4,
            rank=2,
            gang_hosts=("host-a",) * 4,
            checkpoint="/fixture",
            local_device=2,
        )
        self.assertTrue(config["reuse_world_store"])
        self.assertEqual(config["options"].shards.shard_count, 1)
        self.assertEqual(config["gpu_devices"], (0, 1, 2, 3))
        documents = publications(config["group"], "gpu")
        _, prepared = prepared_package(config["group"], documents)
        self.assertEqual(prepared["manifest"].manifest.backend, "fabric_gpu")
        self.assertEqual(len(prepared["manifest"].manifest.tensors[0].shards), 4)

    def test_supported_static_shapes(self):
        for n, p, ranks in [(1, 2, 2), (2, 3, 2), (3, 5, 2), (5, 5, 1)]:
            group, docs = fixture(n, p, ranks)
            topology.validate_group(group)
            exchange._prepare_package(group, package(group, docs))

    def test_invalid_memberships_and_host_assignments(self):
        mutations = [
            lambda g: g["members"][1].update(instance_id="i0"),
            lambda g: g["members"][1].update(shards=[]),
            lambda g: g["members"][0]["shards"][0].update(publisher_rank=True),
            lambda g: g["members"][0]["shards"][0].update(index=1),
            lambda g: g["members"][0]["shards"][0].update(host_id="host-1"),
            lambda g: g["members"][0].update(world_size=0),
        ]
        for mutate in mutations:
            group, _ = fixture()
            mutate(group)
            with self.assertRaises((ValueError, exchange.ExchangeError)):
                topology.validate_group(group)

    def test_bounded_frames_and_invalid_generation(self):
        with self.assertRaisesRegex(exchange.ExchangeError, "MESSAGE_TOO_LARGE"):
            exchange.decode(b"x" * (exchange.MAX_MESSAGE + 1))
        with self.assertRaisesRegex(exchange.ExchangeError, "MESSAGE_TOO_LARGE"):
            exchange.encode({"value": "x" * exchange.MAX_MESSAGE})
        for frame in (
            b'{"instance_id":"a","instance_id":"b"}',
            b'{"value":NaN}',
            b'{"value":"\xff"}',
            b'{"value":' + b"[" * 2048 + b"]" * 2048 + b"}",
        ):
            with self.assertRaisesRegex(exchange.ExchangeError, "INVALID_MESSAGE"):
                exchange.decode(frame)
        for value in (None, 1, True, "not-a-uuid"):
            group, _ = fixture()
            group["generation"] = value
            with self.assertRaisesRegex(
                topology.ShardConfigError, "INVALID_GENERATION"
            ):
                topology.validate_group(group)

    def test_host_and_gpu_packages_decode_once_and_keep_host_identity(self):
        group, _ = fixture(n=2, p=2)
        for kind, handles in (("host", 4), ("gpu", 16)):
            with self.subTest(kind=kind):
                docs = publications(group, memory_kind=kind)
                wire, staged = prepared_package(group, docs)
                received = exchange.decode(exchange.encode(wire))
                with patch.object(
                    exchange, "encode", side_effect=AssertionError("second encoding")
                ):
                    self.assertEqual(exchange._prepare_package(group, received), staged)
                self.assertEqual(len(staged["capabilities"]), handles)
                self.assertTrue(
                    all(len(handle) == 64 for handle in staged["capabilities"].values())
                )
                manifest = staged["manifest"].manifest
                weight, scale = sorted(
                    manifest.tensors, key=lambda t: t.name, reverse=True
                )
                self.assertEqual(
                    [(p.host_id, p.row_begin, p.row_end) for p in weight.shards],
                    [(p.host_id, p.row_begin, p.row_end) for p in scale.shards],
                )
                self.assertEqual(
                    manifest.backend, "fabric_gpu" if kind == "gpu" else "fabric_egm"
                )

                changed = deepcopy(wire)
                changed["documents"][0]["layers"][0]["segments"][0]["weight"][
                    "handle"
                ] = "bad!"
                with self.assertRaisesRegex(ValueError, "invalid capability encoding"):
                    exchange._prepare_package(group, changed)
                changed = deepcopy(wire)
                changed["documents"][0]["layers"][0]["segments"][0]["row_begin"] += 1
                with self.assertRaisesRegex(ValueError, "descriptor geometry"):
                    exchange._prepare_package(group, changed)
                changed = deepcopy(wire)
                changed["documents"][0]["layers"][0]["segments"][0].pop("scale")
                with self.assertRaisesRegex(ValueError, "INVALID_FIELDS"):
                    exchange._prepare_package(group, changed)
                changed_group = deepcopy(group)
                changed_group["generation"] = str(UUID(int=999))
                with self.assertRaisesRegex(
                    ValueError, "generation/count/revision/hash mismatch"
                ):
                    exchange._prepare_package(changed_group, wire)
                changed_group = deepcopy(group)
                changed_group["members"][0]["shards"][0]["host_id"] = "unexpected-host"
                with self.assertRaisesRegex(
                    exchange.ExchangeError, "PACKAGE_HOST_MISMATCH"
                ):
                    exchange._prepare_package(changed_group, wire)


class MemoryStore:
    """Shared test Store with real blocking semantics and bounded waits."""

    def __init__(self):
        self.data = {}
        self.condition = threading.Condition()

    def set(self, key, value):
        with self.condition:
            self.data[key] = value
            self.condition.notify_all()

    def get(self, key):
        with self.condition:
            return self.data[key]

    def check(self, keys):
        with self.condition:
            return all(key in self.data for key in keys)

    def compare_set(self, key, expected, desired):
        with self.condition:
            current = self.data.get(key, b"")
            if current == expected:
                self.data[key] = desired
                self.condition.notify_all()
                return desired
            return current

    def wait(self, keys, timeout):
        deadline = time.monotonic() + timeout.total_seconds()
        with self.condition:
            while not all(k in self.data for k in keys):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("store timeout")
                self.condition.wait(remaining)

    def add(self, key, amount):
        with self.condition:
            result = int(self.data.get(key, b"0")) + amount
            self.data[key] = str(result).encode()
            self.condition.notify_all()
            return result

    def delete_key(self, key):
        with self.condition:
            return self.data.pop(key, None) is not None


class StartupTestCase(unittest.TestCase):
    def setUp(self):
        self.runtime = EngramRuntime()

    def prepare(self, store, *, install=None, **kwargs):
        return EngramStartup(
            store,
            publish=shard_publisher.publish,
            install=self.runtime.install if install is None else install,
            **kwargs,
        ).run()


class IntraInstanceTest(StartupTestCase):
    def _mock_publisher(self, docs):
        by_index = {d["shard"]: d for d in docs}
        stub = mock.patch.object(
            shard_publisher,
            "publish",
            lambda checkpoint, revision, host, generation, shard, count, **kw: by_index[
                shard
            ],
        )
        stub.start()
        self.addCleanup(stub.stop)

    def test_only_rank_zero_exchanges_and_all_ranks_receive_identical_backing(self):
        env = cohort_env(2)
        config, docs = derived(env, world_size=4)
        store = MemoryStore()
        calls = []

        def exchange_for_test(cfg, local, deadline, **kwargs):
            calls.append(local)
            return prepared_package(cfg["group"], docs)

        with tempfile.TemporaryDirectory() as directory, patch.object(
            exchange, "_exchange_group", exchange_for_test
        ), patch.object(
            exchange, "_parse_publications", wraps=exchange._parse_publications
        ) as prepare:
            self._mock_publisher(docs[:1])
            envs = [dict(env) for _ in range(4)]
            with ThreadPoolExecutor(4) as pool:
                futures = [
                    pool.submit(
                        self.prepare,
                        store,
                        rank=i,
                        world_size=4,
                        environ=envs[i],
                    )
                    for i in range(4)
                ]
                results = [f.result() for f in futures]
            self.assertEqual(len(calls), 1)
            self.assertEqual(prepare.call_count, 4)
            self.assertTrue(all(r is None for r in results))
            self.assertFalse(
                {
                    k: v
                    for k, v in store.data.items()
                    if not k.startswith("engram/start_count/")
                }
            )
            self.assertEqual(envs, [env] * 4)
            self.assertEqual(self.runtime.prepared()[1].shards.shard_count, 2)
            staged, _ = self.runtime.prepared()
            self.assertIsNotNone(staged)
            self.assertEqual(staged["generation"], config["group"]["generation"])
            # Serving must not materialize any bundle files.
            self.assertEqual(list(Path(directory).glob("**/*.json")), [])

    def test_bad_p_fails_before_any_store_side_effect(self):
        store = MemoryStore()
        env = cohort_env(2)
        env[topology.COUNT_ENV] = "1"
        with self.assertRaisesRegex(ValueError, "STORAGE_HOST_COUNT"):
            self.prepare(store, rank=0, world_size=2, environ=env)
        self.assertFalse(
            {
                k: v
                for k, v in store.data.items()
                if not k.startswith("engram/start_count/")
            }
        )

    def test_inconsistent_worker_group_never_publishes_package(self):
        env = cohort_env(2, timeout=0.2)
        config, docs = derived(env, world_size=2)
        store = MemoryStore()
        prefix = "engram/" + config["group"]["generation"] + "/attempt/1/i0/"
        store.set(prefix + "config/1", b"wrong-group")
        self._mock_publisher(docs[:1])
        with self.assertRaisesRegex(
            exchange.ExchangeError, "INSTANCE_GROUP_CONFIG_MISMATCH"
        ):
            self.prepare(store, rank=0, world_size=2, environ=env)
        self.assertNotIn(prefix + "package", store.data)

    def test_self_published_shards_describe_their_own_host(self):
        from unittest.mock import patch

        env = cohort_env(2)
        env.pop("RTP_ENGRAM_LOCAL_DEVICE")
        env.pop("RTP_ENGRAM_LOCAL_NUMA")
        config, docs = derived(env, world_size=2)
        store = MemoryStore()
        published = []

        def publish(checkpoint, revision, host_id, generation, shard, shards, **kwargs):
            self.assertEqual(checkpoint, env[topology.CHECKPOINT_ENV])
            self.assertEqual(revision, config["group"]["model_revision"])
            self.assertEqual(generation, config["group"]["generation"])
            self.assertEqual(shards, 2)
            self.assertEqual(
                host_id, "127.0.0.1"
            )  # the shard's own host, not the member's
            self.assertIsNotNone(kwargs["deadline"])
            self.assertEqual(kwargs["device"], 1)
            self.assertEqual(kwargs["max_allocation_bytes"], 128 << 30)
            self.assertEqual(kwargs["budget_bytes"], 256 << 30)
            published.append((host_id, shard))
            return docs[shard]

        def exchange_for_test(cfg, local, deadline, **kwargs):
            return prepared_package(cfg["group"], docs)

        with patch.object(shard_publisher, "publish", publish), patch.object(
            exchange, "_exchange_group", exchange_for_test
        ), self.assertLogs(
            "rtp_llm.models_py.modules.dsv41.engram_distributed.exchange", "WARNING"
        ):
            with ThreadPoolExecutor(2) as pool:
                results = [
                    pool.submit(
                        self.prepare,
                        store,
                        rank=i,
                        world_size=2,
                        environ=dict(env),
                        local_device=1,
                    )
                    for i in range(2)
                ]
                [result.result() for result in results]
        self.assertEqual(published, [("127.0.0.1", 0)])
        staged, _ = self.runtime.prepared()
        self.assertIsNotNone(staged)
        self.assertEqual(staged["generation"], config["group"]["generation"])
        self.assertFalse(
            {
                k: v
                for k, v in store.data.items()
                if not k.startswith("engram/start_count/")
            }
        )  # descriptors went through Store, not files

    def test_unconfigured_models_do_nothing(self):
        self.assertIsNone(self.prepare(None, rank=0, world_size=8, environ={}))

    def test_publish_and_assembly_failures_wake_local_waiters(self):
        secret = "/private/checkpoint secret-capability=" + "x" * 4096
        for failure_at in ("publish", "assembly", "remote_verify"):
            with self.subTest(failure_at=failure_at):
                env = cohort_env(2, timeout=5)
                config, docs = derived(env, world_size=2)
                store = MemoryStore()
                if failure_at == "remote_verify":
                    original_error = exchange._StartupPeerFailure(
                        1,
                        "verify",
                        error_class="file_not_found",
                        cause_code="FILE_NOT_FOUND",
                    )
                else:
                    original_error = FileNotFoundError(secret)

                def allocate(*args, **kwargs):
                    if failure_at == "publish":
                        raise original_error
                    return docs[0]

                def exchange_for_test(*args, **kwargs):
                    raise original_error

                started = time.monotonic()
                with mock.patch.object(
                    shard_publisher, "publish", allocate
                ), mock.patch.object(exchange, "_exchange_group", exchange_for_test):
                    with ThreadPoolExecutor(2) as pool:
                        jobs = [
                            pool.submit(
                                self.prepare,
                                store,
                                rank=rank,
                                world_size=2,
                                environ=dict(env),
                            )
                            for rank in range(2)
                        ]
                        self.assertIs(jobs[0].exception(timeout=2), original_error)
                        peer = jobs[1].exception(timeout=2)
                self.assertLess(time.monotonic() - started, 2)
                self.assertIsInstance(peer, exchange._StartupPeerFailure)
                self.assertEqual(
                    (peer.rank, peer.phase),
                    (0, "publish" if failure_at == "publish" else "assembly"),
                )
                self.assertEqual(peer.error_class, "file_not_found")
                self.assertEqual(peer.cause_code, "FILE_NOT_FOUND")
                self.assertNotIn("private", str(peer))
                prefix = f'engram/{config["group"]["generation"]}/attempt/1/i0/'
                payload = store.get(prefix.rsplit("i0/", 1)[0] + "failure")
                self.assertLessEqual(len(payload), 512)
                self.assertNotIn(b"secret-capability", payload)
                self.assertNotIn(prefix + "package", store.data)
                self.assertIsNone(self.runtime.prepared())

    def test_first_failure_is_retained_even_when_success_keys_exist(self):
        store = MemoryStore()
        error = exchange.ExchangeError(
            "ASSEMBLY_FAILED", error_class="manifest", cause_code="INVALID_INTEGER"
        )
        exchange._publish_failure(store, "failed", 3, "assembly", error)
        first = store.get("failed")
        exchange._publish_failure(
            store, "failed", 0, "package", RuntimeError("secondary")
        )
        self.assertEqual(store.get("failed"), first)
        store.set("package", b"ready")
        with self.assertRaises(exchange._StartupPeerFailure) as raised:
            exchange._wait(
                store, ["package"], time.monotonic() + 1, failure_key="failed"
            )
        self.assertEqual(raised.exception.rank, 3)
        self.assertEqual(raised.exception.cause_code, "INVALID_INTEGER")

    def test_failure_reporting_cannot_replace_original_error(self):
        store = MemoryStore()
        original = FileNotFoundError("private filename")
        with mock.patch.object(
            store, "compare_set", side_effect=RuntimeError("store disconnected")
        ):
            try:
                try:
                    raise original
                except Exception as error:
                    exchange._publish_failure(store, "failed", 0, "publish", error)
                    raise
            except FileNotFoundError as actual:
                self.assertIs(actual, original)
        self.assertFalse(store.check(["failed"]))

    def test_failure_diagnostics_are_bounded_and_sanitized(self):
        store = MemoryStore()
        error = exchange.ExchangeError("secret-capability")
        # A caller can mutate exception attributes after construction; the
        # sender must still apply the shared allowlist at its own boundary.
        error.error_class = "private filename"
        error.cause_code = "secret-capability"
        exchange._publish_failure(store, "failed", 0, "publish", error)
        self.assertNotIn(b"private", store.get("failed"))
        self.assertNotIn(b"secret-capability", store.get("failed"))
        for payload in (
            b"x" * 513,
            exchange.encode(
                dict(rank=True, phase="publish", error_class=None, cause_code=None)
            ),
            exchange.encode(
                dict(rank=0, phase="private-path", error_class=None, cause_code=None)
            ),
        ):
            with self.subTest(payload=payload[:30]):
                store.set("malformed", payload)
                with self.assertRaisesRegex(
                    exchange.ExchangeError, "INVALID_INSTANCE_FAILURE"
                ):
                    exchange._wait(
                        store, ["never"], time.monotonic() + 1, failure_key="malformed"
                    )


class RankSpaceTest(unittest.TestCase):
    """A member is either a whole gang or one host of a gang shared by members."""

    def test_shared_gang_members_are_member_local(self):
        env = cohort_env(p=2, instances=2, index=1)
        env[topology.RANKS_ENV] = "4"
        group = topology.build_group(env, 8)["group"]
        self.assertEqual([m["world_size"] for m in group["members"]], [4, 4])
        self.assertEqual(
            [sh["publisher_rank"] for m in group["members"] for sh in m["shards"]],
            [0, 0],
        )

    def test_whole_gang_members_keep_global_ranks(self):
        env = cohort_env(p=2, instances=2, index=1)
        group = topology.build_group(env, 4)["group"]
        self.assertEqual([m["world_size"] for m in group["members"]], [4, 4])
        self.assertEqual(
            [sh["publisher_rank"] for m in group["members"] for sh in m["shards"]],
            [0, 0],
        )

    def test_ranks_per_instance_must_describe_the_gang(self):
        env = cohort_env(p=2, instances=2, index=1)
        env[topology.RANKS_ENV] = "3"
        with self.assertRaises(topology.ShardConfigError):
            topology.build_group(env, 8)


class DeploymentIdentityTest(unittest.TestCase):
    def AssertUuid(self, value):
        self.assertEqual(str(UUID(value)), value)

    def test_physical_p_matches_explicit_hosts(self):
        for p in (2, 3, 5, 18):
            env = cohort_env(p, instances=p)
            group = topology.build_group(env, 4)["group"]
            self.assertEqual(
                [m["shards"][0]["host_id"] for m in group["members"]],
                [f"127.0.0.{i+1}" for i in range(p)],
            )
        with self.assertRaisesRegex(topology.ShardConfigError, "MUST_MATCH_COHORT"):
            topology.build_group(cohort_env(4, instances=2), 8)

    def test_bare_hosts_take_the_shared_port(self):
        explicit = topology.build_group(cohort_env(2), 4)["group"]
        env = cohort_env(2)
        env[topology.COHORT_ENV] = "127.0.0.1,127.0.0.2"
        env[topology.PORT_ENV] = "29500"
        self.assertEqual(topology.build_group(env, 4)["group"], explicit)
        del env[topology.PORT_ENV]
        defaulted = topology.build_group(env, 4)["group"]
        self.assertEqual(defaulted["members"][0]["shards"][0]["host_id"], "127.0.0.1")
        self.assertEqual(
            topology.build_group(dict(env, **{topology.PORT_ENV: "29500"}), 4)["group"],
            explicit,
        )

    def test_instance_index_is_discovered_from_the_local_address(self):
        env = cohort_env(2)
        del env[topology.INDEX_ENV]
        # Do not assume the test host's hostname resolves to loopback.
        with mock.patch.object(topology, "_local_addresses", lambda: {"127.0.0.1"}):
            self.assertEqual(topology.build_group(env, 4)["index"], 0)
        with mock.patch.object(topology, "_local_addresses", lambda: {"127.0.0.2"}):
            self.assertEqual(topology.build_group(env, 4)["index"], 1)
        with mock.patch.object(topology, "_local_addresses", lambda: {"10.9.9.9"}):
            with self.assertRaisesRegex(
                topology.ShardConfigError, "INSTANCE_INDEX_REQUIRED"
            ):
                topology.build_group(env, 4)
        with mock.patch.object(
            topology, "_local_addresses", lambda: {"127.0.0.1", "127.0.0.2"}
        ):
            with self.assertRaisesRegex(
                topology.ShardConfigError, "AMBIGUOUS_LOCAL_ADDRESS"
            ):
                topology.build_group(env, 4)

    def test_generation_is_not_a_user_input(self):
        env = cohort_env(2)
        expected = topology.build_group(env, 4)["group"]
        env["RTP_ENGRAM_GENERATION"] = "ignored-legacy-setting"
        self.assertEqual(topology.build_group(env, 4)["group"], expected)
        env[topology.COHORT_ENV] = "127.0.0.1:10001,127.0.0.1:10002"
        with self.assertRaisesRegex(topology.ShardConfigError, "DISTINCT_STORAGE_HOST"):
            topology.build_group(env, 4)

    def test_wait_uses_bounded_operation_not_blocking_get(self):
        class Store:
            def get(self, key):
                raise AssertionError("unbounded get must not implement wait")

            def wait(self, keys, timeout):
                self.keys, self.timeout = keys, timeout.total_seconds()
                raise RuntimeError("timeout")

        store = Store()
        with self.assertRaisesRegex(exchange.ExchangeError, "WAIT_FAILED_OR_TIMEOUT"):
            exchange._wait(store, ["a", "b"], time.monotonic() + 0.1)
        self.assertEqual(store.keys, ["a", "b"])
        self.assertTrue(0 < store.timeout <= 0.1)


class TopologySourceTest(unittest.TestCase):
    def test_reuse_gang_topology_without_store_or_hostname_lookup(self):
        env = {
            "RTP_ENGRAM_STORAGE": "host_shared",
            topology.COUNT_ENV: "2",
            topology.REVISION_ENV: "rev",
            topology.CHECKPOINT_ENV: "/model",
        }
        with mock.patch.object(
            topology,
            "_local_addresses",
            side_effect=AssertionError("must reuse RTP IP"),
        ):
            cfg = topology.build_group(
                env, 4, rank=2, gang_hosts=["10.0.0.1"] * 2 + ["10.0.0.2"] * 2
            )
        self.assertEqual(cfg["index"], 1)
        self.assertEqual(cfg["rank_base"], 2)
        self.assertTrue(cfg["reuse_world_store"])
        self.assertEqual([m["world_size"] for m in cfg["group"]["members"]], [2, 2])

    def test_explicit_list_does_not_hide_multihost_gang(self):
        env = cohort_env(4, instances=4, index=3)
        # Second of two independent 2-host instances: local ranks start at zero.
        cfg = topology.build_group(
            env, 8, rank=5, gang_hosts=["127.0.0.3"] * 4 + ["127.0.0.4"] * 4
        )
        self.assertEqual(cfg["rank_base"], 4)
        self.assertEqual(cfg["group"]["members"][3]["world_size"], 4)
        self.assertFalse(cfg["reuse_world_store"])
        env[topology.INDEX_ENV] = "2"
        with self.assertRaisesRegex(topology.ShardConfigError, "INDEX_HOST_MISMATCH"):
            topology.build_group(
                env, 8, rank=5, gang_hosts=["127.0.0.3"] * 4 + ["127.0.0.4"] * 4
            )

    def test_invalid_rank_layouts_fail_before_allocation(self):
        for hosts, error in [
            (["a", "a", "b"], "UNEVEN"),
            (["a", "b", "a", "b"], "NONCONTIGUOUS"),
        ]:
            with self.assertRaisesRegex(topology.ShardConfigError, error):
                topology._gang_parts(hosts, len(hosts))
        with self.assertRaisesRegex(topology.ShardConfigError, "INVALID_GANG"):
            topology._gang_parts(["a"], 4)
        env = cohort_env(2)
        env[topology.RANKS_ENV] = "4"
        with self.assertRaisesRegex(
            topology.ShardConfigError, "RANKS_PER_INSTANCE_MISMATCH"
        ):
            topology.build_group(
                env, 4, gang_hosts=["127.0.0.1"] * 2 + ["127.0.0.2"] * 2
            )

    def test_platform_directory_is_topology_not_an_enable_switch(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "hosts").write_text("10.0.0.1\n10.0.0.2\n")
            env = {topology.GANGINFO_ENV: directory}
            self.assertIsNone(topology.build_group(env, 4))
            env.update(
                {
                    "RTP_ENGRAM_STORAGE": "host_shared",
                    topology.COUNT_ENV: "2",
                    topology.INDEX_ENV: "0",
                    topology.REVISION_ENV: "rev",
                    topology.CHECKPOINT_ENV: "/model",
                }
            )
            cfg = topology.build_group(
                env, 4, gang_hosts=["10.0.0.1"] * 2 + ["10.0.0.2"] * 2
            )
            self.assertTrue(cfg["reuse_world_store"])
            env[topology.COHORT_ENV] = "127.0.0.1,127.0.0.2"
            cfg = topology.build_group(env, 4)
            self.assertEqual(cfg["coordinator"]["host"], "127.0.0.1")
        with self.assertRaisesRegex(topology.ShardConfigError, "INVALID_GANGINFO"):
            topology._ganginfo_cohort({topology.GANGINFO_ENV: "/nonexistent/ganginfo"})
        self.assertIsNone(topology._ganginfo_cohort({}))

    def test_legacy_tls_and_descriptor_config_is_rejected(self):
        env = cohort_env(2)
        env["RTP_ENGRAM_TLS_DIR"] = "/not-present/pki"
        with self.assertRaisesRegex(
            topology.ShardConfigError, "REMOVED_ENGRAM_TLS_CONFIG"
        ):
            topology.build_config(env, 2, gang_hosts=["127.0.0.1", "127.0.0.2"])
        del env["RTP_ENGRAM_TLS_DIR"]
        hosts = ["127.0.0.1", "127.0.0.2"]
        self.assertNotIn("tls", topology.build_config(env, 2, gang_hosts=hosts))
        env["RTP_ENGRAM_LOCAL_DESCRIPTORS"] = "/obsolete/shard.json"
        with self.assertRaisesRegex(
            topology.ShardConfigError, "REMOVED_LOCAL_DESCRIPTOR"
        ):
            topology.build_config(env, 2, gang_hosts=hosts)

    def test_missing_framework_topology_does_not_guess(self):
        env = {"RTP_ENGRAM_STORAGE": "host_shared", topology.COUNT_ENV: "2"}
        with self.assertRaisesRegex(
            topology.ShardConfigError, "GANG_TOPOLOGY_REQUIRED"
        ):
            topology.build_group(env, 4)
        with self.assertRaisesRegex(
            topology.ShardConfigError, "STORAGE_HOST_COUNT_MUST_MATCH_COHORT"
        ):
            topology.build_group(env, 4, gang_hosts=["a"] * 4)

    def _annotation(self, directory, parts):
        payload = {name: {"name": name, "ip": ip} for name, ip in parts}
        body = json.dumps(payload).replace('"', '\\"')
        path = Path(directory) / "annotations"
        path.write_text('app.c2.io/biz-detail-ganginfo="%s"\n' % body)
        return str(path)

    def test_gang_annotation_discovers_single_host_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._annotation(
                directory,
                [
                    ("prefill_part0", "10.0.0.1"),
                    ("prefill_part1", "10.0.0.2"),
                    ("prefill_part2", "10.0.0.3"),
                    ("prefill_part3", "10.0.0.4"),
                ],
            )
            env = {
                "RTP_ENGRAM_STORAGE": "host_shared",
                topology.COUNT_ENV: "4",
                topology.REVISION_ENV: "rev",
                topology.CHECKPOINT_ENV: "/model",
                topology.ANNOTATION_ENV: path,
            }
            cfg = topology.build_group(env, 4, rank=0, gang_hosts=["10.0.0.1"] * 4)
            self.assertFalse(cfg["reuse_world_store"])
            self.assertEqual(cfg["index"], 0)
            self.assertEqual(
                cfg["coordinator"],
                {"host": "10.0.0.1", "port": topology.DEFAULT_COHORT_PORT},
            )
            self.assertEqual(
                [m["shards"][0]["host_id"] for m in cfg["group"]["members"]],
                ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"],
            )
            with self.assertLogs(topology.__name__, level="INFO") as logs:
                topology.build_group(env, 4, rank=0, gang_hosts=["10.0.0.1"] * 4)
            self.assertIn(
                "source=annotation instances=4 p=4 index=0 "
                "coordinator=10.0.0.1:%d local=10.0.0.1" % topology.DEFAULT_COHORT_PORT,
                logs.output[0],
            )
            # Every instance keeps its own rank space; only the address differs.
            cfg = topology.build_group(env, 4, rank=0, gang_hosts=["10.0.0.4"] * 4)
            self.assertEqual(cfg["index"], 3)
            self.assertEqual(cfg["rank_base"], 0)
            env[topology.PORT_ENV] = "29999"
            cfg = topology.build_group(env, 4, rank=0, gang_hosts=["10.0.0.1"] * 4)
            self.assertEqual(cfg["coordinator"]["port"], 29999)

    def test_gang_annotation_keeps_single_instance_multihost(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._annotation(
                directory, [("decode_part0", "10.0.0.1"), ("decode_part1", "10.0.0.2")]
            )
            env = {
                "RTP_ENGRAM_STORAGE": "host_shared",
                topology.COUNT_ENV: "2",
                topology.REVISION_ENV: "rev",
                topology.CHECKPOINT_ENV: "/model",
                topology.ANNOTATION_ENV: path,
            }
            cfg = topology.build_group(
                env, 8, rank=5, gang_hosts=["10.0.0.1"] * 4 + ["10.0.0.2"] * 4
            )
            self.assertTrue(cfg["reuse_world_store"])
            self.assertEqual(cfg["index"], 1)
            self.assertEqual(cfg["rank_base"], 4)
            self.assertEqual(
                [m["shards"][0]["host_id"] for m in cfg["group"]["members"]],
                ["10.0.0.1", "10.0.0.2"],
            )

    def test_gang_annotation_leaves_multihost_instances_alone(self):
        # A 4-pod gang of two 2-host instances keeps the per-instance layout:
        # the annotation cannot say how pods group, so it must not be guessed.
        with tempfile.TemporaryDirectory() as directory:
            path = self._annotation(
                directory,
                [("prefill_part%d" % i, "10.0.0.%d" % (i + 1)) for i in range(4)],
            )
            env = {
                "RTP_ENGRAM_STORAGE": "host_shared",
                topology.COUNT_ENV: "2",
                topology.REVISION_ENV: "rev",
                topology.CHECKPOINT_ENV: "/model",
                topology.ANNOTATION_ENV: path,
            }
            cfg = topology.build_group(
                env, 8, rank=0, gang_hosts=["10.0.0.1"] * 4 + ["10.0.0.2"] * 4
            )
            self.assertTrue(cfg["reuse_world_store"])
            self.assertEqual(
                [m["shards"][0]["host_id"] for m in cfg["group"]["members"]],
                ["10.0.0.1", "10.0.0.2"],
            )

    def test_gang_annotation_must_list_the_local_host(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._annotation(
                directory,
                [("prefill_part0", "10.0.0.1"), ("prefill_part1", "10.0.0.2")],
            )
            env = {
                "RTP_ENGRAM_STORAGE": "host_shared",
                topology.COUNT_ENV: "2",
                topology.REVISION_ENV: "rev",
                topology.CHECKPOINT_ENV: "/model",
                topology.ANNOTATION_ENV: path,
            }
            with self.assertRaisesRegex(
                topology.ShardConfigError, "LOCAL_HOST_NOT_IN_GANGINFO"
            ):
                topology.build_group(env, 4, gang_hosts=["10.0.0.9"] * 4)

    def test_gang_annotation_is_optional_but_not_ignored_when_configured(self):
        env = {
            "RTP_ENGRAM_STORAGE": "host_shared",
            topology.COUNT_ENV: "2",
            topology.REVISION_ENV: "rev",
            topology.CHECKPOINT_ENV: "/model",
        }
        with mock.patch.object(
            topology, "DEFAULT_ANNOTATION", "/nonexistent/annotations"
        ):
            with self.assertRaisesRegex(
                topology.ShardConfigError, "STORAGE_HOST_COUNT_MUST_MATCH_COHORT"
            ):
                topology.build_group(env, 4, gang_hosts=["10.0.0.1"] * 4)
        env[topology.ANNOTATION_ENV] = "/nonexistent/annotations"
        with self.assertRaisesRegex(topology.ShardConfigError, "INVALID_GANGINFO"):
            topology.build_group(env, 4, gang_hosts=["10.0.0.1"] * 4)

    def test_resolved_checkpoint_wins_over_an_object_store_uri(self):
        # Whale passes CHECKPOINT_PATH as oss://; the launcher resolves it to a
        # local fuse mount before the engine starts, and that is what we read.
        env = {
            "RTP_ENGRAM_STORAGE": "host_shared",
            topology.COUNT_ENV: "2",
            topology.REVISION_ENV: "rev",
            "CHECKPOINT_PATH": "oss://bucket/model/?OSS_ENDPOINT=internal",
        }
        hosts = ["10.0.0.1"] * 2 + ["10.0.0.2"] * 2
        cfg = topology.build_group(env, 4, gang_hosts=hosts, checkpoint="/mnt/fuse/abc")
        self.assertEqual(cfg["checkpoint"], "/mnt/fuse/abc")
        env[topology.CHECKPOINT_ENV] = "/explicit"
        cfg = topology.build_group(env, 4, gang_hosts=hosts, checkpoint="/mnt/fuse/abc")
        self.assertEqual(cfg["checkpoint"], "/explicit")

    def test_explicit_cohort_still_overrides_the_annotation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._annotation(
                directory,
                [("prefill_part%d" % i, "10.0.0.%d" % (i + 1)) for i in range(4)],
            )
            env = {
                "RTP_ENGRAM_STORAGE": "host_shared",
                topology.COUNT_ENV: "2",
                topology.REVISION_ENV: "rev",
                topology.CHECKPOINT_ENV: "/model",
                topology.ANNOTATION_ENV: path,
                topology.COHORT_ENV: "127.0.0.1,127.0.0.2",
            }
            cfg = topology.build_group(env, 4, gang_hosts=["127.0.0.1"] * 4)
            self.assertEqual(cfg["coordinator"]["host"], "127.0.0.1")
            self.assertEqual(len(cfg["group"]["members"]), 2)


class RevisionTest(unittest.TestCase):
    def test_revision_is_derived_from_the_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            index = Path(directory) / "model.safetensors.index.json"
            index.write_text('{"weight_map": {"a": "a.safetensors"}}')
            env = cohort_env(2)
            del env[topology.REVISION_ENV]
            env[topology.CHECKPOINT_ENV] = directory
            first = topology.build_group(env, 4)["group"]["model_revision"]
            self.assertTrue(first.startswith("ckpt@"))
            # Identical weights give identical identity, wherever they are mounted.
            other = cohort_env(2)
            del other[topology.REVISION_ENV]
            other[topology.CHECKPOINT_ENV] = directory
            self.assertEqual(
                first, topology.build_group(other, 4)["group"]["model_revision"]
            )
            index.write_text('{"weight_map": {"a": "b.safetensors"}}')
            self.assertNotEqual(
                first, topology.build_group(env, 4)["group"]["model_revision"]
            )
            # An explicit revision still wins, and a checkpoint without either
            # index is refused rather than given an accidental identity.
            explicit = dict(env, **{topology.REVISION_ENV: "release-1019"})
            self.assertEqual(
                topology.build_group(explicit, 4)["group"]["model_revision"],
                "release-1019",
            )
            empty = Path(directory) / "empty"
            empty.mkdir()
            with self.assertRaisesRegex(
                topology.ShardConfigError, "MODEL_REVISION_REQUIRED"
            ):
                topology.build_group(
                    dict(env, **{topology.CHECKPOINT_ENV: str(empty)}), 4
                )


class StartupScopeTest(StartupTestCase):
    def test_two_multihost_instances_each_publish_their_local_shards(self):
        base = cohort_env(4, instances=4)
        base.pop(topology.INDEX_ENV)
        group = topology.build_group(
            base, 4, gang_hosts=["127.0.0.1"] * 2 + ["127.0.0.2"] * 2
        )["group"]
        _, docs = derived(cohort_env(4, instances=4), world_size=2)
        cohort = MemoryStore()
        stores = [MemoryStore(), MemoryStore()]
        published, calls = [], []

        def publish(checkpoint, revision, host_id, generation, shard, shards, **kw):
            self.assertEqual(host_id, docs[shard]["host_id"])
            self.assertEqual(generation, docs[shard]["generation"])
            published.append(shard)
            return docs[shard]

        def exchange_for_test(cfg, local, deadline, **kwargs):
            calls.append(cfg["instance_id"])
            return exchange._exchange_metadata(
                cohort,
                cfg["group"],
                cfg["instance_id"],
                local,
                deadline,
                lambda documents: prepared_package(group, documents),
                lambda result: exchange._prepare_package(group, result),
                group_digest=exchange.digest(exchange.encode(cfg["group"])),
            )

        with mock.patch.object(shard_publisher, "publish", publish), patch.object(
            exchange, "_exchange_group", exchange_for_test
        ):
            with ThreadPoolExecutor(8) as pool:
                jobs = []
                for gang in range(2):
                    hosts = [f"127.0.0.{2*gang+1}"] * 2 + [f"127.0.0.{2*gang+2}"] * 2
                    for rank in range(4):
                        jobs.append(
                            pool.submit(
                                self.prepare,
                                stores[gang],
                                rank=rank,
                                world_size=4,
                                environ=dict(base),
                                gang_hosts=hosts,
                            )
                        )
                [job.result() for job in jobs]
        self.assertEqual(sorted(published), list(range(4)))
        self.assertEqual(sorted(calls), ["i0", "i1", "i2", "i3"])


class CohortStoreTest(unittest.TestCase):
    def run_member(
        self, store, group, docs, member, *, build=None, verify=None, timeout=2
    ):
        owned = {sh["index"] for sh in group["members"][member]["shards"]}
        return exchange._exchange_metadata(
            store,
            group,
            "i%d" % member,
            [doc for doc in docs if doc["shard"] in owned],
            time.monotonic() + timeout,
            build or (lambda documents: prepared_package(group, documents)),
            verify or (lambda result: exchange._prepare_package(group, result)),
            group_digest=exchange.digest(exchange.encode(group)),
        )

    def test_every_member_verifies_before_any_member_returns(self):
        group, docs = fixture()
        store = MemoryStore()
        verifying, release = threading.Event(), threading.Event()

        def verify(result):
            prepared = exchange._prepare_package(group, result)
            verifying.set()
            self.assertTrue(release.wait(1))
            return prepared

        with ThreadPoolExecutor(2) as pool:
            coordinator = pool.submit(self.run_member, store, group, docs, 0)
            peer = pool.submit(self.run_member, store, group, docs, 1, verify=verify)
            self.assertTrue(verifying.wait(1))
            self.assertFalse(coordinator.done())
            self.assertFalse(peer.done())
            release.set()
            self.assertEqual(coordinator.result(2), peer.result(2))
        self.assertFalse(store.data)  # all successful exchange keys were released

    def test_unknown_member_and_changed_republication_are_rejected(self):
        group, docs = fixture()
        store = MemoryStore()
        with self.assertRaisesRegex(exchange.ExchangeError, "UNKNOWN_INSTANCE"):
            exchange._exchange_metadata(
                store,
                group,
                "outsider",
                [],
                time.monotonic() + 1,
                None,
                None,
                group_digest=exchange.digest(exchange.encode(group)),
            )
        store.set("member/i1", b"old publication")
        with self.assertRaisesRegex(
            exchange.ExchangeError, "CONFLICTING_REPUBLICATION"
        ):
            self.run_member(store, group, docs, 1)

    def test_missing_member_and_missing_verification_are_bounded(self):
        group, docs = fixture()
        for prepublish in (False, True):
            with self.subTest(prepublish=prepublish):
                store = MemoryStore()
                if prepublish:
                    store.set(
                        "member/i1",
                        exchange.encode(
                            dict(
                                group_digest=exchange.digest(exchange.encode(group)),
                                documents=[docs[1]],
                            )
                        ),
                    )
                started = time.monotonic()
                with self.assertRaisesRegex(
                    exchange.ExchangeError, "INSTANCE_STARTUP_TIMEOUT"
                ):
                    self.run_member(store, group, docs, 0, timeout=0.15)
                self.assertLess(time.monotonic() - started, 0.8)
                self.assertNotIn("ready", store.data)

    def test_assembly_failures_wake_peer_with_safe_categories(self):
        for kind in ("checkpoint", "descriptor", "manifest"):
            with self.subTest(kind=kind):
                group, docs = fixture()
                store = MemoryStore()
                if kind == "checkpoint":

                    def build(documents):
                        raise FileNotFoundError("/private/checkpoint capability=secret")

                    expected = ("file_not_found", "FILE_NOT_FOUND")
                else:
                    if kind == "descriptor":
                        docs[1]["hash_abi"] = "sha256:" + "d" * 64
                        expected = ("validation", "VALIDATION_FAILED")
                    else:
                        docs[0]["layers"][0]["layer_id"] = "private-layer"
                        expected = ("manifest", "INVALID_INTEGER")
                    build = lambda documents: prepared_package(group, documents)
                with ThreadPoolExecutor(2) as pool:
                    coordinator = pool.submit(
                        self.run_member, store, group, docs, 0, build=build
                    )
                    peer = pool.submit(self.run_member, store, group, docs, 1)
                    self.assertIsNotNone(coordinator.exception(2))
                    error = peer.exception(2)
                self.assertIsInstance(error, exchange._StartupPeerFailure)
                self.assertEqual((error.error_class, error.cause_code), expected)
                self.assertLessEqual(len(store.get("failure")), 512)
                self.assertNotIn(b"private", store.get("failure"))
                self.assertNotIn(b"secret", store.get("failure"))
                self.assertNotIn("ready", store.data)

    def test_verification_failure_reaches_coordinator_before_release(self):
        group, docs = fixture()
        store = MemoryStore()
        original = ValueError("private verification details")

        def verify(result):
            raise original

        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(self.run_member, store, group, docs, 0)
            second = pool.submit(self.run_member, store, group, docs, 1, verify=verify)
            self.assertIs(second.exception(2), original)
            self.assertIsInstance(first.exception(2), exchange._StartupPeerFailure)
        self.assertNotIn("ready", store.data)

    def test_wrong_group_or_malformed_submission_never_builds_package(self):
        group, docs = fixture()
        for value, code in (
            ([], "INVALID_MESSAGE"),
            (dict(group_digest="wrong", documents=[docs[1]]), "GROUP_MISMATCH"),
        ):
            store = MemoryStore()
            store.set("member/i1", exchange.encode(value))
            with self.assertRaisesRegex(exchange.ExchangeError, code):
                self.run_member(store, group, docs, 0, timeout=0.1)
            self.assertNotIn("package", store.data)

    def test_ack_mismatch_never_releases_peers(self):
        group, docs = fixture()
        store = MemoryStore()
        original_set = store.set

        def corrupt(key, value):
            original_set(key, b"wrong" if key == "verified/i1" else value)

        with mock.patch.object(store, "set", corrupt), ThreadPoolExecutor(2) as pool:
            first = pool.submit(self.run_member, store, group, docs, 0)
            second = pool.submit(self.run_member, store, group, docs, 1)
            self.assertEqual(first.exception(2).code, "ACK_MISMATCH")
            self.assertEqual(second.exception(2).cause_code, "ACK_MISMATCH")
        self.assertNotIn("ready", store.data)

    def test_broken_logging_cannot_block_success_or_hide_original_failure(self):
        group, docs = fixture()
        for failed in (False, True):
            store = MemoryStore()
            original = OSError("private checkpoint")

            def build(documents):
                if failed:
                    raise original
                return prepared_package(group, documents)

            with mock.patch.object(
                logging.Logger, "log", side_effect=RuntimeError("log failed")
            ), mock.patch.object(
                logging.Logger, "error", side_effect=RuntimeError("log failed")
            ), ThreadPoolExecutor(
                2
            ) as pool:
                first = pool.submit(self.run_member, store, group, docs, 0, build=build)
                second = pool.submit(self.run_member, store, group, docs, 1)
                if failed:
                    self.assertIs(first.exception(2), original)
                    self.assertIsInstance(
                        second.exception(2), exchange._StartupPeerFailure
                    )
                else:
                    self.assertEqual(first.result(2), second.result(2))


class RealStoreTest(StartupTestCase):
    @classmethod
    def setUpClass(cls):
        # This is an integration gate, not optional coverage. A missing Torch
        # dependency must fail CI instead of reporting a successful skip.
        from torch.distributed import TCPStore

        cls.TCPStore = TCPStore

    def setUp(self):
        super().setUp()
        self.server = self.TCPStore(
            "127.0.0.1", 0, None, True, timedelta(seconds=20), wait_for_workers=False
        )

    def client(self):
        return self.TCPStore(
            "127.0.0.1", self.server.port, None, False, timedelta(seconds=20)
        )

    def tearDown(self):
        del self.server

    def test_cohort_reconnect_uses_fresh_attempt_and_rejects_wrong_group(self):
        from copy import deepcopy

        group, _ = fixture()
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            address = listener.getsockname()
        deadline = time.monotonic() + 3
        owner = exchange._open_store(
            group,
            "i0",
            address,
            deadline,
            group_digest=exchange.digest(exchange.encode(group)),
        )
        first = exchange._open_store(
            group,
            "i1",
            address,
            deadline,
            group_digest=exchange.digest(exchange.encode(group)),
        )
        owner.set("package", b"stale")
        self.assertEqual(first.get("package"), b"stale")
        second = exchange._open_store(
            group,
            "i1",
            address,
            deadline,
            group_digest=exchange.digest(exchange.encode(group)),
        )
        self.assertFalse(second.check(["package"]))
        wrong = deepcopy(group)
        wrong["model_revision"] = "different"
        with self.assertRaisesRegex(exchange.ExchangeError, "GROUP_MISMATCH"):
            exchange._open_store(
                wrong,
                "i1",
                address,
                deadline,
                group_digest=exchange.digest(exchange.encode(wrong)),
            )
        del second, first, owner

    def test_cohort_failure_reaches_real_client_and_releases_listener(self):

        group, docs = fixture()
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            address = listener.getsockname()
        original = FileNotFoundError("/private/checkpoint")

        def build(*_):
            raise original

        with patch.object(exchange, "_assemble_package", build), ThreadPoolExecutor(
            2
        ) as pool:
            futures = [
                pool.submit(
                    exchange._exchange_group,
                    dict(
                        group=group,
                        group_digest=exchange.digest(exchange.encode(group)),
                        instance_id="i%d" % member,
                        coordinator=dict(host=address[0], port=address[1]),
                        reuse_world_store=False,
                        checkpoint=None,
                    ),
                    [docs[0], docs[2]] if member == 0 else [docs[1]],
                    time.monotonic() + 3,
                    world_store=None,
                    attempt=1,
                )
                for member in range(2)
            ]
            self.assertIs(futures[0].exception(4), original)
            peer = futures[1].exception(4)
            self.assertEqual(peer.cause_code, "FILE_NOT_FOUND")
            self.assertNotIn("private", str(peer))
        with socket.socket() as client:
            client.settimeout(1)
            self.assertNotEqual(client.connect_ex(address), 0)

    def test_disjoint_waits_are_not_collectives_and_keep_per_call_deadline(self):
        a, b = self.client(), self.client()
        started = time.monotonic()
        with ThreadPoolExecutor(2) as pool:
            done = pool.submit(exchange._wait, a, ["ready"], time.monotonic() + 2)
            missing = pool.submit(
                exchange._wait,
                b,
                ["never"],
                time.monotonic() + 0.15,
                failure_key="absent_failure",
            )
            self.server.set("ready", b"1")
            done.result(timeout=3)
            with self.assertRaisesRegex(exchange.ExchangeError, "TIMEOUT"):
                missing.result(timeout=3)
        self.assertLess(time.monotonic() - started, 3)

    def test_reuse_world_store_ignores_previous_attempt(self):
        hosts = ["127.0.0.1"] * 2 + ["127.0.0.2"] * 2
        env = cohort_env(2, timeout=5)
        env.pop(topology.COHORT_ENV)
        env.pop(topology.INDEX_ENV)
        env.update(RTP_ENGRAM_LOCAL_DEVICE="0", RTP_ENGRAM_LOCAL_NUMA="0")
        cfg = topology.build_group(env, 4, gang_hosts=hosts)
        _, docs = derived(cohort_env(2), 2)
        for doc in docs:
            doc["generation"] = cfg["group"]["generation"]
        old = f'engram/{cfg["group"]["generation"]}/attempt/1/package'
        self.server.set(old, b"invalid old package")
        old_failure = old.rsplit("/", 1)[0] + "/failure"
        exchange._publish_failure(
            self.server, old_failure, 0, "publish", RuntimeError("old failure")
        )
        for rank in range(4):
            self.server.add(f"engram/start_count/{rank}", 1)
        original = exchange._assemble_package
        published = []

        def allocate(checkpoint, revision, host, generation, shard, count, **kwargs):
            published.append(shard)
            return docs[shard]

        clients = [self.client() for _ in range(4)]
        with patch.object(shard_publisher, "publish", allocate), patch.object(
            exchange,
            "_assemble_package",
            lambda g, d, c: original(g, d, c, sample_rows=samples),
        ), patch.object(
            exchange, "_open_store", side_effect=AssertionError("unexpected new Store")
        ), patch.object(
            exchange, "_exchange_metadata", wraps=exchange._exchange_metadata
        ) as metadata:
            with ThreadPoolExecutor(4) as pool:
                pending = [
                    pool.submit(
                        self.prepare,
                        clients[r],
                        rank=r,
                        world_size=4,
                        gang_hosts=hosts,
                        environ=dict(env),
                    )
                    for r in range(4)
                ]
                [p.result(timeout=8) for p in pending]
        self.assertEqual(sorted(published), [0, 1])
        self.assertEqual(metadata.call_count, 2)
        self.assertEqual(
            {call.args[2] for call in metadata.call_args_list}, {"i0", "i1"}
        )
        self.assertIsNotNone(self.runtime.prepared())
        self.assertEqual(self.server.get(old), b"invalid old package")
        self.assertTrue(self.server.check([old_failure]))
        prefix = f'engram/{cfg["group"]["generation"]}/attempt/2/'
        for key in (
            "package",
            "ready",
            "member/i0",
            "verified/i1",
            "departed/i0",
            "i0/package",
            "i1/package",
        ):
            self.assertFalse(self.server.check([prefix + key]), key)
        self.server.set("unrelated/world/key", b"alive")
        self.assertEqual(self.client().get("unrelated/world/key"), b"alive")
        for rank in range(4):
            self.assertEqual(self.server.get(f"engram/start_count/{rank}"), b"2")

    def test_two_cp4_instances_share_from_platform_gang_annotation(self):
        # One platform gang contains two independent compute worlds. Both
        # launchers start at world rank zero; each world has its own TCPStore.
        second = self.TCPStore(
            "127.0.0.1", 0, None, True, timedelta(seconds=20), wait_for_workers=False
        )
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations"
            members = {
                f"prefill_part{i}": {
                    "name": f"prefill_part{i}",
                    "ip": f"127.0.0.{i + 1}",
                }
                for i in range(2)
            }
            path.write_text(
                "app.c2.io/biz-detail-ganginfo="
                + json.dumps(json.dumps(members))
                + "\n"
            )
            env = cohort_env(2, timeout=10)
            env.pop(topology.COHORT_ENV)
            env.pop(topology.INDEX_ENV)
            env.update(
                {topology.ANNOTATION_ENV: str(path), topology.PORT_ENV: str(port)}
            )
            groups = [
                topology.build_config(env, 4, gang_hosts=[f"127.0.0.{i + 1}"] * 4)
                for i in range(2)
            ]
            self.assertEqual(groups[0]["group"], groups[1]["group"])
            self.assertEqual([g["index"] for g in groups], [0, 1])
            self.assertEqual([g["rank_base"] for g in groups], [0, 0])
            self.assertTrue(all(not g["reuse_world_store"] for g in groups))
            fixture_env = cohort_env(2)
            fixture_env[topology.COHORT_ENV] = f"127.0.0.1:{port},127.0.0.2:{port}"
            _, docs = derived(fixture_env, 4)
            stores = [self.server, second]
            clients = [
                [
                    self.TCPStore(
                        "127.0.0.1", server.port, None, False, timedelta(seconds=20)
                    )
                    for _ in range(4)
                ]
                for server in stores
            ]
            original = exchange._assemble_package
            published = []

            def allocate(
                checkpoint, revision, host, generation, shard, count, **kwargs
            ):
                self.assertEqual(host, docs[shard]["host_id"])
                self.assertEqual(generation, docs[shard]["generation"])
                self.assertEqual(count, 2)
                published.append(shard)
                return docs[shard]

            exchanged = []
            original_exchange = exchange._exchange_metadata

            def record_exchange(store, group, instance_id, *args, group_digest):
                # Retain only IDs: retaining Store arguments in a Mock would
                # keep the temporary listener alive after startup returns.
                exchanged.append(instance_id)
                return original_exchange(
                    store, group, instance_id, *args, group_digest=group_digest
                )

            with patch.object(shard_publisher, "publish", allocate), patch.object(
                exchange,
                "_assemble_package",
                lambda g, d, c: original(g, d, c, sample_rows=samples),
            ), patch.object(
                self.runtime,
                "install",
                wraps=self.runtime.install,
            ) as staged, patch.object(
                exchange, "_exchange_metadata", record_exchange
            ):
                with ThreadPoolExecutor(8) as pool:
                    pending = [
                        pool.submit(
                            self.prepare,
                            clients[i][rank],
                            rank=rank,
                            world_size=4,
                            gang_hosts=[f"127.0.0.{i + 1}"] * 4,
                            environ=dict(env),
                        )
                        for i in range(2)
                        for rank in range(4)
                    ]
                    [job.result(timeout=15) for job in pending]
            self.assertEqual(sorted(published), [0, 1])
            self.assertEqual(staged.call_count, 8)
            self.assertEqual(sorted(exchanged), ["i0", "i1"])
            first = staged.call_args_list[0].args[0]
            self.assertTrue(
                all(call.args[0] == first for call in staged.call_args_list)
            )
            for i, server in enumerate(stores):
                prefix = f'engram/{groups[i]["group"]["generation"]}/attempt/1/i{i}/'
                self.assertFalse(server.check([prefix + "package"]))
                for rank in range(4):
                    self.assertEqual(server.get(f"engram/start_count/{rank}"), b"1")
            # The temporary cohort TCPStore must release its listener at READY.
            with socket.socket() as client:
                client.settimeout(1)
                self.assertNotEqual(client.connect_ex(("127.0.0.1", port)), 0)

    def test_partial_restart_fails_boundedly_without_consuming_old_package(self):
        env = cohort_env(2, timeout=0.15)
        cfg = topology.build_group(env, 2)
        self.server.add("engram/start_count/1", 1)
        self.server.set(
            f'engram/{cfg["group"]["generation"]}/attempt/1/i0/package', b"stale"
        )
        start = time.monotonic()
        with self.assertRaisesRegex(exchange.ExchangeError, "TIMEOUT"):
            self.prepare(self.client(), rank=1, world_size=2, environ=env)
        self.assertLess(time.monotonic() - start, 3)
        self.assertIsNone(self.runtime.prepared())

    def test_install_failure_is_broadcast_before_receipt(self):
        env = cohort_env(2, timeout=8)
        cfg, docs = derived(env, world_size=2)
        clients = [self.client(), self.client()]
        failure = ValueError("injected local install failure")

        def reject_package(_package, _options):
            raise failure

        with patch.object(
            shard_publisher, "publish", return_value=docs[0]
        ), patch.object(
            exchange,
            "_exchange_group",
            return_value=prepared_package(cfg["group"], docs),
        ), ThreadPoolExecutor(
            2
        ) as pool:
            jobs = [
                pool.submit(
                    self.prepare,
                    clients[rank],
                    rank=rank,
                    world_size=2,
                    environ=dict(env),
                    install=reject_package if rank else self.runtime.install,
                )
                for rank in range(2)
            ]
            peer, local = (job.exception(timeout=3) for job in jobs)
        self.assertIs(local, failure)
        self.assertIsInstance(peer, exchange._StartupPeerFailure)
        self.assertEqual((peer.rank, peer.phase), (1, "stage"))
        prefix = f'engram/{cfg["group"]["generation"]}/attempt/1/i0/'
        self.assertFalse(self.server.check([prefix + "received/1"]))

    def test_rank_failure_notifies_every_waiter_in_the_current_gang_attempt(self):
        hosts = ["127.0.0.1"] * 2 + ["127.0.0.2"] * 2
        env = cohort_env(2, timeout=10)
        env.pop(topology.COHORT_ENV)
        env.pop(topology.INDEX_ENV)
        cfg = topology.build_group(env, 4, gang_hosts=hosts)
        _, docs = derived(cohort_env(2), 2)
        for doc in docs:
            doc["generation"] = cfg["group"]["generation"]
        for phase, failed_rank in (("publish", 2), ("assembly", 0)):
            with self.subTest(phase=phase):
                original_error = PermissionError("private checkpoint capability")

                def allocate(
                    checkpoint, revision, host, generation, shard, count, **kw
                ):
                    if phase == "publish" and shard == 1:
                        raise original_error
                    return docs[shard]

                def assemble(*args):
                    raise original_error

                clients = [self.client() for _ in range(4)]
                started = time.monotonic()
                with patch.object(shard_publisher, "publish", allocate), patch.object(
                    exchange, "_assemble_package", assemble
                ):
                    with ThreadPoolExecutor(4) as pool:
                        pending = [
                            pool.submit(
                                self.prepare,
                                clients[rank],
                                rank=rank,
                                world_size=4,
                                gang_hosts=hosts,
                                environ=dict(env),
                            )
                            for rank in range(4)
                        ]
                        errors = [job.exception(timeout=3) for job in pending]
                self.assertLess(time.monotonic() - started, 3)
                self.assertIs(errors[failed_rank], original_error)
                for rank, error in enumerate(errors):
                    if rank == failed_rank:
                        continue
                    self.assertIsInstance(error, exchange._StartupPeerFailure)
                    self.assertEqual((error.rank, error.phase), (failed_rank, phase))
                    self.assertEqual(error.error_class, "permission")
                    self.assertEqual(error.cause_code, "PERMISSION_DENIED")
                    self.assertNotIn("private", str(error))
                self.assertIsNone(self.runtime.prepared())


class ResidentOwnerTest(unittest.TestCase):
    def test_discovered_gpu_owner_defaults_to_all_local_ranks(self):
        generation = mock.Mock()
        generation.network.peers = [("127.0.0.1", 29484)]
        generation.options.policy.memory_kind = "gpu"
        with (
            patch.object(sys, "argv", ["resident", "--discover"]),
            patch.dict(os.environ, {"WORLD_SIZE": "4", "LOCAL_WORLD_SIZE": "4"}),
            patch.object(resident.signal, "signal"),
            patch.object(resident, "ResidentGeneration", return_value=generation),
            patch.object(topology, "_instance_index", return_value=0),
            patch.object(resident, "serve_shard") as serve,
        ):
            resident.main()
        self.assertEqual(serve.call_args.kwargs["gpu_devices"], (0, 1, 2, 3))


if __name__ == "__main__":
    unittest.main()
