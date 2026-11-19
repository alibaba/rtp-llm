import copy
import os
import socket
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import psutil
from rtp_llm.test.smoke.scr_checkpoint_probe import (
    assert_clean_checkpoint,
    kv_backing_snapshot,
    tcp_snapshot,
)


def clean_snapshot():
    return {
        "network": {
            "pids": [123],
            "internal_tcp": [],
            "external_tcp": [],
            "tcp_listeners": [],
        },
        "participants": [
            {
                "pid": 123,
                "gpu_kv_bytes": 4096,
                "host_kv": {
                    "gpu_pools": [{"ptr": 8192, "bytes": 4096}],
                    "host_pools": [],
                    "active_pinned_bytes": 0,
                },
                "pinned": {
                    "pid": 123,
                    "initialized": True,
                    "errors": 0,
                    "active_bytes": 0,
                    "driver_calls": 1,
                    "allocations": [],
                },
            }
        ],
    }


class CheckpointResourceAssertionTest(unittest.TestCase):
    def test_clean_state_keeps_gpu_kv(self):
        assert_clean_checkpoint(clean_snapshot())

    def test_non_kv_pinned_residual_is_allowed(self):
        snapshot = clean_snapshot()
        snapshot["participants"][0]["pinned"].update(
            active_bytes=2304, allocations=[{"ptr": 1, "bytes": 2304}]
        )
        assert_clean_checkpoint(snapshot)

    def test_host_kv_pinned_residual_fails(self):
        snapshot = clean_snapshot()
        snapshot["participants"][0]["host_kv"]["active_pinned_bytes"] = 4096
        with self.assertRaisesRegex(AssertionError, "Host KV pinned resources"):
            assert_clean_checkpoint(snapshot)

    def test_failed_observation_fails(self):
        for field, value in [
            ("initialized", False),
            ("errors", 1),
            ("pid", 456),
            ("driver_calls", 0),
        ]:
            with self.subTest(field=field):
                snapshot = clean_snapshot()
                snapshot["participants"][0]["pinned"][field] = value
                with self.assertRaises(AssertionError):
                    assert_clean_checkpoint(snapshot)

    def test_missing_model_and_missing_participants_fail(self):
        snapshot = clean_snapshot()
        snapshot["participants"][0]["gpu_kv_bytes"] = 0
        with self.assertRaises(AssertionError):
            assert_clean_checkpoint(snapshot)
        snapshot["participants"] = []
        with self.assertRaises(AssertionError):
            assert_clean_checkpoint(snapshot)

    def test_external_connection_fails_even_with_internal_connections(self):
        snapshot = clean_snapshot()
        snapshot["network"]["internal_tcp"] = [{"pid": 123}]
        snapshot["network"]["external_tcp"] = [{"pid": 123}]
        with self.assertRaisesRegex(AssertionError, "External TCP"):
            assert_clean_checkpoint(snapshot)

    def test_listeners_are_allowed(self):
        snapshot = clean_snapshot()
        snapshot["network"]["tcp_listeners"] = [
            {"pid": 123, "local": [ip, 8000]} for ip in ("0.0.0.0", "::")
        ]
        assert_clean_checkpoint(snapshot)

    def test_four_tp_backends_do_not_require_internal_connections(self):
        snapshot = clean_snapshot()
        participant = snapshot["participants"][0]
        snapshot["participants"] = []
        for pid in range(123, 127):
            backend = copy.deepcopy(participant)
            backend["pid"] = backend["pinned"]["pid"] = pid
            snapshot["participants"].append(backend)
        for count in (0, 1, 10):
            with self.subTest(internal_connections=count):
                snapshot["network"]["internal_tcp"] = [{"pid": 123}] * count
                assert_clean_checkpoint(snapshot, expected_tp_size=4)
        snapshot["participants"].pop()
        with self.assertRaisesRegex(AssertionError, "Expected 4"):
            assert_clean_checkpoint(snapshot, expected_tp_size=4)


class KvBackingObservationTest(unittest.TestCase):
    gpu_log = (
        "[INFO] [123:123] [RANK 0][host][rtp_llm/cpp/cache/block_tree_cache/"
        "block_pool/DeviceBlockPool.cc:328][initializeCacheBuffer()] backing selected: "
        "pool_name=default actual_backing=GPU is_cuda=1 is_pinned=0 "
        "ptr=0x2000 total_size=4096 bytes memory_layouts=1\n"
    )
    host_log = (
        "[INFO] [123:123] [RANK 0][host][rtp_llm/cpp/cache/block_tree_cache/"
        "block_pool/HostBlockPool.cc:107][init()] backing selected: "
        "pool_name=block_tree_host_full payload_bytes=512 stride_bytes=512 "
        "physical_block_count=8 total_size=4096 bytes total_size_mb=0.01 ptr=0x1000\n"
    )

    def test_only_live_host_kv_ranges_count(self):
        participants = clean_snapshot()["participants"]
        participants[0]["pinned"].update(
            active_bytes=10496,
            allocations=[
                {"ptr": 0xF00, "bytes": 8192},
                {"ptr": 0x10000, "bytes": 2304},
            ],
        )
        result = kv_backing_snapshot(participants, self.gpu_log + self.host_log)[123]
        self.assertEqual(result["active_pinned_bytes"], 4096)
        self.assertEqual(len(result["host_pools"]), 1)
        participants[0]["host_kv"] = result
        with self.assertRaisesRegex(AssertionError, "Host KV pinned resources"):
            assert_clean_checkpoint({**clean_snapshot(), "participants": participants})
        participants[0]["pinned"]["allocations"].pop(0)
        with self.assertRaisesRegex(RuntimeError, "Host KV backing not fully observed"):
            kv_backing_snapshot(participants, self.gpu_log + self.host_log)
        result = kv_backing_snapshot(participants, self.gpu_log)[123]
        self.assertEqual(result["active_pinned_bytes"], 0)
        participants[0]["host_kv"] = result
        assert_clean_checkpoint({**clean_snapshot(), "participants": participants})

    def test_host_pinned_device_backing_is_detected(self):
        participants = clean_snapshot()["participants"]
        participants[0]["pinned"]["allocations"] = [{"ptr": 0x3000, "bytes": 4096}]
        host_log = self.gpu_log.replace(
            "is_cuda=1 is_pinned=0", "is_cuda=0 is_pinned=1"
        )
        host_log = host_log.replace("actual_backing=GPU", "actual_backing=CPU_PINNED")
        host_log = host_log.replace("ptr=0x2000", "ptr=0x3000")
        result = kv_backing_snapshot(participants, self.gpu_log + host_log)[123]
        self.assertEqual(result["active_pinned_bytes"], 4096)

    def test_multiple_host_pools_are_counted(self):
        participants = clean_snapshot()["participants"]
        participants[0]["pinned"]["allocations"] = [
            {"ptr": 0x1000, "bytes": 4096},
            {"ptr": 0x3000, "bytes": 4096},
        ]
        second = self.host_log.replace("ptr=0x1000", "ptr=0x3000")
        result = kv_backing_snapshot(
            participants, self.gpu_log + self.host_log + second
        )
        self.assertEqual(result[123]["active_pinned_bytes"], 8192)

    def test_missing_backing_logs_and_unobserved_pool_fail(self):
        participants = clean_snapshot()["participants"]
        with self.assertRaisesRegex(RuntimeError, "GPU KV backing log missing"):
            kv_backing_snapshot(participants, "")
        with self.assertRaisesRegex(RuntimeError, "unobserved checkpoint process"):
            kv_backing_snapshot(
                participants,
                self.gpu_log + self.host_log.replace("[123:123]", "[456:456]"),
            )

    def test_unparsable_host_backing_log_fails(self):
        with self.assertRaisesRegex(RuntimeError, "Cannot parse KV backing log"):
            kv_backing_snapshot(
                clean_snapshot()["participants"],
                self.gpu_log + self.host_log.replace("[123:123]", "[unknown:unknown]"),
            )

    def test_host_vmm_diagnostics_do_not_hide_unobserved_host_kv(self):
        participants = clean_snapshot()["participants"]
        participants[0]["pinned"].update(host_vmm_calls=1, host_vmm_created_bytes=4096)
        participants[0]["host_kv"] = kv_backing_snapshot(participants, self.gpu_log)[
            123
        ]
        assert_clean_checkpoint({**clean_snapshot(), "participants": participants})
        with self.assertRaisesRegex(RuntimeError, "Host KV backing not fully observed"):
            kv_backing_snapshot(participants, self.gpu_log + self.host_log)

    def test_partially_observed_host_pool_fails(self):
        participants = clean_snapshot()["participants"]
        participants[0]["pinned"]["allocations"] = [{"ptr": 0x1000, "bytes": 256}]
        with self.assertRaisesRegex(RuntimeError, "Host KV backing not fully observed"):
            kv_backing_snapshot(participants, self.gpu_log + self.host_log)

    def test_malformed_host_backing_log_fails(self):
        with self.assertRaises((KeyError, ValueError)):
            kv_backing_snapshot(
                clean_snapshot()["participants"],
                self.gpu_log
                + self.host_log.replace("total_size=4096", "total_size=unknown"),
            )


class TcpObservationTest(unittest.TestCase):
    def test_local_connections_need_no_peer_pairing_or_backend_membership(self):
        def connection(fd, local, remote, status=psutil.CONN_ESTABLISHED):
            return SimpleNamespace(fd=fd, laddr=local, raddr=remote, status=status)

        def process(pid, connections):
            return SimpleNamespace(
                pid=pid,
                children=lambda recursive: [],
                is_running=lambda: True,
                status=lambda: psutil.STATUS_RUNNING,
                net_connections=lambda kind: connections,
            )

        child = process(
            124,
            [
                connection(1, ("192.0.2.1", 4000), ("192.0.2.1", 5000)),
                connection(2, ("0.0.0.0", 4000), (), psutil.CONN_LISTEN),
                connection(3, ("::", 6000), (), psutil.CONN_LISTEN),
            ],
        )
        root = process(
            123,
            [
                connection(1, ("192.0.2.1", 7000), ("192.0.2.2", 8000)),
                connection(2, ("127.0.0.1", 7000), ("127.0.0.1", 8000)),
            ],
        )
        root.children = lambda recursive: [child]
        with patch(
            "rtp_llm.test.smoke.scr_checkpoint_probe.psutil.Process", return_value=root
        ), patch(
            "rtp_llm.test.smoke.scr_checkpoint_probe.psutil.net_if_addrs",
            return_value={
                "eth0": [SimpleNamespace(family=socket.AF_INET, address="192.0.2.1")]
            },
        ):
            snapshot = tcp_snapshot(123)
        self.assertEqual(snapshot["pids"], [123, 124])
        self.assertEqual(len(snapshot["internal_tcp"]), 2)
        self.assertEqual(len(snapshot["tcp_listeners"]), 2)
        self.assertEqual(snapshot["external_tcp"][0]["remote"], ["192.0.2.2", 8000])

    def test_ipv4_ipv6_and_non_established_connections(self):
        internal = [
            ("127.0.0.1", psutil.CONN_ESTABLISHED),
            ("127.0.0.2", psutil.CONN_ESTABLISHED),
            ("::1", psutil.CONN_ESTABLISHED),
            ("::ffff:127.0.0.1", psutil.CONN_ESTABLISHED),
            ("192.0.2.1", psutil.CONN_SYN_SENT),
            ("2001:db8::1", psutil.CONN_CLOSE_WAIT),
            ("::ffff:192.0.2.1", psutil.CONN_ESTABLISHED),
            ("fe80::1%eth0", psutil.CONN_ESTABLISHED),
        ]
        external = [
            ("192.0.2.2", psutil.CONN_SYN_SENT),
            ("2001:db8::2", psutil.CONN_CLOSE_WAIT),
            ("::ffff:192.0.2.2", psutil.CONN_ESTABLISHED),
            ("10.0.0.2", psutil.CONN_ESTABLISHED),
        ]
        root = SimpleNamespace(
            pid=123,
            children=lambda recursive: [],
            is_running=lambda: True,
            status=lambda: psutil.STATUS_RUNNING,
            net_connections=lambda kind: [
                SimpleNamespace(
                    fd=i, laddr=("0.0.0.0", 1234), raddr=(ip, 80), status=status
                )
                for i, (ip, status) in enumerate(internal + external)
            ],
        )
        with patch(
            "rtp_llm.test.smoke.scr_checkpoint_probe.psutil.Process", return_value=root
        ), patch(
            "rtp_llm.test.smoke.scr_checkpoint_probe.psutil.net_if_addrs",
            return_value={
                "eth0": [
                    SimpleNamespace(family=family, address=address)
                    for family, address in [
                        (socket.AF_INET, "192.0.2.1"),
                        (socket.AF_INET6, "2001:db8::1"),
                        (socket.AF_INET6, "fe80::1%eth0"),
                        (socket.AF_UNIX, "/tmp/non-ip-socket"),
                    ]
                ]
            },
        ):
            result = tcp_snapshot(123)
        self.assertEqual(
            [entry["remote"][0] for entry in result["internal_tcp"]],
            [ip for ip, _ in internal],
        )
        self.assertEqual(
            [entry["remote"][0] for entry in result["external_tcp"]],
            [ip for ip, _ in external],
        )

    def test_permission_failure_is_not_an_empty_snapshot(self):
        root = SimpleNamespace(
            pid=123,
            children=lambda recursive: [],
            is_running=lambda: True,
            status=lambda: psutil.STATUS_RUNNING,
            net_connections=lambda kind: (_ for _ in ()).throw(
                psutil.AccessDenied(123)
            ),
        )
        with patch(
            "rtp_llm.test.smoke.scr_checkpoint_probe.psutil.Process", return_value=root
        ):
            with self.assertRaises(psutil.AccessDenied):
                tcp_snapshot(123)

    def test_real_local_connections_and_listeners_are_allowed(self):
        # Non-loopback IPv4/IPv6 classification is covered by the fixtures
        # above; a real socket test must not depend on host VPN/bridge routes.
        with socket.socket() as listener, socket.socket() as client:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            client.connect(listener.getsockname())
            connection, _ = listener.accept()
            with connection:
                snapshot = tcp_snapshot(os.getpid())
                connected_fds = {client.fileno(), connection.fileno()}
                self.assertTrue(
                    connected_fds.issubset(
                        {entry["fd"] for entry in snapshot["internal_tcp"]}
                    )
                )
                self.assertFalse(
                    connected_fds.intersection(
                        {entry["fd"] for entry in snapshot["external_tcp"]}
                    )
                )
                self.assertIn(
                    listener.fileno(),
                    {entry["fd"] for entry in snapshot["tcp_listeners"]},
                )


if __name__ == "__main__":
    unittest.main()
