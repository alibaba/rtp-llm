# SPDX-License-Identifier: Apache-2.0
import json
import logging
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, patch

from rtp_llm.models_py.modules.dsv41.engram_distributed import (
    resident,
    resident_network as net,
)
from rtp_llm.models_py.modules.dsv41.engram_distributed.exchange import (
    ExchangeError,
    decode,
    encode,
)
from rtp_llm.models_py.modules.dsv41.engram_distributed.test.test_startup import (
    fixture,
    publications,
)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ResidentNetworkTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.group, self.docs = fixture(n=1, p=1)
        self.env = {
            "RTP_ENGRAM_STORAGE": "host_shared",
            "ENGRAM_SHARD_COUNT": "1",
            net.TRANSPORT_ENV: "tcp",
            net.PORT_ENV: str(free_port()),
            "RTP_ENGRAM_MODEL_REVISION": self.group["model_revision"],
            "RTP_ENGRAM_CHECKPOINT": str(self.root),
            "RTP_ENGRAM_COHORT_TIMEOUT": "0.25",
        }
        self.env = self.pod_env(0, {"prefill_part0": {"ip": "127.0.0.1"}}, self.env)
        self.docs[0]["generation"] = self.env["RTP_ENGRAM_RESIDENT_GENERATION"]

    def pod_env(self, part, members, env):
        pod = self.root / str(part)
        pod.mkdir(mode=0o700, exist_ok=True)
        annotations = pod / "annotations"
        annotations.write_text(
            f"app.c2.io/biz-detail-ganginfo={json.dumps(json.dumps(members))}\n"
        )
        (pod / "labels").write_text(
            f'gang-version-hash="release-one"\napp.c2.io/gang-part-name="prefill-part{part}"\n'
        )
        result = {
            **env,
            "ENGRAM_SHARD_COUNT": str(len(members)),
            "GANG_ANNOCATION_PATH": str(annotations),
            net.STATE_ENV: str(pod),
        }
        result["RTP_ENGRAM_RESIDENT_GENERATION"] = net.c2_topology(result)[0]
        return result

    def generation(self, name, **extra):
        directory = self.root / name
        directory.mkdir(mode=0o700, exist_ok=True)
        return resident.ResidentGeneration(
            {**self.env, net.STATE_ENV: str(directory), **extra}
        )

    def test_numeric_endpoints_connect_without_dns(self):
        owner = self.generation("owner")
        consumer = self.generation("worker")
        with net.OwnerEndpoint(owner, 0) as endpoint:
            endpoint.publish(self.docs[0])
            with patch.object(
                socket, "getaddrinfo", side_effect=AssertionError("DNS used")
            ):
                self.assertEqual(consumer.network.request(0, 1), self.docs[0])

    def test_ready_read_reconnect_without_shared_files_and_log_redaction(self):
        owner = self.generation("owner")
        consumer = self.generation("worker")
        with (
            net.OwnerEndpoint(owner, 0) as endpoint,
            patch.object(resident, "checkpoint_samples", return_value={}),
        ):
            with self.assertRaisesRegex(ExchangeError, "TIMEOUT"):
                consumer.load()
            endpoint.publish(self.docs[0])
            with self.assertLogs(net._LOGGER, logging.INFO) as logs:
                first = consumer.load()
                second = self.generation("worker").load()
            self.assertEqual(first["generation"], second["generation"])
            self.assertEqual(first["capabilities"], second["capabilities"])
            self.assertIn("resident_owners_connected", "\n".join(logs.output))
            self.assertNotIn(
                self.docs[0]["layers"][0]["segments"][0]["weight"]["handle"],
                "\n".join(logs.output),
            )
            self.assertFalse(list(self.root.rglob("ready.json")))
            for file in (self.root / "worker").rglob("*"):
                if file.is_file():
                    self.assertNotIn(b"handle", file.read_bytes())

    @unittest.skipUnless(
        sys.platform == "linux", "requires multiple loopback addresses"
    )
    def test_c2_gang_reconnect_and_reject_outside_source(self):
        members = {
            "prefill_part1": {"ip": "127.0.0.2"},
            "prefill_part0": {"ip": "127.0.0.1"},
        }
        first_env, second_env = (
            self.pod_env(part, members, self.env) for part in range(2)
        )
        first, second = (
            resident.ResidentGeneration(e) for e in (first_env, second_env)
        )
        self.assertEqual(first.generation, second.generation)
        _, documents = fixture(n=2, p=2)
        for document in documents:
            document["generation"] = first.generation

        with (
            net.OwnerEndpoint(first, 0) as left,
            net.OwnerEndpoint(second, 1) as right,
            patch.object(resident, "checkpoint_samples", return_value={}),
        ):
            left.publish(documents[0])
            right.publish(documents[1])
            before = first.load()
            sibling = second.load()
            after = resident.ResidentGeneration(first_env).load()
            self.assertEqual(before["capabilities"], after["capabilities"])
            self.assertEqual(sibling["capabilities"], after["capabilities"])
            self.assertEqual(second.network.request(0, 1), documents[0])
            with self.assertLogs(net._LOGGER, logging.WARNING) as logs:
                with socket.create_connection(
                    left.server.server_address,
                    timeout=1,
                    source_address=("127.0.0.3", 0),
                ) as outsider:
                    self.assertEqual(outsider.recv(1), b"")
            self.assertIn("reason=outside_c2_gang", "\n".join(logs.output))

        labels = Path(first_env[net.STATE_ENV]) / "labels"
        labels.write_text(labels.read_text().replace("release-one", "release-two"))
        with self.assertRaisesRegex(ValueError, "GANG_IDENTITY_MISMATCH"):
            resident.ResidentGeneration(first_env)
        self.assertFalse(list(self.root.rglob("ready.json")))
        self.assertFalse(list(self.root.rglob("owner.key")))

    def test_replacement_allocation_is_rejected_across_worker_objects(self):
        owner = self.generation("owner")
        with patch.object(resident, "checkpoint_samples", return_value={}):
            with net.OwnerEndpoint(owner, 0) as endpoint:
                endpoint.publish(self.docs[0])
                self.generation("worker").load()
            altered = deepcopy(self.docs[0])
            # Same generation/revision, different valid Fabric allocation.
            import base64

            altered["layers"][0]["segments"][0]["weight"]["handle"] = base64.b64encode(
                b"x" * 64
            ).decode()
            with net.OwnerEndpoint(owner, 0) as replacement:
                replacement.publish(altered)
                with self.assertRaisesRegex(ExchangeError, "ALLOCATION_CHANGED"):
                    self.generation("worker").load()

    def test_owner_loss_times_out_only_new_attach(self):
        owner = self.generation("owner")
        with net.OwnerEndpoint(owner, 0) as endpoint:
            endpoint.publish(self.docs[0])
            document = self.generation("worker").network.request(0, 1)
        start = time.monotonic()
        with self.assertRaisesRegex(ExchangeError, "TIMEOUT"):
            self.generation("worker").load()
        self.assertLess(time.monotonic() - start, 1)
        self.assertEqual(document, self.docs[0])

    def test_withdrawn_owner_cannot_publish_a_different_allocation(self):
        owner = self.generation("owner")
        with net.OwnerEndpoint(owner, 0) as endpoint:
            endpoint.publish(self.docs[0])
            endpoint.withdraw()
            with self.assertRaisesRegex(ExchangeError, "CONFLICTING_REPUBLICATION"):
                endpoint.publish(self.docs[0])
            with self.assertRaisesRegex(ConnectionError, "NOT_READY"):
                self.generation("worker").network.request(0, 1)

    def test_connect_time_is_counted_in_rpc_deadline(self):
        client = self.generation("worker").network
        connection = MagicMock()
        with (
            patch.object(net.socket, "socket", return_value=connection),
            patch.object(net.time, "monotonic", side_effect=[0.0, 2.0]),
        ):
            with self.assertRaises(TimeoutError):
                client.request(0, 1)
        connection.__enter__.return_value.sendall.assert_not_called()

    def test_oversized_request_cannot_read_or_change_owner(self):
        owner = self.generation("owner")
        with net.OwnerEndpoint(owner, 0) as endpoint:
            endpoint.publish(self.docs[0])
            with socket.create_connection(owner.network.peers[0], timeout=1) as sock:
                sock.sendall(struct.pack("!I", 2**32 - 1))
                self.assertEqual(sock.recv(1), b"")
            self.assertEqual(
                self.generation("worker").network.request(0, 1), self.docs[0]
            )
            self.assertEqual(endpoint.rejections, 1)

    def test_response_replay_rejected(self):
        client = self.generation("worker").network
        reply = dict(
            protocol=net._PROTOCOL,
            kind="reply",
            generation=client.generation,
            revision=client.revision,
            shard=0,
            nonce="old" * 21 + "x",
            status="ready",
            document=self.docs[0],
        )
        with socket.socket() as listener:
            listener.bind(client.peers[0])
            listener.listen(1)

            def fake_owner():
                conn, _ = listener.accept()
                with conn:
                    net._read_frame(conn, net._REQUEST_LIMIT, time.monotonic() + 1)
                    net._send_frame(conn, encode(reply))

            thread = threading.Thread(target=fake_owner)
            thread.start()
            try:
                with self.assertRaisesRegex(ExchangeError, "RESPONSE_MISMATCH"):
                    client.request(0, 1)
            finally:
                thread.join(2)

    def test_response_requires_exact_integer_fields(self):
        client = self.generation("worker").network
        for field in ("protocol", "shard"):
            with self.subTest(field=field), socket.socket() as listener:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind(client.peers[0])
                listener.listen(1)

                def fake_owner():
                    conn, _ = listener.accept()
                    with conn:
                        request = decode(
                            net._read_frame(
                                conn, net._REQUEST_LIMIT, time.monotonic() + 1
                            )
                        )
                        reply = {
                            **request,
                            "kind": "reply",
                            "status": "ready",
                            "document": self.docs[0],
                        }
                        reply[field] = bool(reply[field])
                        net._send_frame(conn, encode(reply))

                thread = threading.Thread(target=fake_owner)
                thread.start()
                try:
                    with self.assertRaisesRegex(ExchangeError, "RESPONSE_MISMATCH"):
                        client.request(0, 1)
                finally:
                    thread.join(2)

    def test_slow_peer_cannot_extend_read_deadline(self):
        left, right = socket.socketpair()
        with left, right:
            left.sendall(struct.pack("!I", 10) + b"x")
            start = time.monotonic()
            with self.assertRaises(TimeoutError):
                net._read_frame(right, 100, start + 0.05)
            self.assertLess(time.monotonic() - start, 0.2)

    def test_gpu_owner_publishes_gpu_backing(self):
        generation = self.generation("owner", RTP_ENGRAM_STORAGE="gpu_shared")
        group = {**self.group, "generation": generation.generation}
        (document,) = publications(group, memory_kind="gpu")
        env = {**self.env, "RTP_ENGRAM_GPU_RESERVE_BYTES": "4096"}
        with patch.object(resident, "ShardPublisher") as factory:
            publisher = factory.return_value
            publisher.publish.return_value = document
            publisher.release.return_value = []
            resident.serve_shard(generation, env, 0, "host-0", 0, lambda: True, (0, 1))
        options = publisher.publish.call_args.kwargs
        self.assertEqual(options["memory_kind"], "gpu")
        self.assertEqual(options["gpu_devices"], (0, 1))
        self.assertEqual(options["gpu_reserve_bytes"], 4096)
        publisher.release.assert_called_once_with(generation.generation)

    def test_legacy_transport_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "RESIDENT_C2_TCP_REQUIRED"):
            self.generation("worker", RTP_ENGRAM_RESIDENT_DIR=str(self.root))


if __name__ == "__main__":
    unittest.main()
