# SPDX-License-Identifier: Apache-2.0
import json
import logging
import os
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, patch

from cryptography.fernet import Fernet

from rtp_llm.models_py.modules.dsv41.engram_distributed import (
    resident,
    resident_network as net,
)
from rtp_llm.models_py.modules.dsv41.engram_distributed.exchange import (
    ExchangeError,
    decode,
    encode,
)
from rtp_llm.models_py.modules.dsv41.engram_distributed.test.test_startup import fixture


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
        self.key = Fernet.generate_key()
        keyfile = self.root / "key"
        keyfile.write_bytes(self.key)
        keyfile.chmod(0o600)
        self.env = {
            "RTP_ENGRAM_STORAGE": "host_shared",
            "ENGRAM_SHARD_COUNT": "1",
            net.TRANSPORT_ENV: "tcp",
            net.KEY_ENV: str(keyfile),
            net.ENDPOINTS_ENV: f"127.0.0.1:{free_port()}",
            "RTP_ENGRAM_RESIDENT_GENERATION": self.group["generation"],
            "RTP_ENGRAM_MODEL_REVISION": self.group["model_revision"],
            "RTP_ENGRAM_CHECKPOINT": str(self.root),
            "RTP_ENGRAM_COHORT_TIMEOUT": "0.25",
        }

    def generation(self, name, **extra):
        directory = self.root / name
        directory.mkdir(mode=0o700, exist_ok=True)
        return resident.ResidentGeneration(
            {**self.env, net.STATE_ENV: str(directory), **extra}
        )

    def test_ready_read_reconnect_without_shared_files_and_log_redaction(self):
        owner = self.generation("owner")
        consumer = self.generation("worker")
        with net.OwnerEndpoint(owner, 0) as endpoint, patch.object(
            resident, "checkpoint_samples", return_value={}
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
            self.assertNotIn(self.key.decode(), "\n".join(logs.output))
            self.assertFalse(list(self.root.rglob("ready.json")))
            for file in (self.root / "worker").rglob("*"):
                if file.is_file():
                    self.assertNotIn(b"handle", file.read_bytes())

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
        with patch.object(
            net.socket, "create_connection", return_value=connection
        ), patch.object(net.time, "monotonic", side_effect=[0.0, 2.0]):
            with self.assertRaises(TimeoutError):
                client.request(0, 1)
        connection.__enter__.return_value.sendall.assert_not_called()

    def test_wrong_key_and_oversized_request_cannot_read_or_change_owner(self):
        owner = self.generation("owner")
        with net.OwnerEndpoint(owner, 0) as endpoint:
            endpoint.publish(self.docs[0])
            wrong = Fernet(Fernet.generate_key())
            request = dict(
                protocol=1,
                kind="get",
                generation=owner.generation,
                revision=owner.revision,
                shard=0,
                nonce="a" * 64,
            )
            with socket.create_connection(owner.network.peers[0], timeout=1) as sock:
                net._send_frame(sock, wrong.encrypt(encode(request)))
                self.assertEqual(sock.recv(1), b"")
            with socket.create_connection(owner.network.peers[0], timeout=1) as sock:
                sock.sendall(struct.pack("!I", 2**32 - 1))
                self.assertEqual(sock.recv(1), b"")
            self.assertEqual(
                self.generation("worker").network.request(0, 1), self.docs[0]
            )
            self.assertEqual(endpoint.rejections, 2)

    def test_response_replay_rejected_even_when_correctly_encrypted(self):
        client = self.generation("worker").network
        reply = dict(
            protocol=1,
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
                    net._send_frame(conn, client.cipher.encrypt(encode(reply)))

            thread = threading.Thread(target=fake_owner)
            thread.start()
            try:
                with self.assertRaisesRegex(ExchangeError, "RESPONSE_MISMATCH"):
                    client.request(0, 1)
            finally:
                thread.join(2)

    def test_authenticated_response_requires_exact_integer_fields(self):
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
                            client.cipher.decrypt(
                                net._read_frame(
                                    conn, net._REQUEST_LIMIT, time.monotonic() + 1
                                )
                            )
                        )
                        reply = {
                            **request,
                            "kind": "reply",
                            "status": "ready",
                            "document": self.docs[0],
                        }
                        reply[field] = bool(reply[field])
                        net._send_frame(conn, client.cipher.encrypt(encode(reply)))

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

    def test_key_permissions_and_mixed_transports_fail_closed(self):
        Path(self.env[net.KEY_ENV]).chmod(0o644)
        with self.assertRaisesRegex(ExchangeError, "UNSAFE"):
            self.generation("worker")
        Path(self.env[net.KEY_ENV]).chmod(0o600)
        with self.assertRaisesRegex(ValueError, "CONFLICTING_RESIDENT_TRANSPORT"):
            self.generation("worker", RTP_ENGRAM_RESIDENT_DIR=str(self.root))

    def test_gang_discovery_uses_dedicated_owner_ports(self):
        env = {k: v for k, v in self.env.items() if k != net.ENDPOINTS_ENV}
        with patch.object(net.config, "_cohort", return_value=None), patch.object(
            net.config, "_ganginfo_cohort", return_value=None
        ), patch.object(
            net.config, "_local_addresses", return_value={"10.1.0.1"}
        ), patch.object(
            net.config,
            "_annotation_cohort",
            return_value=[("10.1.0.1", 29483), ("10.1.0.2", 29483)],
        ):
            self.assertEqual(
                net.endpoints(env, 2), (("10.1.0.1", 29484), ("10.1.0.2", 29484))
            )

    def test_worker_sigkill_sigterm_and_abort_leave_owner_and_peer_alive(self):
        self.group, self.docs = fixture(n=2, p=2)
        ports = [free_port(), free_port()]
        self.env.update(ENGRAM_SHARD_COUNT="2", RTP_ENGRAM_COHORT_TIMEOUT="3")
        self.env[net.ENDPOINTS_ENV] = f"127.0.0.1:{ports[0]},localhost:{ports[1]}"
        processes = []

        def child(code, state, *args):
            directory = self.root / state
            directory.mkdir(mode=0o700, exist_ok=True)
            env = {**os.environ, **self.env, net.STATE_ENV: str(directory)}
            log = (directory / f"process-{len(processes)}.log").open("w")
            self.addCleanup(log.close)
            proc = subprocess.Popen(
                [sys.executable, "-c", code, *args], env=env, stdout=log, stderr=log
            )
            processes.append(proc)
            return proc

        def cleanup():
            for p in reversed(processes):
                if p.poll() is None:
                    p.kill()
                p.wait(timeout=5)

        self.addCleanup(cleanup)
        owner_code = """
import json, sys
from rtp_llm.models_py.modules.dsv41.engram_distributed import resident
document=json.loads(sys.argv[1])
class Publisher:
 def publish(self,*a,**kw): return document
 def release(self,*a): return []
resident.ShardPublisher=Publisher
sys.argv=[sys.argv[0],'--shard',sys.argv[2],'--host-id','fixture-'+sys.argv[2]]
resident.main()
"""
        owners = [
            child(owner_code, f"owner-{i}", json.dumps(self.docs[i]), str(i))
            for i in range(2)
        ]
        consumer = self.generation("observer")
        with patch.object(resident, "checkpoint_samples", return_value={}):
            consumer.load()
        worker_code = """
import os, resource, sys, time
from pathlib import Path
from rtp_llm.models_py.modules.dsv41.engram_distributed import resident, EngramStartup, EngramRuntime
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
resident.checkpoint_samples=lambda *_:{}
runtime=EngramRuntime()
EngramStartup(None,rank=0,world_size=1,publish=None,install=runtime.install).run()
Path(sys.argv[1]).write_text(str(os.getpid()))
while True: time.sleep(.02)
"""

        def ready(proc, path):
            deadline = time.monotonic() + 5
            while (
                not path.exists()
                and proc.poll() is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.02)
            self.assertTrue(path.exists())

        sibling = child(worker_code, "sibling", str(self.root / "sibling-ready"))
        ready(sibling, self.root / "sibling-ready")
        for sig in (signal.SIGKILL, signal.SIGTERM, signal.SIGABRT):
            marker = self.root / f"ready-{sig.name}"
            worker = child(worker_code, "worker", str(marker))
            ready(worker, marker)
            worker.send_signal(sig)
            self.assertEqual(worker.wait(timeout=5), -sig)
            self.assertIsNone(sibling.poll())
            self.assertTrue(all(p.poll() is None for p in owners))
            with patch.object(resident, "checkpoint_samples", return_value={}):
                self.assertEqual(
                    consumer.load()["generation"], self.group["generation"]
                )
        replacement = child(worker_code, "worker", str(self.root / "replacement-ready"))
        ready(replacement, self.root / "replacement-ready")
        self.assertFalse(list(self.root.rglob("ready.json")))


if __name__ == "__main__":
    unittest.main()
