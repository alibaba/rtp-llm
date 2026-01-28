import shutil
import threading
import unittest
from contextlib import contextmanager

from rtp_llm.test.utils.port_util import *


class TestExpiredLockFile(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_lock_file_creation(self):
        lock_file = self.test_dir / "port_8080.lock"
        with ExpiredLockFile(lock_file, 8080):
            self.assertTrue(lock_file.exists())
            with open(lock_file) as f:
                data = json.load(f)
                self.assertEqual(data["port"], 8080)
                self.assertEqual(data["ttl"], 3600)
                self.assertEqual(data["pid"], os.getpid())
                self.assertAlmostEqual(time.time(), data["timestamp"], delta=1)

    def test_lock_file_cleanup(self):
        lock_file = self.test_dir / "port_8080.lock"
        with ExpiredLockFile(lock_file, 8080):
            self.assertTrue(lock_file.exists())
        self.assertFalse(lock_file.exists())

    def test_concurrent_locks(self):
        lock_file = self.test_dir / "port_8080.lock"
        with ExpiredLockFile(lock_file, 8080):
            # should raise PortInUseError for the 2nd time try to lock
            with self.assertRaises(PortInUseError):
                with ExpiredLockFile(lock_file, 8080):
                    pass


class TestPortManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.port_manager = PortManager(self.test_dir, start_port=12000)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @contextmanager
    def occupy_port(self, port):
        """simulate port in use"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("", port))
            yield sock
        finally:
            sock.close()

    def test_simple_get_consecutive_ports_basic(self):
        num_ports = 3
        ports, locks = self.port_manager.get_consecutive_ports(num_ports)
        try:
            self.assertEqual(len(ports), num_ports)
            self.assertEqual(len(locks), num_ports)

            # check ports are consecutive
            self.assertTrue(all(b == a + 1 for a, b in zip(ports, ports[1:])))

            # check lock file exists
            self.assertTrue(
                all((self.test_dir / f"port_{port}.lock").exists() for port in ports)
            )
        finally:
            for lock in locks:
                lock.__exit__(None, None, None)

    def test_get_consecutive_ports_with_occupied(self):
        start_port = self.port_manager.start_port
        with self.occupy_port(start_port):
            ports, locks = self.port_manager.get_consecutive_ports(2)
            try:
                self.assertNotIn(start_port, ports)
                self.assertEqual(len(ports), 2)
                # should skip the first occupied port and get the next consecutive port range
                self.assertEqual(ports[0], 12001)
                self.assertEqual(ports[1], 12002)
            finally:
                for lock in locks:
                    lock.__exit__(None, None, None)

    def test_get_consecutive_ports_reuse(self):
        ports1, locks1 = self.port_manager.get_consecutive_ports(2)
        original_ports = ports1[:]

        # simulate all locks released
        for lock in locks1:
            lock.__exit__(None, None, None)

        # 2nd time try to get ports
        ports2, locks2 = self.port_manager.get_consecutive_ports(2)
        try:
            # should reuse the released ports
            self.assertEqual(ports2, original_ports)
        finally:
            for lock in locks2:
                lock.__exit__(None, None, None)

    def test_concurrent_access(self):
        def get_ports():
            return PortManager(self.test_dir).get_consecutive_ports(20)

        threads = []
        results = []

        for _ in range(10):
            thread = threading.Thread(target=lambda: results.append(get_ports()))
            threads.append(thread)

        for thread in threads:
            thread.start()

        # wait for all thread finish working
        for thread in threads:
            thread.join()

        try:
            all_ports = set()
            for ports, _ in results:
                if ports:
                    # make sure no overlapping ports
                    self.assertTrue(
                        all(port not in all_ports for port in ports),
                        f"Found overlapping ports: {ports} in all_ports: {all_ports}",
                    )
                    # make sure all ports are consecutive
                    self.assertTrue(
                        all(b == a + 1 for a, b in zip(ports, ports[1:])),
                        f"ports should be consecutive: {ports}",
                    )
                    all_ports.update(ports)
            self.assertEqual(len(all_ports), 200)
        finally:
            # release all locks
            for _, locks in results:
                if locks:
                    for lock in locks:
                        lock.__exit__(None, None, None)

    def test_cleanup_stale_locks(self):
        lock_file = self.test_dir / "port_8080.lock"
        with open(lock_file, "w") as f:
            json.dump(
                {
                    "port": 8080,
                    "pid": os.getpid(),
                    "timestamp": time.time() - 7200,
                    "ttl": 3600,
                },
                f,
            )

        self.port_manager.cleanup_stale_locks()
        self.assertFalse(lock_file.exists())


class TestPortsContext(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_simple_get_consecutive_ports(self):
        with PortsContext(self.test_dir, num_ports=2) as ports:
            self.assertEqual(len(ports), 2)
            self.assertEqual(ports[1], ports[0] + 1)

    def test_port_release(self):
        lock_file = None
        # 1st time acquired the port
        with PortsContext(self.test_dir, num_ports=1) as ports:
            first_port = ports[0]
            lock_file = self.test_dir / f"port_{first_port}.lock"
            self.assertTrue(lock_file.exists())

        # exit context, lock file should be released
        self.assertFalse(lock_file.exists())

    def test_concurrent_access(self):
        """test the case that concurrent allocates ports without conflict"""
        results = []

        def get_ports():
            try:
                with PortsContext(self.test_dir, num_ports=2) as ports:
                    # simulate use the port
                    time.sleep(0.1)
                    results.append(ports)
                return True
            except Exception:
                return False

        threads = []
        for _ in range(5):
            thread = threading.Thread(target=get_ports)
            threads.append(thread)

        for thread in threads:
            thread.start()

        # wait for all thread finish working
        for thread in threads:
            thread.join()

        all_ports = set()
        for ports in results:
            # make sure no overlapping ports
            self.assertTrue(all(port not in all_ports for port in ports))
            all_ports.update(ports)
        self.assertEqual(len(all_ports), 10)

    def test_exception_handling(self):
        """test the case that lock file is released when exception raised"""

        class TestException(Exception):
            pass

        ports_obtained = None
        try:
            with PortsContext(self.test_dir, num_ports=2) as ports:
                ports_obtained = ports
                self.assertTrue(
                    all(
                        (self.test_dir / f"port_{port}.lock").exists() for port in ports
                    )
                )
                raise TestException("simulate exception raised")
        except TestException:
            self.assertFalse(
                all((self.test_dir / f"port_{port}.lock").exists() for port in ports)
            )

    def test_multiple_contexts(self):
        """test multiple embedded PortsContext"""
        with PortsContext(self.test_dir, num_ports=20) as ports1:
            with PortsContext(self.test_dir, num_ports=30) as ports2:
                # make sure no overlapping ports between two groups
                self.assertTrue(set(ports1).isdisjoint(set(ports2)))

                # make sure all ports are consecutive
                self.assertTrue(
                    all(b == a + 1 for a, b in zip(ports1, ports1[1:])),
                    "ports1 should be consecutive",
                )
                self.assertTrue(
                    all(b == a + 1 for a, b in zip(ports2, ports2[1:])),
                    "ports2 should be consecutive",
                )


if __name__ == "__main__":
    unittest.main()
