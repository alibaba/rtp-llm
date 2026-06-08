import unittest
import threading
import time

from rtp_llm.omni.engine.stream_channel import StreamChannel


class TestStreamChannel(unittest.TestCase):
    def test_emit_and_recv(self):
        ch = StreamChannel(maxsize=10)
        ch.emit({"token": 1})
        ch.emit({"token": 2})
        self.assertEqual(ch.recv(timeout=1.0), {"token": 1})
        self.assertEqual(ch.recv(timeout=1.0), {"token": 2})

    def test_close_and_drain(self):
        ch = StreamChannel(maxsize=10)
        ch.emit({"token": 1})
        ch.emit({"token": 2})
        ch.close()
        chunks = list(ch)
        self.assertEqual(chunks, [{"token": 1}, {"token": 2}])

    def test_recv_after_close_returns_none(self):
        ch = StreamChannel(maxsize=10)
        ch.close()
        result = ch.recv(timeout=0.1)
        self.assertIsNone(result)

    def test_is_closed(self):
        ch = StreamChannel(maxsize=10)
        self.assertFalse(ch.closed)
        ch.close()
        self.assertTrue(ch.closed)

    def test_threaded_producer_consumer(self):
        ch = StreamChannel(maxsize=5)
        received = []

        def consumer():
            for chunk in ch:
                received.append(chunk)

        t = threading.Thread(target=consumer)
        t.start()
        for i in range(10):
            ch.emit(i)
        ch.close()
        t.join(timeout=5.0)
        self.assertEqual(received, list(range(10)))

    def test_backpressure(self):
        ch = StreamChannel(maxsize=2)
        ch.emit("a")
        ch.emit("b")
        blocked = [False]

        def producer():
            ch.emit("c")
            blocked[0] = True

        t = threading.Thread(target=producer)
        t.start()
        time.sleep(0.1)
        self.assertFalse(blocked[0])
        ch.recv(timeout=1.0)
        t.join(timeout=2.0)
        self.assertTrue(blocked[0])

    def test_pending_count(self):
        ch = StreamChannel(maxsize=10)
        self.assertEqual(ch.pending, 0)
        ch.emit("a")
        ch.emit("b")
        self.assertEqual(ch.pending, 2)
        ch.recv(timeout=1.0)
        self.assertEqual(ch.pending, 1)


if __name__ == "__main__":
    unittest.main()
