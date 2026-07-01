import unittest
import threading

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
        producer_ready = threading.Event()
        producer_done = threading.Event()

        def producer():
            producer_ready.set()
            ch.emit("c")
            producer_done.set()

        t = threading.Thread(target=producer)
        t.start()
        producer_ready.wait(timeout=2.0)
        self.assertFalse(producer_done.wait(timeout=0.05))
        ch.recv(timeout=1.0)
        t.join(timeout=2.0)
        self.assertTrue(producer_done.is_set())

    def test_close_when_queue_full_does_not_block(self):
        ch = StreamChannel(maxsize=2)
        ch.emit("a")
        ch.emit("b")
        done = threading.Event()

        def closer():
            ch.close()
            done.set()

        t = threading.Thread(target=closer)
        t.start()
        t.join(timeout=2.0)
        self.assertTrue(done.is_set())
        self.assertTrue(ch.closed)

    def test_pending_count(self):
        ch = StreamChannel(maxsize=10)
        self.assertEqual(ch.pending, 0)
        ch.emit("a")
        ch.emit("b")
        self.assertEqual(ch.pending, 2)
        ch.recv(timeout=1.0)
        self.assertEqual(ch.pending, 1)


    def test_close_on_full_queue_does_not_block(self):
        ch = StreamChannel(maxsize=2)
        ch.emit("a")
        ch.emit("b")
        # Queue is now full; close must not block
        completed = [False]

        def closer():
            ch.close()
            completed[0] = True

        t = threading.Thread(target=closer)
        t.start()
        t.join(timeout=2.0)
        self.assertTrue(completed[0], "close() blocked on a full queue")
        self.assertTrue(ch.closed)

    def test_close_on_full_queue_preserves_all_chunks(self):
        ch = StreamChannel(maxsize=3)
        ch.emit("a")
        ch.emit("b")
        ch.emit("c")
        # Queue is full; close should not displace any enqueued data
        ch.close()
        chunks = list(ch)
        self.assertEqual(chunks, ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
