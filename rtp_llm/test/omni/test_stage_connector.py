import unittest

import torch

from rtp_llm.omni.engine.stage_connector import (
    SharedMemoryConnector,
    StageConnector,
    StageOutput,
)


class TestStageOutput(unittest.TestCase):
    def test_default_fields(self):
        output = StageOutput()
        self.assertIsNone(output.token_ids)
        self.assertIsNone(output.embeddings)
        self.assertIsNone(output.audio_waveform)
        self.assertIsNone(output.image_tensor)
        self.assertEqual(output.metadata, {})

    def test_with_token_ids(self):
        output = StageOutput(token_ids=[1, 2, 3])
        self.assertEqual(output.token_ids, [1, 2, 3])

    def test_with_embeddings(self):
        emb = torch.randn(10, 128)
        output = StageOutput(embeddings=emb)
        self.assertTrue(torch.equal(output.embeddings, emb))


class TestSharedMemoryConnector(unittest.TestCase):
    def setUp(self):
        self.connector = SharedMemoryConnector()

    def test_put_and_get_by_name(self):
        output = StageOutput(token_ids=[1, 2, 3])
        self.connector.put("req_1", "thinker", output)
        retrieved = self.connector.get("req_1", "thinker")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.token_ids, [1, 2, 3])

    def test_get_nonexistent_returns_none(self):
        result = self.connector.get("nonexistent", "thinker")
        self.assertIsNone(result)

    def test_cleanup(self):
        self.connector.put("req_1", "thinker", StageOutput(token_ids=[1]))
        self.connector.put("req_1", "talker", StageOutput(token_ids=[2]))
        self.connector.put("req_2", "thinker", StageOutput(token_ids=[3]))
        self.connector.cleanup("req_1")
        self.assertIsNone(self.connector.get("req_1", "thinker"))
        self.assertIsNone(self.connector.get("req_1", "talker"))
        self.assertIsNotNone(self.connector.get("req_2", "thinker"))

    def test_cleanup_nonexistent_is_noop(self):
        self.connector.cleanup("nonexistent")

    def test_put_overwrites_existing(self):
        self.connector.put("req_1", "thinker", StageOutput(token_ids=[1]))
        self.connector.put("req_1", "thinker", StageOutput(token_ids=[2]))
        retrieved = self.connector.get("req_1", "thinker")
        self.assertEqual(retrieved.token_ids, [2])

    def test_is_abstract_base(self):
        self.assertIsInstance(self.connector, StageConnector)

    def test_open_stream(self):
        ch = self.connector.open_stream("req_1", "thinker", "talker")
        ch.emit({"token": 42})
        chunk = ch.recv(timeout=1.0)
        self.assertEqual(chunk, {"token": 42})

    def test_open_stream_same_pair_returns_same_channel(self):
        ch1 = self.connector.open_stream("req_1", "thinker", "talker")
        ch2 = self.connector.open_stream("req_1", "thinker", "talker")
        self.assertIs(ch1, ch2)

    def test_cleanup_closes_streams(self):
        ch = self.connector.open_stream("req_1", "thinker", "talker")
        self.connector.cleanup("req_1")
        self.assertTrue(ch.closed)


if __name__ == "__main__":
    unittest.main()
