"""A KVCM SDK read must use the URI and checksum returned by its own query."""

import json
import unittest
from unittest.mock import Mock

from btc_kvcm import PaceMetadata, PaceSpec, PaceStorageBackend


class DescriptorTest(unittest.TestCase):
    def setUp(self):
        self.key = b"key\x00one"
        self.other = b"key\x00two"
        self.specs = [PaceSpec("key", 0, (8,)), PaceSpec("value", 8, (8,))]
        self.metadata = object.__new__(PaceMetadata)
        self.metadata._handle = None
        self.metadata._specs = {0: self.specs}
        self.checksum = {"ch3:0": "1" * 16}
        self.metadata._call = Mock(return_value=[(0, json.dumps([
            0, self.key.hex(), {spec.name: ("pace://" + spec.name, self.checksum) for spec in self.specs},
        ]))])
        self.storage = object.__new__(PaceStorageBackend)
        self.storage._connected = True
        self.storage._payload = 16
        self.storage._metadata = self.metadata
        self.storage._specs = self.metadata._specs
        self.storage._host = Mock()
        self.storage._host.offset.return_value = 32
        self.storage._host.read_many.return_value = True

    def test_query_and_read_use_the_same_proof(self):
        descriptions = self.storage.describe([(0, self.key), (0, self.other)])
        self.assertEqual(set(descriptions), {(0, self.key)})
        self.metadata._call.return_value.clear()
        self.assertTrue(self.storage.read_described([(0, self.key, object(), descriptions[0, self.key])]))
        self.metadata._call.assert_called_once_with("describe", rows=[
            (0, int.from_bytes(self.key, "big", signed=True), self.key.hex()),
            (0, int.from_bytes(self.other, "big", signed=True), self.other.hex()),
        ])
        self.storage._host.read_many.assert_called_once_with([
            ("pace://key", self.specs[0], 32, self.checksum),
            ("pace://value", self.specs[1], 40, self.checksum),
        ])

    def test_wrong_key_and_partial_group_do_not_read(self):
        descriptor = self.storage.describe([(0, self.key)])[0, self.key]
        with self.assertRaisesRegex(ValueError, "another cache page"):
            self.storage.read_described([(0, self.other, object(), descriptor)])
        self.metadata._call.return_value = []
        self.assertFalse(self.storage.read_many([(0, self.key, object())]))
        self.storage._host.read_many.assert_not_called()
