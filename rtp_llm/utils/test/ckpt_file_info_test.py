import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.utils.ckpt_file_info import CkptFileInfo


class CheckpointByteStreamTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "weights.safetensors"
        self.tensors = {
            "first": torch.arange(35, dtype=torch.int32),
            "last": torch.arange(63, dtype=torch.uint8),
        }
        save_file(self.tensors, self.path)
        self.checkpoint = CkptFileInfo(str(self.path))

    def test_stream_preserves_bytes_and_tensor_offsets(self):
        for name, tensor in self.tensors.items():
            expected = tensor.view(torch.uint8).numpy().tobytes()
            output = bytearray(len(expected))
            self.checkpoint.copy_tensor_bytes(name, output)
            self.assertEqual(output, expected)

    def test_stream_handles_short_reads(self):
        class ShortRead(io.FileIO):
            def readinto(self, buffer):
                with buffer[:7] as short:
                    return super().readinto(short)

        output = bytearray(63)
        with patch(
            "rtp_llm.utils.ckpt_file_info.open",
            lambda path, mode, buffering: ShortRead(path, mode),
            create=True,
        ):
            self.checkpoint.copy_tensor_bytes("last", output)
        self.assertEqual(output, self.tensors["last"].numpy().tobytes())

    def test_stream_rejects_wrong_size_and_readonly_destination(self):
        for output in (bytearray(62), bytes(63)):
            with self.assertRaisesRegex(ValueError, "Destination"):
                self.checkpoint.copy_tensor_bytes("last", output)

    def test_stream_rejects_a_truncated_checkpoint(self):
        name = max(self.tensors, key=self.checkpoint.metadata.get)
        size = self.tensors[name].numel() * self.tensors[name].element_size()
        with self.path.open("r+b") as file:
            file.truncate(self.path.stat().st_size - 1)
        with self.assertRaisesRegex(EOFError, "Truncated checkpoint tensor"):
            self.checkpoint.copy_tensor_bytes(name, bytearray(size))


if __name__ == "__main__":
    unittest.main()
