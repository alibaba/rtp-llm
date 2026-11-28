import errno
import mmap
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from rtp_llm.models_py import weight_mapper


@unittest.skipUnless(os.name == "posix", "private POSIX mapping")
class PytorchArchiveReaderTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "pytorch_model.bin"
        base = torch.arange(24).reshape(4, 6)
        self.state = {"base": base, "view": base[1:, 1::2]}

    def read(self):
        return dict(weight_mapper.get_all_weights([str(self.path)], device="cpu"))

    def test_private_views_keep_storage_alive_and_leave_archive_unchanged(self):
        torch.save({"state_dict": self.state}, self.path)
        with mock.patch.object(torch, "load", wraps=torch.load) as load:
            weights = self.read()
        load.assert_called_once_with(
            str(self.path), map_location="cpu", weights_only=True, mmap=True
        )
        view = weights["view"]
        self.assertEqual(view.stride(), (6, 2))
        self.assertEqual(view.storage_offset(), 7)
        view[0, 0] = -1
        self.assertEqual(weights["base"][1, 1].item(), -1)
        # Both views outlived the reader. Another reader must see the file
        # contents, not the in-place edit in the first reader's private map.
        self.assertTrue(torch.equal(self.read()["view"], self.state["view"]))

    def test_legacy_and_shared_mapping_policy_use_the_eager_reader(self):
        for legacy in (True, False):
            with self.subTest(legacy=legacy):
                torch.save(
                    self.state, self.path, _use_new_zipfile_serialization=not legacy
                )
                flags = mmap.MAP_PRIVATE if legacy else mmap.MAP_SHARED
                with torch.serialization.set_default_mmap_options(flags), mock.patch.object(
                    torch, "load", wraps=torch.load
                ) as load:
                    weights = self.read()
                load.assert_called_once_with(
                    str(self.path), map_location="cpu", weights_only=True, mmap=False
                )
                weights["view"][0, 0] = -1
                self.assertEqual(weights["base"][1, 1].item(), -1)
                self.assertTrue(torch.equal(self.read()["base"], self.state["base"]))

    def test_unsupported_filesystem_only_falls_back_on_mapping_capability_errors(self):
        torch.save(self.state, self.path)
        with mock.patch.object(
            mmap, "mmap", side_effect=OSError(errno.ENODEV, "cannot map")
        ), mock.patch.object(torch, "load", wraps=torch.load) as load:
            self.assertTrue(torch.equal(self.read()["base"], self.state["base"]))
        load.assert_called_once_with(
            str(self.path), map_location="cpu", weights_only=True, mmap=False
        )
        for code in (errno.ENOTCONN, errno.EIO, errno.ENOMEM):
            with self.subTest(errno=code), mock.patch.object(
                mmap, "mmap", side_effect=OSError(code, "failed")
            ), mock.patch.object(torch, "load") as load:
                with self.assertRaises(OSError) as failure:
                    self.read()
                self.assertEqual(failure.exception.errno, code)
                load.assert_not_called()

    def test_corrupt_zip_is_not_retried(self):
        self.path.write_bytes(b"PK\x03\x04truncated")
        with mock.patch.object(torch, "load", wraps=torch.load) as load:
            with self.assertRaises(RuntimeError):
                self.read()
        self.assertEqual(load.call_count, 1)

    def test_non_cpu_reader_and_safetensors_do_not_use_archive_mapping_policy(self):
        from safetensors.torch import save_file

        torch.save(self.state, self.path)
        safe = self.path.with_suffix(".safetensors")
        save_file({"base": self.state["base"]}, str(safe))
        with mock.patch.object(
            weight_mapper, "_can_map_cpu_archive", side_effect=AssertionError
        ):
            meta = dict(weight_mapper.get_all_weights([str(self.path)], device="meta"))
            self.assertEqual(meta["base"].device.type, "meta")
            actual = dict(weight_mapper.get_all_weights([str(safe)], device="cpu"))
            self.assertTrue(torch.equal(actual["base"], self.state["base"]))


if __name__ == "__main__":
    unittest.main()
