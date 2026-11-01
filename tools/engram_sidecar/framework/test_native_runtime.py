import unittest

from tools.engram_sidecar.framework.native_runtime import assert_single_compute_library


class NativeRuntimeTest(unittest.TestCase):
    def test_segments_and_aliases_of_one_file_are_not_duplicates(self):
        text = "\n".join([
            "1000-2000 r--p 00000000 08:01 42 /build/librtp_compute_ops.so",
            "2000-3000 r-xp 00001000 08:01 42 /build/librtp_compute_ops.so",
            "4000-5000 r--p 00000000 08:01 42 /libs/librtp_compute_ops.so",
            "6000-7000 rw-p 00000000 00:00 0 [heap]",
        ])
        self.assertEqual(len(assert_single_compute_library(text)), 2)

    def test_copied_compute_has_private_callback_state(self):
        text = "\n".join([
            "1000-2000 r-xp 00001000 08:01 42 /build/librtp_compute_ops.so",
            "3000-4000 r-xp 00001000 08:01 43 /repo/librtp_compute_ops.real.so",
        ])
        with self.assertRaisesRegex(RuntimeError, "found 2"):
            assert_single_compute_library(text)

    def test_device_is_part_of_identity_and_missing_or_deleted_is_error(self):
        for text in ("", "1000-2000 r-xp 00001000 08:01 42 /build/librtp_compute_ops.so (deleted)",
                     "1000-2000 r-xp 00001000 08:01 42 /a/librtp_compute_ops.so\n"
                     "3000-4000 r-xp 00001000 08:02 42 /b/librtp_compute_ops.so"):
            with self.assertRaises(RuntimeError):
                assert_single_compute_library(text)


if __name__ == "__main__":
    unittest.main()
