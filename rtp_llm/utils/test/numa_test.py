import ctypes
import errno
import os
import unittest
from unittest import mock

from rtp_llm.utils import numa


class NumaTest(unittest.TestCase):
    def test_automatic_policy_depends_on_allowed_memory_nodes(self):
        for memory, allowed, expected in [
            ("0-1", "0-1", "0,1"),
            ("0-3", "1,3", "1,3"),
            ("0-1", "1", None),
            ("0-3", "0-3", None),
            ("0", "0", None),
        ]:
            with self.subTest(memory=memory, allowed=allowed), mock.patch.dict(
                os.environ, {}, clear=True
            ), mock.patch.object(numa.sys, "platform", "linux"), mock.patch.object(
                numa.Path,
                "read_text",
                side_effect=[memory, f"Name:\tpython\nMems_allowed_list:\t{allowed}\n"],
            ), mock.patch.object(
                numa, "_set_interleave"
            ) as apply:
                numa.configure_numa_interleave()
                self.assertEqual(os.environ.get("NUMA_INTERLEAVE_NODES"), expected)
                if expected:
                    apply.assert_called_once_with(set(map(int, expected.split(","))))
                else:
                    apply.assert_not_called()

    def test_explicit_override_and_disable_skip_detection(self):
        for value in ("", "2-3"):
            with self.subTest(value=value), mock.patch.dict(
                os.environ, {"NUMA_INTERLEAVE_NODES": value}, clear=True
            ), mock.patch.object(
                numa, "_available_memory_nodes"
            ) as detect, mock.patch.object(
                numa, "_set_interleave"
            ) as apply:
                numa.configure_numa_interleave()
                detect.assert_not_called()
                if value:
                    apply.assert_called_once_with({2, 3})
                else:
                    apply.assert_not_called()

    def test_permission_failure_is_visible_and_explicit_mode_fails(self):
        for configured in (None, "0,1"):
            env = {} if configured is None else {"NUMA_INTERLEAVE_NODES": configured}
            with self.subTest(configured=configured), mock.patch.dict(
                os.environ, env, clear=True
            ), mock.patch.object(numa.sys, "platform", "linux"), mock.patch.object(
                numa, "_available_memory_nodes", return_value={0, 1}
            ), mock.patch.object(
                numa, "_set_interleave", side_effect=PermissionError("denied")
            ), mock.patch(
                "sys.stderr"
            ) as stderr:
                if configured is None:
                    numa.configure_numa_interleave()
                    self.assertNotIn("NUMA_INTERLEAVE_NODES", os.environ)
                    self.assertTrue(stderr.write.called)
                else:
                    with self.assertRaises(RuntimeError):
                        numa.configure_numa_interleave()

    def test_libnuma_mask_and_errno(self):
        word_bits = ctypes.sizeof(ctypes.c_ulong) * 8
        with mock.patch.object(numa.ctypes, "CDLL") as load:
            policy = load.return_value.set_mempolicy
            policy.return_value = 0
            numa._set_interleave({1, word_bits + 2})
            mode, mask, maxnode = policy.call_args.args
            self.assertEqual(
                (mode, list(mask), maxnode), (3, [2, 4], 2 * word_bits + 1)
            )
            policy.return_value = -1
            with mock.patch.object(numa.ctypes, "get_errno", return_value=errno.EPERM):
                with self.assertRaises(OSError) as error:
                    numa._set_interleave({0, 1})
            self.assertEqual(error.exception.errno, errno.EPERM)


if __name__ == "__main__":
    unittest.main()
