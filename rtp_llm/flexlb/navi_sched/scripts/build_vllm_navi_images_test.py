#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).with_name("build_vllm_navi_images.py")
SPEC = importlib.util.spec_from_file_location("build_vllm_navi_images", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BuildVllmNaviImagesTest(unittest.TestCase):
    @mock.patch.object(MODULE.time, "sleep")
    @mock.patch.object(MODULE, "get_run")
    def test_wait_accepts_both_success_statuses(self, get_run, _sleep):
        for status in ("COMPLETE", "SUCCESS"):
            with self.subTest(status=status):
                expected = {
                    "status": status,
                    "isFinished": True,
                    "isFailure": False,
                }
                get_run.return_value = expected
                self.assertEqual(
                    expected,
                    MODULE.wait_for_success("repo", 123, 0),
                )

    @mock.patch.object(MODULE.time, "sleep")
    @mock.patch.object(MODULE, "get_run")
    def test_wait_rejects_failed_or_unknown_terminal_status(self, get_run, _sleep):
        for result in (
            {"status": "FAILED", "isFinished": True, "isFailure": True},
            {"status": "CANCELED", "isFinished": True, "isFailure": False},
        ):
            with self.subTest(result=result):
                get_run.return_value = result
                with self.assertRaises(RuntimeError):
                    MODULE.wait_for_success("repo", 123, 0)

    @mock.patch.object(MODULE, "run")
    def test_find_image_selects_latest_exact_match(self, run):
        run.return_value = "\n".join(
            [
                "prefix_20260930010101_accelerated",
                "prefix_20260930010102_containerd_accelerated",
                "prefix_20260930010103_accelerated",
            ]
        )
        image = MODULE.find_image("repo", r"prefix_\d{14}_accelerated")
        self.assertEqual("repo:prefix_20260930010103_accelerated", image)

    @mock.patch.object(MODULE, "run")
    def test_verify_image_checks_multi_arch_platforms(self, run):
        run.side_effect = [
            "sha256:abc\n",
            '{"manifests":['
            '{"platform":{"os":"linux","architecture":"amd64"}},'
            '{"platform":{"os":"linux","architecture":"arm64"}}]}',
        ]
        result = MODULE.verify_image(
            "repo:tag", {"linux/amd64", "linux/arm64"}
        )
        self.assertEqual("sha256:abc", result["digest"])
        self.assertEqual(["linux/amd64", "linux/arm64"], result["platforms"])

    @mock.patch.object(MODULE, "run")
    def test_verify_image_checks_single_arch_config(self, run):
        run.side_effect = [
            "sha256:def\n",
            '{"schemaVersion":2}',
            '{"os":"linux","architecture":"amd64"}',
        ]
        result = MODULE.verify_image("repo:tag", {"linux/amd64"})
        self.assertEqual(["linux/amd64"], result["platforms"])


if __name__ == "__main__":
    unittest.main()
