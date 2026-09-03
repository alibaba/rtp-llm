import os
import unittest
from unittest.mock import patch

from rtp_llm.utils import expandable_segments as es


class ExpandableSegmentsTest(unittest.TestCase):
    def setUp(self):
        self.saved_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        self.saved_marker = os.environ.get(es._REQUESTED_ENV)
        es._reset_for_testing()

    def tearDown(self):
        es._reset_for_testing()
        if self.saved_conf is None:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        else:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = self.saved_conf
        if self.saved_marker is None:
            os.environ.pop(es._REQUESTED_ENV, None)
        else:
            os.environ[es._REQUESTED_ENV] = self.saved_marker

    def test_alloc_conf_without_expandable(self):
        self.assertEqual(
            es._alloc_conf_without_expandable(
                "expandable_segments:True, max_split_size_mb:128"
            ),
            "max_split_size_mb:128",
        )
        self.assertEqual(
            es._alloc_conf_without_expandable("garbage_collection_threshold:0.9"),
            "garbage_collection_threshold:0.9",
        )
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            "garbage_collection_threshold:0.9, expandable_segments:FALSE"
        )
        self.assertFalse(es.prepare_expandable_segments())

    def test_live_config_preserves_other_options(self):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            "garbage_collection_threshold:0.9,expandable_segments:True"
        )
        with patch.object(es, "_set_live"):
            es.prepare_expandable_segments()
        self.assertEqual(
            es._live_conf(True),
            "garbage_collection_threshold:0.9,expandable_segments:True",
        )

    def test_prepare_defers_and_runtime_enables(self):
        os.environ[
            "PYTORCH_CUDA_ALLOC_CONF"
        ] = "expandable_segments:True,max_split_size_mb:128"
        applied = []

        def fake_set_live(enabled):
            applied.append(enabled)
            es._live = enabled

        with patch.object(es, "_set_live", side_effect=fake_set_live):
            self.assertTrue(es.prepare_expandable_segments())
            self.assertEqual(
                os.environ["PYTORCH_CUDA_ALLOC_CONF"], "max_split_size_mb:128"
            )
            self.assertEqual(applied, [False])
            self.assertTrue(es.enable_runtime_expandable())
            self.assertEqual(applied, [False, True])
            self.assertNotIn(es._REQUESTED_ENV, os.environ)
            self.assertFalse(es.enable_runtime_expandable())
            self.assertTrue(es.is_runtime_expandable_active())

    def test_spawn_marker_recovers_request_after_parent_strips_env(self):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        os.environ[es._REQUESTED_ENV] = "1"
        with patch.object(es, "_set_live") as set_live:
            self.assertTrue(es.prepare_expandable_segments())
            set_live.assert_called_once_with(False)
        self.assertEqual(
            os.environ["PYTORCH_CUDA_ALLOC_CONF"], "max_split_size_mb:128"
        )

    def test_prepare_is_idempotent(self):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        with patch.object(es, "_set_live") as set_live:
            self.assertTrue(es.prepare_expandable_segments())
            self.assertTrue(es.prepare_expandable_segments())
            set_live.assert_called_once_with(False)

    def test_disabled_context_restores_runtime_setting(self):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        applied = []

        def fake_set_live(enabled):
            applied.append(enabled)
            es._live = enabled

        with patch.object(es, "_set_live", side_effect=fake_set_live):
            es.prepare_expandable_segments()
            es.enable_runtime_expandable()
            with es.expandable_segments_disabled():
                self.assertTrue(es._active)
                self.assertFalse(es._live)
            self.assertTrue(es.is_runtime_expandable_active())
        self.assertEqual(applied, [False, True, False, True])

    def test_disabled_context_restores_after_exception(self):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        applied = []

        def fake_set_live(enabled):
            applied.append(enabled)
            es._live = enabled

        with patch.object(es, "_set_live", side_effect=fake_set_live):
            es.prepare_expandable_segments()
            es.enable_runtime_expandable()
            with self.assertRaisesRegex(RuntimeError, "inside-vmm"):
                with es.expandable_segments_disabled():
                    raise RuntimeError("inside-vmm")
            self.assertTrue(es.is_runtime_expandable_active())
        self.assertEqual(applied, [False, True, False, True])

    def test_unrequested_is_inert(self):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        with patch.object(es, "_set_live") as set_live:
            self.assertFalse(es.prepare_expandable_segments())
            self.assertFalse(es.enable_runtime_expandable())
            set_live.assert_not_called()


if __name__ == "__main__":
    unittest.main()
