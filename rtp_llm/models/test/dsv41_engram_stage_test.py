"""Stage lifetime, aggregate quotas and isolated Engram forward frames."""

import dataclasses
import pickle
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

from rtp_llm.models.test.dsv41_cache_codec_test import load_codec_source
from rtp_llm.models.test.dsv41_engram_hash_test import engram as hashing
from rtp_llm.models.test.dsv41_engram_weights_test import (
    host,
    linear,
    loading,
    write_checkpoint,
)

with patch.dict(
    sys.modules,
    {
        "rtp_llm.models_py.modules.dsv4.v41_engram_hash": hashing,
        "rtp_llm.models_py.modules.dsv4.v41_engram_host": host,
        "rtp_llm.models_py.modules.dsv4.v41_engram_weights": loading,
        "rtp_llm.models_py.modules.dsv4.v41_mxfp8": linear,
    },
):
    stage_module = load_codec_source("_dsv41_engram_stage_test", "v41_engram_stage.py")


class V41EngramStageTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config, _ = write_checkpoint(self.root, ((1, 10), (14, 31)))
        self.config.text.update(engram_compressed_vocab_size=3, engram_vocab_size=2)
        self.sources = loading.plan_stage_engram_sources(
            self.config, self.root, range(21), "a" * 64
        )

    def hasher(self):
        layout = hashing.V41EngramLayout.from_config(self.config.text, 2)
        return hashing.V41EngramHasher(layout, [0, 1, 2, 0, 1, 2, 1])

    def stage(self, sources=None, **kwargs):
        sources = self.sources if sources is None else sources
        args = dict(
            device="cpu",
            max_host_bytes=sum(s.table.nbytes for s in sources.values()),
            max_weight_bytes=sum(s.weight_bytes for s in sources.values()),
            directory=self.root,
            chunk_tokens=2,
            backend="eager",
        )
        args.update(kwargs)
        result = stage_module.V41EngramStage(
            sources, self.hasher() if sources else None, **args
        )
        self.addCleanup(result.close)
        return result

    @staticmethod
    def windows():
        return torch.tensor(
            [
                [0, -1, -1, -1],
                [4, 0, -1, -1],
                [6, 4, 0, -1],
                [-1, 6, 4, 0],
                [3, 1, 6, 4],
            ]
        )

    def test_lifecycle_two_layers_and_explicit_forward_equivalence(self):
        stage = self.stage()
        with self.assertRaisesRegex(RuntimeError, "not ready"):
            stage.prepare(self.windows())
        self.assertIs(stage.initialize(), stage)
        self.assertIs(stage.initialize(), stage)
        self.assertEqual(stage.layer_ids, (1, 14))
        frame = stage.prepare(self.windows())
        hidden = torch.randn(5, 4, 32).bfloat16()
        expected_hashes = self.hasher()(self.windows(), backend="eager")
        for index, layer in enumerate(stage.layer_ids):
            expected = stage._modules[layer](
                hidden, expected_hashes.ids[:, index], expected_hashes.valid_rows
            )
            torch.testing.assert_close(
                stage.apply(layer, hidden, frame), expected, rtol=0, atol=0
            )
        self.assertEqual(set(stage.__dict__) & {"hashes", "frame", "windows"}, set())
        stage.close()
        self.assertEqual(stage.state, "closed")
        self.assertEqual(stage._modules, {})
        with self.assertRaisesRegex(RuntimeError, "not ready"):
            stage.apply(1, hidden, frame)
        with self.assertRaisesRegex(RuntimeError, "new owner"):
            stage.initialize()

    def test_aggregate_admission_precedes_any_table_attachment(self):
        with patch.object(
            stage_module,
            "_PinnedSharedTable",
            side_effect=AssertionError("table attachment"),
        ):
            for kwargs in (
                {"max_host_bytes": self.sources[1].table.nbytes},
                {"max_weight_bytes": self.sources[1].weight_bytes},
                {"max_host_bytes": True},
                {"max_weight_bytes": -1},
            ):
                with self.assertRaisesRegex(ValueError, "budget"):
                    self.stage(**kwargs)

    def test_stage_one_has_no_file_device_or_hash_work(self):
        with patch.object(
            Path, "open", side_effect=AssertionError("file I/O")
        ), patch.object(
            torch.cuda, "current_device", side_effect=AssertionError("CUDA query")
        ), patch.object(
            stage_module,
            "_PinnedSharedTable",
            side_effect=AssertionError("table attach"),
        ):
            stage = self.stage(
                {}, device="cuda", backend="flashinfer", directory="/missing"
            )
            with stage:
                self.assertEqual(stage.layer_ids, ())
                self.assertIsNone(stage.prepare(None))
                self.assertEqual((stage.host_bytes, stage.weight_bytes), (0, 0))

    def test_interleaved_frames_and_cp_permutations_remain_independent(self):
        with self.stage() as stage:
            windows = self.windows()
            order = torch.tensor([4, 1, 3, 0, 2])
            first, second = stage.prepare(windows), stage.prepare(windows[order])
            hidden = torch.randn(5, 4, 32).bfloat16()
            output = stage.apply(1, hidden, first)
            actual = stage.apply(1, hidden[order], second)
            torch.testing.assert_close(actual, output[order], rtol=0, atol=0)
            torch.testing.assert_close(
                stage.apply(1, hidden, first), output, rtol=0, atol=0
            )
            alias = hidden.clone()
            self.assertIs(stage.apply(1, alias, first, out=alias), alias)
            torch.testing.assert_close(alias, output, rtol=0, atol=0)

    def test_two_stage_owners_share_tables_but_reject_each_others_frames(self):
        with self.stage() as first, self.stage() as second:
            for layer in first.layer_ids:
                self.assertEqual(
                    first._tables[layer]._path.stat().st_ino,
                    second._tables[layer]._path.stat().st_ino,
                )
            frame = first.prepare(self.windows())
            hidden = torch.ones(5, 4, 32, dtype=torch.bfloat16)
            with self.assertRaisesRegex(ValueError, "different stage"):
                second.apply(1, hidden, frame)
            paths = [table._path for table in first._tables.values()]
            first.close()
            self.assertTrue(all(path.exists() for path in paths))
            second.apply(14, hidden, second.prepare(self.windows()))
        self.assertTrue(all(not path.exists() for path in paths))

    def test_startup_failure_releases_prior_layers_and_never_publishes_ready(self):
        stage = self.stage()
        original = stage_module.load_engram_module

        def fail_second(source, *args, **kwargs):
            if source.layer_id == 14:
                raise RuntimeError("projection load failed")
            return original(source, *args, **kwargs)

        with patch.object(stage_module, "load_engram_module", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "projection load failed"):
                stage.initialize()
        self.assertEqual(stage.state, "closed")
        self.assertEqual(stage._tables, {})
        self.assertEqual(stage._modules, {})
        self.assertEqual(list(self.root.glob("rtp_llm_engram_*.json")), [])

    def test_failed_teardown_retains_weights_and_can_be_retried(self):
        stage = self.stage().initialize()
        table = stage._tables[14]
        with patch.object(table, "close", side_effect=RuntimeError("drain failed")):
            with self.assertRaisesRegex(RuntimeError, "retain owner"):
                stage.close()
        self.assertEqual(stage.state, "closing")
        self.assertEqual(set(stage._tables), {14})
        self.assertEqual(set(stage._modules), {1, 14})
        with self.assertRaisesRegex(RuntimeError, "not ready"):
            stage.prepare(self.windows())
        stage.close()
        self.assertEqual(stage.state, "closed")
        self.assertEqual(stage._modules, {})

    def test_failed_startup_cleanup_retains_an_owner_for_retry(self):
        stage = self.stage()
        original = stage_module.load_engram_module
        with ExitStack() as stack:

            def fail_second(source, *args, **kwargs):
                if source.layer_id == 14:
                    stack.enter_context(
                        patch.object(
                            stage._tables[1],
                            "close",
                            side_effect=RuntimeError("drain failed"),
                        )
                    )
                    raise RuntimeError("weight load failed")
                return original(source, *args, **kwargs)

            stack.enter_context(
                patch.object(
                    stage_module, "load_engram_module", side_effect=fail_second
                )
            )
            with self.assertRaisesRegex(
                RuntimeError, "cleanup must be retried"
            ) as error:
                stage.initialize()
            self.assertIn("weight load failed", str(error.exception.__cause__))
            self.assertEqual(stage.state, "closing")
            self.assertEqual(set(stage._tables), {1})
            self.assertEqual(set(stage._modules), {1})
        stage.close()
        self.assertEqual(stage.state, "closed")

    def test_wrong_layout_checkpoint_owner_and_fork_are_rejected(self):
        wrong = dict(self.sources)
        wrong[14] = dataclasses.replace(
            wrong[14],
            table=dataclasses.replace(wrong[14].table, checkpoint_sha256="b" * 64),
        )
        with self.assertRaisesRegex(ValueError, "mix deployment"):
            self.stage(wrong)
        wrong = dict(self.sources)
        wrong[2] = wrong.pop(1)
        with self.assertRaisesRegex(ValueError, "ownership"):
            self.stage(wrong)
        stage = self.stage()
        with patch.object(stage_module.os, "getpid", return_value=stage._pid + 1):
            with self.assertRaisesRegex(RuntimeError, "after fork"):
                stage.initialize()
            with self.assertRaisesRegex(RuntimeError, "after fork"):
                stage.close()
        with self.assertRaisesRegex(TypeError, "pickled"):
            pickle.dumps(stage)

    def test_empty_frames_and_invalid_frame_rows_do_not_write_output(self):
        with self.stage() as stage:
            frame = stage.prepare(torch.empty(0, 4, dtype=torch.int64))
            output = stage.apply(1, torch.empty(0, 4, 32, dtype=torch.bfloat16), frame)
            self.assertEqual(output.shape, (0, 4, 32))
            hidden = torch.ones(5, 4, 32, dtype=torch.bfloat16)
            out = torch.full_like(hidden, 42)
            with self.assertRaisesRegex(ValueError, "rows/layout"):
                stage.apply(1, hidden, frame, out=out)
            with self.assertRaisesRegex(ValueError, "not owned"):
                stage.apply(2, hidden, frame, out=out)
            torch.testing.assert_close(out, torch.full_like(out, 42))


if __name__ == "__main__":
    unittest.main()
