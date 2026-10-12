"""Qualify device frequency construction on held NAS output projections."""

import hashlib
import importlib
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4 import rope
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_attention_owner_sm120 import (
    make_layer,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_cuda_frequency_reference import (
    cuda_frequency_reference,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_output_precision import (
    output_projection_reference,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_tp8_projection import tp8_projection
from rtp_llm.tools.dsv41_attention_replay import (
    NASReader,
    native_attention,
    official_attention,
)
from rtp_llm.tools.dsv41_output_projection_replay import execute_case, immutable_sources
from rtp_llm.tools.dsv41_query_owner_replay import tensor_digest
from rtp_llm.tools.dsv41_reference_probe import model_args


class CUDAFrequencySM120Test(unittest.TestCase):
    @classmethod
    @torch.inference_mode()
    def setUpClass(cls):
        if torch.cuda.get_device_capability() != (12, 0):
            raise RuntimeError("CUDA frequency qualification requires leased SM120")
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.bfloat16)
        torch.set_float32_matmul_precision("highest")
        path = Path(os.environ["DSV41_TEST_ATTENTION_BOUNDARY_TENSORS"])
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != os.environ["DSV41_TEST_ATTENTION_BOUNDARY_SHA256"]
        ):
            raise ValueError("Held projection tensors changed")
        cls.cases = torch.load(path, map_location="cpu", weights_only=True)[
            "projections"
        ]
        if [(c["step"], c["layer"]) for c in cls.cases] != [(2, 6), (1, 6), (0, 0)]:
            raise ValueError("Held projection coverage differs")
        root = Path(os.environ["DSV41_TEST_CHECKPOINT"])
        sys.path.insert(0, str(root / "inference"))
        cls.official, converter, cls.kernel = [
            importlib.import_module(n) for n in ("model", "convert", "kernel")
        ]
        for module in (cls.official, converter, cls.kernel):
            if Path(module.__file__).resolve() != root / "inference" / (
                module.__name__ + ".py"
            ):
                raise ValueError("Released module import differs")
        args = model_args(cls.official, root, 256)
        config = json.loads((root / "config.json").read_text())["text_config"]
        reader = NASReader(root)
        cls.owners = {
            layer: native_attention(args, config, reader, layer) for layer in (0, 6)
        }
        cls.shards = {
            layer: official_attention(cls.official, converter, args, reader, layer)
            for layer in (0, 6)
        }

    def exact(self, left, right):
        left, right = left.cpu().contiguous(), right.cpu().contiguous()
        self.assertEqual(left.dtype, right.dtype)
        self.assertEqual(left.numel(), right.numel())
        self.assertTrue(
            torch.equal(
                left.reshape(-1).view(torch.uint8), right.reshape(-1).view(torch.uint8)
            )
        )

    def projection(self, case):
        owner = self.owners[case["layer"]]
        start, rows = case["identity"]["start_pos"], len(case["identity"]["input_ids"])
        return owner._project_output(
            case["attention_before"].cuda().clone(),
            owner.freqs_cis[start : start + rows],
        )

    @torch.inference_mode()
    def test_all_held_tables_outputs_restoration_and_sources(self):
        memo = rope._FREQS_CIS_CACHE
        memo_sources = {
            key: (id(value), tensor_digest(value)) for key, value in memo.items()
        }
        with tp8_projection(True), output_projection_reference(
            True, kernel=self.kernel
        ):
            for case in self.cases:
                owner = self.owners[case["layer"]]
                original = owner.freqs_cis
                self.exact(case["native_freqs_full"], original)
                self.exact(case["outputs"]["candidate"], self.projection(case))
                sources = immutable_sources(owner, self.shards[case["layer"]])
                before = {name: tensor_digest(value) for name, value in sources.items()}
                with cuda_frequency_reference(True, attentions=[owner]):
                    self.assertIs(rope._FREQS_CIS_CACHE, memo)
                    self.exact(case["released_freqs_full"], owner.freqs_cis)
                    self.exact(case["outputs"]["released"], self.projection(case))
                    self.assertEqual(
                        before,
                        {name: tensor_digest(value) for name, value in sources.items()},
                    )
                self.assertIs(owner.freqs_cis, original)
                self.exact(case["outputs"]["candidate"], self.projection(case))
        self.assertIs(rope._FREQS_CIS_CACHE, memo)
        self.assertEqual(
            memo_sources,
            {key: (id(value), tensor_digest(value)) for key, value in memo.items()},
        )
        print(
            'DSV41_CUDA_FREQ_HELD {"cases": 3, "tables": 3, "restorations": 3, "sources_per_case": 54}',
            flush=True,
        )

    def test_disabled_distinct_owner_and_geometry_admission(self):
        owner = self.owners[6]
        original = owner.freqs_cis
        with cuda_frequency_reference(False, attentions=[object()]):
            self.assertIs(original, owner.freqs_cis)
        for enabled, owners in (
            (1, ()),
            (True, ()),
            (True, (object(),)),
            (True, (owner, owner)),
        ):
            with self.assertRaises(ValueError), cuda_frequency_reference(
                enabled, attentions=owners
            ):
                pass
        with patch.object(owner, "freqs_cis", original.real):
            with self.assertRaisesRegex(
                ValueError, "admitted"
            ), cuda_frequency_reference(True, attentions=[owner]):
                pass

    def test_unselected_nested_and_exception_restore(self):
        owner, other = self.owners[6], self.owners[0]
        original, unselected = owner.freqs_cis, other.freqs_cis
        with cuda_frequency_reference(True, attentions=[owner]):
            selected = owner.freqs_cis
            self.assertIs(other.freqs_cis, unselected)
            with cuda_frequency_reference(False), cuda_frequency_reference(
                True, attentions=[owner]
            ):
                self.exact(selected, owner.freqs_cis)
            self.assertIs(owner.freqs_cis, selected)
            with self.assertRaisesRegex(RuntimeError, "frequency sentinel"):
                with cuda_frequency_reference(True, attentions=[owner, other]):
                    raise RuntimeError("frequency sentinel")
            self.assertIs(owner.freqs_cis, selected)
            self.assertIs(other.freqs_cis, unselected)
        self.assertIs(owner.freqs_cis, original)

    @torch.inference_mode()
    def test_frequency_parameter_families(self):
        for ratio in (0, 1, 2):
            owner = make_layer(2 if ratio else 0, ratio)
            args = (
                owner._rope_dim,
                owner._rope_max_seq_len,
                owner._rope_o_seq_len,
                owner._rope_base,
                owner._rope_factor,
                owner._rope_beta_fast,
                owner._rope_beta_slow,
            )
            with torch.device(owner.freqs_cis.device):
                expected = self.official.precompute_freqs_cis(*args)
            with cuda_frequency_reference(True, attentions=[owner]):
                self.exact(expected, owner.freqs_cis)
        print('DSV41_CUDA_FREQ_PARAMETERS {"ratios": 3}', flush=True)

    @torch.inference_mode()
    def test_changing_projection_component_graphs(self):
        for case in (self.cases[0], self.cases[2]):
            owner = self.owners[case["layer"]]
            original = owner.freqs_cis
            rows = len(case["identity"]["input_ids"])
            attention = (
                case["attention_before"].cuda().reshape(rows, 64, 512).contiguous()
            )
            before = attention.clone()
            positions = (
                torch.arange(rows, device="cuda", dtype=torch.int64)
                + case["identity"]["start_pos"]
            )
            with cuda_frequency_reference(True, attentions=[owner]), tp8_projection(
                True
            ), output_projection_reference(True, kernel=self.kernel):

                def execute():
                    return owner._project_output(
                        attention.clone(), owner.freqs_cis[positions]
                    )

                execute()
                execute()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    result = execute()
                for step in range(3):
                    attention.copy_(before * (1 + 0.25 * step))
                    positions.copy_(
                        torch.arange(rows, device="cuda")
                        + case["identity"]["start_pos"]
                        + step
                    )
                    graph.replay()
                    raw, _ = execute_case(
                        owner,
                        self.shards[case["layer"]],
                        self.official,
                        self.kernel,
                        attention.clone(),
                        owner.freqs_cis[positions],
                        self.shards[case["layer"]][0].freqs_cis[positions],
                    )
                    self.exact(raw["outputs"]["released"], result)
            self.assertIs(owner.freqs_cis, original)
        print('DSV41_CUDA_FREQ_GRAPHS {"graphs": 2, "changed_replays": 6}', flush=True)


if __name__ == "__main__":
    unittest.main()
