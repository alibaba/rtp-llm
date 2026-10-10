"""Qualify scoped output GEMMs against frozen held NAS projections."""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import AttentionV41FP8
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_output_precision import (
    output_projection_reference,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_query_precision import (
    query_projection_reference,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_tp8_projection import tp8_projection
from rtp_llm.models_py.modules.dsv4.v41_mxfp8 import V41MXFP8Linear
from rtp_llm.tools.dsv41_attention_replay import NASReader, native_attention
from rtp_llm.tools.dsv41_output_projection_replay import admit_owner
from rtp_llm.tools.dsv41_query_boundary_replay import sha256
from rtp_llm.tools.dsv41_reference_probe import model_args


class V41OutputPrecisionSM120Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib
        import sys
        from types import SimpleNamespace

        if torch.cuda.get_device_capability() != (12, 0):
            raise RuntimeError(
                "Output GEMM qualification requires leased SM120 hardware"
            )
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.bfloat16)
        torch.set_float32_matmul_precision("highest")
        cls.root = Path(os.environ["DSV41_TEST_CHECKPOINT"])
        artifacts = Path(os.environ["DSV41_TEST_OUTPUT_ARTIFACT_ROOT"])
        repo = Path(__file__).resolve().parents[6]
        sys.path.insert(0, str(cls.root / "inference"))
        official = importlib.import_module("model")
        args = model_args(official, cls.root, 256)
        config = json.loads((cls.root / "config.json").read_text())["text_config"]
        reader = NASReader(cls.root)
        cls.owners = [
            native_attention(args, config, reader, layer) for layer in range(3)
        ]
        cls.cases = []
        cls.queries = {}
        for prompt in ("france", "arithmetic"):
            name = f"output-projection-{prompt}-1"
            report = json.loads((artifacts / f"{name}.json").read_text())
            manifest = json.loads((artifacts / f"{name}.run.json").read_text())
            admission = SimpleNamespace(
                **{
                    field: Path(manifest[field])
                    for field in (
                        "history_report",
                        "history_artifact",
                        "query_report",
                        "query_artifact",
                        "owner_report",
                        "owner_artifact",
                    )
                }
            )
            _, held, _, _, _ = admit_owner(admission, cls.root, repo)
            queries = torch.load(
                admission.query_artifact, map_location="cpu", weights_only=True
            )
            for case in queries["cases"]:
                cls.queries[prompt, case["stream"], case["layer"], case["step"]] = case
            if reader.hashes != report["weight_sha256"]:
                raise ValueError(
                    "Output qualification weights differ from the held replay"
                )
            path = artifacts / f"{name}.pt"
            if sha256(path) != report["tensor_artifact_sha256"]:
                raise ValueError("Qualified output tensor digest differs")
            raw = torch.load(path, map_location="cpu", weights_only=True)
            for case in raw["cases"]:
                cls.cases.append(
                    (
                        prompt,
                        case,
                        held[f"{case['stream']}_inputs"][case["step"]][case["layer"]]
                        .cuda()
                        .contiguous(),
                    )
                )
        if len(cls.cases) != 42:
            raise ValueError("Output qualification requires all 42 held cases")

    def exact(self, expected, actual):
        expected = expected.cpu().contiguous().reshape(-1)
        actual = actual.cpu().contiguous().reshape(-1)
        self.assertEqual(expected.dtype, actual.dtype)
        self.assertEqual(expected.shape, actual.shape)
        self.assertTrue(
            torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
        )

    def projection(self, owner, case, *, out=None):
        attention = case["attention_before"].cuda().clone()
        start = case["identity"]["start_pos"]
        rows = len(case["identity"]["input_ids"])
        freqs = owner.freqs_cis[start : start + rows]
        self.exact(case["native_freqs"], freqs)
        return owner._project_output(attention, freqs, out=out)

    @torch.inference_mode()
    def test_all_native_controls(self):
        with tp8_projection(True):
            for prompt, case, _ in self.cases:
                with self.subTest(
                    prompt=prompt,
                    layer=case["layer"],
                    step=case["step"],
                    stream=case["stream"],
                ):
                    self.exact(
                        case["outputs"]["native"],
                        self.projection(self.owners[case["layer"]], case),
                    )

    @torch.inference_mode()
    def test_composed_outputs_and_unselected_kv(self):
        with tp8_projection(True):
            for _, case, value in self.cases:
                owner = self.owners[case["layer"]]
                sources = (
                    case["attention_before"],
                    owner.wo_a_w,
                    owner.wo_a_s,
                    owner._wo_a_bf16_v41,
                    owner.wo_b.weight,
                    owner.wo_b._sm120_linear.scales,
                    owner.freqs_cis,
                )
                before = [source.clone() for source in sources]
                kv = owner._lin(owner.wkv, value).clone()
                with output_projection_reference(True):
                    self.exact(
                        case["outputs"]["candidate"], self.projection(owner, case)
                    )
                    self.exact(kv, owner._lin(owner.wkv, value))
                self.exact(case["outputs"]["native"], self.projection(owner, case))
                for original, current in zip(before, sources):
                    self.exact(original, current)
        print(
            "DSV41_OUTPUT_COMPOSED "
            + json.dumps(
                {
                    "released_outputs_exact": 42,
                    "native_restorations_exact": 42,
                    "unselected_kv_exact": 42,
                }
            ),
            flush=True,
        )

    @torch.inference_mode()
    def test_nested_query_and_output_contexts(self):
        with tp8_projection(True), query_projection_reference(
            True
        ), output_projection_reference(True):
            for prompt, case, value in self.cases:
                owner = self.owners[case["layer"]]
                expected = self.queries[
                    prompt, case["stream"], case["layer"], case["step"]
                ]
                self.exact(expected["released_a"], owner._lin(owner.wq_a, value))
                self.exact(case["outputs"]["candidate"], self.projection(owner, case))
        print(
            "DSV41_OUTPUT_NESTED " + json.dumps({"query_output_controls_exact": 42}),
            flush=True,
        )

    @torch.inference_mode()
    def test_caller_output_buffer(self):
        _, case, _ = self.cases[0]
        owner = self.owners[case["layer"]]
        out = torch.empty_like(case["outputs"]["candidate"], device="cuda")
        with tp8_projection(True), output_projection_reference(True):
            result = self.projection(owner, case, out=out)
        self.assertEqual(result.data_ptr(), out.data_ptr())
        self.exact(case["outputs"]["candidate"], out)

    @torch.inference_mode()
    def test_exception_restores_selection_and_patches(self):
        _, case, value = self.cases[0]
        owner = self.owners[case["layer"]]
        original_project, original_dense = (
            AttentionV41FP8._project_output,
            V41MXFP8Linear.forward_quantized,
        )
        kv = owner._lin(owner.wkv, value).clone()

        def fail(*args, **kwargs):
            raise RuntimeError("output scope sentinel")

        with patch.object(AttentionV41FP8, "_project_output", fail):
            with output_projection_reference(True):
                with self.assertRaisesRegex(RuntimeError, "output scope sentinel"):
                    self.projection(owner, case)
                self.exact(kv, owner._lin(owner.wkv, value))
            self.assertIs(AttentionV41FP8._project_output, fail)
            self.assertIs(V41MXFP8Linear.forward_quantized, original_dense)
        self.assertIs(AttentionV41FP8._project_output, original_project)

    @torch.inference_mode()
    def test_composed_single_token_cuda_graphs(self):
        checked = 0
        for prompt, case, _ in self.cases:
            selected_step = 2 if prompt == "france" else 1
            if (case["stream"], case["layer"], case["step"]) != (
                "official",
                0,
                selected_step,
            ):
                continue
            owner = self.owners[0]
            attention = case["attention_before"].cuda()
            start = case["identity"]["start_pos"]
            freqs = owner.freqs_cis[start : start + 1]
            graph = torch.cuda.CUDAGraph()
            with tp8_projection(True), output_projection_reference(True):
                owner._project_output(attention.clone(), freqs)
                owner._project_output(attention.clone(), freqs)
                torch.cuda.synchronize()
                with torch.cuda.graph(graph):
                    result = owner._project_output(attention.clone(), freqs)
                graph.replay()
            self.exact(case["outputs"]["candidate"], result)
            checked += 1
        self.assertEqual(checked, 2)
        print(
            "DSV41_OUTPUT_GRAPHS "
            + json.dumps({"single_token_outputs_exact": checked}),
            flush=True,
        )

    def test_disabled_context(self):
        project, dense = (
            AttentionV41FP8._project_output,
            V41MXFP8Linear.forward_quantized,
        )
        with output_projection_reference(False):
            self.assertIs(AttentionV41FP8._project_output, project)
            self.assertIs(V41MXFP8Linear.forward_quantized, dense)


if __name__ == "__main__":
    unittest.main()
