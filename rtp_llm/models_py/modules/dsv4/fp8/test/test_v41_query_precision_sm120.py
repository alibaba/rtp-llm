"""Qualify native query GEMM selection against frozen held NAS queries."""

import hashlib
import importlib
import json
import os
import sys
import unittest
from pathlib import Path

import torch

from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import AttentionV41FP8
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_query_precision import (
    query_projection_reference,
)
from rtp_llm.models_py.modules.dsv4.v41_mxfp8 import V41MXFP8Linear
from rtp_llm.tools.dsv41_attention_replay import NASReader, native_attention
from rtp_llm.tools.dsv41_query_boundary_replay import rotate, validate_history
from rtp_llm.tools.dsv41_reference_probe import model_args


class V41QueryPrecisionSM120Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability() != (12, 0):
            raise RuntimeError("Query diagnostic requires leased SM120 hardware")
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.bfloat16)
        torch.set_float32_matmul_precision("highest")
        cls.root = Path(os.environ["DSV41_TEST_CHECKPOINT"])
        artifacts = Path(os.environ["DSV41_TEST_QUERY_ARTIFACT_ROOT"])
        repo = Path(__file__).resolve().parents[6]
        sys.path.insert(0, str(cls.root / "inference"))
        official = importlib.import_module("model")
        args = model_args(official, cls.root, 256)
        config = json.loads((cls.root / "config.json").read_text())["text_config"]
        cls.reader = NASReader(cls.root)
        cls.owners = [
            native_attention(args, config, cls.reader, layer) for layer in range(3)
        ]
        cls.cases = []
        for prompt, query_name in (
            ("france", "query-boundary-france-3"),
            ("arithmetic", "query-boundary-arithmetic-1"),
        ):
            history_name = f"engram-history-{prompt}-sparse-1"
            history = json.loads((artifacts / f"{history_name}.json").read_text())
            inputs = validate_history(
                cls.root, repo, history, artifacts / f"{history_name}.input.pt"
            )
            query = json.loads((artifacts / f"{query_name}.json").read_text())
            query_path = artifacts / f"{query_name}.pt"
            if (
                hashlib.sha256(query_path.read_bytes()).hexdigest()
                != query["tensor_artifact_sha256"]
            ):
                raise ValueError("Qualified query tensor digest differs")
            saved = torch.load(query_path, map_location="cpu", weights_only=True)
            for case in saved["cases"]:
                value = inputs[f"{case['stream']}_inputs"][case["step"]][
                    case["layer"]
                ].cuda()
                cls.cases.append(
                    (prompt, case, value, history["identities"][case["step"]])
                )
        if len(cls.cases) != 42:
            raise ValueError("Query diagnostic requires all 42 held cases")

    def exact(self, expected, actual):
        expected = expected.reshape(-1, expected.shape[-1]).cpu().contiguous()
        actual = actual.reshape(-1, actual.shape[-1]).cpu().contiguous()
        self.assertEqual(expected.shape, actual.shape)
        self.assertEqual(expected.dtype, actual.dtype)
        self.assertTrue(
            torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
        )

    def query(self, owner, value, identity):
        a = owner._lin(owner.wq_a, value)
        qr = owner._rmsnorm_weighted(a, owner.q_norm)
        b = owner._lin(owner.wq_b, qr)
        start = identity["start_pos"]
        freqs = owner.freqs_cis[start : start + len(identity["input_ids"])]
        return rotate(owner, b, freqs, start), a

    @torch.inference_mode()
    def test_all_native_controls(self):
        for _, case, value, identity in self.cases:
            with self.subTest(
                stream=case["stream"], step=case["step"], layer=case["layer"]
            ):
                result, _ = self.query(self.owners[case["layer"]], value, identity)
                self.exact(case["native_query"], result)

    @torch.inference_mode()
    def test_combined_queries_and_unselected_kv(self):
        for _, case, value, identity in self.cases:
            owner = self.owners[case["layer"]]
            sources = (
                value,
                owner.wq_a.weight,
                owner.wq_b.weight,
                owner.q_norm,
                owner.freqs_cis,
            )
            before = [source.clone() for source in sources]
            kv = owner._lin(owner.wkv, value).clone()
            with query_projection_reference(True):
                result, _ = self.query(owner, value, identity)
                selected_kv = owner._lin(owner.wkv, value)
            self.exact(case["released_query"], result)
            self.exact(kv, selected_kv)
            restored, _ = self.query(owner, value, identity)
            self.exact(case["native_query"], restored)
            for original, current in zip(before, sources):
                self.assertTrue(
                    torch.equal(
                        original.contiguous().view(torch.uint8),
                        current.contiguous().view(torch.uint8),
                    )
                )
        print(
            "DSV41_QUERY_COMBINED "
            + json.dumps(
                {
                    "released_queries_exact": 42,
                    "native_restorations_exact": 42,
                    "unselected_kv_exact": 42,
                }
            ),
            flush=True,
        )

    @torch.inference_mode()
    def test_first_and_second_projection_selection(self):
        for _, case, value, identity in self.cases:
            owner = self.owners[case["layer"]]
            with query_projection_reference(True, first=True, second=False):
                _, a = self.query(owner, value, identity)
            self.exact(case["released_a"], a)
            with query_projection_reference(True, first=False, second=True):
                result, a = self.query(owner, value, identity)
            self.exact(case["native_a"], a)
            if torch.equal(case["native_qr"], case["released_qr"]):
                self.exact(case["released_query"], result)

    @torch.inference_mode()
    def test_output_buffer_and_scope_restoration(self):
        _, case, value, _ = self.cases[0]
        owner = self.owners[case["layer"]]
        original_linear, original_dense = (
            AttentionV41FP8._lin,
            V41MXFP8Linear.forward_quantized,
        )
        out = torch.empty_like(case["released_a"], device="cuda").reshape(
            -1, owner.wq_a.N
        )
        with self.assertRaisesRegex(RuntimeError, "scope sentinel"):
            with query_projection_reference(True):
                result = owner._lin(owner.wq_a, value, out=out)
                self.assertEqual(result.data_ptr(), out.data_ptr())
                self.exact(case["released_a"], result)
                raise RuntimeError("scope sentinel")
        self.assertIs(AttentionV41FP8._lin, original_linear)
        self.assertIs(V41MXFP8Linear.forward_quantized, original_dense)

    @torch.inference_mode()
    def test_combined_query_cuda_graph(self):
        checked = 0
        for _, case, value, identity in self.cases:
            if (case["stream"], case["step"], case["layer"]) != ("official", 1, 0):
                continue
            owner = self.owners[0]
            graph = torch.cuda.CUDAGraph()
            with query_projection_reference(True):
                self.query(owner, value, identity)
                self.query(owner, value, identity)
                torch.cuda.synchronize()
                with torch.cuda.graph(graph):
                    result, _ = self.query(owner, value, identity)
                graph.replay()
            self.exact(case["released_query"], result)
            checked += 1
        self.assertEqual(checked, 2)
        print(
            "DSV41_QUERY_GRAPHS " + json.dumps({"single_token_queries_exact": checked}),
            flush=True,
        )

    def test_disabled_and_empty_selection(self):
        with query_projection_reference(False, first=False, second=False):
            pass
        with self.assertRaisesRegex(ValueError, "selected projection"):
            with query_projection_reference(True, first=False, second=False):
                pass


if __name__ == "__main__":
    unittest.main()
