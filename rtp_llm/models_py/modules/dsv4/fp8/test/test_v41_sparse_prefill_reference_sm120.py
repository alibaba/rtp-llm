"""Qualify scoped sparse prefill against admitted order and numerical artifacts."""

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import _v41_attention_triton as decode
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_attention_sm120 as prefill
from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import AttentionV41FP8
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_sparse_precision import (
    sparse_sink_div_rn,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_sparse_prefill_reference import (
    sparse_prefill_reference,
    swa_first_plan,
)
from rtp_llm.tools.dsv41_query_boundary_replay import sha256


class V41SparsePrefillReferenceSM120Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability() != (12, 0):
            raise RuntimeError(
                "Sparse prefill qualification requires leased SM120 hardware"
            )
        torch.set_num_threads(1)
        cls.cases = []
        artifacts = Path(os.environ["DSV41_TEST_SPARSE_ARTIFACT_ROOT"])
        for prompt in ("france", "arithmetic"):
            path = artifacts / f"sparse-order-{prompt}-1.pt"
            report = json.loads(path.with_suffix(".json").read_text())
            if sha256(path) != report["tensor_artifact_sha256"]:
                raise ValueError("Sparse order tensor evidence changed")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if (
                len(payload["cases"]) != 6
                or payload["identities"] != report["identities"]
            ):
                raise ValueError("Sparse order case coverage differs")
            for case in payload["cases"]:
                cls.cases.append((prompt, case))
        if len(cls.cases) != 12:
            raise ValueError(
                "Sparse prefill qualification requires all twelve held cases"
            )

    def exact(self, expected, actual):
        expected, actual = expected.cpu().contiguous(), actual.cpu().contiguous()
        self.assertEqual(expected.shape, actual.shape)
        self.assertEqual(expected.dtype, actual.dtype)
        self.assertTrue(
            torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
        )

    def inputs(self, case):
        return {
            name: value.cuda().clone()
            for name, value in case["operands_before"].items()
        }

    def global_rows(self, inputs):
        return inputs["kv"].shape[0] - inputs["q"].shape[0]

    def fake_scope(self, inputs, *, fail=False):
        def native(owner, qkv, common, **kwargs):
            if fail:
                raise RuntimeError("prefill scope sentinel")
            return prefill.sparse_prefill_attention(*inputs.values())

        owner = SimpleNamespace(
            compress_ratio=2 if self.global_rows(inputs) else 0, window_size=128
        )
        qkv = SimpleNamespace(qr=inputs["q"][:, 0])
        common = SimpleNamespace(
            cp_on=False, batch_size=1, prefix_lengths=torch.zeros(1, dtype=torch.int32)
        )
        kwargs = dict(
            kv=inputs["kv"],
            indices=inputs["indices"],
            topk_length=inputs["lengths"],
            profile_name="scoped_test",
        )
        return native, owner, qkv, common, kwargs

    @torch.inference_mode()
    def test_all_held_controls_plans_and_restorations(self):
        with sparse_sink_div_rn(True):
            for prompt, case in self.cases:
                with self.subTest(
                    prompt=prompt, layer=case["layer"], stream=case["stream"]
                ):
                    inputs = self.inputs(case)
                    saved = {name: value.clone() for name, value in inputs.items()}
                    self.exact(
                        case["outputs"]["native"]["maintained"],
                        prefill.sparse_prefill_attention(*inputs.values()),
                    )
                    rows = self.global_rows(inputs)
                    ordered, lengths = swa_first_plan(
                        inputs["indices"],
                        inputs["lengths"],
                        inputs["kv"].shape[0],
                        rows,
                    )
                    self.exact(
                        case["plans_before"]["reference_compact"]["indices"], ordered
                    )
                    self.exact(
                        case["plans_before"]["reference_compact"]["lengths"], lengths
                    )
                    with sparse_prefill_reference(True, global_rows=rows):
                        self.exact(
                            case["outputs"]["reference_compact"][
                                "libdevice_exp_fusion_on"
                            ],
                            prefill.sparse_prefill_attention(*inputs.values()),
                        )
                    self.exact(
                        case["outputs"]["native"]["maintained"],
                        prefill.sparse_prefill_attention(*inputs.values()),
                    )
                    for name, value in inputs.items():
                        self.exact(saved[name], value)
        print(
            "DSV41_SPARSE_PREFILL_HELD "
            + json.dumps(
                {
                    "native_controls": 12,
                    "corrected_outputs": 12,
                    "restorations": 12,
                    "plan_tensors": 24,
                    "source_tensors": 60,
                }
            ),
            flush=True,
        )

    @torch.inference_mode()
    def test_bound_owner_and_unselected_prefill_decode(self):
        original_decode = decode._attention
        with sparse_sink_div_rn(True, decode=False):
            for _, case in self.cases:
                inputs = self.inputs(case)
                native, owner, qkv, common, kwargs = self.fake_scope(inputs)
                with patch.object(AttentionV41FP8, "_prefill_sparse_attention", native):
                    with sparse_prefill_reference(True):
                        self.assertIs(decode._attention, original_decode)
                        self.exact(
                            case["outputs"]["native"]["maintained"],
                            prefill.sparse_prefill_attention(*inputs.values()),
                        )
                        self.exact(
                            case["outputs"]["reference_compact"][
                                "libdevice_exp_fusion_on"
                            ],
                            AttentionV41FP8._prefill_sparse_attention(
                                owner, qkv, common, **kwargs
                            ),
                        )
                        self.exact(
                            case["outputs"]["native"]["maintained"],
                            prefill.sparse_prefill_attention(*inputs.values()),
                        )
        print(
            "DSV41_SPARSE_PREFILL_SCOPE "
            + json.dumps(
                {
                    "bound_outputs": 12,
                    "unselected_prefill": 24,
                    "decode_provider_unchanged": True,
                }
            ),
            flush=True,
        )

    @torch.inference_mode()
    def test_nested_direct_sink_disabled_and_restore(self):
        from contextlib import ExitStack

        for reference_first in (False, True):
            for _, case in self.cases:
                inputs = self.inputs(case)
                original = prefill._sparse_prefill
                contexts = (
                    sparse_prefill_reference(
                        True, global_rows=self.global_rows(inputs)
                    ),
                    sparse_sink_div_rn(True),
                )
                if not reference_first:
                    contexts = contexts[::-1]
                with ExitStack() as stack:
                    for context in contexts:
                        stack.enter_context(context)
                    selected = prefill._sparse_prefill
                    with sparse_prefill_reference(False):
                        self.assertIs(prefill._sparse_prefill, selected)
                        self.exact(
                            case["outputs"]["reference_compact"][
                                "libdevice_exp_fusion_on"
                            ],
                            prefill.sparse_prefill_attention(*inputs.values()),
                        )
                self.assertIs(prefill._sparse_prefill, original)
        print(
            "DSV41_SPARSE_PREFILL_NESTED " + json.dumps({"both_orders_exact": 24}),
            flush=True,
        )

    @torch.inference_mode()
    def test_exception_restores_owner_token_and_kernel(self):
        _, case = self.cases[2]
        inputs = self.inputs(case)
        native, owner, qkv, common, kwargs = self.fake_scope(inputs, fail=True)
        original_kernel, original_owner = (
            prefill._sparse_prefill,
            AttentionV41FP8._prefill_sparse_attention,
        )
        with sparse_sink_div_rn(True, decode=False), patch.object(
            AttentionV41FP8, "_prefill_sparse_attention", native
        ):
            with sparse_prefill_reference(True):
                with self.assertRaisesRegex(RuntimeError, "prefill scope sentinel"):
                    AttentionV41FP8._prefill_sparse_attention(
                        owner, qkv, common, **kwargs
                    )
                self.exact(
                    case["outputs"]["native"]["maintained"],
                    prefill.sparse_prefill_attention(*inputs.values()),
                )
            self.assertIs(AttentionV41FP8._prefill_sparse_attention, native)
        self.assertIs(prefill._sparse_prefill, original_kernel)
        self.assertIs(AttentionV41FP8._prefill_sparse_attention, original_owner)

    def test_rejects_unqualified_owner_workspaces_and_hardware(self):
        _, case = self.cases[2]
        inputs = self.inputs(case)
        native, owner, qkv, common, kwargs = self.fake_scope(inputs)
        with patch.object(
            AttentionV41FP8, "_prefill_sparse_attention", native
        ), sparse_prefill_reference(True):
            for field, value in (
                ("cp_on", True),
                ("batch_size", 2),
                ("prefix_lengths", torch.ones(1, dtype=torch.int32)),
            ):
                original = getattr(common, field)
                setattr(common, field, value)
                with self.assertRaisesRegex(ValueError, "initial workspace"):
                    AttentionV41FP8._prefill_sparse_attention(
                        owner, qkv, common, **kwargs
                    )
                setattr(common, field, original)
            for invalid in (dict(kwargs, kv=inputs["kv"][:-1]),):
                with self.assertRaisesRegex(ValueError, "initial workspace"):
                    AttentionV41FP8._prefill_sparse_attention(
                        owner, qkv, common, **invalid
                    )
            with self.assertRaisesRegex(ValueError, "initial workspace"):
                AttentionV41FP8._prefill_sparse_attention(
                    owner,
                    SimpleNamespace(qr=torch.empty(129, 512, device="cuda")),
                    common,
                    **kwargs,
                )
        with patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)):
            with self.assertRaisesRegex(ValueError, "SM120"):
                with sparse_prefill_reference(True):
                    self.fail("Non-SM120 candidate admitted")
            with sparse_prefill_reference(False):
                pass

    @torch.inference_mode()
    def test_masked_empty_rows_and_extreme_sinks(self):
        q = torch.zeros(3, 4, 512, device="cuda", dtype=torch.bfloat16)
        kv = torch.full((33, 1, 512), 0.25, device="cuda", dtype=torch.bfloat16)
        indices = (
            torch.arange(33, device="cuda", dtype=torch.int64)[None, None]
            .expand(3, 1, -1)
            .contiguous()
        )
        indices[1].fill_(-1)
        indices[0, :, 0] = 1000
        lengths = torch.tensor([0, 33, 33], device="cuda", dtype=torch.int64)
        sinks = torch.tensor(
            [-1000, 1000, -100, 100], device="cuda", dtype=torch.float32
        )
        saved = [value.clone() for value in (q, kv, indices, lengths, sinks)]
        with sparse_prefill_reference(True, global_rows=0):
            output = prefill.sparse_prefill_attention(q, kv, indices, lengths, sinks)
            self.assertTrue(output.isfinite().all())
            self.assertEqual(torch.count_nonzero(output[:2]).item(), 0)
            expected = (
                (33 * 0.25 / (33 + sinks.exp()))
                .bfloat16()[None, :, None]
                .expand(1, 4, 512)
            )
            self.exact(expected, output[2:])
            empty = prefill.sparse_prefill_attention(
                q[:0], kv, indices[:0], lengths[:0], sinks
            )
            self.assertEqual(empty.shape, q[:0].shape)
        for original, value in zip(saved, (q, kv, indices, lengths, sinks)):
            self.exact(original, value)

    @torch.inference_mode()
    def test_changed_input_prefill_component_graphs(self):
        checked = 0
        for layer in (0, 2):
            selected = [
                case
                for prompt, case in self.cases
                if case["layer"] == layer and case["stream"] == "official"
            ]
            self.assertEqual(len(selected), 2)
            inputs = self.inputs(selected[0])
            graph = torch.cuda.CUDAGraph()
            with sparse_prefill_reference(True, global_rows=self.global_rows(inputs)):
                prefill.sparse_prefill_attention(*inputs.values())
                prefill.sparse_prefill_attention(*inputs.values())
                torch.cuda.synchronize()
                with torch.cuda.graph(graph):
                    output = prefill.sparse_prefill_attention(*inputs.values())
                for case in selected:
                    for name, value in inputs.items():
                        value.copy_(case["operands_before"][name])
                    graph.replay()
                    self.exact(
                        case["outputs"]["reference_compact"]["libdevice_exp_fusion_on"],
                        output,
                    )
            checked += 1
        print(
            "DSV41_SPARSE_PREFILL_GRAPHS "
            + json.dumps({"graphs": checked, "changed_input_outputs": 4}),
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()
