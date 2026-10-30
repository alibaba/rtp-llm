"""SM90 sparse-prefill dispatch, sparse selection, and graph regressions."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa import sparse_fmha, sparse_prefill_sm90
from rtp_llm.models_py.modules.qwen4_exp.test.sparse_prefill_sm90_test_utils import (
    errors,
    fixture,
    fp64_reference,
    metadata,
    save,
)


class SparsePrefillLaunchConfigTest(unittest.TestCase):
    def test_bounded_launch_selection(self):
        cases = [
            (1, 128, "NVIDIA H100", (64, 4)),
            (1, 129, "NVIDIA H100", (32, 4)),
            (1, 512, "NVIDIA H100", (32, 4)),
            (1, 513, "NVIDIA H100", (16, 1)),
            (1, 5119, "NVIDIA H100", (16, 1)),
            (1, 5120, "NVIDIA H100", (64, 2)),
            (1, 10240, "NVIDIA H100", (64, 2)),
            (1, 15360, "NVIDIA H100", (64, 2)),
            (1, 20480, "NVIDIA H100", (64, 2)),
            (1, 20481, "NVIDIA H100", (16, 1)),
            (2, 128, "NVIDIA H100", (32, 4)),
            (2, 5120, "NVIDIA H100", (16, 1)),
            (2, 20480, "NVIDIA H100", (16, 1)),
            (1, 1024, "NVIDIA H20", (32, 4)),
            (1, 1025, "NVIDIA H20", (16, 1)),
            (1, 5120, "NVIDIA H20", (16, 1)),
            (1, 20480, "NVIDIA H20", (16, 1)),
        ]
        for batch, length, device_name, expected in cases:
            with self.subTest(batch=batch, length=length, device_name=device_name):
                self.assertEqual(
                    sparse_prefill_sm90._prefill_launch_config(
                        batch, length, device_name
                    ),
                    expected,
                )


class SparsePrefillSm90Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.version.hip is not None or torch.cuda.get_device_capability()[0] != 9:
            raise unittest.SkipTest("requires an SM90 CUDA GPU")
        cls.old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    @classmethod
    def tearDownClass(cls):
        torch.backends.cuda.matmul.allow_tf32 = cls.old_tf32

    def _run_cases(self, name, cases):
        record = {"status": "RUNNING", "metadata": metadata(), "cases": []}
        try:
            for label, values, options in cases:
                q, k, v, ids = values
                before = [t.clone() for t in values]
                reference = fp64_reference(*values)
                eligible = sparse_prefill_sm90.supports_sm90_sparse_prefill(
                    *values, **options
                )
                direct = sparse_prefill_sm90.try_sm90_sparse_prefill(*values, **options)
                self.assertEqual(direct is not None, eligible)
                with patch.object(
                    sparse_prefill_sm90,
                    "try_sm90_sparse_prefill",
                    wraps=sparse_prefill_sm90.try_sm90_sparse_prefill,
                ) as dispatch:
                    output = sparse_fmha.sparse_prefill_attn(
                        *values, validate_selected=False, **options
                    )
                    self.assertEqual(dispatch.call_count, 1)
                if eligible:
                    torch.testing.assert_close(output, direct, atol=0, rtol=0)
                with patch.object(
                    sparse_prefill_sm90, "try_sm90_sparse_prefill", return_value=None
                ):
                    baseline = sparse_fmha.sparse_prefill_attn(
                        *(tensor.contiguous() for tensor in values),
                        validate_selected=False,
                        **options
                    )
                item = {
                    "label": label,
                    "shape": list(q.shape),
                    "kv_shape": list(k.shape),
                    "strides": [list(t.stride()) for t in values],
                    "candidate_fp64": errors(output, reference),
                    "baseline_fp64": errors(baseline, reference),
                    "baseline_layout": "contiguous copies of the same logical inputs",
                    "candidate_baseline": errors(output, baseline),
                    "inputs_unchanged": all(
                        torch.equal(a, b) for a, b in zip(before, values)
                    ),
                    "dtype": str(output.dtype),
                    "output_stride": list(output.stride()),
                    "sm90_supported": sparse_prefill_sm90.supports_sm90_sparse_prefill(
                        *values, **options
                    ),
                }
                record["cases"].append(item)
                save(name, record)
                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertTrue(item["inputs_unchanged"])
                for key in ("candidate_fp64", "baseline_fp64", "candidate_baseline"):
                    self.assertTrue(item[key]["pass"], (label, key, item[key]))
            record["status"] = "PASS"
        except Exception as exc:
            record.update(status="FAIL", error=repr(exc))
            raise
        finally:
            save(name, record)

    def test_mini(self):
        cases = [(str(s), fixture(s), {}) for s in (129, 257)]
        for _, values, _ in cases:
            self.assertTrue(sparse_prefill_sm90.supports_sm90_sparse_prefill(*values))
        self._run_cases("mini", cases)

    def test_selection_and_strides(self):
        values = fixture(131, batch=2, strided=True)
        self.assertTrue(sparse_prefill_sm90.supports_sm90_sparse_prefill(*values))
        ids = values[3]
        ids[:, 0] = -1
        ids[:, 1] = -1
        ids[:, 1, -1] = 130
        ids[:, 2] = -1
        ids[:, 2, :3] = torch.tensor([127, 127, 128], device="cuda", dtype=torch.int32)
        ids[:, 3, 8:2000] = -7
        ids[:, 4, :4] = torch.tensor(
            [131, 999, -5, 130], device="cuda", dtype=torch.int32
        )
        packed = torch.empty((2, 131, 2052), device="cuda", dtype=torch.int32)
        sparse_prefill_sm90._compact_sm90_selection[(2 * 131,)](
            ids,
            packed,
            ids.stride(0),
            ids.stride(1),
            S=131,
            T=131,
            WIDTH=2051,
            BLOCK=4096,
            num_warps=8,
        )
        original_cpu, packed_cpu = ids.cpu(), packed.cpu()
        for batch in range(2):
            for row in range(131):
                source = original_cpu[batch, row]
                expected = source[(source >= 0) & (source < 131)]
                count = int(packed_cpu[batch, row, -1])
                self.assertEqual(count, expected.numel())
                self.assertTrue(torch.equal(packed_cpu[batch, row, :count], expected))
                self.assertTrue(bool((packed_cpu[batch, row, count:2051] == -1).all()))
        self._run_cases("selection_and_strides", [("B2-strided-generic", values, {})])
        output = sparse_fmha.sparse_prefill_attn(*values, validate_selected=False)
        self.assertTrue(bool((output[:, :, 0] == 0).all()))
        with self.assertRaises(ValueError):
            sparse_fmha.sparse_prefill_attn(*values)

    def test_fallback_and_guard(self):
        cpu = [
            torch.empty((1, 12, 128, 256)),
            torch.empty((1, 1, 128, 256)),
            torch.empty((1, 1, 128, 256)),
            torch.empty((1, 128, 2051), dtype=torch.int32),
        ]
        with patch(
            "torch.cuda.get_device_capability",
            side_effect=AssertionError("CPU must return early"),
        ):
            self.assertFalse(sparse_prefill_sm90.supports_sm90_sparse_prefill(*cpu))
        values = fixture(128)
        for cap in ((8, 0), (12, 0)):
            with patch("torch.cuda.get_device_capability", return_value=cap):
                self.assertFalse(
                    sparse_prefill_sm90.supports_sm90_sparse_prefill(*values)
                )
        with patch("torch.version.hip", "mock"):
            self.assertFalse(sparse_prefill_sm90.supports_sm90_sparse_prefill(*values))
        q, k, v, ids = values
        for unsupported in (
            (q[:, :6], k, v, ids),
            (q[..., :128], k[..., :128], v[..., :128], ids),
            (q, k, v, ids[..., :2050]),
        ):
            self.assertFalse(
                sparse_prefill_sm90.supports_sm90_sparse_prefill(*unsupported)
            )
        oversized = (
            q[:, :, :1].expand(1, 12, 20481, 256),
            k[:, :, :1].expand(1, 1, 20481, 256),
            v[:, :, :1].expand(1, 1, 20481, 256),
            ids[:, :1].expand(1, 20481, 2051),
        )
        self.assertFalse(sparse_prefill_sm90.supports_sm90_sparse_prefill(*oversized))
        cases = []
        for label, data, kwargs in (
            ("prefix", fixture(128, kv_length=193), {}),
            ("short", fixture(127), {}),
            ("B3", fixture(128, batch=3), {}),
            ("tile_override", fixture(128), {"blk_k": 32}),
        ):
            self.assertFalse(
                sparse_prefill_sm90.supports_sm90_sparse_prefill(*data, **kwargs)
            )
            cases.append((label, data, kwargs))
        self._run_cases("fallback", cases)

    def test_graph_input_and_selection_refresh(self):
        record = {"status": "RUNNING", "metadata": metadata(), "cases": []}
        try:
            values = fixture(129, strided=True)
            self.assertTrue(sparse_prefill_sm90.supports_sm90_sparse_prefill(*values))
            for _ in range(3):
                sparse_fmha.sparse_prefill_attn(*values, validate_selected=False)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    sparse_fmha.sparse_prefill_attn(*values, validate_selected=False)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = sparse_fmha.sparse_prefill_attn(
                    *values, validate_selected=False
                )
            for seed in (47, 53):
                updated = fixture(129, strided=True, seed=seed)
                updated[3][:, 0] = -1
                updated[3][:, 1] = -1
                updated[3][:, 1, -1] = 128
                for destination, source in zip(values, updated):
                    destination.copy_(source)
                graph.replay()
                reference = fp64_reference(*values)
                eager = sparse_fmha.sparse_prefill_attn(
                    *values, validate_selected=False
                )
                metric = errors(captured, reference)
                exact = errors(captured, eager, 0.0, 0.0)
                record["cases"].append(
                    {"seed": seed, "fp64": metric, "graph_eager_exact": exact}
                )
                save("graph", record)
                self.assertTrue(metric["pass"], metric)
                self.assertTrue(exact["pass"], exact)
            record["status"] = "PASS"
        except Exception as exc:
            record.update(status="FAIL", error=repr(exc))
            raise
        finally:
            save("graph", record)

    def test_qk_amplitude(self):
        cases = []
        for amplitude in (0.25, 1.0, 2.0, 4.0):
            q, k, v, ids = fixture(129)
            q.mul_(amplitude)
            k.mul_(amplitude)
            cases.append((str(amplitude), (q, k, v, ids), {}))
        self._run_cases("amplitude", cases)


if __name__ == "__main__":
    unittest.main()
