"""Independent 656->576 contract tests; GPU tests use the selected CUDA device.

Run e.g. CUDA_VISIBLE_DEVICES=4 python trtllm_kv_compat_test.py. The scalar
CPU oracle does not import the native Attention benchmark or FlashInfer.
"""

import importlib.util
import unittest
from pathlib import Path

import torch


def _module():
    for root in Path(__file__).resolve().parents:
        path = root / "rtp_llm/models_py/triton_kernels/sparse_mla/trtllm_kv_compat.py"
        if path.is_file():
            spec = importlib.util.spec_from_file_location("trtllm_kv_compat", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("trtllm_kv_compat.py")


FP8 = torch.float8_e4m3fn


def saturated_fp8(value):
    return value.float().clamp(-448, 448).to(FP8)


def fixture(k=64):
    generator = torch.Generator(device="cpu").manual_seed(571)
    q = torch.randn((8, 2, 576), generator=generator).bfloat16()
    q[0, 0, :2] = torch.tensor([1000.0, -1000.0]).bfloat16()
    kv = torch.empty((5, 64, 656), dtype=torch.uint8)
    values = (torch.randn((5, 64, 512), generator=generator) * 31).to(FP8)
    scales = torch.rand((5, 64, 4), generator=generator) / 3 + 0.001
    rope = torch.randn((5, 64, 64), generator=generator).bfloat16()
    # A scale-rounding case: FP32 1.063 -> BF16 1.0625 -> FP8 1.0,
    # whereas directly converting FP32 1.063 to FP8 gives 1.125.
    values[1, 0, 0] = 1.0
    scales[1, 0, 0] = 1.063
    # Unlike FP32-scale Prefill upconvert, the actual Decode reader rounds
    # scale first. These land on opposite sides of an FP8 boundary.
    values[1, 0, 128] = 3.0
    scales[1, 0, 1] = 0.5607638359069824
    # Independently expose the product-BF16 rounding boundary after scale was
    # already rounded: -58.125 -> BF16 -58 -> FP8 -56, not direct FP8 -60.
    values[1, 0, 256] = -30.0
    scales[1, 0, 2] = 1.936505675315857
    rope[1, 0, :2] = torch.tensor([1000.0, -1000.0]).bfloat16()
    scales[1, 1].fill_(torch.finfo(torch.float32).tiny)
    kv[..., :512].copy_(values.view(torch.uint8))
    kv[..., 512:528].copy_(scales.contiguous().view(torch.uint8))
    kv[..., 528:].copy_(rope.contiguous().view(torch.uint8))
    kv[0].fill_(255)  # Forbidden page contents must never leak into output.
    req = torch.tensor([0, 1, -1, 2, 0, 0, 1, 0], dtype=torch.int32)
    # Strided rows exercise the table stride rather than assume tight packing.
    table = torch.tensor(
        [[1, 3, 0, 6, -1, 77], [2, 4, 1, 3, 0, 77]], dtype=torch.int32
    )[:, :5]
    lengths = torch.tensor([130, 64, 20, 20, 0, 400, 65, -1], dtype=torch.int32)
    topk = torch.full((8, k), -1, dtype=torch.int32)
    for row, tokens in enumerate(
        (
            [0, 63, 64, 65, -1, 0, 200, 128, 10000, 1],
            [0, 63, 64],
            [0, 1],
            [0],
            [0],
            [192, 256],
            [64, 0, 64],
            [0],
        )
    ):
        topk[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.int32)
    q[[2, 3, 4, 5, 7]] = float("nan")
    return dict(q=q, kv=kv, topk=topk, req_ids=req, block_table=table, seq_lens=lengths)


def oracle(inputs):
    q, kv, topk = (inputs[name].cpu() for name in ("q", "kv", "topk"))
    req, table, lengths = (
        inputs[name].cpu() for name in ("req_ids", "block_table", "seq_lens")
    )
    rows, k = topk.shape
    sources = torch.full((rows, k), -1, dtype=torch.int64)
    indices = torch.full((rows, k), -1, dtype=torch.int32)
    counts = torch.zeros(rows, dtype=torch.int32)
    kv_result = torch.zeros((rows, k, 576), dtype=FP8)
    q_result = torch.zeros_like(q, dtype=FP8)
    for row in range(rows):
        selected = []
        request = int(req[row])
        for logical in topk[row].tolist():
            if not (0 <= request < table.shape[0] and 0 <= logical < int(lengths[row])):
                continue
            block, offset = divmod(logical, kv.shape[1])
            if block >= table.shape[1]:
                continue
            page = int(table[request, block])
            if not 0 < page < kv.shape[0]:
                continue
            selected.append(page * kv.shape[1] + offset)
        counts[row] = len(selected)
        for col, physical in enumerate(selected):
            page, offset = divmod(physical, kv.shape[1])
            packed = kv[page, offset]
            value = packed[:512].view(FP8).float()
            scale = packed[512:528].view(torch.float32).bfloat16().float()
            scale = scale.repeat_interleave(128)
            latent = (value * scale).bfloat16()
            rope = packed[528:].view(torch.bfloat16)
            kv_result[row, col].copy_(saturated_fp8(torch.cat((latent, rope))))
            sources[row, col] = physical
            indices[row, col] = row * k + col
        if selected:
            q_result[row].copy_(saturated_fp8(q[row]))
        else:
            indices[row, 0] = row * k
    return dict(
        q_out=q_result,
        kv_out=kv_result,
        source_indices=sources,
        indices_out=indices,
        counts_out=counts,
        lengths_out=counts.clamp_min(1),
    )


def nonfinite_fixture():
    inputs = fixture()
    inputs["q"][0, 0, :3] = torch.tensor(
        [float("nan"), float("inf"), -float("inf")], dtype=torch.bfloat16
    )
    inputs["q"][0, 0, 512] = float("nan")
    # Row zero selects physical page 1, slot 0 first. Exercise each source
    # of a non-finite latent independently, without affecting index validity.
    packed = inputs["kv"][1, 0]
    latent = packed[:512].view(FP8)
    latent.copy_(torch.ones(512).to(FP8))
    latent[0] = float("nan")
    latent[257] = -1.0
    latent[385] = -1.0
    packed[512:528].view(torch.float32).copy_(
        torch.tensor([1.0, float("nan"), float("inf"), -float("inf")])
    )
    packed[528:].view(torch.bfloat16)[:3].copy_(
        torch.tensor([float("nan"), float("inf"), -float("inf")]).bfloat16()
    )
    return inputs


class CompatOracleTest(unittest.TestCase):
    def test_explicit_bf16_rounding_is_observable(self):
        value = torch.tensor([1.063], dtype=torch.float32)
        self.assertNotEqual(
            float(saturated_fp8(value)[0]), float(saturated_fp8(value.bfloat16())[0])
        )
        self.assertEqual(float(saturated_fp8(value.bfloat16())[0]), 1.0)

    def test_decode_rounds_scale_before_multiplying(self):
        value = torch.tensor([3.0], dtype=FP8).float()
        scale = torch.tensor([0.5607638359069824], dtype=torch.float32)
        prefill = saturated_fp8((value * scale).bfloat16())
        decode = saturated_fp8((value * scale.bfloat16().float()).bfloat16())
        self.assertEqual(float(prefill[0]), 1.625)
        self.assertEqual(float(decode[0]), 1.75)
        self.assertEqual(float(oracle(fixture())["kv_out"][0, 0, 128]), 1.75)

    def test_decode_also_rounds_product_before_fp8(self):
        value = torch.tensor([-30.0], dtype=FP8).float()
        scale = torch.tensor([1.936505675315857]).bfloat16().float()
        direct = saturated_fp8(value * scale)
        decode = saturated_fp8((value * scale).bfloat16())
        self.assertEqual(float(direct[0]), -60.0)
        self.assertEqual(float(decode[0]), -56.0)
        self.assertEqual(float(oracle(fixture())["kv_out"][0, 0, 256]), -56.0)

    def test_mapping_preserves_duplicates_and_causal_bounds(self):
        result = oracle(fixture())
        self.assertEqual(result["counts_out"].tolist(), [6, 2, 0, 0, 0, 0, 3, 0])
        self.assertEqual(
            result["source_indices"][0, :6].tolist(), [64, 127, 192, 193, 64, 65]
        )
        self.assertEqual(result["source_indices"][6, :3].tolist(), [256, 128, 256])
        self.assertEqual(result["lengths_out"].tolist(), [6, 2, 1, 1, 1, 1, 3, 1])

    def test_valid_nan_propagates_and_infinity_saturates(self):
        result = oracle(nonfinite_fixture())
        q = result["q_out"][0, 0].float()
        kv = result["kv_out"][0, 0].float()
        self.assertTrue(bool(torch.isnan(q[[0, 512]]).all()))
        self.assertEqual(q[1:3].tolist(), [448.0, -448.0])
        self.assertTrue(bool(torch.isnan(kv[[0, 512]]).all()))
        self.assertTrue(bool(torch.isnan(kv[128:256]).all()))
        self.assertEqual(
            kv[[256, 257, 384, 385, 513, 514]].tolist(),
            [448.0, -448.0, -448.0, 448.0, 448.0, -448.0],
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CompatCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _module()

    def make(self, k=64):
        inputs = {name: tensor.cuda() for name, tensor in fixture(k).items()}
        # .cuda() may densify the CPU view. Reintroduce a padded row stride.
        padded = torch.empty((2, 7), dtype=torch.int32, device="cuda")
        padded[:, :5].copy_(inputs["block_table"])
        inputs["block_table"] = padded[:, :5]
        rows, heads, _ = inputs["q"].shape
        outputs = dict(
            q_out=torch.empty_like(inputs["q"], dtype=FP8),
            kv_out=torch.empty((rows, k, 576), dtype=FP8, device="cuda"),
            source_indices=torch.empty((rows, k), dtype=torch.int64, device="cuda"),
            indices_out=torch.empty((rows, k), dtype=torch.int32, device="cuda"),
            counts_out=torch.empty(rows, dtype=torch.int32, device="cuda"),
            lengths_out=torch.empty(rows, dtype=torch.int32, device="cuda"),
        )
        outputs["kv_out"].view(torch.uint8).fill_(127)
        return inputs, outputs

    def run_conversion(self, inputs, outputs):
        self.module.convert_selected_kv(**inputs, **outputs)

    def assert_result(self, inputs, outputs):
        expected = oracle(inputs)
        for name in ("source_indices", "indices_out", "counts_out", "lengths_out"):
            self.assertTrue(torch.equal(outputs[name].cpu(), expected[name]), name)
        self.assertTrue(
            torch.equal(
                outputs["q_out"].view(torch.uint8).cpu(),
                expected["q_out"].view(torch.uint8),
            )
        )
        for row, count in enumerate(expected["counts_out"].tolist()):
            end = max(count, 1)
            actual = outputs["kv_out"][row, :end].view(torch.uint8).cpu()
            desired = expected["kv_out"][row, :end].view(torch.uint8)
            self.assertTrue(torch.equal(actual, desired), f"KV row {row}")

    def test_quantization_mapping_and_empty_rows(self):
        for k in (64, 192, 2048):
            with self.subTest(k=k):
                inputs, outputs = self.make(k)
                before = inputs["kv"].clone()
                self.run_conversion(inputs, outputs)
                self.assert_result(inputs, outputs)
                self.assertTrue(torch.equal(before, inputs["kv"]))
                for row, count in enumerate(outputs["counts_out"].cpu().tolist()):
                    tail = outputs["kv_out"][row, max(count, 1) :].view(torch.uint8)
                    self.assertTrue(bool((tail == 127).all()), "unused tail modified")

    def test_all_valid_duplicates_at_full_topk(self):
        inputs, outputs = self.make(2048)
        inputs["topk"].zero_()
        inputs["req_ids"].zero_()
        inputs["seq_lens"].fill_(1)
        inputs["q"].zero_()
        self.run_conversion(inputs, outputs)
        self.assert_result(inputs, outputs)
        self.assertEqual(outputs["counts_out"].cpu().tolist(), [2048] * 8)

    def test_valid_nan_propagation_and_infinity_saturation(self):
        inputs, outputs = self.make()
        special = nonfinite_fixture()
        for name, value in special.items():
            inputs[name].copy_(value)
        self.run_conversion(inputs, outputs)
        expected = oracle(special)
        for name in ("q_out", "kv_out"):
            # Compare NaN positions rather than NaN sign/payload encodings;
            # every finite value, including saturated infinities, must be exact.
            actual = outputs[name][0, 0].float().cpu()
            desired = expected[name][0, 0].float()
            mask = torch.isnan(desired)
            self.assertTrue(torch.equal(torch.isnan(actual), mask), name)
            self.assertTrue(torch.equal(actual[~mask], desired[~mask]), name)
        for name in ("source_indices", "indices_out", "counts_out", "lengths_out"):
            self.assertTrue(torch.equal(outputs[name].cpu(), expected[name]), name)

    def test_mask_empty_output_ignores_nan_and_preserves_live_rows(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            counts = torch.tensor([0, 1, 0], dtype=torch.int32, device="cuda")
            out = torch.full((3, 2, 512), 7.0, dtype=dtype, device="cuda")
            out[0].fill_(float("nan"))
            self.module.mask_empty_output(out, counts)
            self.assertTrue(bool((out[0] == 0).all()))
            self.assertTrue(bool((out[1] == 7).all()))
            self.assertTrue(bool((out[2] == 0).all()))

    def test_graph_replay_refreshes_inputs_without_allocating(self):
        inputs, outputs = self.make()
        for _ in range(3):
            self.run_conversion(inputs, outputs)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        self.run_conversion(inputs, outputs)
        self.assertEqual(torch.cuda.memory_allocated(), baseline)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.run_conversion(inputs, outputs)
        inputs["topk"][0].fill_(-1)
        inputs["topk"][4, :3] = torch.tensor([2, 3, 2], device="cuda")
        inputs["seq_lens"][4] = 4
        inputs["q"].fill_(1.063)
        inputs["block_table"][0, 0] = 2
        inputs["kv"][2, 2, 528:].view(torch.bfloat16).fill_(3.0)
        graph.replay()
        self.assert_result(inputs, outputs)

    def test_four_dimensional_kv_and_alternate_stream(self):
        inputs, outputs = self.make()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.module.convert_selected_kv(
                **dict(inputs, kv=inputs["kv"].unsqueeze(2)), **outputs
            )
        torch.cuda.current_stream().wait_stream(stream)
        self.assert_result(inputs, outputs)

    def test_empty_table_or_cache_never_reads_poisoned_query(self):
        for empty in ("block_table", "kv"):
            with self.subTest(empty=empty):
                inputs, outputs = self.make()
                inputs[empty] = inputs[empty][:0]
                inputs["q"].fill_(float("nan"))
                self.run_conversion(inputs, outputs)
                self.assert_result(inputs, outputs)
                self.assertEqual(outputs["counts_out"].cpu().tolist(), [0] * 8)

    def test_zero_rows_is_a_noop(self):
        inputs, outputs = self.make()
        for name in ("q", "topk", "req_ids", "seq_lens"):
            inputs[name] = inputs[name][:0]
        outputs = {name: tensor[:0] for name, tensor in outputs.items()}
        baseline = torch.cuda.memory_allocated()
        self.run_conversion(inputs, outputs)
        self.assertEqual(torch.cuda.memory_allocated(), baseline)

    def test_compaction_physical_address_is_int64(self):
        # Test address arithmetic without allocating a >128-TiB fake cache.
        topk = torch.full((1, 64), -1, dtype=torch.int32, device="cuda")
        topk[0, 0] = 65535
        req = torch.zeros(1, dtype=torch.int32, device="cuda")
        table = torch.tensor([[2**31 - 2]], dtype=torch.int32, device="cuda")
        lengths = torch.full((1,), 65536, dtype=torch.int32, device="cuda")
        source = torch.empty((1, 64), dtype=torch.int64, device="cuda")
        indices = torch.empty((1, 64), dtype=torch.int32, device="cuda")
        count = torch.empty(1, dtype=torch.int32, device="cuda")
        trt_length = torch.empty_like(count)
        self.module._compact_indices[(1,)](
            topk,
            req,
            table,
            lengths,
            source,
            indices,
            count,
            trt_length,
            K=64,
            BLOCK_K=64,
            REQUESTS=1,
            TABLE_COLS=1,
            PAGE=65536,
            PAGES=2**31,
            TABLE_STRIDE=1,
        )
        self.assertEqual(int(source[0, 0]), (2**31 - 2) * 65536 + 65535)
        self.assertEqual(int(count[0]), 1)

    def test_rejects_invalid_storage_contract(self):
        inputs, outputs = self.make()
        with self.assertRaisesRegex(ValueError, "source_indices"):
            self.run_conversion(
                inputs, dict(outputs, source_indices=outputs["indices_out"])
            )
        with self.assertRaisesRegex(ValueError, "K must"):
            self.run_conversion(dict(inputs, topk=inputs["topk"][:, :63]), outputs)
        with self.assertRaisesRegex(ValueError, "same CUDA device"):
            self.run_conversion(
                dict(inputs, seq_lens=inputs["seq_lens"].cpu()), outputs
            )
        with self.assertRaisesRegex(ValueError, "q must be contiguous"):
            self.run_conversion(
                dict(
                    inputs, q=inputs["q"].transpose(0, 1).contiguous().transpose(0, 1)
                ),
                outputs,
            )


if __name__ == "__main__":
    unittest.main()
