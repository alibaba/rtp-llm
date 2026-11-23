import json
import os
import unittest

import torch

import rtp_llm.models
from rtp_llm.cpp.cuda_graph.tests.cuda_graph_test_utils import (
    CudaGraphTestModelBuilder,
    ModelBuildConfig,
)
from rtp_llm.cpp.cuda_graph.tests.libtest_cuda_graph_runner import CudaGraphRunner
from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInputs, get_typemeta


class InputEmbeddingGraphIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.cuda.set_device(0)
        cls.builder = CudaGraphTestModelBuilder(
            ModelBuildConfig(
                model_path=os.environ.get("PG_GRAPH_MODEL_PATH", "/mnt/nas1/hf/Qwen3-0.6B"),
                max_seq_len=64,
                max_total_tokens=512,
                hack_layer_num=0,
                act_type="BF16",
            )
        )
        cls.built = cls.builder.build_model(init_kv_cache=True)
        cls.model = cls.built.model
        print("MODEL_LAYERS=" + str(cls.built.layer_num), flush=True)
        cls.prefill = CudaGraphRunner()
        cls.prefill.init_generation_prefill(
            cls.model, 2, 64, 1, [8, 16, 32, 64], cls.built.hidden_size
        )
        cls.decode = CudaGraphRunner()
        cls.decode.init_decode(cls.model, cls.built.hidden_size, 64, 1, [1, 2])

    def inputs(self, lengths, *, decode_step=None, segments=()):
        batch = len(lengths)
        prefill = decode_step is None
        count = sum(lengths) if prefill else batch
        x = PyModelInputs()
        x.input_ids = torch.arange(20, 20 + count, dtype=torch.int32, device="cuda")
        if not prefill:
            x.input_ids += decode_step
        x.input_hiddens = torch.zeros((count, self.built.hidden_size), dtype=torch.bfloat16, device="cuda")
        a = PyAttentionInputs()
        a.is_prefill = prefill
        a.is_target_verify = False
        a.dtype = get_typemeta(torch.zeros(1, dtype=torch.bfloat16))
        host = lambda values: torch.tensor(values, dtype=torch.int32).pin_memory()
        a.input_lengths = host(lengths if prefill else [1] * batch)
        a.prefix_lengths = host([0] * batch if prefill else [])
        a.sequence_lengths = host([] if prefill else [n + decode_step for n in lengths])
        a.sequence_lengths_plus_1_device = (a.sequence_lengths + 1).cuda()
        cu = [0]
        for n in (lengths if prefill else [1] * batch):
            cu.append(cu[-1] + n)
        a.cu_seqlens = host(cu if prefill else [0] * (batch + 1))
        a.cu_seqlens_device = a.cu_seqlens.cuda()
        a.cu_kv_seqlens_device = a.cu_seqlens_device.clone()
        a.decode_cu_seqlens_device = host([] if prefill else list(range(batch + 1))).cuda()
        a.total_tokens = count
        a.context_total_kv_length = count if prefill else 0
        blocks = host(list(range(1, batch + 1))).reshape(batch, 1)
        a.kv_cache_kernel_block_id = blocks
        a.kv_cache_kernel_block_id_device = blocks.cuda()
        a.kv_cache_block_id = blocks
        a.kv_cache_block_id_device = a.kv_cache_kernel_block_id_device
        offsets = []
        skipped = 0
        for n in (lengths if prefill else [1] * batch):
            offsets.extend([skipped] * n)
            skipped += (max(lengths) if prefill else 1) - n
        a.padding_offset = host(offsets).cuda()
        x.attention_inputs = a
        if prefill:
            x.combo_position_ids = torch.cat([torch.arange(n, device="cuda", dtype=torch.int32) for n in lengths])
        else:
            x.combo_position_ids = host([n + decode_step for n in lengths]).cuda()
        x.attention_inputs.combo_position_ids = x.combo_position_ids
        CudaGraphRunner.prepare_device_mirrors(x)
        if segments:
            x.input_embeddings_locs = host([loc for loc, _ in segments])
            x.input_embeddings = [tensor for _, tensor in segments]
        return x

    def run_path(self, lengths, segments, prefill_graph, decode_graph, eager_padding=0):
        for cache in self.built.kv_cache._layer_tensors:
            cache.zero_()
        outputs = []
        for step in [None, 0, 1, 2]:
            input_lengths = lengths + [eager_padding] if step is None and eager_padding else lengths
            x = self.inputs(input_lengths, decode_step=step, segments=segments if step is None else ())
            if step is None and eager_padding:
                x.input_ids[sum(lengths):] = 0
            runner = self.prefill if step is None and prefill_graph else self.decode if step is not None and decode_graph else None
            if runner is not None:
                self.assertTrue(runner.canRun(x), "Graph must execute, not silently fall back")
                result = runner.forward(x)
                self.assertGreater(runner.getCurrentRealGraphSize(), 0)
            else:
                result = self.model.forward(x)
            count = sum(lengths) if step is None else len(lengths)
            outputs.append(result.hidden_states[:count].detach().clone())
        torch.cuda.synchronize()
        return outputs

    @torch.inference_mode()
    def test_embedding_prefill_and_decode_against_eager(self):
        reports = []
        width = self.built.hidden_size
        # Alternate overridden/plain requests and shrink/grow a shared capture bucket.
        for lengths, layout in [([8], [(1, 3)]), ([4], []), ([4], [(1, 2)]), ([5, 11], [(0, 2), (7, 4)]), ([16], [])]:
            segments = []
            for loc, size in layout:
                ids = torch.arange(100 + loc, 100 + loc + size, device="cuda", dtype=torch.int32)
                replacement = self.model.embed_tokens(ids).clone().float()
                self.assertEqual(replacement.shape, (size, width))
                segments.append((loc, replacement))
            reference = self.run_path(lengths, segments, False, False)
            padded_reference = self.run_path(lengths, segments, False, False, eager_padding=4) if lengths == [4] else reference
            if segments:
                plain = self.run_path(lengths, (), False, False)
                self.assertGreater((reference[0].float() - plain[0].float()).abs().max().item(), 1e-3)
            for pg, dg in [(False, False), (False, True), (True, False), (True, True)]:
                actual = self.run_path(lengths, segments, pg, dg)
                errors = []
                for step, (got, expected) in enumerate(zip(actual, reference)):
                    self.assertTrue(torch.isfinite(got).all().item())
                    delta = got.float() - expected.float()
                    error = delta.abs().max().item()
                    errors.append(error)
                    print("GRAPH_COMPARE=" + json.dumps(dict(lengths=lengths, segments=len(segments), prefill_graph=pg, decode_graph=dg, step=step, max_abs=error, rmse=delta.square().mean().sqrt().item(), reference_rms=expected.float().square().mean().sqrt().item())), flush=True)
                    if pg and lengths == [4]:
                        padded_delta = got.float() - padded_reference[step].float()
                        print("PADDED_EAGER_COMPARE=" + json.dumps(dict(segments=len(segments), decode_graph=dg, step=step, max_abs=padded_delta.abs().max().item(), rmse=padded_delta.square().mean().sqrt().item())), flush=True)
                    # Follow cuda_graph_decode_padding.py: compare at the shape
                    # actually executed. BF16 kernels can differ across token
                    # counts even without a graph. Keep the cross-shape errors
                    # above as diagnostics; do not relax the numeric tolerance.
                    same_shape_expected = padded_reference[step] if pg else expected
                    with self.subTest(lengths=lengths, segments=len(segments), prefill_graph=pg, decode_graph=dg, step=step):
                        torch.testing.assert_close(got, same_shape_expected, rtol=1e-2, atol=1e-2)
                reports.append(dict(lengths=lengths, segments=len(segments), prefill_graph=pg, decode_graph=dg, max_abs=errors))
        print("INPUT_EMBEDDING_GRAPH_RESULT=" + json.dumps(reports), flush=True)


if __name__ == "__main__":
    unittest.main()
