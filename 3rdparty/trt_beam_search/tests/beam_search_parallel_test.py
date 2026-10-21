import unittest

import torch

from sampling_test_support import (add_inputs, check_frozen_sources, clones, exact,
                                   load_ops, sample_inputs, stage_inputs)


class BeamSearchParallelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; absence must not be reported as a passing skip")
        cls.ops = load_ops()
        check_frozen_sources()

    def test_frozen_reference_hashes(self):
        check_frozen_sources()

    def check_add(self, values, topk):
        original, candidate = clones(values), clones(values)
        a = self.ops.add_cum(*original, topk, True)
        b = self.ops.add_cum(*candidate, topk, False)
        exact(a, b)
        exact(original[1:], candidate[1:])

    def test_add_actual_shapes(self):
        for batch in (1, 2):
            for beams, topk in ((1, 256), (128, 512), (256, 2048), (512, 2048)):
                for dtype in (torch.float32, torch.float16):
                    with self.subTest(batch=batch, beams=beams, topk=topk, dtype=dtype):
                        self.check_add(add_inputs(batch, beams, topk, dtype), topk)

    def test_add_odd_slots_finished_diversity(self):
        for beams, topk, tail in ((127, 255, 7), (257, 513, 13), (1, 1, 0)):
            for dtype in (torch.float32, torch.float16):
                self.check_add(add_inputs(2, beams, topk, dtype, tail, states=True), topk)

    def test_add_float_boundaries(self):
        for dtype in (torch.float32, torch.float16):
            values = add_inputs(2, 128, 257, dtype, states=True)
            values[0][:, :8] = torch.tensor([0., -0., float("inf"), -float("inf"), float("nan"),
                                           1., -1., torch.finfo(dtype).tiny], device="cuda", dtype=dtype)
            self.check_add(values, 257)

    def check_stage(self, state, output, alias=False, cba=False, tiled=False, v2=True):
        a = self.ops.stage3(clones(state), output, 257, alias, cba, tiled, v2, True)
        b = self.ops.stage3(clones(state), output, 257, alias, cba, tiled, v2, False)
        exact(a, b)

    def test_stage3_dynamic_and_fixed_alias(self):
        for batch in (1, 2):
            for beams, output in ((1, 128), (128, 256), (256, 1024), (512, 1024), (257, 127), (128, 128)):
                for dtype in (torch.float32, torch.float16):
                    with self.subTest(batch=batch, beams=beams, output=output, dtype=dtype):
                        self.check_stage(stage_inputs(batch, beams, output, dtype), output)
            for dtype in (torch.float32, torch.float16):
                self.check_stage(stage_inputs(batch, 256, 256, dtype, states=True), 256, alias=True)

    def test_stage3_finished_and_slot_permutation(self):
        for beams, output in ((128, 256), (256, 1024), (257, 127)):
            self.check_stage(stage_inputs(2, beams, output, states=True), output)

    def test_stage3_cba_tiled_v1_fallback(self):
        for dtype in (torch.float32, torch.float16):
            state = stage_inputs(2, 4, 4, dtype, states=True)
            self.check_stage(state, 4, cba=True)
            self.check_stage(state, 4, tiled=True, alias=True)
            self.check_stage(stage_inputs(2, 4, 4, dtype, states=True, v2=False), 4, v2=False)

    def test_full_sampler_actual_shapes(self):
        for batch in (1, 2):
            for beams, output in ((1, 128), (128, 128), (128, 256), (256, 256), (256, 1024)):
                with self.subTest(batch=batch, beams=beams, output=output):
                    source = sample_inputs(batch, beams)
                    exact(self.ops.sample(*clones(source), output, True),
                          self.ops.sample(*clones(source), output, False))

    def test_full_sampler_ties_masks_and_odd_shapes(self):
        for pattern in ("ties", "near", "masked"):
            for beams, output, vocab in ((127, 257, 2051), (257, 128, 4099), (4, 4, 257)):
                source = sample_inputs(2, beams, vocab, pattern=pattern)
                exact(self.ops.sample(*clones(source), output, True),
                      self.ops.sample(*clones(source), output, False))

    def graph_compare(self, source_fn, run):
        source = source_fn(100)
        # Explicit side-stream warmup before separate reference/candidate capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run(clones(source), True)
                run(clones(source), False)
        torch.cuda.current_stream().wait_stream(stream)
        left, right = clones(source), clones(source)
        ga, gb = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(ga):
            a = run(left, True)
        with torch.cuda.graph(gb):
            b = run(right, False)
        addresses = [[x.data_ptr() for x in side] for side in (left, right)]
        for seed in (101, 102, 103):
            new = source_fn(seed)
            eager = run(clones(new), True)
            for source_tensor, x, y in zip(new, left, right):
                x.copy_(source_tensor)
                y.copy_(source_tensor)
            ga.replay()
            gb.replay()
            exact(eager, a)
            exact(a, b)
            self.assertEqual(addresses, [[x.data_ptr() for x in side] for side in (left, right)])

    def test_graph_add_changed_content(self):
        self.graph_compare(lambda seed: add_inputs(2, 257, 513, states=True, seed=seed),
                           lambda x, ref: self.ops.add_cum(*x, 513, ref))

    def test_graph_stage3_changed_content_and_alias(self):
        def source(seed):
            x = stage_inputs(2, 128, 128, states=True, seed=seed)
            x[0].add_(seed % 3).remainder_(128 * 257)
            x[5] = x[5].roll(seed % 3, dims=1)
            x[7] = x[7].flip(0) if seed % 2 else x[7]
            return x
        self.graph_compare(source, lambda x, ref: self.ops.stage3(x, 128, 257, True, False, False, True, ref))

if __name__ == "__main__":
    unittest.main()
