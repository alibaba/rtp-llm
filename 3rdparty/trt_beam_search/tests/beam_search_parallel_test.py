import math
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

    def _run_live_and_frozen(self, source, output):
        frozen = self.ops.sample(*clones(source), output, True)
        live = self.ops.sample(*clones(source), output, False)
        exact(frozen, live)
        return live

    def _check_math_beams(self, actual, history, inputs, lengths, parents, tokens, scores):
        actual_ids, actual_inputs, actual_lengths, actual_scores, actual_parents = (
            value.cpu() for value in actual
        )
        self.assertTrue(torch.equal(actual_parents[0].long(), parents))
        self.assertTrue(torch.equal(actual_inputs[0], inputs[0, parents]))
        self.assertTrue(torch.equal(actual_lengths[0], lengths[0, parents] + 1))
        expected_history = history[0, parents].clone()
        positions = lengths[0, parents].long()
        expected_history[torch.arange(parents.numel(), device="cpu"), positions] = tokens.to(torch.int32)
        self.assertTrue(torch.equal(actual_ids[0], expected_history))
        torch.testing.assert_close(actual_scores[0].double(), scores, atol=2e-5, rtol=2e-6)

    def test_large_variable_beams_masked_full_vs_compact_math_reference(self):
        # A synthetic non-contiguous vocabulary, with the production workload's
        # dimensions. No checkpoint, tokenizer, or model-specific IDs are needed.
        vocab, subset_size = 217216, 65537
        subset_cpu = torch.arange(subset_size, device="cpu", dtype=torch.long) * 3 + 1
        subset = subset_cpu.cuda()
        allowed = torch.zeros(vocab, device="cuda", dtype=torch.bool)
        allowed[subset] = True
        background = -40.0
        history = torch.tensor([[[7, 11, 0, 0]]], device="cpu", dtype=torch.int32)
        inputs = torch.tensor([[2]], device="cpu", dtype=torch.int32)
        lengths = inputs.clone()
        reference_cum = torch.zeros(1, device="cpu", dtype=torch.float64)
        states = None
        for step, (beams, output) in enumerate(((1, 500), (500, 1000))):
            with self.subTest(step=step, beams=beams, output=output):
                count = 1000 if step == 0 else 4
                local_columns = torch.cat((
                    torch.tensor([0, subset_size - 1], device="cpu"),
                    torch.arange(1, count - 1, device="cpu"),
                ))
                compact_columns = (
                    torch.arange(beams, device="cpu")[:, None] * 137 + local_columns
                ) % subset_size
                canonical_columns = subset_cpu[compact_columns]
                if step == 0:
                    candidate_values = (-torch.arange(count, device="cpu").float() * .005)[None, :]
                else:
                    candidate_values = torch.tensor([[0., -4., -12., -20.]], device="cpu").expand(beams, -1).clone()
                    candidate_values[:, 1] -= torch.arange(beams, device="cpu").float() * .0001

                # FP64 oracle accounts for every allowed background column by
                # multiplicity. Only competitive candidates need enumeration.
                values64 = candidate_values.double()
                background_log_mass = math.log(subset_size - count) + background
                log_z = torch.logsumexp(torch.cat((
                    values64,
                    torch.full((beams, 1), background_log_mass, device="cpu", dtype=torch.float64),
                ), dim=1), dim=1)
                candidate_scores = values64 - log_z[:, None] + reference_cum[:, None]
                ordered, indices = torch.topk(candidate_scores.flatten(), output + 1)
                self.assertGreater(float((ordered[:output - 1] - ordered[1:output]).min()), 1e-4)
                self.assertGreater(float(ordered[output - 1] - ordered[output]), 1e-4)
                self.assertLess(float((background - log_z + reference_cum).max()), float(ordered[output - 1]))
                indices, scores = indices[:output], ordered[:output]
                parents = indices // count
                tokens = canonical_columns.flatten()[indices]

                # Both samplers still receive actual full-width / compact-width
                # matrices. Mask BEFORE the wrapper's in-place log_softmax.
                full = torch.full((1, beams, vocab), background, device="cuda", dtype=torch.float32)
                full[0].scatter_(1, canonical_columns.cuda(), candidate_values.cuda())
                compact = full.index_select(-1, subset)
                full.masked_fill_(~allowed.view(1, 1, vocab), -torch.inf)
                if states is None:
                    initial = [history.cuda(), inputs.cuda(), lengths.cuda(), reference_cum.float()[None, :].cuda()]
                    states = [clones(initial), clones(initial)]
                results = []
                for route, logits in enumerate((full, compact)):
                    result = self._run_live_and_frozen([logits, *states[route]], output)
                    if route == 1:
                        # History already contains canonical IDs, exactly as the
                        # dispatcher contract requires. Restore only NEW tokens.
                        position = int(lengths[0, 0])
                        new_ids = result[0][:, :, position].long()
                        self.assertTrue(bool(((new_ids >= 0) & (new_ids < subset_size)).all()))
                        result[0][:, :, position] = subset[new_ids].to(torch.int32)
                    self._check_math_beams(result, history, inputs, lengths, parents, tokens, scores)
                    results.append(result)
                exact(results[0][1:3], results[1][1:3])
                exact(results[0][4], results[1][4])
                torch.testing.assert_close(results[0][3], results[1][3], atol=2e-5, rtol=2e-6)
                position = int(lengths[0, 0])
                exact(results[0][0][:, :, :position + 1], results[1][0][:, :, :position + 1])
                states = [[r[0], r[1], r[2], r[3]] for r in results]
                history = history[:, parents, :].clone()
                history[0, :, position] = tokens.to(torch.int32)
                inputs = inputs[:, parents].clone()
                lengths = lengths[:, parents] + 1
                reference_cum = scores
                del full, compact

    def test_unmasked_full_vocab_changes_cross_parent_ranking(self):
        # The winning token is allowed in BOTH distributions, but omitted mass
        # differs by parent, so the winning parent changes without a pre-mask.
        subset = torch.tensor([1, 3], device="cuda", dtype=torch.long)
        logits = torch.tensor([[[-100., 0., 0., -10., -100.],
                                [-100., 0., -100., -1., -100.]]], device="cuda")
        ids = torch.tensor([[[7, 11, 0], [13, 17, 0]]], device="cuda", dtype=torch.int32)
        lengths = torch.full((1, 2), 2, device="cuda", dtype=torch.int32)
        cum = torch.zeros(1, 2, device="cuda")
        masked = logits.clone()
        masked[:, :, [0, 2, 4]] = -torch.inf
        full = self._run_live_and_frozen([logits, ids, lengths, lengths, cum], 1)
        constrained = self._run_live_and_frozen([masked, ids, lengths, lengths, cum], 1)
        compact = self._run_live_and_frozen([logits.index_select(-1, subset), ids, lengths, lengths, cum], 1)
        self.assertTrue(bool(((compact[0][:, :, 2] >= 0) & (compact[0][:, :, 2] < subset.numel())).all()))
        compact[0][:, :, 2] = subset[compact[0][:, :, 2].long()].to(torch.int32)
        for source, result in ((logits, full), (masked, constrained)):
            oracle = torch.log_softmax(source.cpu().double(), dim=-1).flatten()
            index = oracle.argmax().reshape(1)
            self._check_math_beams(result, ids.cpu(), lengths.cpu(), lengths.cpu(),
                                   index // 5, index % 5, oracle[index])
        self.assertEqual(int(full[4][0, 0]), 1)
        self.assertEqual(int(constrained[4][0, 0]), 0)
        exact(constrained[:3], compact[:3])
        exact(constrained[4], compact[4])
        torch.testing.assert_close(constrained[3], compact[3], atol=2e-5, rtol=2e-6)

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
