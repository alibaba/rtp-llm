import unittest
from unittest.mock import patch

import torch
from rtp_llm.models_py.sampling import adapter, runtime
from rtp_llm.models_py.sampling.draft_logits import DraftLogitsCache
from rtp_llm.models_py.sampling.vllm_rejection import rejection_sample


def verify_requests_individually(
    logits: torch.Tensor,
    draft_probs: torch.Tensor | None,
    draft_ids: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
    do_sample: torch.Tensor,
    forced: list[bool],
    caps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, rows, vocab = logits.shape
    proposals = draft_ids.cpu().tolist()
    limits = caps.cpu().tolist()
    local_positions = torch.arange(rows, device=logits.device, dtype=torch.int32)
    row_bounds = local_positions.new_tensor([0, rows])
    mapping = torch.zeros_like(local_positions)
    forced_rates = torch.ones(rows - 1, device=logits.device)
    outputs: list[torch.Tensor] = []
    lengths: list[torch.Tensor] = []
    for request in range(batch):
        limit = min(rows - 1, max(0, limits[request]))
        packed = [-1] + [
            token if column < limit else -1
            for column, token in enumerate(proposals[request])
        ]
        cache = (
            DraftLogitsCache(draft_probs[request : request + 1].log())
            if draft_probs is not None
            else None
        )
        sampled, count = rejection_sample(
            target_logits=logits[request].reshape(rows, vocab),
            draft_logits_cache=cache,
            draft_sampled=torch.tensor(packed, device=logits.device, dtype=torch.int64),
            cu_num_logits=row_bounds,
            pos=positions[request : request + 1] + local_positions,
            idx_mapping=mapping[:1],
            expanded_idx_mapping=mapping,
            expanded_local_pos=local_positions,
            temperature=do_sample[request : request + 1].float(),
            seed=seeds[request : request + 1],
            num_speculative_steps=rows - 1,
            synthetic_conditional_rates=forced_rates if forced[request] else None,
            skip_masked_vocab_blocks=True,
        )
        outputs.append(
            torch.where(
                local_positions.unsqueeze(0) < count.unsqueeze(1), sampled, -1
            ).to(torch.int32)
        )
        lengths.append(count)
    return torch.cat(outputs), torch.cat(lengths)


class DirectOutputTest(unittest.TestCase):
    def test_cached_graph_refreshes_inputs_and_owns_outputs(self) -> None:
        from rtp_llm.models_py.sampling import rejection_graph

        torch.manual_seed(456)
        batch, steps, vocab = 16, 5, 8193
        for forced in (False, True):
            old_outputs = []
            for iteration in range(2):
                logits = torch.randn(batch, steps + 1, vocab, device="cuda")
                ids = torch.randint(
                    vocab, (batch, steps), device="cuda", dtype=torch.int32
                )
                seeds = (
                    torch.arange(batch, device="cuda", dtype=torch.int64) + iteration
                )
                positions = seeds + iteration * 17
                sample = seeds % 2 == 0
                caps = (seeds % (steps + 1)).int()
                args = (
                    logits,
                    None,
                    ids,
                    seeds,
                    positions,
                    sample,
                    [forced] * batch,
                    caps,
                )
                expected, expected_lengths = adapter._rejection_sample_logits_eager(
                    *args, clean_output=True
                )
                output = torch.full(
                    (batch, steps + 1), -777, device="cuda", dtype=torch.int32
                )
                lengths = torch.full((batch,), -777, device="cuda", dtype=torch.int32)
                self.assertTrue(
                    rejection_graph.try_replay(
                        logits,
                        ids,
                        seeds,
                        positions,
                        sample,
                        [forced] * batch,
                        caps,
                        output,
                        lengths,
                    )
                )
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                torch.testing.assert_close(lengths, expected_lengths, rtol=0, atol=0)
                old_outputs.append((output, output.clone(), lengths, lengths.clone()))
            for output, snapshot, lengths, saved_lengths in old_outputs:
                torch.testing.assert_close(output, snapshot, rtol=0, atol=0)
                torch.testing.assert_close(lengths, saved_lengths, rtol=0, atol=0)

    def test_indirect_graph_queued_inputs_from_another_stream(self) -> None:
        from rtp_llm.models_py.sampling import rejection_graph

        torch.cuda.synchronize()
        producer = torch.cuda.Stream()
        consumer = torch.cuda.current_stream()
        saved = []
        first_logits = None
        first_snapshot = None
        for iteration in range(8):
            with torch.cuda.stream(producer):
                logits = torch.randn(16, 6, 8193, device="cuda")
                ids = torch.randint(8193, (16, 5), device="cuda", dtype=torch.int32)
                seeds = torch.arange(16, device="cuda", dtype=torch.int64) + iteration
                positions = seeds + iteration * 19
                sample = seeds % 2 == 0
                expected, counts = adapter._rejection_sample_logits_eager(
                    logits, None, ids, seeds, positions, sample, clean_output=True
                )
                if iteration == 0:
                    first_logits = logits
                    first_snapshot = logits.clone()
            consumer.wait_stream(producer)
            tokens = torch.empty((16, 6), device="cuda", dtype=torch.int32)
            lengths = torch.empty(16, device="cuda", dtype=torch.int32)
            self.assertTrue(
                rejection_graph.try_replay(
                    logits,
                    ids,
                    seeds,
                    positions,
                    sample,
                    None,
                    None,
                    tokens,
                    lengths,
                )
            )
            saved.append((tokens, lengths, expected, counts))
            # The Graph must protect each producer allocation until its read
            # completes, even when the caller immediately releases the tensor.
            del logits
        torch.cuda.synchronize()
        torch.testing.assert_close(first_logits, first_snapshot, rtol=0, atol=0)
        for tokens, lengths, expected, counts in saved:
            torch.testing.assert_close(tokens, expected, rtol=0, atol=0)
            torch.testing.assert_close(lengths, counts, rtol=0, atol=0)

    def test_graph_stream_isolation_and_external_capture_fallback(self) -> None:
        from rtp_llm.models_py.sampling import rejection_graph

        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        saved = []
        for stream in streams:
            with torch.cuda.stream(stream):
                logits = torch.randn(16, 6, 513, device="cuda")
                ids = torch.randint(513, (16, 5), device="cuda", dtype=torch.int32)
                seeds = torch.arange(16, device="cuda", dtype=torch.int64)
                sample = seeds % 2 == 0
                tokens = torch.empty((16, 6), device="cuda", dtype=torch.int32)
                lengths = torch.empty(16, device="cuda", dtype=torch.int32)
                args = (logits, ids, seeds, seeds, sample, None, None, tokens, lengths)
                expected, counts = adapter._rejection_sample_logits_eager(
                    logits, None, ids, seeds, seeds, sample, clean_output=True
                )
                # Finish old entries so eviction cannot intentionally fall back.
                torch.cuda.synchronize()
                self.assertTrue(rejection_graph.try_replay(*args))
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    self.assertFalse(rejection_graph.try_replay(*args))
                    adapter.rejection_sample_logits(
                        logits,
                        None,
                        ids,
                        seeds,
                        seeds,
                        sample,
                        output_ids=tokens,
                        output_lengths=lengths,
                        clean_output=True,
                    )
                graph.replay()
                saved.append((tokens, lengths, expected, counts))
        torch.cuda.synchronize()
        for tokens, lengths, expected, counts in saved:
            torch.testing.assert_close(tokens, expected, rtol=0, atol=0)
            torch.testing.assert_close(lengths, counts, rtol=0, atol=0)

    def test_matches_legacy_output_and_preserves_buffers(self) -> None:
        torch.manual_seed(123)
        for steps, dense in [(0, False), (3, True), (5, False)]:
            with self.subTest(steps=steps, dense=dense):
                batch, vocab = 8, 131
                logits = torch.randn(batch, steps + 1, vocab, device="cuda")
                probs = (
                    torch.softmax(
                        torch.randn(batch, steps, vocab, device="cuda"), -1
                    )
                    if dense
                    else None
                )
                ids = torch.randint(
                    vocab, (batch, steps), device="cuda", dtype=torch.int32
                )
                seeds = torch.arange(batch, device="cuda", dtype=torch.int64)
                positions = seeds + 17
                sample = seeds % 2 == 0
                caps = (seeds % (steps + 1)).int()
                forced = [False, True, True, False, False, True, False, True]
                args = (logits, probs, ids, seeds, positions, sample, forced, caps)
                expected, lengths = adapter.rejection_sample_logits(*args)
                storage = torch.full(
                    (batch, steps + 3), -777, device="cuda", dtype=torch.int32
                )
                output = storage[:, 1:-1]
                counts = torch.full_like(lengths, -777)
                actual, actual_lengths = adapter.rejection_sample_logits(
                    *args,
                    output_ids=output,
                    output_lengths=counts,
                    clean_output=True,
                )
                self.assertEqual(actual.data_ptr(), output.data_ptr())
                self.assertEqual(actual_lengths.data_ptr(), counts.data_ptr())
                torch.testing.assert_close(
                    actual, expected.clamp_min(0), rtol=0, atol=0
                )
                torch.testing.assert_close(actual_lengths, lengths, rtol=0, atol=0)
                self.assertTrue((storage[:, (0, -1)] == -777).all().item())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    adapter.rejection_sample_logits(
                        *args,
                        output_ids=output,
                        output_lengths=counts,
                        clean_output=True,
                    )
                output.fill_(-777)
                counts.fill_(-777)
                graph.replay()
                torch.testing.assert_close(
                    output, expected.clamp_min(0), rtol=0, atol=0
                )
                torch.testing.assert_close(counts, lengths, rtol=0, atol=0)


class DeferredRejectionTest(unittest.TestCase):
    def test_warmup_rng_and_error_handling(self) -> None:
        before = torch.cuda.get_rng_state().clone()
        self.assertEqual(runtime.warmup(8193), "")
        self.assertTrue(torch.equal(before, torch.cuda.get_rng_state()))
        with patch.object(
            adapter, "rejection_sample_logits", side_effect=ImportError("missing dependency")
        ):
            self.assertIn("ImportError", runtime.warmup(8193))
        with patch.object(
            adapter, "rejection_sample_logits", side_effect=RuntimeError("CUDA execution failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "CUDA execution failed"):
                runtime.warmup(8193)

    def test_batched_graph_replay_matches_individual_requests(self) -> None:
        steps = 3
        for dense, dtype, vocab in [(False, torch.int32, 131), (True, torch.int64, 8193)]:
            with self.subTest(dense=dense, dtype=dtype):
                batch = 8
                generator = torch.Generator(device="cuda").manual_seed(3041)
                logits = torch.randn(
                    batch, steps + 1, vocab, device="cuda", generator=generator
                )
                draft = torch.empty(batch, 2 * steps, device="cuda", dtype=dtype)[:, ::2]
                draft.copy_(logits[:, :steps].argmax(-1))
                q = logits[:, :steps].softmax(-1) if dense else None
                seeds = torch.arange(batch, device="cuda", dtype=torch.int64) + 17
                positions = seeds * 31 + (1 << 33)
                random = torch.ones(batch, device="cuda", dtype=torch.bool)
                caps = torch.full((batch,), steps, device="cuda", dtype=torch.int32)
                forced = [False, False, True, True, False, True, False, False]
                output = torch.empty(batch, steps + 1, device="cuda", dtype=torch.int32)
                lengths = torch.empty(batch, device="cuda", dtype=torch.int32)

                def draw() -> tuple[torch.Tensor, torch.Tensor]:
                    return adapter.rejection_sample_logits(
                        logits,
                        q,
                        draft,
                        seeds,
                        positions,
                        random,
                        forced,
                        caps,
                        output_ids=output,
                        output_lengths=lengths,
                    )

                for _ in range(3):
                    draw()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual, actual_lengths = draw()
                self.assertEqual(actual.data_ptr(), output.data_ptr())
                self.assertEqual(actual_lengths.data_ptr(), lengths.data_ptr())
                for replay in range(2):
                    logits.normal_(generator=generator)
                    logits[:, :, -2:] = -torch.inf
                    draft.copy_(logits[:, :steps].argmax(-1))
                    logits[:, :steps].scatter_add_(
                        2,
                        draft.long().unsqueeze(-1),
                        torch.full((batch, steps, 1), 3.0, device="cuda"),
                    )
                    if q is not None:
                        q.copy_(logits[:, :steps].softmax(-1))
                        q.mul_(0.35).scatter_add_(
                            2,
                            draft.long().unsqueeze(-1),
                            torch.full((batch, steps, 1), 0.65, device="cuda"),
                        )
                    seeds.add_(1009)
                    positions.add_(137)
                    random.copy_(
                        torch.tensor(
                            [(index + replay) % 3 != 0 for index in range(batch)],
                            device="cuda",
                        )
                    )
                    caps.copy_(
                        torch.tensor(
                            [(index + replay) % 6 - 1 for index in range(batch)],
                            device="cuda",
                            dtype=torch.int32,
                        )
                    )
                    expected, expected_lengths = verify_requests_individually(
                        logits, q, draft, seeds, positions, random, forced, caps
                    )
                    original_q = q.clone() if q is not None else None
                    graph.replay()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(
                        actual_lengths, expected_lengths, rtol=0, atol=0
                    )
                    if q is not None:
                        torch.testing.assert_close(q, original_q, rtol=0, atol=0)

    def draw(
        self,
        target: torch.Tensor,
        q: torch.Tensor | None,
        draft: torch.Tensor,
        seeds: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return adapter.rejection_sample_logits(
            target.log(),
            q,
            draft,
            seeds,
            positions,
            torch.ones(target.shape[0], device="cuda", dtype=torch.bool),
        )

    def test_each_rejection_position_and_bonus(self) -> None:
        target = torch.zeros(4, 4, 4, device="cuda")
        for request in range(4):
            for position in range(3):
                target[request, position, 0 if position < request else 2] = 1
        target[:, 3, 3] = 1
        draft = torch.zeros(4, 3, dtype=torch.int32, device="cuda")
        seeds = torch.arange(4, device="cuda", dtype=torch.int64)
        q = torch.zeros_like(target[:, :-1])
        q[:, :, 0] = 1
        random = torch.tensor([False, True, False, True], device="cuda")
        expected = [[2, -1, -1, -1], [0, 2, -1, -1], [0, 0, 2, -1], [0, 0, 0, 3]]
        for probs in (None, q):
            with self.subTest(dense=probs is not None):
                output, lengths = adapter.rejection_sample_logits(
                    target.log(), probs, draft, seeds, seeds, random
                )
                self.assertEqual(output.cpu().tolist(), expected)
                self.assertEqual(lengths.cpu().tolist(), [1, 2, 3, 4])

    def test_dense_draft_uses_residual_instead_of_target(self) -> None:
        batch = 64
        target = torch.tensor([0.0, 0.3, 0.7], device="cuda").expand(batch, 2, 3)
        draft_probs = torch.tensor([0.6, 0.4, 0], device="cuda").expand(batch, 1, 3)
        draft = torch.zeros(batch, 1, dtype=torch.int32, device="cuda")
        seeds = torch.arange(batch, dtype=torch.int64, device="cuda") + 17
        output, lengths = self.draw(target, draft_probs, draft, seeds, seeds)
        self.assertTrue(bool((output[:, 0] == 2).all()))
        self.assertTrue(bool((output[:, 1] == -1).all()))
        self.assertTrue(bool((lengths == 1).all()))

    def test_bonus_distribution_and_masked_tokens(self) -> None:
        batch = 20000
        target = torch.zeros(batch, 2, 4, device="cuda")
        target[:, 0, 0] = 1
        target[:, 1] = torch.tensor([0.6, 0.3, 0.1, 0], device="cuda")
        seeds = torch.arange(batch, dtype=torch.int64, device="cuda") + 20261106
        output, lengths = self.draw(
            target,
            None,
            torch.zeros(batch, 1, device="cuda", dtype=torch.int32),
            seeds,
            seeds,
        )
        self.assertTrue(bool((lengths == 2).all()))
        observed = (
            torch.bincount(output[:, 1].long(), minlength=4).float().cpu() / batch
        )
        expected = torch.tensor([0.6, 0.3, 0.1, 0])
        self.assertTrue(bool((torch.abs(observed - expected) < 0.015).all()), observed)
        self.assertEqual(int((output[:, 1] == 3).sum()), 0)

    def test_zero_draft_steps(self) -> None:
        target = torch.tensor([[[0.0, 1.0]]], device="cuda")
        seeds = torch.zeros(1, dtype=torch.int64, device="cuda")
        output, lengths = self.draw(
            target,
            None,
            torch.empty(1, 0, device="cuda", dtype=torch.int32),
            seeds,
            seeds,
        )
        self.assertEqual(output.item(), 1)
        self.assertEqual(lengths.item(), 1)

    def test_acceptance_and_correction_preserve_target_distribution(self) -> None:
        batch = 50000
        generator = torch.Generator(device="cuda").manual_seed(73)
        p = torch.tensor([0.2, 0.3, 0.5], device="cuda")
        q = torch.tensor([0.6, 0.4, 0.0], device="cuda")
        proposals = torch.multinomial(
            q, batch, replacement=True, generator=generator
        ).reshape(batch, 1)
        seeds = torch.arange(batch, device="cuda", dtype=torch.int64) + 9001
        output, lengths = self.draw(
            p.expand(batch, 2, 3), q.expand(batch, 1, 3), proposals, seeds, seeds
        )
        frequency = torch.bincount(output[:, 0].long(), minlength=3).float() / batch
        self.assertTrue(bool((torch.abs(frequency - p) < 0.01).all()), frequency)
        self.assertAlmostEqual(float((lengths == 2).float().mean()), 0.5, delta=0.01)

    def test_mixed_greedy_force_and_cap(self) -> None:
        target = torch.tensor([[[0, 1.0, 0], [0, 0, 1.0]]], device="cuda").repeat(
            4, 1, 1
        )
        seeds = torch.arange(4, device="cuda", dtype=torch.int64)
        output, lengths = adapter.rejection_sample_logits(
            target.log(),
            None,
            torch.zeros(4, 1, device="cuda", dtype=torch.int32),
            seeds,
            seeds,
            torch.tensor([False, True, False, True], device="cuda"),
            [False, True, True, True],
            torch.tensor([1, 1, 0, 0], device="cuda"),
        )
        self.assertEqual(output.cpu().tolist(), [[1, -1], [0, 2], [1, -1], [1, -1]])
        self.assertEqual(lengths.cpu().tolist(), [1, 2, 1, 1])

    def test_cumulative_logprobs_follow_final_tokens(self) -> None:
        target = torch.tensor([[[0.5, 0.5, 0], [0, 0.25, 0.75]]], device="cuda")
        cumulative = torch.full((2,), -2.0, device="cuda")
        seeds = torch.zeros(1, device="cuda", dtype=torch.int64)
        output, lengths = adapter.rejection_sample_logits(
            target.log(),
            None,
            torch.zeros(1, 1, device="cuda", dtype=torch.int32),
            seeds,
            seeds,
            torch.ones(1, device="cuda", dtype=torch.bool),
            [True],
            None,
            cumulative,
        )
        expected = (
            target.gather(2, output.long().unsqueeze(2)).squeeze(2).log().cumsum(1) - 2
        )
        torch.testing.assert_close(cumulative, expected.reshape(-1))
        self.assertEqual(lengths.item(), 2)

    def test_padded_large_vocab_invalid_draft_and_shift(self) -> None:
        logits = torch.full((3, 4, 154883), -torch.inf, device="cuda")
        logits[:, :, 8193] = 2
        logits[:, :, 150001] = 1
        draft = torch.full((3, 3), -1, dtype=torch.int32, device="cuda")
        seeds = torch.arange(3, dtype=torch.int64, device="cuda")
        random = torch.ones(3, dtype=torch.bool, device="cuda")
        first, lengths = adapter.rejection_sample_logits(
            logits, None, draft, seeds, seeds, random
        )
        second, _ = adapter.rejection_sample_logits(
            logits + 10000, None, draft, seeds, seeds, random
        )
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(bool(((first[:, 0] == 8193) | (first[:, 0] == 150001)).all()))
        self.assertEqual(lengths.cpu().tolist(), [1, 1, 1])


class FusedDraftLogTest(unittest.TestCase):
    def test_fused_float32_path_does_not_materialize_log(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(8921)
        logits = torch.randn(4, 4, 131, device="cuda", generator=generator)
        q = logits[:, :3].softmax(-1)
        draft = q.argmax(-1).int()
        seeds = torch.arange(4, device="cuda", dtype=torch.int64)
        do_sample = torch.ones(4, device="cuda", dtype=torch.bool)
        for enabled in (False, None):
            with self.subTest(fuse_draft_probs_log=enabled):
                def draw():
                    return adapter._rejection_sample_logits_eager(
                        logits, q, draft, seeds, seeds, do_sample, fuse_draft_probs_log=enabled
                    )

                draw()
                torch.cuda.synchronize()
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU]
                ) as profile:
                    draw()
                    torch.cuda.synchronize()
                operators = {event.key for event in profile.key_averages()}
                self.assertEqual("aten::log" in operators, enabled is False)

    def test_toggle_matches_materialized_log_with_strided_probabilities(self) -> None:
        for batch, steps, vocab, dtype, strided in [
            (1, 1, 7, torch.float32, False),
            (8, 3, 131, torch.float16, True),
            (8, 3, 131, torch.bfloat16, True),
            (20, 3, 8193, torch.float32, True),
            (8, 3, 131, torch.float64, True),
        ]:
            with self.subTest(shape=(batch, steps, vocab), dtype=dtype, strided=strided):
                generator = torch.Generator(device="cuda").manual_seed(5119)
                logits = torch.randn(
                    batch, steps + 1, vocab, generator=generator, device="cuda"
                )
                logits[:, :, -2:] = -torch.inf
                q = (
                    torch.randn(
                        batch, steps, vocab, generator=generator, device="cuda"
                    )
                    .softmax(-1)
                    .to(dtype)
                )
                q[:, :, -2:] = 0
                q /= q.sum(-1, keepdim=True)
                if strided:
                    storage = torch.full(
                        (batch * 2, steps * 2, vocab),
                        -7,
                        device="cuda",
                        dtype=dtype,
                    )
                    view = storage[::2, ::2]
                    view.copy_(q)
                    q = view
                original_q = q.clone()
                original_logits = logits.clone()
                draft = q.argmax(-1).int()
                seeds = (
                    torch.arange(batch, device="cuda", dtype=torch.int64) + 107
                )
                positions = seeds * 13 + (1 << 33)
                do_sample = seeds % 3 != 0
                args = (logits, q, draft, seeds, positions, do_sample)
                expected, expected_lengths = adapter._rejection_sample_logits_eager(
                    *args, fuse_draft_probs_log=False
                )
                actual, lengths = adapter._rejection_sample_logits_eager(
                    *args, fuse_draft_probs_log=True
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    lengths, expected_lengths, rtol=0, atol=0
                )
                torch.testing.assert_close(q, original_q, rtol=0, atol=0)
                torch.testing.assert_close(
                    logits, original_logits, rtol=0, atol=0
                )
                if strided:
                    self.assertTrue(bool((storage[1::2] == -7).all()))
                    self.assertTrue(bool((storage[::2, 1::2] == -7).all()))

    def test_cache_flag_preserves_temperature_and_row_mapping(self) -> None:
        batch, steps, vocab = 4, 3, 131
        generator = torch.Generator(device="cuda").manual_seed(9137)
        logits = torch.randn(
            batch, steps + 1, vocab, device="cuda", generator=generator
        )
        q = torch.randn(
            batch, steps, vocab, device="cuda", generator=generator
        ).softmax(-1)
        q[:, :, -2:] = 0
        q /= q.sum(-1, keepdim=True)
        draft = (
            torch.cat([torch.full((batch, 1), -1, device="cuda"), q.argmax(-1)], dim=1)
            .long()
            .flatten()
        )
        local = torch.arange(steps + 1, device="cuda", dtype=torch.int32).repeat(batch)
        mapping = torch.arange(batch, device="cuda", dtype=torch.int32)
        row_mapping = mapping.flip(0)
        temperature = torch.tensor([0.0, 0.5, 1.0, 1.7], device="cuda")
        seeds = mapping.long() * 1009 + 373

        def draw(cache: DraftLogitsCache) -> tuple[torch.Tensor, torch.Tensor]:
            return rejection_sample(
                target_logits=logits.flatten(0, 1),
                draft_logits_cache=cache,
                draft_sampled=draft,
                cu_num_logits=torch.arange(batch + 1, device="cuda", dtype=torch.int32)
                * (steps + 1),
                pos=local.long() + (1 << 33),
                idx_mapping=mapping,
                expanded_idx_mapping=mapping.repeat_interleave(steps + 1),
                expanded_local_pos=local,
                temperature=temperature,
                seed=seeds,
                num_speculative_steps=steps,
            )

        probabilities = q.to(torch.bfloat16)
        expected, expected_lengths = draw(DraftLogitsCache(probabilities.log()))
        actual, lengths = draw(
            DraftLogitsCache(probabilities.flip(0), row_mapping, values_are_probs=True)
        )
        torch.testing.assert_close(lengths, expected_lengths, rtol=0, atol=0)
        valid = local.reshape(batch, steps + 1) < lengths[:, None]
        torch.testing.assert_close(actual[valid], expected[valid], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
