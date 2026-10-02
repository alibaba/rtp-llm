"""CPU protocol tests for the FastAFD rank scheduler and error paths."""

import queue
import threading
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.distributed import fast_afd


class _Mailbox:
    def __init__(self, buffered=False):
        self.messages = defaultdict(queue.Queue)
        self.local = threading.local()
        self.max_pending = 0
        self.max_pending_by_pair = defaultdict(int)
        self.pending_headers_by_pair = defaultdict(int)
        self.max_headers_by_pair = defaultdict(int)
        self.lock = threading.Lock()
        self.buffered = buffered

    def set_rank(self, rank):
        self.local.rank = rank

    def send(self, tensor, dst, group):
        self.assert_group(group)
        received = threading.Event()
        pair = (self.local.rank, dst)
        is_header = tensor.dtype == torch.int64 and tensor.numel() == 9
        mailbox = self.messages[pair]
        mailbox.put((tensor.clone(), received, is_header))
        with self.lock:
            self.max_pending = max(self.max_pending, mailbox.qsize())
            self.max_pending_by_pair[pair] = max(
                self.max_pending_by_pair[pair], mailbox.qsize()
            )
            if is_header:
                self.pending_headers_by_pair[pair] += 1
                self.max_headers_by_pair[pair] = max(
                    self.max_headers_by_pair[pair],
                    self.pending_headers_by_pair[pair],
                )
        if not self.buffered and not received.wait(timeout=5):
            raise TimeoutError("send had no matching recv")

    def recv(self, tensor, src, group):
        self.assert_group(group)
        pair = (src, self.local.rank)
        incoming, received, is_header = self.messages[pair].get(timeout=5)
        if is_header:
            with self.lock:
                self.pending_headers_by_pair[pair] -= 1
        if incoming.shape != tensor.shape or incoming.dtype != tensor.dtype:
            received.set()
            raise AssertionError(
                f"message mismatch: {incoming.shape}/{incoming.dtype} "
                f"vs {tensor.shape}/{tensor.dtype}"
            )
        tensor.copy_(incoming)
        received.set()
        return tensor

    @staticmethod
    def assert_group(group):
        if group != fast_afd.Group.DP_AND_TP:
            raise AssertionError(f"wrong process group: {group}")


class _FakeFusedMoe(torch.nn.Module):
    topk_ids_dtype = torch.int64
    includes_shared_expert = False
    expert_num = 4

    def __init__(self, offset=0, fail=False):
        super().__init__()
        self.offset = offset
        self.fail = fail
        self.batch_sizes = []

    def forward(self, hidden_states, topk_weights, topk_ids, activation):
        self.batch_sizes.append(hidden_states.shape[0])
        if self.fail:
            raise RuntimeError("test expert failure")
        assert activation == "SiGLU"
        assert topk_ids.dtype == torch.int64
        routed = ((topk_ids.float() + 1) * topk_weights).sum(dim=1, keepdim=True)
        return hidden_states + routed.to(hidden_states.dtype) + self.offset


def _client(service_rank=2):
    return fast_afd.FastAFDClient(
        service_rank,
        hidden_size=4,
        top_k=2,
        device="cpu",
        activation_dtype=torch.float16,
        expert_count=4,
    )


def _service(layers=None):
    return fast_afd.FastAFDExpertService(
        attention_ranks=[0, 1],
        fused_moe_by_layer=layers or {0: _FakeFusedMoe(), 1: _FakeFusedMoe(2)},
        hidden_size=4,
        top_k=2,
        device="cpu",
        activation_dtype=torch.float16,
        expert_count=4,
    )


def _input(tokens):
    hidden = torch.ones((tokens, 4), dtype=torch.float16)
    ids = torch.tensor([[0, 2]] * tokens, dtype=torch.int32).reshape(tokens, 2)
    weights = torch.tensor([[0.25, 0.75]] * tokens, dtype=torch.float32).reshape(
        tokens, 2
    )
    return hidden, ids, weights


class FastAFDProtocolTest(unittest.TestCase):
    def _run(self, attention_tasks, service, buffered=False):
        mailbox = _Mailbox(buffered=buffered)
        self.last_mailbox = mailbox
        outputs = {}
        errors = {}

        def run_rank(rank, task):
            mailbox.set_rank(rank)
            try:
                outputs[rank] = task()
            except Exception as exc:
                errors[rank] = exc

        with patch.object(fast_afd, "send", mailbox.send), patch.object(
            fast_afd, "recv", mailbox.recv
        ):
            threads = [
                threading.Thread(target=run_rank, args=(2, service.serve_until_done)),
                threading.Thread(target=run_rank, args=(0, attention_tasks[0])),
                threading.Thread(target=run_rank, args=(1, attention_tasks[1])),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
            self.assertFalse(
                any(thread.is_alive() for thread in threads), "protocol hung"
            )
        return outputs, errors

    def test_same_layer_requests_are_combined_and_split_by_rank(self):
        fused_moe = _FakeFusedMoe()

        def task(rank, tokens):
            def run():
                client = _client()
                client.begin_step()
                try:
                    hidden, ids, weights = _input(tokens)
                    if rank == 1:
                        hidden.fill_(7)
                        ids[:] = torch.tensor([1, 3], dtype=torch.int32)
                    return client.forward(0, hidden, ids, weights)
                finally:
                    client.finish()

            return run

        outputs, errors = self._run(
            {0: task(0, 2), 1: task(1, 3)},
            _service({0: fused_moe}),
        )
        self.assertEqual(errors, {})
        self.assertEqual(fused_moe.batch_sizes, [5])
        self.assertEqual(outputs[0].shape, (2, 4))
        self.assertEqual(outputs[1].shape, (3, 4))
        self.assertTrue(torch.all(outputs[0] == 3.5))
        self.assertTrue(torch.all(outputs[1] == 10.5))

    def test_different_layer_positions_do_not_wait_for_alignment(self):
        first = _FakeFusedMoe()
        second = _FakeFusedMoe(2)

        def task(layer_order):
            def run():
                client = _client()
                client.begin_step()
                try:
                    return [
                        client.forward(layer_idx, *_input(1))
                        for layer_idx in layer_order
                    ]
                finally:
                    client.finish()

            return run

        outputs, errors = self._run(
            {0: task([0, 1]), 1: task([1, 0])},
            _service({0: first, 1: second}),
        )
        self.assertEqual(errors, {})
        self.assertEqual(first.batch_sizes, [1, 1])
        self.assertEqual(second.batch_sizes, [1, 1])
        self.assertTrue(torch.all(outputs[0][0] == 3.5))
        self.assertTrue(torch.all(outputs[1][0] == 5.5))

    def test_large_round_falls_back_to_bounded_expert_batches(self):
        fused_moe = _FakeFusedMoe()

        def task():
            client = _client()
            client.begin_step()
            try:
                return client.forward(0, *_input(2))
            finally:
                client.finish()

        with patch.object(fast_afd, "_MAX_TOKENS", 3):
            outputs, errors = self._run({0: task, 1: task}, _service({0: fused_moe}))
        self.assertEqual(errors, {})
        self.assertEqual(fused_moe.batch_sizes, [2, 2])
        self.assertEqual(outputs[0].shape, (2, 4))
        self.assertEqual(outputs[1].shape, (2, 4))

    def test_failed_aggregate_reports_to_every_rank(self):
        fused_moe = _FakeFusedMoe(fail=True)

        def task():
            client = _client()
            client.begin_step()
            try:
                with self.assertRaisesRegex(RuntimeError, "EXPERT_FAILED"):
                    client.forward(0, *_input(1))
            finally:
                client.finish()

        _, errors = self._run({0: task, 1: task}, _service({0: fused_moe}))
        self.assertEqual(fused_moe.batch_sizes, [2])
        self.assertNotIn(0, errors)
        self.assertNotIn(1, errors)
        self.assertIn(2, errors)
        self.assertIn("ranks [0, 1] expert failed", str(errors[2]))

    def test_consecutive_idle_steps_then_work_remain_aligned(self):
        service = _service()
        three_steps = SimpleNamespace(
            serve_until_done=lambda: [service.serve_until_done() for _ in range(3)]
        )

        def task(sizes_by_step):
            client = _client()

            def run():
                outputs = []
                for sizes in sizes_by_step:
                    client.begin_step()
                    try:
                        for tokens in sizes:
                            outputs.append(client.forward(0, *_input(tokens)))
                    finally:
                        client.finish()
                return outputs

            return run

        outputs, errors = self._run(
            {0: task([[], [], [1]]), 1: task([[2], [1], []])},
            three_steps,
            buffered=True,
        )
        self.assertEqual(errors, {})
        self.assertEqual([out.shape[0] for out in outputs[0]], [1])
        self.assertEqual([out.shape[0] for out in outputs[1]], [2, 1])
        # Even when send can buffer, the finish ACK prevents a silent rank
        # from piling up later-step control messages.
        self.assertLessEqual(self.last_mailbox.max_headers_by_pair[(0, 2)], 1)
        self.assertTrue(
            all(mailbox.empty() for mailbox in self.last_mailbox.messages.values())
        )

    def test_global_idle_only_when_every_attention_rank_finishes_without_requests(self):
        service = _service()

        def serve_two_steps():
            idle_flags = []
            for _ in range(2):
                service.serve_until_done()
                idle_flags.append(service.global_idle)
            return idle_flags

        def task(rank):
            def run():
                client = _client()
                client.begin_step()
                client.finish()
                idle_flags = [client.global_idle]
                client.begin_step()
                self.assertFalse(client.global_idle)
                if rank == 0:
                    client.forward(0, *_input(1))
                client.finish()
                idle_flags.append(client.global_idle)
                return idle_flags

            return run

        outputs, errors = self._run(
            {0: task(0), 1: task(1)},
            SimpleNamespace(serve_until_done=serve_two_steps),
        )
        self.assertEqual(errors, {})
        self.assertEqual(outputs[0], [True, False])
        self.assertEqual(outputs[1], [True, False])
        self.assertEqual(outputs[2], [True, False])

    def test_serial_rank_shutdown_does_not_wait_for_other_rank(self):
        service = _service()
        first_rank_stopped = threading.Event()

        def serve_until_stopped():
            rounds = 0
            while not service.serve_until_done():
                rounds += 1
            return rounds

        def first_rank():
            client = _client()
            client.begin_step()
            client.finish()
            client.stop()
            first_rank_stopped.set()
            client.stop()  # The shutdown hook may be called more than once.
            with self.assertRaisesRegex(RuntimeError, "was stopped"):
                client.begin_step()

        def second_rank():
            client = _client()
            client.begin_step()
            client.finish()
            if not first_rank_stopped.wait(timeout=3):
                raise TimeoutError("first rank STOP waited for the other rank")
            client.begin_step()
            try:
                result = client.forward(0, *_input(1))
            finally:
                client.finish()
            client.stop()
            return result

        outputs, errors = self._run(
            {0: first_rank, 1: second_rank},
            SimpleNamespace(serve_until_done=serve_until_stopped),
        )
        self.assertEqual(errors, {})
        self.assertTrue(first_rank_stopped.is_set())
        self.assertEqual(outputs[2], 2)
        self.assertTrue(torch.all(outputs[1] == 3.5))
        self.assertEqual(service._live_attention_ranks, set())

    def test_stop_before_first_step_exits_service(self):
        service = _service()
        first_rank_stopped = threading.Event()

        def first_rank():
            _client().stop()
            first_rank_stopped.set()

        def second_rank():
            if not first_rank_stopped.wait(timeout=3):
                raise TimeoutError("first rank STOP was not acknowledged")
            _client().stop()

        outputs, errors = self._run({0: first_rank, 1: second_rank}, service)
        self.assertEqual(errors, {})
        self.assertTrue(outputs[2])
        self.assertTrue(service.serve_until_done())

    def test_failed_client_does_not_send_stop_to_exited_service(self):
        client = _client()
        client._failed = True
        with patch.object(fast_afd, "send") as send_message:
            client.stop()
        send_message.assert_not_called()

    def test_variable_microbatch_counts_empty_request_and_reuse(self):
        clients = [_client(), _client()]

        def task(rank, sizes):
            def run():
                client = clients[rank]
                client.begin_step()
                result = []
                try:
                    for tokens in sizes:
                        for layer_idx in (0, 1):
                            hidden, ids, weights = _input(tokens)
                            result.append(
                                client.forward(layer_idx, hidden, ids, weights)
                            )
                finally:
                    client.finish()
                return result

            return run

        service = _service()
        outputs, errors = self._run({0: task(0, [2, 1]), 1: task(1, [0])}, service)
        self.assertEqual(errors, {})
        self.assertEqual([item.shape[0] for item in outputs[0]], [2, 2, 1, 1])
        self.assertEqual([item.shape[0] for item in outputs[1]], [0, 0])
        self.assertTrue(torch.all(outputs[0][0] == 3.5))
        self.assertTrue(torch.all(outputs[0][1] == 5.5))

        # The same clients and expert service are reused for the next engine step.
        outputs, errors = self._run({0: task(0, [1]), 1: task(1, [2])}, service)
        self.assertEqual(errors, {})
        self.assertEqual(outputs[0][0].shape, (1, 4))
        self.assertEqual(outputs[1][0].shape, (2, 4))

    def test_invalid_local_shape_aborts_without_stranding_other_rank(self):
        def bad_rank():
            client = _client()
            client.begin_step()
            hidden, _, weights = _input(1)
            with self.assertRaises(ValueError):
                client.forward(
                    0, hidden, torch.empty((1, 1), dtype=torch.int32), weights
                )
            client.finish()

        def good_rank():
            client = _client()
            client.begin_step()
            try:
                client.forward(0, *_input(1))
            finally:
                client.finish()

        _, errors = self._run({0: bad_rank, 1: good_rank}, _service())
        self.assertIn(2, errors)
        self.assertIn("attention rank 0 aborted", str(errors[2]))
        self.assertNotIn(0, errors)
        self.assertIn("STEP_FAILED", str(errors[1]))

    def test_unknown_layer_rejected_and_drain_continues(self):
        def bad_rank():
            client = _client()
            client.begin_step()
            with self.assertRaisesRegex(RuntimeError, "BAD_LAYER"):
                client.forward(99, *_input(1))
            client.finish()

        def good_rank():
            client = _client()
            client.begin_step()
            try:
                client.forward(0, *_input(1))
            finally:
                client.finish()

        _, errors = self._run({0: bad_rank, 1: good_rank}, _service())
        self.assertIn(2, errors)
        self.assertIn("BAD_LAYER", str(errors[2]))
        self.assertNotIn(0, errors)
        self.assertIn("STEP_FAILED", str(errors[1]))

    def test_failed_step_stops_other_rank_before_second_step(self):
        for failure_kind in ("abort", "rejected_request"):
            with self.subTest(failure_kind=failure_kind):
                entered_second_step = threading.Event()

                def failing_rank():
                    client = _client()
                    client.begin_step()
                    try:
                        if failure_kind == "abort":
                            hidden, _, weights = _input(1)
                            client.forward(
                                0,
                                hidden,
                                torch.empty((1, 1), dtype=torch.int32),
                                weights,
                            )
                        else:
                            client.forward(99, *_input(1))
                    finally:
                        client.finish()

                def healthy_rank():
                    client = _client()
                    client.begin_step()
                    try:
                        client.forward(0, *_input(1))
                    finally:
                        client.finish()
                    entered_second_step.set()
                    client.begin_step()
                    try:
                        client.forward(0, *_input(1))
                    finally:
                        client.finish()

                _, errors = self._run({0: failing_rank, 1: healthy_rank}, _service())
                self.assertFalse(entered_second_step.is_set())
                self.assertIn("STEP_FAILED", str(errors[1]))
                self.assertIn(2, errors)
                if failure_kind == "abort":
                    self.assertIsInstance(errors[0], ValueError)
                    self.assertIn("topk_ids must have shape", str(errors[0]))
                else:
                    self.assertIn("BAD_LAYER", str(errors[0]))

    def test_expert_failure_is_reported_to_client(self):
        def failing_rank():
            client = _client()
            client.begin_step()
            try:
                with self.assertRaisesRegex(RuntimeError, "EXPERT_FAILED"):
                    client.forward(0, *_input(1))
            finally:
                client.finish()

        def other_rank():
            client = _client()
            client.begin_step()
            client.finish()

        _, errors = self._run(
            {0: failing_rank, 1: other_rank},
            _service({0: _FakeFusedMoe(fail=True)}),
        )
        self.assertIn(2, errors)
        self.assertIn("aborted", str(errors[2]))
        self.assertNotIn(0, errors)
        self.assertIn("STEP_FAILED", str(errors[1]))

    def test_unsupported_configuration_fails_before_communication(self):
        with self.assertRaisesRegex(ValueError, "float16 and bfloat16"):
            fast_afd.FastAFDClient(2, 4, 2, "cpu", torch.float32, 4)
        with self.assertRaisesRegex(ValueError, "contiguous attention ranks"):
            fast_afd.FastAFDExpertService(
                [1], {0: _FakeFusedMoe()}, 4, 2, "cpu", torch.float16, 4
            )


if __name__ == "__main__":
    unittest.main()
