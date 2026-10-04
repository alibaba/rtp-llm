"""CPU protocol tests for the FastAFD rank scheduler and error paths."""

import queue
import threading
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.models_py.distributed import fast_afd


class _TestControlTransport:
    def send(self, tensor, dst):
        raise AssertionError("control transport was not connected to a mailbox")

    def recv(self, tensor, src):
        raise AssertionError("control transport was not connected to a mailbox")


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
        self.control_messages = 0
        self.data_messages = 0

    def set_rank(self, rank):
        self.local.rank = rank

    def send(self, tensor, dst, group):
        self.assert_group(group)
        if tensor.ndim != 2:
            raise AssertionError("control message sent through the data group")
        self._send(tensor, dst, "data")

    def send_control(self, tensor, dst):
        self.assert_control_tensor(tensor)
        self._send(tensor, dst, "control")

    def _send(self, tensor, dst, channel):
        received = threading.Event()
        pair = (self.local.rank, dst)
        is_header = channel == "control" and tensor.numel() == 9
        mailbox = self.messages[(channel, *pair)]
        mailbox.put((tensor.clone(), received, is_header))
        with self.lock:
            if channel == "control":
                self.control_messages += 1
            else:
                self.data_messages += 1
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
        if tensor.ndim != 2:
            raise AssertionError("control message received through the data group")
        return self._recv(tensor, src, "data")

    def recv_control(self, tensor, src):
        self.assert_control_tensor(tensor)
        return self._recv(tensor, src, "control")

    def _recv(self, tensor, src, channel):
        pair = (src, self.local.rank)
        incoming, received, is_header = self.messages[(channel, *pair)].get(timeout=5)
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

    @staticmethod
    def assert_control_tensor(tensor):
        if (
            tensor.device.type != "cpu"
            or tensor.dtype != torch.int64
            or tensor.ndim != 1
            or tensor.numel() not in (1, 9)
        ):
            raise AssertionError(
                "control messages must be fixed-size CPU int64 tensors"
            )


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
        control_transport=_TestControlTransport(),
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
        control_transport=_TestControlTransport(),
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

        with (
            patch.object(fast_afd, "send", mailbox.send),
            patch.object(fast_afd, "recv", mailbox.recv),
            patch.object(_TestControlTransport, "send", mailbox.send_control),
            patch.object(_TestControlTransport, "recv", mailbox.recv_control),
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

    def test_control_messages_use_cpu_transport_separate_from_payloads(self):
        def active_rank():
            client = _client()
            client.begin_step()
            result = client.forward(0, *_input(1))
            client.finish()
            return result

        def idle_rank():
            client = _client()
            client.begin_step()
            client.finish()

        outputs, errors = self._run({0: active_rank, 1: idle_rank}, _service())
        self.assertEqual(errors, {})
        self.assertTrue(torch.all(outputs[0] == 3.5))
        # REQUEST + two FINISH headers, two request ACKs + two FINISH ACKs.
        self.assertEqual(self.last_mailbox.control_messages, 7)
        # Only hidden, ids, weights and output travel through the data group.
        self.assertEqual(self.last_mailbox.data_messages, 4)

    def test_default_control_transport_uses_dedicated_process_group(self):
        process_group = object()
        with (
            patch.object(
                fast_afd, "get_fast_afd_control_group", return_value=process_group
            ) as get_group,
            patch.object(torch.distributed, "is_initialized", return_value=False),
        ):
            # Explicit CUDA placement needs no GPU allocation at construction;
            # control buffers must still be allocated on CPU.
            client = fast_afd.FastAFDClient(2, 4, 2, "cuda:0", torch.float16, 4)
            service = fast_afd.FastAFDExpertService(
                [0, 1], {0: _FakeFusedMoe()}, 4, 2, "cuda:0", torch.float16, 4
            )
        self.assertEqual(get_group.call_count, 2)
        self.assertIs(client._control_transport.group, process_group)
        self.assertIs(service._control_transport.group, process_group)

        def receive_status(tensor, src, group):
            self.assertEqual(src, 2)
            self.assertIs(group, process_group)
            _Mailbox.assert_control_tensor(tensor)
            tensor.fill_(fast_afd._Status.OK)

        with (
            patch.object(torch.distributed, "send") as control_send,
            patch.object(torch.distributed, "recv", side_effect=receive_status),
            patch.object(fast_afd, "send") as data_send,
            patch.object(fast_afd, "recv") as data_recv,
        ):
            client.begin_step()
            client.finish()
        header = control_send.call_args.args[0]
        _Mailbox.assert_control_tensor(header)
        self.assertEqual(header[1].item(), fast_afd._Kind.FINISH)
        self.assertEqual(control_send.call_args.args[1], 2)
        self.assertIs(control_send.call_args.kwargs["group"], process_group)
        data_send.assert_not_called()
        data_recv.assert_not_called()

        def receive_header(tensor, src, group):
            self.assertIn(src, (0, 1))
            self.assertIs(group, process_group)
            _Mailbox.assert_control_tensor(tensor)
            tensor.copy_(fast_afd._header(fast_afd._Kind.FINISH, 0))

        with (
            patch.object(torch.distributed, "send") as control_send,
            patch.object(torch.distributed, "recv", side_effect=receive_header),
            patch.object(fast_afd, "send") as data_send,
            patch.object(fast_afd, "recv") as data_recv,
        ):
            self.assertFalse(service.serve_until_done())
        self.assertTrue(service.global_idle)
        self.assertEqual(control_send.call_count, 2)
        for call in control_send.call_args_list:
            _Mailbox.assert_control_tensor(call.args[0])
            self.assertEqual(call.args[0].item(), fast_afd._Status.ALL_IDLE)
            self.assertIs(call.kwargs["group"], process_group)
        data_send.assert_not_called()
        data_recv.assert_not_called()

    def test_injected_control_transport_does_not_initialize_a_group(self):
        transport = MagicMock()
        with patch.object(fast_afd, "get_fast_afd_control_group") as get_group:
            client = fast_afd.FastAFDClient(
                2, 4, 2, "cpu", torch.float16, 4, control_transport=transport
            )
            service = fast_afd.FastAFDExpertService(
                [0, 1],
                {0: _FakeFusedMoe()},
                4,
                2,
                "cpu",
                torch.float16,
                4,
                control_transport=transport,
            )
        get_group.assert_not_called()
        self.assertIs(client._control_transport, transport)
        self.assertIs(service._control_transport, transport)

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
        with patch.object(client._control_transport, "send") as send_message:
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

    def test_multi_expert_topology_rejects_gaps_duplicates_and_wrong_leader(self):
        for expert_ranks in ([], [2, 4], [2, 2], [3, 4], [3, 2]):
            with (
                self.subTest(expert_ranks=expert_ranks),
                patch.object(fast_afd, "_ExpertTeam") as team,
            ):
                with self.assertRaisesRegex(ValueError, "contiguous expert ranks"):
                    fast_afd.FastAFDClient(
                        2,
                        4,
                        2,
                        "cpu",
                        torch.float16,
                        4,
                        control_transport=_TestControlTransport(),
                        expert_ranks=expert_ranks,
                    )
                with self.assertRaisesRegex(ValueError, "contiguous expert ranks"):
                    fast_afd.FastAFDExpertService(
                        [0, 1],
                        {0: _FakeFusedMoe()},
                        4,
                        2,
                        "cpu",
                        torch.float16,
                        4,
                        control_transport=_TestControlTransport(),
                        expert_ranks=expert_ranks,
                    )
                team.assert_not_called()

    def test_multi_expert_topology_matches_world_and_endpoint_roles(self):
        for rank in range(4):
            with (
                self.subTest(rank=rank),
                patch.object(torch.distributed, "is_initialized", return_value=True),
                patch.object(torch.distributed, "get_world_size", return_value=4),
                patch.object(torch.distributed, "get_rank", return_value=rank),
                patch.object(fast_afd, "_ExpertTeam") as team,
            ):
                kwargs = {
                    "hidden_size": 4,
                    "top_k": 2,
                    "device": "cpu",
                    "activation_dtype": torch.float16,
                    "expert_count": 4,
                    "control_transport": _TestControlTransport(),
                    "expert_ranks": [2, 3],
                }
                if rank < 2:
                    endpoint = fast_afd.FastAFDClient(2, **kwargs)
                    with self.assertRaisesRegex(ValueError, "on an expert rank"):
                        fast_afd.FastAFDExpertService(
                            [0, 1], {0: _FakeFusedMoe()}, **kwargs
                        )
                else:
                    endpoint = fast_afd.FastAFDExpertService(
                        [0, 1], {0: _FakeFusedMoe()}, **kwargs
                    )
                    with self.assertRaisesRegex(ValueError, "on an attention rank"):
                        fast_afd.FastAFDClient(2, **kwargs)
                self.assertEqual(endpoint.expert_ranks, (2, 3))
                # Even AG ranks must participate in ordered process-group
                # creation, although only EG ranks execute its collectives.
                team.assert_called_once_with((2, 3), torch.device("cpu"))

        with (
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=5),
            patch.object(fast_afd, "_ExpertTeam") as team,
            self.assertRaisesRegex(ValueError, "final world rank"),
        ):
            fast_afd.FastAFDClient(
                2,
                4,
                2,
                "cpu",
                torch.float16,
                4,
                control_transport=_TestControlTransport(),
                expert_ranks=[2, 3],
            )
        team.assert_not_called()

    def test_expert_shards_must_partition_all_experts_evenly(self):
        with (
            patch.object(fast_afd, "_ExpertTeam") as team,
            self.assertRaisesRegex(ValueError, "divide evenly"),
        ):
            fast_afd.FastAFDExpertService(
                [0, 1],
                {0: _FakeFusedMoe()},
                4,
                2,
                "cpu",
                torch.float16,
                4,
                control_transport=_TestControlTransport(),
                expert_ranks=[2, 3, 4],
            )
        team.assert_not_called()


if __name__ == "__main__":
    unittest.main()
