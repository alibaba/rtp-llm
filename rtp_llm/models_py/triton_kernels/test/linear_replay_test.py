"""Whole-window correctness for both LINEAR replay recurrences."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensorMode

from rtp_llm.models_py.triton_kernels import linear_replay as replay_module
from rtp_llm.models_py.triton_kernels.linear_replay import (
    finalize_linear_replay,
    linear_serial_replay,
)


class LinearReplayInputContractTest(unittest.TestCase):
    def test_pitched_group_views_and_explicit_group_id_without_gpu(self):
        with FakeTensorMode():

            def empty(shape, dtype=torch.float32):
                return torch.empty(shape, device="cuda:0", dtype=dtype)

            batch, steps, heads, value_heads, dim = 2, 4, 2, 4, 128
            channels = (2 * heads + value_heads) * dim
            cache = SimpleNamespace(
                k=empty((8, steps, heads, dim)),
                u=empty((8, steps, value_heads, dim)),
                g=empty((8, steps, value_heads, 1)),
                conv_inputs=empty((8, steps, channels), torch.bfloat16),
                slot_generations=empty((8,), torch.int64),
                log_epochs=empty((8,), torch.int64),
                valid_counts=empty((8,), torch.int32),
                error_flags=empty((8,), torch.int32),
            )
            inputs = SimpleNamespace(
                slot_ids=empty((batch,), torch.int32),
                slot_generations=empty((batch,), torch.int64),
                prev_accept_lengths=empty((batch,), torch.int32),
                history_valid_lengths=empty((batch,), torch.int32),
                history_epochs=empty((batch,), torch.int64),
                verify_epochs=empty((batch,), torch.int64),
                init_kinds=empty((batch,), torch.int32),
                state_read_block_ids=empty((3, batch + 2), torch.int32)[:, :batch],
                active_block_ids=empty((3, batch + 2), torch.int32)[:, :batch],
            )
            packed = empty((batch * steps, channels + 3), torch.bfloat16)
            q = packed[:, : heads * dim]
            k = packed[:, heads * dim : 2 * heads * dim]
            v = packed[:, 2 * heads * dim : channels]
            args = (
                q,
                k,
                v,
                empty((batch * steps, value_heads), torch.bfloat16),
                empty((batch * steps, value_heads), torch.bfloat16),
                empty((channels, 4), torch.bfloat16),
                empty((value_heads,)),
                empty((value_heads,)),
                empty((16, value_heads, dim, dim)),
                empty((16, 3, channels), torch.bfloat16),
                cache,
                inputs,
            )
            with patch.object(
                replay_module, "_rebuild_state_kernel"
            ) as rebuild, patch.object(
                replay_module, "_replay_conv_kernel"
            ) as conv, patch.object(
                replay_module, "_serial_verify_kernel"
            ) as verify:
                output = linear_serial_replay(*args, group_id=2, vector_gate=False)
            self.assertEqual(output.shape, (batch * steps, value_heads, dim))
            rebuild.__getitem__.return_value.assert_called_once()
            conv.__getitem__.return_value.assert_called_once()
            verify.__getitem__.return_value.assert_called_once()
            with self.assertRaisesRegex(ValueError, "cache group"):
                linear_serial_replay(*args, group_id=3, vector_gate=False)
            with self.assertRaisesRegex(ValueError, "log capacity"):
                finalize_linear_replay(cache, inputs, steps + 1)


class LinearReplayTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        torch.manual_seed(1234)

    def _case(self, vector_gate, steps=4, width=4, key_dim=128, value_dim=128):
        batch, heads, value_heads = 5, 2, 2 if vector_gate else 4
        channels = 2 * heads * key_dim + value_heads * value_dim
        gate_dim = key_dim if vector_gate else 1
        slots, blocks = batch + 1, 3 * batch + 1

        def zeros(shape, dtype=torch.float32):
            return torch.zeros(shape, dtype=dtype, device="cuda")

        cache = SimpleNamespace(
            group_id=1,
            k=zeros((slots, steps, heads, key_dim)),
            u=zeros((slots, steps, value_heads, value_dim)),
            g=zeros((slots, steps, value_heads, gate_dim)),
            conv_inputs=zeros((slots, steps, channels), torch.bfloat16),
            slot_generations=zeros((slots,), torch.int64),
            log_epochs=zeros((slots,), torch.int64),
            valid_counts=zeros((slots,), torch.int32),
            error_flags=zeros((slots,), torch.int32),
        )
        inputs = SimpleNamespace(
            slot_ids=torch.tensor([4, 1, 3, 2, -1], device="cuda", dtype=torch.int32),
            slot_generations=torch.ones(batch, device="cuda", dtype=torch.int64),
            prev_accept_lengths=zeros((batch,), torch.int32),
            history_valid_lengths=zeros((batch,), torch.int32),
            history_epochs=zeros((batch,), torch.int64),
            verify_epochs=torch.ones(batch, device="cuda", dtype=torch.int64),
            init_kinds=torch.ones(batch, device="cuda", dtype=torch.int32),
            state_read_block_ids=zeros((2, batch + 3), torch.int32)[:, :batch],
            active_block_ids=zeros((2, batch + 3), torch.int32)[:, :batch],
        )
        inputs.state_read_block_ids[1] = torch.arange(1, batch + 1, device="cuda")
        inputs.active_block_ids[1] = torch.arange(
            batch + 1, 2 * batch + 1, device="cuda"
        )
        ssm = torch.randn(blocks, value_heads, key_dim, value_dim, device="cuda") * 0.1
        conv = torch.randn(
            blocks, width - 1, channels, device="cuda", dtype=torch.bfloat16
        )
        weights = (
            torch.randn(channels, width, device="cuda", dtype=torch.bfloat16) * 0.2
        )
        alog = torch.randn(value_heads, device="cuda") * 0.1
        bias = torch.randn(value_heads * gate_dim, device="cuda") * 0.1
        return (
            batch,
            heads,
            value_heads,
            channels,
            cache,
            inputs,
            ssm,
            conv,
            weights,
            alog,
            bias,
        )

    def _reference(
        self,
        raw,
        a,
        b,
        weights,
        alog,
        bias,
        state,
        history,
        heads,
        value_heads,
        key_dim,
        value_dim,
        vector_gate,
    ):
        batch, steps, _ = raw.shape
        outputs, states, histories = [], [], []
        for t in range(steps):
            window = torch.cat((history, raw[:, t : t + 1]), dim=1)
            products = window.float() * weights.T.float().unsqueeze(0)
            if vector_gate:
                convolved = products.sum(dim=1)
                convolved = convolved * torch.sigmoid(convolved)
            else:
                convolved = torch.zeros_like(products[:, 0])
                for tap in range(weights.shape[1]):
                    convolved = convolved + products[:, tap]
                convolved = convolved / (1 + torch.exp(-convolved))
            convolved = convolved.to(raw.dtype).float()
            q, k, v = torch.split(
                convolved,
                [heads * key_dim, heads * key_dim, value_heads * value_dim],
                dim=-1,
            )
            k_raw = k.reshape(batch, heads, key_dim)
            q_raw = convolved[:, : heads * key_dim].reshape(batch, heads, key_dim)
            q = q_raw / torch.sqrt((q_raw * q_raw).sum(-1, keepdim=True) + 1e-6)
            k = k_raw / torch.sqrt((k_raw * k_raw).sum(-1, keepdim=True) + 1e-6)
            q = q.repeat_interleave(value_heads // heads, dim=1) * key_dim**-0.5
            k = k.repeat_interleave(value_heads // heads, dim=1)
            v = v.reshape(batch, value_heads, value_dim)
            if vector_gate:
                x = a[:, t].reshape(batch, value_heads, key_dim).float() + bias.reshape(
                    value_heads, key_dim
                )
                gate = -20 * torch.sigmoid(torch.exp(alog)[None, :, None] * x)
                state = state * torch.exp(gate).unsqueeze(-1)
                beta = torch.sigmoid(b[:, t].float())
            else:
                x = a[:, t].float() + bias
                gate = -torch.exp(alog) * F.softplus(x)
                state = state * torch.exp(gate)[:, :, None, None]
                beta = torch.sigmoid(b[:, t].float()).to(b.dtype).float()
            u = (v - (state * k.unsqueeze(-1)).sum(-2)) * beta.unsqueeze(-1)
            state = state + k.unsqueeze(-1) * u.unsqueeze(-2)
            outputs.append((state * q.unsqueeze(-1)).sum(-2).to(raw.dtype))
            history = window[:, 1:]
            states.append(state.clone())
            histories.append(history.clone())
        return torch.stack(outputs, 1), states, histories

    def _run_rounds(
        self, vector_gate, steps, width, key_dim, value_dim, *, use_cuda_graph=False
    ):
        (
            batch,
            heads,
            value_heads,
            channels,
            cache,
            inputs,
            ssm,
            conv,
            weights,
            alog,
            bias,
        ) = self._case(vector_gate, steps, width, key_dim, value_dim)
        reference_state = ssm[inputs.state_read_block_ids[1].long()].clone()
        reference_conv = conv[inputs.state_read_block_ids[1].long()].clone()
        previous_states = previous_histories = None
        graph = graph_output = None
        graph_raw = graph_a = graph_b = None
        graph_addresses = None
        for round_id in range(3):
            if round_id == 2:
                order = torch.tensor([2, 0, 3, 1, 4], device="cuda")
                previous_states = [state[order] for state in previous_states]
                previous_histories = [history[order] for history in previous_histories]
                inputs.slot_ids.copy_(inputs.slot_ids[order])
                inputs.active_block_ids.copy_(inputs.active_block_ids[:, order])
            if round_id:
                accepted = [
                    min(i + 1, steps) if round_id == 1 else max(1, steps - i)
                    for i in range(batch)
                ]
                reference_state = torch.stack(
                    [previous_states[a - 1][i] for i, a in enumerate(accepted)]
                )
                reference_conv = torch.stack(
                    [previous_histories[a - 1][i] for i, a in enumerate(accepted)]
                )
                inputs.prev_accept_lengths.copy_(
                    torch.tensor(accepted, device="cuda", dtype=torch.int32)
                )
                inputs.history_valid_lengths.fill_(steps)
                inputs.history_epochs.fill_(round_id)
                inputs.verify_epochs.fill_(round_id + 1)
                inputs.init_kinds.zero_()
                inputs.state_read_block_ids.copy_(inputs.active_block_ids)
                if round_id == 1:
                    inputs.active_block_ids[1] += batch
            raw = torch.randn(
                batch, steps, channels + 7, dtype=torch.bfloat16, device="cuda"
            )[:, :, :channels]
            gate_dim = key_dim if vector_gate else 1
            a = torch.randn(
                batch,
                steps,
                value_heads * gate_dim,
                device="cuda",
                dtype=torch.bfloat16,
            )
            b = torch.randn(
                batch, steps, value_heads, device="cuda", dtype=torch.bfloat16
            )
            if use_cuda_graph:
                if graph_raw is None:
                    graph_raw, graph_a, graph_b = raw, a, b
                else:
                    graph_raw.copy_(raw)
                    graph_a.copy_(a)
                    graph_b.copy_(b)
                raw, a, b = graph_raw, graph_a, graph_b
                addresses = tuple(
                    tensor.data_ptr()
                    for tensor in (raw, a, b, ssm, conv, *vars(inputs).values())
                )
                if graph_addresses is None:
                    graph_addresses = addresses
                self.assertEqual(addresses, graph_addresses)
            # Retain strided projected views, as Qwen's qkvz projection does.
            q, k, v = torch.split(
                raw.reshape(batch * steps, channels),
                [heads * key_dim, heads * key_dim, value_heads * value_dim],
                dim=-1,
            )
            expected, previous_states, previous_histories = self._reference(
                raw,
                a,
                b,
                weights,
                alog,
                bias,
                reference_state,
                reference_conv,
                heads,
                value_heads,
                key_dim,
                value_dim,
                vector_gate,
            )

            def run_replay():
                output = linear_serial_replay(
                    q,
                    k,
                    v,
                    a.flatten(0, 1),
                    b.flatten(0, 1),
                    weights,
                    alog,
                    bias,
                    ssm,
                    conv,
                    cache,
                    inputs,
                    group_id=1,
                    vector_gate=vector_gate,
                    lower_bound=-20 if vector_gate else None,
                )
                finalize_linear_replay(cache, inputs, steps)
                return output

            if use_cuda_graph:
                if graph is None:
                    # Warmup/capture must not consume the first real window.
                    slots = inputs.slot_ids.clone()
                    inputs.slot_ids.fill_(-1)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        run_replay()
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        graph_output = run_replay()
                    torch.cuda.current_stream().wait_stream(stream)
                    inputs.slot_ids.copy_(slots)
                graph.replay()
                actual = graph_output
            else:
                actual = run_replay()
            actual = actual.reshape(batch, steps, value_heads, value_dim)
            torch.testing.assert_close(
                actual[:-1], expected[:-1], rtol=0.02, atol=0.005
            )
            torch.testing.assert_close(
                actual[-1], torch.zeros_like(actual[-1]), rtol=0, atol=0
            )
            destination = inputs.active_block_ids[1, :-1].long()
            torch.testing.assert_close(
                ssm[destination], reference_state[:-1], rtol=2e-4, atol=2e-5
            )
            torch.testing.assert_close(
                conv[destination], reference_conv[:-1], rtol=0, atol=0
            )
            self.assertEqual(cache.valid_counts[4].item(), steps)
            self.assertEqual(cache.log_epochs[4].item(), round_id + 1)

    def test_kda_all_accept_lengths_cross_page_and_in_place(self):
        self._run_rounds(True, 4, 4, 128, 128)

    def test_qwen_grouped_heads_and_scalar_gate(self):
        self._run_rounds(False, 4, 4, 128, 128)

    def test_kda_cuda_graph_multiround_accept_and_slot_reordering(self):
        self._run_rounds(True, 4, 4, 128, 128, use_cuda_graph=True)

    def test_qwen_cuda_graph_multiround_accept_and_slot_reordering(self):
        self._run_rounds(False, 4, 4, 128, 128, use_cuda_graph=True)

    def test_qwen_variable_verify_and_conv_width(self):
        for steps, width in ((1, 2), (2, 3), (6, 4)):
            with self.subTest(steps=steps, width=width):
                self._run_rounds(False, steps, width, 64, 96)

    def test_cold_empty_prefix_and_cuda_graph(self):
        (
            batch,
            heads,
            value_heads,
            channels,
            cache,
            inputs,
            ssm,
            conv,
            weights,
            alog,
            bias,
        ) = self._case(False, steps=2)
        inputs.init_kinds.fill_(2)
        inputs.state_read_block_ids.fill_(-1)
        qkv = torch.randn(batch * 2, channels, device="cuda", dtype=torch.bfloat16)
        q, k, v = torch.split(
            qkv, [heads * 128, heads * 128, value_heads * 128], dim=-1
        )
        gate = torch.randn(batch * 2, value_heads, device="cuda", dtype=torch.bfloat16)

        def run():
            return linear_serial_replay(
                q,
                k,
                v,
                gate,
                gate,
                weights,
                alog,
                bias,
                ssm,
                conv,
                cache,
                inputs,
                group_id=1,
                vector_gate=False,
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            expected = run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run()
        for _ in range(2):
            graph.replay()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            ssm[inputs.active_block_ids[1, :-1].long()],
            torch.zeros_like(ssm[1:batch]),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
