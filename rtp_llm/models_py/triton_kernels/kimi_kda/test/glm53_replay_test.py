"""Accepted-chain replay, page rollover and graph remapping against full tapes."""

import json
import os
import unittest
from pathlib import Path

import torch
from rtp_llm.models_py.triton_kernels.causal_conv1d import causal_conv1d_update
from rtp_llm.models_py.triton_kernels.kimi_kda import fused_recurrent_kda
from rtp_llm.models_py.triton_kernels.kimi_kda.glm53_replay import (
    KDAReplayWorkspace,
    commit_kda_replay,
)
from rtp_llm.models_py.triton_kernels.kimi_kda.glm53_short_conv import (
    glm53_kda_short_conv_decode,
    glm53_kda_short_conv_verify,
)


class Glm53ReplayTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(
            torch.cuda.is_available(), "GPU validation must not silently skip"
        )
        torch.manual_seed(9341)

    def _case(self, batch, tokens, layers=2):
        heads, dim, page = 4, 128, 128
        channels = heads * dim
        pages = batch * (tokens + 2) + 1
        lengths = torch.tensor(
            ([128, 129, 130, 131073] * batch)[:batch], device="cuda", dtype=torch.int32
        )
        table = torch.zeros(batch, 1035, device="cuda", dtype=torch.int32)
        for n in range(batch):
            start = (int(lengths[n]) - 2) // page
            ids = torch.arange(
                n * (tokens + 2) + 1,
                (n + 1) * (tokens + 2) + 1,
                device="cuda",
                dtype=torch.int32,
            )
            table[n, start : start + tokens + 2] = ids.flip(0)
        if batch > 1:
            table[-1].zero_()  # graph padding; never touch physical page zero
        x = (
            torch.randn(
                batch * tokens, 3 * channels, device="cuda", dtype=torch.bfloat16
            )
            * 0.2
        )
        gate = (
            torch.randn(batch * tokens, channels, device="cuda", dtype=torch.bfloat16)
            * 0.2
        )
        beta = torch.randn(batch * tokens, heads, device="cuda", dtype=torch.bfloat16)
        conv_weight = (
            torch.randn(3 * channels, 4, device="cuda", dtype=torch.float32) * 0.2
        )
        alog = torch.randn(heads, device="cuda", dtype=torch.float32) * 0.1
        bias = torch.randn(channels, device="cuda", dtype=torch.float32) * 0.1
        # Pad both page pitches; pointer descriptors must honor physical strides.
        state_seed = torch.randn(pages, heads * dim * dim + 32, device="cuda") * 0.02
        conv_seed = (
            torch.randn(pages, 9 * channels + 32, device="cuda", dtype=torch.bfloat16)
            * 0.2
        )
        old_states, old_convs, workspaces, outputs = [], [], [], []
        for _ in range(layers):
            old_state = state_seed.clone()[:, : heads * dim * dim].view(
                pages, heads, dim, dim
            )
            old_conv = conv_seed.clone()[:, : 9 * channels].view(pages, 3, 3 * channels)
            new_state = state_seed.clone()[:, : heads * dim * dim].view(
                pages, heads, dim, dim
            )
            new_conv = conv_seed.clone()[:, : 9 * channels].view(pages, 3, 3 * channels)
            workspace = KDAReplayWorkspace(
                batch, heads, dim, x.device, new_conv, new_state
            )
            old_qkv = (
                causal_conv1d_update(
                    x.view(batch, tokens, -1).transpose(1, 2),
                    old_conv.transpose(1, 2),
                    conv_weight,
                    activation="silu",
                    block_map=table,
                    seq_size_per_block=page,
                    sequence_lengths=lengths,
                )
                .transpose(1, 2)
                .reshape(batch, tokens, 3, heads, dim)
            )
            fused_qkv = glm53_kda_short_conv_verify(
                x, conv_weight, new_conv, table, lengths, page, batch, tokens, workspace
            )
            for part, expected in zip(fused_qkv, old_qkv.unbind(2)):
                live = batch - 1 if batch > 1 else batch
                # The legacy convolution reads/writes page zero for graph
                # padding. Replay suppresses those cache writes; recurrent
                # output masking remains identical for the padded request.
                torch.testing.assert_close(
                    part.view_as(expected)[:live], expected[:live], rtol=0, atol=0
                )
            args = dict(
                g=gate.view(batch, tokens, heads, dim),
                beta=beta.view(batch, tokens, heads),
                A_log=alog,
                dt_bias=bias,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                use_qk_l2norm_in_kernel=True,
                lower_bound=-5.0,
                block_map=table,
                sequence_lengths=lengths,
                seq_size_per_block=page,
            )
            old_output, _ = fused_recurrent_kda(
                **args,
                q=old_qkv[:, :, 0].contiguous(),
                k=old_qkv[:, :, 1].contiguous(),
                v=old_qkv[:, :, 2].contiguous(),
                initial_state=old_state,
            )
            new_output, _ = fused_recurrent_kda(
                **args,
                q=fused_qkv[0].view(batch, tokens, heads, dim),
                k=fused_qkv[1].view(batch, tokens, heads, dim),
                v=fused_qkv[2].view(batch, tokens, heads, dim),
                initial_state=new_state,
                store_states=False,
            )
            torch.testing.assert_close(new_output, old_output, rtol=0, atol=0)
            torch.testing.assert_close(
                new_state,
                state_seed[:, : heads * dim * dim].view_as(new_state),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                new_conv, conv_seed[:, : 9 * channels].view_as(new_conv), rtol=0, atol=0
            )
            workspace.capture_gates(gate, beta, batch, tokens)
            workspaces.append(workspace)
            old_states.append(old_state)
            old_convs.append(old_conv)
            outputs.append(new_output)
        descriptors = torch.tensor(
            [w.descriptor() + [alog.data_ptr(), bias.data_ptr()] for w in workspaces],
            device="cuda",
            dtype=torch.uint64,
        )
        return locals()

    def _check_commit(self, case, accepted):
        c = case
        for w in c["workspaces"]:
            w.state.copy_(
                c["state_seed"][:, : c["heads"] * c["dim"] ** 2].view_as(w.state)
            )
            w.conv.copy_(c["conv_seed"][:, : 9 * c["channels"]].view_as(w.conv))
        commit_kda_replay(c["descriptors"], c["workspaces"], accepted, c["page"], -5.0)
        for layer, w in enumerate(c["workspaces"]):
            for n in range(c["batch"]):
                steps = int(accepted[n])
                if steps == 0 or (c["batch"] > 1 and n == c["batch"] - 1):
                    continue
                seq = int(c["lengths"][n])
                for t in range(steps):
                    if (seq + t) % c["page"] != 0 and t != steps - 1:
                        continue
                    src = int(c["table"][n, (seq - 1) // c["page"] + t])
                    dst = int(c["table"][n, (seq + t - 1) // c["page"]])
                    torch.testing.assert_close(
                        w.state[dst], c["old_states"][layer][src], rtol=3e-5, atol=3e-6
                    )
                    torch.testing.assert_close(
                        w.conv[dst], c["old_convs"][layer][src], rtol=0, atol=0
                    )
            torch.testing.assert_close(
                w.state[0],
                c["state_seed"][0, : c["heads"] * c["dim"] ** 2].view_as(w.state[0]),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                w.conv[0],
                c["conv_seed"][0, : 9 * c["channels"]].view_as(w.conv[0]),
                rtol=0,
                atol=0,
            )

    def test_verify_and_batched_replay_all_accept_lengths(self):
        for batch in (1, 2, 4, 7):
            for tokens in (1, 2, 4, 8):
                with self.subTest(batch=batch, tokens=tokens):
                    case = self._case(batch, tokens)
                    for length in range(tokens + 1):
                        accepted = torch.tensor(
                            [(length + n) % (tokens + 1) for n in range(batch)],
                            device="cuda",
                            dtype=torch.int32,
                        )
                        self._check_commit(case, accepted)

    def test_graph_commit_changes_acceptance_without_recapture(self):
        case = self._case(4, 4)
        accepted = torch.ones(4, device="cuda", dtype=torch.int32)
        for _ in range(3):
            self._check_commit(case, accepted)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            commit_kda_replay(
                case["descriptors"], case["workspaces"], accepted, 128, -5.0
            )
        for lengths in ([4, 2, 1, 0], [1, 4, 3, 0]):
            accepted.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
            for w in case["workspaces"]:
                w.state.copy_(
                    case["state_seed"][:, : case["heads"] * case["dim"] ** 2].view_as(
                        w.state
                    )
                )
                w.conv.copy_(
                    case["conv_seed"][:, : 9 * case["channels"]].view_as(w.conv)
                )
            graph.replay()
            actual = [(w.state.clone(), w.conv.clone()) for w in case["workspaces"]]
            self._check_commit(case, accepted)
            for w, (state, conv) in zip(case["workspaces"], actual):
                torch.testing.assert_close(w.state, state, rtol=0, atol=0)
                torch.testing.assert_close(w.conv, conv, rtol=0, atol=0)

    def test_verify_graph_refreshes_payload_pages_and_gates(self):
        c = self._case(4, 4, layers=1)
        w = c["workspaces"][0]

        def verify():
            qkv = glm53_kda_short_conv_verify(
                c["x"],
                c["conv_weight"],
                w.conv,
                c["table"],
                c["lengths"],
                128,
                4,
                4,
                w,
            )
            w.capture_gates(c["gate"], c["beta"], 4, 4)
            return fused_recurrent_kda(
                q=qkv[0].view(4, 4, 4, 128),
                k=qkv[1].view(4, 4, 4, 128),
                v=qkv[2].view(4, 4, 4, 128),
                g=c["gate"].view(4, 4, 4, 128),
                beta=c["beta"].view(4, 4, 4),
                initial_state=w.state,
                A_log=c["alog"],
                dt_bias=c["bias"],
                lower_bound=-5.0,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                use_qk_l2norm_in_kernel=True,
                block_map=c["table"],
                sequence_lengths=c["lengths"],
                seq_size_per_block=128,
                store_states=False,
            )[0]

        for _ in range(3):
            verify()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = verify()
        c["table"][:3].copy_(c["table"][:3].flip(0))
        c["lengths"][:3].copy_(c["lengths"][:3].flip(0))
        c["x"].mul_(0.7)
        c["gate"].add_(0.3)
        graph.replay()
        captured_payload = w.qkv.clone()
        expected = verify()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            w.qkv[:, :4, :4], captured_payload[:, :4, :4], rtol=0, atol=0
        )

    def test_merged_input_projection_preserves_all_four_slices(self):
        # The beta and low-rank slices need not have aligned widths. Padding
        # belongs only after the final slice; the TP-local head order is fixed.
        width, heads = 4096, 16
        weights = [
            torch.randn(n, width, device="cuda", dtype=torch.bfloat16) * 0.01
            for n in (3 * heads * 128, heads, 128, 128)
        ]
        packed = torch.cat(weights).t().contiguous()
        for rows in (1, 2, 4, 16, 64):
            x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
            outputs = (x @ packed).split([w.shape[0] for w in weights], -1)
            for actual, weight in zip(outputs, weights):
                torch.testing.assert_close(
                    actual,
                    torch.nn.functional.linear(x, weight),
                    rtol=1 / 128,
                    atol=2e-3,
                )

    def test_decode_accepts_packed_projection_row_stride(self):
        channels, batch = 3 * 4 * 128, 4
        packed = torch.randn(batch, channels + 256, device="cuda", dtype=torch.bfloat16)
        x = packed[:, :channels]
        self.assertFalse(x.is_contiguous())
        weight = torch.randn(channels, 4, device="cuda", dtype=torch.float32) * 0.1
        seed = torch.randn(9, 3, channels, device="cuda", dtype=torch.bfloat16)
        old, new = seed.clone(), seed.clone()
        table = torch.tensor(
            [[1, 2], [3, 4], [5, 6], [7, 8]], device="cuda", dtype=torch.int32
        )
        lengths = torch.tensor([128, 129, 128, 129], device="cuda", dtype=torch.int32)
        expected = causal_conv1d_update(
            x.unsqueeze(-1),
            old.transpose(1, 2),
            weight,
            activation="silu",
            block_map=table,
            sequence_lengths=lengths,
            seq_size_per_block=128,
        ).squeeze(-1)
        actual = torch.cat(
            glm53_kda_short_conv_decode(x, weight, new, table, lengths, 128), -1
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(new, old, rtol=0, atol=0)

    def test_workspace_and_commit_timing(self):
        case = self._case(4, 4)
        accepted = torch.full((4,), 4, device="cuda", dtype=torch.int32)

        def run():
            commit_kda_replay(
                case["descriptors"], case["workspaces"], accepted, 128, -5.0
            )

        for _ in range(10):
            run()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        for _ in range(100):
            run()
        end.record()
        end.synchronize()
        payload = sum(w.nbytes for w in case["workspaces"])
        full_tape = (
            2 * 4 * 4 * (case["heads"] * 128 * 128 * 4 + 9 * case["channels"] * 2)
        )
        self.assertLess(payload, full_tape)
        result = dict(
            batch=4,
            verify_width=4,
            layers=2,
            replay_workspace_bytes=payload,
            full_candidate_tape_bytes=full_tape,
            commit_us=start.elapsed_time(end) * 10,
        )
        print("KDA_REPLAY_RESULT=" + json.dumps(result), flush=True)
        output = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if output:
            Path(output, "kda_replay_result.json").write_text(
                json.dumps(result, indent=2)
            )

    def test_repeated_rounds_preserve_committed_state(self):
        c = self._case(4, 4, layers=1)
        w = c["workspaces"][0]
        old_state, old_conv = c["old_states"][0], c["old_convs"][0]
        old_state.copy_(c["state_seed"][:, : 4 * 128 * 128].view_as(old_state))
        old_conv.copy_(c["conv_seed"][:, : 9 * c["channels"]].view_as(old_conv))
        for iteration in range(64):
            old_qkv = (
                causal_conv1d_update(
                    c["x"].view(4, 4, -1).transpose(1, 2),
                    old_conv.transpose(1, 2),
                    c["conv_weight"],
                    activation="silu",
                    block_map=c["table"],
                    sequence_lengths=c["lengths"],
                    seq_size_per_block=128,
                )
                .transpose(1, 2)
                .reshape(4, 4, 3, 4, 128)
            )
            new_qkv = glm53_kda_short_conv_verify(
                c["x"],
                c["conv_weight"],
                w.conv,
                c["table"],
                c["lengths"],
                128,
                4,
                4,
                w,
            )
            args = dict(
                g=c["gate"].view(4, 4, 4, 128),
                beta=c["beta"].view(4, 4, 4),
                A_log=c["alog"],
                dt_bias=c["bias"],
                lower_bound=-5.0,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                use_qk_l2norm_in_kernel=True,
                block_map=c["table"],
                sequence_lengths=c["lengths"],
                seq_size_per_block=128,
            )
            reference, _ = fused_recurrent_kda(
                **args,
                q=old_qkv[:, :, 0].contiguous(),
                k=old_qkv[:, :, 1].contiguous(),
                v=old_qkv[:, :, 2].contiguous(),
                initial_state=old_state,
            )
            output, _ = fused_recurrent_kda(
                **args,
                q=new_qkv[0].view(4, 4, 4, 128),
                k=new_qkv[1].view(4, 4, 4, 128),
                v=new_qkv[2].view(4, 4, 4, 128),
                initial_state=w.state,
                store_states=False,
            )
            torch.testing.assert_close(output, reference, rtol=1 / 128, atol=1e-5)
            w.capture_gates(c["gate"], c["beta"], 4, 4)
            accepted = torch.tensor(
                [1 + (iteration + n) % 4 for n in range(4)],
                device="cuda",
                dtype=torch.int32,
            )
            commit_kda_replay(c["descriptors"], [w], accepted, 128, -5.0)
            for n in range(3):
                seq, steps = int(c["lengths"][n]), int(accepted[n])
                for t in range(steps):
                    if (seq + t) % 128 == 0 or t == steps - 1:
                        src = int(c["table"][n, (seq - 1) // 128 + t])
                        dst = int(c["table"][n, (seq + t - 1) // 128])
                        torch.testing.assert_close(
                            w.state[dst], old_state[src], rtol=3e-5, atol=3e-6
                        )
                        torch.testing.assert_close(
                            w.conv[dst], old_conv[src], rtol=0, atol=0
                        )
                        old_state[dst].copy_(old_state[src].clone())
                        old_conv[dst].copy_(old_conv[src].clone())
            c["lengths"].add_(accepted)


if __name__ == "__main__":
    unittest.main()
