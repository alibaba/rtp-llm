"""Accuracy guard: partial-chunk padding must not be assumed equivalent.

This is a negative control for a discarded optimization, never a service path.
The original failed probe and its numerical differences are preserved in the
task artifacts. Whole dummy requests keep real GEMM sizes unchanged and are
validated independently below.
"""

import itertools
import unittest

import torch
from rtp_llm.models_py.triton_kernels.kimi_kda.chunk import chunk_kda


class PaddingPrecisionTest(unittest.TestCase):
    def test_partial_padding_changes_native_rounding(self):
        self.assertTrue(torch.cuda.is_available())
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(531014)
        mismatched = []
        for lengths in (
            (1, 17, 31, 63, 65, 80, 129, 257, 419),
            (418,) * 7 + (419,) * 32,
        ):
            heads = 16
            shape = (1, sum(lengths), heads, 128)
            inputs = [
                torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1
                for _ in range(4)
            ]
            inputs[3].sub_(4)
            beta = torch.rand(shape[:3], device="cuda")
            state = torch.randn(len(lengths), heads, 128, 128, device="cuda") * 0.1
            a = torch.randn(heads, device="cuda") * 0.1
            bias = torch.randn(heads * 128, device="cuda") * 0.1
            original = (0, *itertools.accumulate(lengths))
            padded_lengths = tuple((n + 63) // 64 * 64 for n in lengths)
            padded = (0, *itertools.accumulate(padded_lengths))
            values = [t.new_zeros((1, padded[-1], heads, 128)) for t in inputs]
            values[3].fill_(-float("inf"))
            pb = beta.new_zeros((1, padded[-1], heads))
            for i, n in enumerate(lengths):
                for dst, src in zip(values, inputs):
                    dst[:, padded[i] : padded[i] + n].copy_(
                        src[:, original[i] : original[i + 1]]
                    )
                pb[:, padded[i] : padded[i] + n].copy_(
                    beta[:, original[i] : original[i + 1]]
                )

            def run(data, b, cu):
                return chunk_kda(
                    *data,
                    b,
                    initial_state=state,
                    cu_seqlens=torch.tensor(cu, dtype=torch.int32, device="cuda"),
                    A_log=a,
                    dt_bias=bias,
                    use_qk_l2norm_in_kernel=True,
                    use_gate_in_kernel=True,
                    safe_gate=True,
                    lower_bound=-5.0,
                    output_final_state=True,
                    return_intermediate_states=True,
                    fuse_state_recurrence=False,
                    chunk_size=64
                )

            old = run(inputs, beta, original)
            new = run(values, pb, padded)
            o = torch.cat(
                [new[0][:, padded[i] : padded[i] + n] for i, n in enumerate(lengths)],
                dim=1,
            )
            for name, x, y in (
                ("output", o, old[0]),
                ("final", new[1], old[1]),
                ("checkpoints", new[2], old[2]),
            ):
                print(
                    name,
                    "equal",
                    bool(torch.equal(x, y)),
                    "max_abs",
                    float((x - y).abs().max()),
                )
                mismatched.append(not torch.equal(x, y))
        self.assertTrue(
            any(mismatched),
            "Do not treat padded partial GEMMs as exact reference arithmetic",
        )

    def test_whole_request_base_padding_is_exact(self):
        from rtp_llm.models_py.triton_kernels.kimi_kda.full_reuse_graph import (
            _pack_inputs,
            _reference,
        )

        self.assertTrue(torch.cuda.is_available())
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(531015)
        lengths = tuple(range(80, 420, 17)) * 3
        heads = 16
        real_cu = (0, *itertools.accumulate(lengths))
        shape = (1, sum(lengths), heads, 128)
        data = [
            torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1
            for _ in range(4)
        ]
        data[3].sub_(4)
        beta = torch.rand(shape[:3], device="cuda")
        state = torch.randn(len(lengths), heads, 128, 128, device="cuda") * 0.1
        a, bias = (
            torch.randn(heads, device="cuda") * 0.1,
            torch.randn(heads * 128, device="cuda") * 0.1,
        )
        cu = torch.tensor(real_cu, dtype=torch.int32, device="cuda")
        expected = chunk_kda(
            *data,
            beta,
            initial_state=state,
            cu_seqlens=cu,
            A_log=a,
            dt_bias=bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=-5.0,
            output_final_state=True,
            return_intermediate_states=True,
            fuse_state_recurrence=False,
            chunk_size=64
        )
        packed, ids, selected = [], [], []
        for length in sorted(set(lengths) | {461, 499}):
            sequences = [i for i, n in enumerate(lengths) if n == length][:2]
            packed.extend([length] * 2)
            ids.extend(sequences + [-1] * (2 - len(sequences)))
            selected.extend(sequences)
        pc = (0, *itertools.accumulate(packed))
        pcu = torch.tensor(pc, dtype=torch.int32, device="cuda")
        inverse = torch.tensor(ids, dtype=torch.int64, device="cuda")
        ps = (1, pc[-1], heads, 128)
        buffers = tuple(t.new_empty(ps) for t in data) + (
            beta.new_empty(ps[:3]),
            state.new_empty((len(packed), heads, 128, 128)),
            torch.empty_like(a),
            torch.empty_like(bias),
        )
        _pack_inputs[
            (
                len(packed),
                (max(max(packed) * heads * 128, heads * 128 * 128) + 1023) // 1024,
            )
        ](
            *data,
            beta,
            state,
            a,
            bias,
            cu,
            inverse,
            *buffers,
            pcu,
            heads * 128,
            heads,
            heads * 128 * 128,
            1024
        )
        indices = torch.tensor(
            [(i, j) for i, n in enumerate(packed) for j in range((n + 63) // 64)],
            dtype=torch.int32,
            device="cuda",
        )
        actual = _reference(buffers, pcu, indices, sum(lengths) * heads)
        real_chunks = (0, *itertools.accumulate((n + 63) // 64 for n in lengths))
        packed_chunks = (0, *itertools.accumulate((n + 63) // 64 for n in packed))
        for slot, seq in enumerate(ids):
            if seq < 0:
                continue
            n = lengths[seq]
            torch.testing.assert_close(
                actual[0][:, pc[slot] : pc[slot] + n],
                expected[0][:, real_cu[seq] : real_cu[seq + 1]],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                actual[1][slot], expected[1][seq], rtol=0, atol=0
            )
            torch.testing.assert_close(
                actual[2][:, packed_chunks[slot] : packed_chunks[slot + 1]],
                expected[2][:, real_chunks[seq] : real_chunks[seq + 1]],
                rtol=0,
                atol=0,
            )
        print(
            "Whole dummy requests: all selected outputs, final states and checkpoints exactly match"
        )


if __name__ == "__main__":
    unittest.main()
