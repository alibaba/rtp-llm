"""Bitwise regression for Qwen GDN on the CUDA varlen path."""

import argparse
from itertools import accumulate

import torch

from rtp_llm.models_py.triton_kernels.causal_conv1d.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from rtp_llm.models_py.triton_kernels.fla import (
    fused_recurrent_gated_delta_rule,
    load_initial_state_from_block_map,
    store_ssm_state_to_block_map,
)
from rtp_llm.models_py.triton_kernels.fla.chunk import chunk_gated_delta_rule
from rtp_llm.models_py.triton_kernels.fla.l2norm import l2norm_fwd


def bit_mismatches(expected, actual):
    assert expected.dtype == actual.dtype and expected.shape == actual.shape
    itemsize = expected.element_size()
    expected = expected.flatten().contiguous().view(torch.uint8)
    actual = actual.flatten().contiguous().view(torch.uint8)
    return (expected != actual).view(-1, itemsize).any(dim=1).count_nonzero().item()


class GdnFixture:
    block = 64
    dim = 128

    def __init__(self, key_heads, value_heads, *, batch_invariant):
        self.value_heads = value_heads
        self.widths = (
            key_heads * self.dim,
            key_heads * self.dim,
            value_heads * self.dim,
        )
        self.channels = sum(self.widths)
        self.batch_invariant = batch_invariant
        torch.manual_seed(813)
        base = {"device": "cuda", "dtype": torch.bfloat16}
        self.raw = [torch.randn(n + 3, self.channels, **base) for n in (193, 149)]
        self.g = [
            -torch.rand(x.size(0), self.value_heads, device="cuda") * 0.06
            for x in self.raw
        ]
        self.beta = [
            torch.rand(x.size(0), self.value_heads, device="cuda") for x in self.raw
        ]
        self.weight = torch.randn(self.channels, 4, **base) * 0.25
        self.blocks = self.ints([[2, 4, 1, 6], [7, 3, 5, 8]])

    @staticmethod
    def ints(values):
        return torch.tensor(values, dtype=torch.int32, device="cuda")

    def qkv(self, mixed, batch=1):
        return [
            value.reshape(batch, -1, width // self.dim, self.dim)
            for value, width in zip(mixed.split(self.widths, dim=-1), self.widths)
        ]

    def run(self, prefill_batches, decode_rows=(0,)):
        positions = [0, 0]
        conv = torch.full(
            (9, 3, self.channels), 17, dtype=torch.bfloat16, device="cuda"
        ).transpose(1, 2)
        ssm = torch.full(
            (9, self.value_heads, self.dim, self.dim),
            17,
            dtype=torch.float32,
            device="cuda",
        )
        target_outputs = []

        for batch in prefill_batches:
            rows, lengths = zip(*batch)
            maps = self.blocks[list(rows)]
            prefixes = self.ints([positions[r] for r in rows])
            cu = self.ints(list(accumulate(lengths, initial=0)))

            def gather(values):
                return torch.cat(
                    [values[r][positions[r] : positions[r] + n] for r, n in batch]
                )

            mixed = causal_conv1d_fn(
                gather(self.raw).T,
                self.weight,
                None,
                conv,
                cu,
                maps,
                prefixes,
                self.block,
            ).T
            initial = torch.empty_like(ssm[: len(rows)])
            load_initial_state_from_block_map(prefixes, maps, ssm, initial, self.block)
            q, k, v = self.qkv(mixed)
            out, h, final = chunk_gated_delta_rule(
                q,
                k,
                v,
                gather(self.g)[None],
                gather(self.beta)[None],
                initial_state=initial,
                output_final_state=True,
                cu_seqlens=cu,
                use_qk_l2norm_in_kernel=True,
                batch_invariant=self.batch_invariant,
            )
            store_ssm_state_to_block_map(
                h, final, prefixes, cu, maps, ssm, self.block, 64
            )
            if 0 in rows:
                index = rows.index(0)
                start = sum(lengths[:index])
                target_outputs.append(out[0, start : start + lengths[index]])
            for r, n in batch:
                positions[r] += n

        target_pages = (positions[0] + self.block - 1) // self.block
        target_slots = self.blocks[0, :target_pages].long()
        result = {
            "prefill": torch.cat(target_outputs, dim=0),
            "conv": conv[target_slots].clone(),
            "ssm": ssm[target_slots].clone(),
        }

        target_outputs.clear()
        maps = self.blocks[list(decode_rows)]
        for _ in range(3):
            seq = self.ints([positions[r] + 1 for r in decode_rows])
            mixed = causal_conv1d_update(
                torch.stack([self.raw[r][positions[r]] for r in decode_rows]),
                conv,
                self.weight,
                activation="silu",
                block_map=maps,
                seq_size_per_block=self.block,
                sequence_lengths=seq,
            )
            q, k, v = self.qkv(mixed, len(decode_rows))
            out, _ = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=torch.stack([self.g[r][positions[r]] for r in decode_rows])[:, None],
                beta=torch.stack([self.beta[r][positions[r]] for r in decode_rows])[
                    :, None
                ],
                initial_state=ssm,
                inplace_final_state=True,
                block_map=maps,
                sequence_lengths=seq,
                seq_size_per_block=self.block,
                use_qk_l2norm_in_kernel=True,
            )
            target_outputs.append(out[decode_rows.index(0), 0])
            for row in decode_rows:
                positions[row] += 1
        result["decode"] = torch.stack(target_outputs)
        result["decode_ssm"] = ssm[
            self.blocks[0, (positions[0] - 1) // self.block]
        ].clone()
        return result


def check_qk_norm(batch_invariant):
    generator = torch.Generator(device="cuda").manual_seed(45)
    qk = torch.randn(
        (193, 128), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    small = l2norm_fwd(qk[:128], batch_invariant=batch_invariant)
    mixed = l2norm_fwd(qk, batch_invariant=batch_invariant)[:128]
    different = bit_mismatches(small, mixed)
    print(f"Q/K norm across the 128-row dispatch boundary: {different}", flush=True)
    if batch_invariant:
        assert different == 0


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--tp4-397b", action="store_true")
    args = parser.parse_args()
    heads = (4, 16) if args.tp4_397b else (16, 32)
    fixture = GdnFixture(*heads, batch_invariant=not args.report_only)
    check_qk_norm(fixture.batch_invariant)
    failures = 0
    for length in (7, 17, 33, 63, 64, 65, 129, 193):
        expected = fixture.run([[(0, length)]])
        cases = {
            "first": [[(0, length), (1, 149)]],
            "last": [[(1, 149), (0, length)]],
        }
        if length > 64:
            cases["restored_at_64"] = [[(0, 64)], [(1, 149), (0, length - 64)]]
        if length > 128:
            cases["restored_at_128"] = [[(0, 128), (1, 149)], [(0, length - 128)]]
        if length > 192:
            cases["restored_at_192"] = [[(0, 192)], [(1, 149), (0, length - 192)]]
        for case, batches in cases.items():
            decode_rows = (0, 1) if case == "first" else (1, 0)
            actual = fixture.run(batches, decode_rows=decode_rows)
            differences = {
                stage: bit_mismatches(tensor, actual[stage])
                for stage, tensor in expected.items()
            }
            print(length, case, differences, flush=True)
            failures += sum(differences.values())
    if not args.report_only:
        assert failures == 0, f"{failures} unequal elements across GDN comparisons"

    # A decode checkpoint, even on an exact page boundary, cannot substitute
    # for the corresponding prefill checkpoint. The serving mode rejects
    # cross-request prefix reuse until it can track the checkpoint's source.
    prefill_checkpoint = fixture.run([[(0, 64)]])["ssm"][-1]
    decode_checkpoint = fixture.run([[(0, 61)]])["decode_ssm"]
    differences = bit_mismatches(prefill_checkpoint, decode_checkpoint)
    print("prefill vs decode checkpoint at token 64:", differences, flush=True)
    assert differences > 0


if __name__ == "__main__":
    main()
