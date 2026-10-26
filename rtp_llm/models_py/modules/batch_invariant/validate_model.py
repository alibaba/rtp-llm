"""Bounded, fixed-token full-model contract check; never trains or samples.

Run with ``python -m rtp_llm.models_py.modules.batch_invariant.validate_model
--model /path/to/checkpoint --output result.json`` on one CUDA GPU.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInputs, get_typemeta

from .kernels import selected_logprobs


class Harness:
    def __init__(self, model):
        self.auto = model
        self.model = model.model
        self.table_width = 8
        self.checks = []

    def inputs(self, sequences, *, prefixes=None, slots=None, decode=False):
        batch = len(sequences)
        prefixes = prefixes or [0] * batch
        slots = slots or list(range(batch))
        lens = [len(s) for s in sequences]
        assert all(n > 0 for n in lens)
        assert not decode or all(n == 1 for n in lens)
        table = torch.tensor(
            [
                [1 + slot * self.table_width + i for i in range(self.table_width)]
                for slot in slots
            ],
            dtype=torch.int32,
        )
        assert int(table.max()) < self.auto.block_nums
        assert (
            max(p + n for p, n in zip(prefixes, lens))
            <= self.table_width * self.auto.tokens_per_block
        )
        a = PyAttentionInputs()
        a.is_prefill = not decode
        a.input_lengths = torch.tensor(lens, dtype=torch.int32)
        a.prefix_lengths = torch.tensor([] if decode else prefixes, dtype=torch.int32)
        a.sequence_lengths = torch.tensor(
            prefixes if decode else [], dtype=torch.int32
        ).pin_memory()
        a.kv_cache_block_id = table
        a.kv_cache_kernel_block_id = table
        a.kv_cache_block_id_device = table.cuda()
        a.kv_cache_kernel_block_id_device = a.kv_cache_block_id_device
        cu = [0]
        for n in lens:
            cu.append(cu[-1] + n)
        a.cu_seqlens_device = torch.tensor(cu, dtype=torch.int32, device="cuda")
        a.padding_offset = torch.tensor(
            [i * max(lens) - cu[i] for i, n in enumerate(lens) for _ in range(n)],
            dtype=torch.int32,
            device="cuda",
        )
        a.dtype = get_typemeta(self.auto.kv_cache.get_layer_cache(0).kv_cache_base)
        tokens = torch.tensor(
            [t for s in sequences for t in s], dtype=torch.int32, device="cuda"
        )
        return PyModelInputs(input_ids=tokens, attention_inputs=a)

    @torch.inference_mode()
    def run(self, sequences, *, device_metadata=False, **kwargs):
        inputs = self.inputs(sequences, **kwargs)
        if device_metadata:
            a = inputs.attention_inputs
            a.input_lengths = a.input_lengths.cuda()
            a.prefix_lengths = a.prefix_lengths.cuda()
            a.sequence_lengths = a.sequence_lengths.cuda()
        hidden = self.model.forward(inputs).hidden_states
        logits = self.model.batch_invariant_lm_head(hidden, self.auto.lm_head_weight)
        targets = torch.tensor([t for s in sequences for t in s], device="cuda")
        return hidden.clone(), logits, selected_logprobs(logits, targets)

    def check(self, name, expected, actual):
        counts = {}
        for kind, a, b in zip(("hidden", "logits", "logprobs"), expected, actual):
            assert a.shape == b.shape and a.dtype == b.dtype, (
                name,
                kind,
                a.shape,
                b.shape,
            )
            assert torch.isfinite(a).all() and torch.isfinite(b).all(), (
                name,
                kind,
                "nonfinite",
            )
            unequal = (
                (a.contiguous().view(torch.uint8) != b.contiguous().view(torch.uint8))
                .sum()
                .item()
            )
            counts[kind] = dict(
                elements=a.numel(),
                unequal_bytes=unequal,
                max_abs=(a.float() - b.float()).abs().max().item(),
            )
        record = dict(case=name, values=counts)
        self.checks.append(record)
        print(json.dumps(record), flush=True)
        assert all(c["unequal_bytes"] == 0 for c in counts.values()), name


@torch.inference_mode()
def validate(model):
    h = Harness(model)
    from rtp_llm.models_py.modules.batch_invariant.attention import (
        BatchInvariantAttention,
    )

    warmup_inputs = h.inputs([[1, 2, 3, 4, 5]])
    cfg = model.model_config.getAttentionConfigs(1)
    qkv = torch.randn(
        5,
        (cfg.head_num + 2 * cfg.kv_head_num) * cfg.size_per_head,
        dtype=torch.bfloat16,
        device="cuda",
    )
    warmup = BatchInvariantAttention(cfg, warmup_inputs.attention_inputs, warmup=True)
    actual_attention = model.model.prepare_fmha_impl(warmup_inputs)
    a = warmup.forward(qkv.clone(), None)
    b = actual_attention.forward(qkv.clone(), model.kv_cache.get_layer_cache(0))
    assert torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )
    print("warmup attention matches allocated KV attention", flush=True)
    # Fixed token IDs avoid tokenizer or RNG differences between executions.
    seq = model.tokenizer.encode(
        "Calculate carefully: if 17 boxes each contain 23 balls and 48 are removed, how many remain? Show the arithmetic."
    )
    seq = (seq * 4)[:65]
    other = model.tokenizer.encode(
        "The quick brown fox jumps over the lazy dog. 123456789"
    )
    ref = h.run([seq])
    h.check("repeat", ref, h.run([seq]))
    h.check("device-metadata-prefill", ref, h.run([seq], device_metadata=True))
    for peers in (1, 3, 6):
        sequences = [other[: 3 + i] for i in range(peers)] + [seq]
        actual = h.run(sequences)
        offset = sum(map(len, sequences[:-1]))
        h.check(
            f"batch-{peers + 1}-target-last", ref, tuple(t[offset:] for t in actual)
        )
        reversed_sequences = list(reversed(sequences))
        h.check(
            f"batch-{peers + 1}-target-first",
            ref,
            tuple(t[: len(seq)] for t in h.run(reversed_sequences)),
        )
    h.check("physical-kv-pages", ref, h.run([seq], slots=[7]))
    for chunk in (1, 7, 16, 33):
        parts = [
            h.run([seq[i : i + chunk]], prefixes=[i]) for i in range(0, len(seq), chunk)
        ]
        h.check(
            f"prefill-chunk-{chunk}",
            ref,
            tuple(torch.cat([p[j] for p in parts]) for j in range(3)),
        )
    for cut in (1, 31, 64):
        parts = [h.run([seq[:cut]])]
        for i in range(cut, len(seq)):
            # Decode alongside a peer with a different prefix length.
            h.run([other[:3]], slots=[1])
            result = h.run(
                [[seq[i]], [other[3]]], prefixes=[i, 3], slots=[0, 1], decode=True
            )
            parts.append(tuple(t[:1] for t in result))
        h.check(
            f"prefill-decode-cut-{cut}",
            ref,
            tuple(torch.cat([p[j] for p in parts]) for j in range(3)),
        )
    # Capture and replay the complete Python model, reusing planned metadata.
    h.run([seq[:32]])
    inputs = h.inputs([[seq[32]]], prefixes=[32], decode=True)
    fmha = h.model.prepare_fmha_impl(inputs, is_cuda_graph=True)

    def forward():
        hidden = h.model.forward(inputs, fmha).hidden_states
        logits = h.model.batch_invariant_lm_head(hidden, model.lm_head_weight)
        return hidden, logits, selected_logprobs(logits, inputs.input_ids)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = forward()
    graph.replay()
    h.check("full-model-graph-replay", tuple(t[32:33] for t in ref), outputs)
    updated = h.inputs([[seq[33]]], prefixes=[33], decode=True)
    inputs.input_ids.copy_(updated.input_ids)
    fmha.prepare_cuda_graph(updated.attention_inputs)
    graph.replay()
    h.check("full-model-graph-new-position", tuple(t[33:34] for t in ref), outputs)
    return h.checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    # Some existing model-loader imports initialize the server's argument parser.
    # This standalone validator owns its command line, so do not forward it there.
    sys.argv = sys.argv[:1]
    from rtp_llm.models_py.standalone.auto_model import AutoModel

    result = dict(model=args.model, started=time.time(), status="running")
    out = Path(args.output)
    try:
        model = AutoModel.from_pretrained(
            args.model, batch_invariant=True, max_total_tokens=8192, tokens_per_block=16
        )
        result.update(
            gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,
            profile=model.model.batch_invariant_policy.profile,
        )
        result["checks"] = validate(model)
        result["status"] = "passed"
    except BaseException as exc:
        result.update(status="failed", error=repr(exc))
        raise
    finally:
        result["ended"] = time.time()
        out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
