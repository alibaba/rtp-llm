"""Time the real CUDA Graph runner and Page-RR metadata without model weights."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch

from rtp_llm.cpp.cuda_graph.tests.cuda_graph_tagged_cache_test import (
    CudaGraphRunner,
    _build_common_inputs,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.page_rr_mla_metadata import (
    PageRRMlaMetadata,
)
from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelOutputs

TAGS = ["full", "indexer_kv"]
PAGE_TOKENS = 4096
KERNEL_PAGE_TOKENS = 128
TABLE_WIDTH = PAGE_TOKENS // KERNEL_PAGE_TOKENS
SHARDS = 8
HIDDEN = 128
MODES = ("host_external", "device_captured")


class AttentionPrepare:
    def __init__(self, inputs):
        self.metadata = PageRRMlaMetadata(PAGE_TOKENS, KERNEL_PAGE_TOKENS, SHARDS, 0)
        self.metadata.prepare_decode(inputs)
        self.check_calls = False
        self.calls = 0

    def prepare_cuda_graph(self, inputs):
        if self.check_calls:
            self.calls += 1
        self.metadata.prepare_decode(inputs, forbid_realloc=True)


class PrepareModel:
    def __init__(self, mode):
        self.mode = mode
        self.cuda_graph_prepares_attention_metadata_in_forward = (
            mode == "device_captured"
        )
        self.backends = []

    def prepare_fmha_impl(self, inputs, is_cuda_graph=False):
        groups = {
            tag: AttentionPrepare(attention)
            for tag, attention in inputs.attention_inputs.items()
        }
        self.backends.extend(groups.values())
        return groups

    def forward(self, inputs, fmha_impl):
        if self.cuda_graph_prepares_attention_metadata_in_forward:
            for tag, impl in fmha_impl.items():
                impl.prepare_cuda_graph(inputs.attention_inputs[tag])
        full, indexer = (fmha_impl[tag].metadata for tag in TAGS)
        # Consume both groups in the captured graph. Small residues preserve an
        # exact BF16 signature while lengths and block IDs change between inputs.
        signature = full.positions_d.remainder(127) + indexer.slot_mapping.remainder(
            113
        )
        return PyModelOutputs(
            inputs.input_hiddens + signature[:, None].to(torch.bfloat16)
        )


def make_inputs(batch, queries, step, length_source):
    attn = PyAttentionInputs()
    verify = queries > 1
    prefix = 1535 + 257 * step
    attn.is_prefill = verify
    attn.is_target_verify = verify
    attn.input_lengths = torch.full((batch,), queries, dtype=torch.int32).pin_memory()
    attn.prefix_lengths = (
        torch.full((batch,), prefix, dtype=torch.int32)
        if verify
        else torch.empty(0, dtype=torch.int32)
    ).pin_memory()
    attn.sequence_lengths = (
        torch.empty(0, dtype=torch.int32)
        if verify
        else torch.full((batch,), prefix - 1, dtype=torch.int32)
    ).pin_memory()
    attn.sequence_lengths_plus_1_device = torch.full(
        (batch,), prefix, dtype=torch.int32, device="cuda"
    )
    attn.cu_seqlens = torch.arange(
        0, (batch + 1) * queries, queries, dtype=torch.int32
    ).pin_memory()
    attn.cu_seqlens_device = attn.cu_seqlens.cuda()
    attn.cu_kv_seqlens_device = torch.arange(
        0,
        (batch + 1) * (queries + prefix),
        queries + prefix,
        dtype=torch.int32,
        device="cuda",
    )
    attn.decode_cu_seqlens_device = torch.arange(
        batch + 1, dtype=torch.int32, device="cuda"
    )
    attn.context_total_kv_length = batch * (queries + prefix)
    if length_source == "cuda":
        attn.input_lengths = attn.input_lengths.cuda()
        attn.prefix_lengths = attn.prefix_lengths.cuda()
        attn.sequence_lengths = attn.sequence_lengths.cuda()
        attn.cu_seqlens = attn.cu_seqlens_device

    tokens = batch * queries
    inputs = _build_common_inputs(
        attn,
        TAGS,
        {"full": step + 1, "indexer_kv": 2 * step + 2},
        batch_size=batch,
        token_count=tokens,
        block_count=TABLE_WIDTH,
    )
    inputs.input_hiddens = torch.zeros(
        (tokens, HIDDEN), dtype=torch.bfloat16, device="cuda"
    )
    return inputs


def make_runner(mode, batch, queries):
    model = PrepareModel(mode)
    runner = CudaGraphRunner()
    runner.init_decode(
        model,
        HIDDEN,
        PAGE_TOKENS,
        TABLE_WIDTH,
        [batch],
        TAGS,
        queries > 1,
        queries,
        0,
        batch,
    )
    return model, runner


def check_replay(model, runner, inputs_ring):
    for backend in model.backends:
        backend.check_calls = True
    outputs = []
    for inputs in inputs_ring:
        if not runner.canRun(inputs):
            raise RuntimeError(f"{model.mode} cannot replay the selected graph")
        outputs.append(runner.forward(inputs).hidden_states.cpu())
    for backend in model.backends:
        backend.check_calls = False
    expected_calls = (
        0 if model.mode == "device_captured" else len(TAGS) * len(inputs_ring)
    )
    actual_calls = sum(backend.calls for backend in model.backends)
    if actual_calls != expected_calls:
        raise RuntimeError(
            f"{model.mode}: got {actual_calls} Python prepare calls, expected {expected_calls}"
        )
    if torch.equal(outputs[0], outputs[1]):
        raise RuntimeError(f"{model.mode} did not reflect changed attention metadata")
    return outputs


def sample(runner, inputs_ring, iterations):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # Materialize event handles outside timing. Event time measures the stream
    # interval, which includes idle gaps when graph submission is CPU-bound.
    start_event.record()
    end_event.record()
    end_event.synchronize()
    start_event.record()
    start_ns = time.perf_counter_ns()
    for index in range(iterations):
        runner.forward(inputs_ring[index % len(inputs_ring)])
    submit_ns = time.perf_counter_ns()
    end_event.record()
    end_event.synchronize()
    complete_ns = time.perf_counter_ns()
    return {
        "submit_us": (submit_ns - start_ns) / (1000 * iterations),
        "complete_us": (complete_ns - start_ns) / (1000 * iterations),
        "stream_us": start_event.elapsed_time(end_event) * 1000 / iterations,
    }


def benchmark_case(batch, queries, length_source, iterations, rounds):
    inputs_ring = [
        make_inputs(batch, queries, step, length_source) for step in range(4)
    ]
    modes = list(MODES)
    runners = {mode: make_runner(mode, batch, queries) for mode in modes}
    expected = check_replay(*runners[modes[0]], inputs_ring)
    for mode in modes[1:]:
        actual = check_replay(*runners[mode], inputs_ring)
        for output, reference in zip(actual, expected):
            torch.testing.assert_close(output, reference, rtol=0, atol=0)
    for _, runner in runners.values():
        sample(runner, inputs_ring, 48)

    samples = {mode: [] for mode in modes}
    for index in range(rounds):
        offset = index % len(modes)
        for mode in modes[offset:] + modes[:offset]:
            samples[mode].append(sample(runners[mode][1], inputs_ring, iterations))
    medians = {
        mode: {
            metric: statistics.median(row[metric] for row in rows) for metric in rows[0]
        }
        for mode, rows in samples.items()
    }
    result = {
        "mode": "decode" if queries == 1 else "target_verify",
        "batch": batch,
        "queries_per_request": queries,
        "input_length_source": length_source,
        "median_us_per_step": medians,
        "samples_us_per_step": samples,
    }
    print("ATTENTION_PREPARE_RESULT " + json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=240)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 32, 128])
    parser.add_argument("--length-source", choices=("cuda", "host"), default="cuda")
    args = parser.parse_args()
    if min(args.iterations, args.rounds, *args.batches) <= 0:
        parser.error("iterations, rounds, and batch sizes must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required for this microbenchmark")
    device = torch.cuda.get_device_properties(0)
    setup = {
        "gpu": device.name,
        "compute_capability": [device.major, device.minor],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "page_tokens": PAGE_TOKENS,
        "kernel_page_tokens": KERNEL_PAGE_TOKENS,
        "table_width": TABLE_WIDTH,
        "metadata_shards": SHARDS,
        "executed_shard": 0,
        "cache_group_tags": TAGS,
        "hidden_width": HIDDEN,
        "iterations_per_sample": args.iterations,
        "rounds": args.rounds,
    }
    print("ATTENTION_PREPARE_SETUP " + json.dumps(setup), flush=True)
    with torch.inference_mode():
        results = [
            benchmark_case(
                batch, queries, args.length_source, args.iterations, args.rounds
            )
            for queries in (1, 4)
            for batch in args.batches
        ]
    output_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    if output_dir:
        (Path(output_dir) / "attention_prepare_microbenchmark.json").write_text(
            json.dumps({"setup": setup, "results": results}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
