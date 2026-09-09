"""Complete RTP-writer + production TRT-compat unit benchmark.

Both branches retain the RTP 656-byte cache and production writer/RoPE/Wkc.
The TRT branch includes the REAL production TopK gather/requantization adapter;
this is not the previous native576-history benchmark. Precision rejection is
retained and causes exit 2, even when experimental timings are requested.
"""

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import inspect
import itertools
import json
import math
import platform
import statistics
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch
from glm53_attention_unit_benchmark import (
    TOLERANCE,
    CommonData,
    capture,
    check_units,
    measure,
)
from glm53_attention_unit_paths import FlashUnit


def backend_module():
    for root in Path(__file__).resolve().parents:
        if (root / "rtp_llm" / "models_py").is_dir():
            sys.path.insert(0, str(root))
            break
    return importlib.import_module(
        "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.trtllm_sparse_impl"
    )


class CompatCommonData(CommonData):
    """Preserve the original fixture, adding H32 as an explicit H64 prefix.

    The original complete-unit fixture intentionally allowed only H8/H16/H64.
    Truncating the SAME H64 source Q/Wkc gives both backends identical H32
    inputs without editing that historical benchmark or its random generator.
    """

    def __init__(self, batch, context, heads, queries=1, seed=53):
        super().__init__(batch, context, 64 if heads == 32 else heads, queries, seed)
        if heads == 32:
            self.heads = heads
            # With T=1, contiguous() may keep the old H64 leading stride.
            # RTP's strided-copy contract requires the canonical H32 stride.
            q = torch.empty(
                (self.tokens, heads, self.q.shape[-1]),
                device=self.q.device,
                dtype=self.q.dtype,
            )
            q.copy_(self.q[:, :heads])
            self.q = q
            self.wkc = self.wkc[:heads].contiguous()


class CompatUnit(FlashUnit):
    def __init__(self, common):
        super().__init__(common)
        self.op = backend_module().TrtllmSparseMlaFp8Op(
            num_heads=common.heads,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            qk_nope_head_dim=192,
            page_size=64,
            softmax_extra_scale=1.0,
            top_k=2048,
            use_cuda_graph=True,
        )
        params = SimpleNamespace(
            batch_indice_d=common.request_ids,
            kvlen_d=common.seq_lens,
            expanded_seq_lens=common.seq_lens,
        )
        inputs = SimpleNamespace(
            is_prefill=False,
            is_target_verify=common.queries > 1,
            is_draft_extend=False,
        )
        self.op.plan(params, common.page_table, inputs)

    def forward(self):
        c = self.common
        self.write_cache(
            self.q_work,
            c.k,
            self.krope_work,
            self.cache,
            c.slots,
            c.positions,
            c.cos_sin,
            512,
            64,
            True,
            "fp8_ds_mla",
        )
        self.copy_rope(self.q_absorbed, self.q_work[..., 192:], 512)
        torch.bmm(
            self.q_work[..., :192].transpose(0, 1),
            c.wkc,
            out=self.q_absorbed[..., :512].transpose(0, 1),
        )
        return self.op.forward(self.q_absorbed, self.cache, c.logical_topk, layer_id=0)


def profile_compat_stages(path, rounds):
    """Diagnostic event markers around the REAL production functions.

    These extra graph events perturb the measured launch sequence. Keep the
    resulting numbers separate from the uninstrumented complete-unit matrix.
    No production function or arithmetic is replaced, only its call is marked.
    """
    module = backend_module()
    events = {
        name: (
            torch.cuda.Event(enable_timing=True, external=True),
            torch.cuda.Event(enable_timing=True, external=True),
        )
        for name in ("convert_selected_kv", "trt_attention", "empty_row_mask")
    }

    def marked(name, function):
        def invoke(*args, **kwargs):
            events[name][0].record()
            result = function(*args, **kwargs)
            events[name][1].record()
            return result

        return invoke

    convert, decode, mask = (
        module.convert_selected_kv,
        path.op._decode,
        module.mask_empty_output,
    )
    try:
        module.convert_selected_kv = marked("convert_selected_kv", convert)
        path.op._decode = marked("trt_attention", decode)
        module.mask_empty_output = marked("empty_row_mask", mask)
        graph, begin, end, _ = capture(path, None)
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        samples = {name: [] for name in events}
        samples["instrumented_complete_unit"] = []
        for _ in range(rounds):
            graph.replay()
            end.synchronize()
            samples["instrumented_complete_unit"].append(
                begin.elapsed_time(end) * 1000.0
            )
            for name, (first, last) in events.items():
                samples[name].append(first.elapsed_time(last) * 1000.0)
    finally:
        module.convert_selected_kv = convert
        path.op._decode = decode
        module.mask_empty_output = mask
    return {
        "qualification": "instrumented graph diagnostics only; additional CUDA event nodes perturb latency; do not substitute for complete_unit_timing or claim core-only speedup",
        "stages": {
            name: {
                "median_us": statistics.median(values),
                "p95_us": sorted(values)[math.ceil(0.95 * len(values)) - 1],
                "samples_us": values,
            }
            for name, values in samples.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--batches", nargs="+", type=int, default=[1])
    parser.add_argument("--contexts", nargs="+", type=int, default=[30000])
    parser.add_argument("--heads", nargs="+", type=int, default=[64, 8])
    parser.add_argument("--queries", nargs="+", type=int, default=[1])
    parser.add_argument("--rounds", type=int, default=31)
    parser.add_argument("--chunk-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--cache-mode", choices=["warm", "cold-l2"], default="warm")
    args = parser.parse_args()
    if min(args.rounds, args.chunk_tokens) < 1:
        parser.error("rounds/chunk-tokens must be positive")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.get_device_capability(0)[0] != 10:
        raise RuntimeError("TRT sparse adapter requires SM10x")
    module = backend_module()
    repo = next(
        parent
        for parent in Path(module.__file__).resolve().parents
        if (parent / "rtp_llm" / "models_py" / "triton_kernels").is_dir()
    )
    source_files = [
        Path(__file__),
        Path(module.__file__),
        Path(inspect.getsourcefile(module.convert_selected_kv)),
        Path(__file__).with_name("glm53_attention_unit_benchmark.py"),
        Path(__file__).with_name("glm53_attention_unit_paths.py"),
    ]
    source_files.extend(
        repo / "rtp_llm" / "models_py" / "triton_kernels" / relative
        for relative in (
            "sparse_mla/fused_qk_rope_cat_cache_mla.py",
            "common/strided_slice_copy.py",
            "sparse_mla/block_index_to_global.py",
        )
    )
    report = dict(
        schema="glm53_trtllm_656_compat_complete_unit_v1",
        args=vars(args),
        started_utc=datetime.now(timezone.utc).isoformat(),
        device=torch.cuda.get_device_name(0),
        device_uuid=str(torch.cuda.get_device_properties(0).uuid),
        host=platform.node(),
        executable=sys.executable,
        gpu_before=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.used,utilization.gpu,clocks.sm",
                "--format=csv",
            ],
            text=True,
            timeout=15,
        ),
        versions={
            name: importlib.metadata.version(name)
            for name in ("torch", "triton", "flashinfer-python", "flash-mla")
        },
        tolerance=TOLERANCE,
        boundary="same pre-RoPE BF16 inputs -> RTP writer/656 cache + Wkc -> FlashMLA or production TRT TopK compatibility adapter -> latent512",
        timing="3 eager + 5 graph warmups; external CUDA events exclude stage/history/JIT; alternating backend order; full unit only",
        flash_scheduler="first-layer per-forward rebuild included",
        flash_low_head_mode="H8/H16/H32 use timed padding to FlashMLA H64, NOT native low-head FlashMLA or demonstrated production TP8 performance",
        h32_fixture="H64 fixture with first 32 Q heads and Wkc heads retained; same shared source for both backends and golden",
        trt_cache="RTP mixed656 history; compact gather/dequant/requant to temporary native576 included in every measured forward",
        conversion_contract="actual FlashMLA cb10b79 decode reader: round stored FP32 scale to BF16, multiply FP8 payload and round product to BF16; saturating RNE E4M3 requantization of absorbed Q and gathered KV; bmm scales 1/16 and 1",
        source_sha256={
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_files
        },
        rows=[],
    )

    def save():
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")

    save()
    rejected = False
    for shape in itertools.product(
        args.batches, args.contexts, args.heads, args.queries
    ):
        batch, context, heads, queries = shape
        shape_dict = dict(batch=batch, context=context, heads=heads, queries=queries)
        try:
            torch.cuda.reset_peak_memory_stats()
            common = CompatCommonData(batch, context, heads, queries, args.seed)
            paths = {"rtp": FlashUnit(common), "trt": CompatUnit(common)}
            common.populate(paths.values(), args.chunk_tokens)
            precision, status = check_units(common, paths)
            row = dict(shape=shape_dict, precision=precision, status=status)
            rejected |= status == "precision_rejected"
            if args.benchmark:
                row["complete_unit_timing"] = measure(
                    paths, args.rounds, args.cache_mode
                )
                row["timing_qualification"] = (
                    "experimental only; strict precision gate is not waived"
                )
            if args.profile_stages:
                row["instrumented_compat_stages"] = profile_compat_stages(
                    paths["trt"], args.rounds
                )
            row["harness_peak_allocated_gib"] = (
                torch.cuda.max_memory_allocated() / 2**30
            )
            report["rows"].append(row)
            save()
            print("RESULT " + json.dumps(row), flush=True)
            del paths, common
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            report["rows"].append(
                dict(
                    shape=shape_dict,
                    status="runtime_failed",
                    traceback=traceback.format_exc(),
                )
            )
            save()
            raise
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["source_sha256_end"] = {
        str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source_files
    }
    if report["source_sha256_end"] != report["source_sha256"]:
        report["exit_reason"] = "source_changed_during_run"
        save()
        raise RuntimeError("Benchmark source changed during the run; discard timings")
    report["exit_reason"] = "precision_rejected" if rejected else "passed"
    save()
    raise SystemExit(2 if rejected else 0)


if __name__ == "__main__":
    main()
