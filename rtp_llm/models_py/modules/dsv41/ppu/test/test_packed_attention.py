"""N5 direct packed operands against the accepted BF16 workspace boundary."""

import hashlib
import io
import json
import math
import os
from pathlib import Path
import struct
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu.attention import flash_mla_sparse_fwd_ppu
from rtp_llm.models_py.modules.dsv41.ppu.graph_attention import _attention_torch
from rtp_llm.models_py.modules.dsv41.ppu.kv_codec import decode_kv, encode_kv
from rtp_llm.models_py.modules.dsv41.ppu.packed_attention import (
    PACKED_ENV, PackedAttentionPlan, compact_prefix, packed_pool_view,
    physical_slots, prepare_packed_attention, load_packed_backend,
)
from rtp_llm.models_py.modules.dsv41.ppu.packed_decode import packed_decode
from rtp_llm.models_py.modules.dsv41.ppu.packed_prefill import packed_prefill


DIAGNOSTICS_ENV = "RTP_V41_PPU_PACKED_DIAGNOSTICS"


def _diagnostic_directory():
    value = os.environ.get(DIAGNOSTICS_ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(DIAGNOSTICS_ENV + " must be explicitly 0 or 1")
    if value == "0":
        return None
    root = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", ""))
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("diagnostics require an existing absolute TEST_UNDECLARED_OUTPUTS_DIR")
    directory = root.resolve() / "n5-packed-b04"
    directory.mkdir(exist_ok=False)
    return directory


def _diagnostic_json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _diagnostic_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_diagnostic_json_value(item) for item in value]
    return value


def _diagnostic_json(path, record):
    payload = json.dumps(_diagnostic_json_value(record), indent=2, sort_keys=True, allow_nan=False).encode()
    if len(payload) > 256 * 1024:
        raise ValueError("diagnostic JSON exceeds 256 KiB")
    with path.open("xb") as output:
        output.write(payload)


def _diagnostic_stages(q, keys, indices, scale, sink):
    # Operands already have the requested dtype/device. In particular, never
    # call .float() here: the independent CPU path must remain true FP64.
    logits = torch.einsum("bhd,bkd->bhk", q, keys) * scale
    logits = logits.masked_fill(indices[:, 0, None] < 0, float("-inf"))
    normalizer = torch.logsumexp(
        torch.cat((logits, sink[None, :, None].expand(q.shape[0], -1, 1)), -1),
        dim=-1, keepdim=True,
    )
    weights = torch.exp(logits - normalizer)
    pv = torch.einsum("bhk,bkd->bhd", weights, keys)
    return {"logits": logits, "weights": weights, "pv": pv}


def _attention_cpu64(q_cpu, bank_cpu, indices_cpu, scale, sink_cpu):
    """Unrounded CPU FP64 mathematical oracle for qualification and diagnostics."""
    if any(t.device.type != "cpu" for t in (q_cpu, bank_cpu, indices_cpu, sink_cpu)):
        raise ValueError("CPU FP64 oracle requires CPU inputs")
    query, bank, sink = q_cpu.double(), bank_cpu.double(), sink_cpu.double()
    keys = bank[:, 0][indices_cpu[:, 0].clamp_min(0).long()]
    stages = _diagnostic_stages(query, keys, indices_cpu, scale, sink)
    if any(value.device.type != "cpu" or value.dtype != torch.float64 for value in stages.values()):
        raise ValueError("CPU oracle stage lost CPU float64 precision")
    return stages, keys


def _diagnostic_metrics(left, right, active=None):
    left, right = left.detach().cpu().double(), right.detach().cpu().double()
    active = torch.ones_like(left, dtype=torch.bool) if active is None else active.expand_as(left)
    finite = torch.isfinite(left) & torch.isfinite(right)
    usable = active & finite
    error = torch.where(usable, (left - right).abs(), torch.zeros_like(left))
    # Same original output budget, evaluated without BF16 rounding. Stage
    # metrics are observations, not new qualification assertions.
    ratio = error / (0.005 + 0.005 * right.abs()).nan_to_num(nan=1.0, posinf=1.0)
    nonfinite_bad = active & ~finite & ~(left == right)
    bad = (usable & (ratio > 1)) | nonfinite_bad
    return {
        "elements": int(active.sum()), "nonfinite_mismatches": int(nonfinite_bad.sum()),
        "max_abs": float(error.max()), "max_budget_ratio": float(ratio.max()),
        "mismatches_at_original_budget": int(bad.sum()),
        "first_mismatch_coordinates": bad.nonzero()[:8].tolist(),
    }


def _diagnostic_e4m3(raw):
    sign, exponent, mantissa = (-1.0 if raw & 128 else 1.0), (raw >> 3) & 15, raw & 7
    if exponent == 15 and mantissa == 7:
        return float("nan")
    return sign * (math.ldexp(mantissa, -9) if exponent == 0
                   else math.ldexp(1.0 + mantissa / 8.0, exponent - 7))


def _diagnostic_decode_rows(rows, width):
    # Independent byte decoder on CPU, not _quant/float8 conversion or a native
    # cache reader. This checks values/scales separately from attention math.
    decoded = []
    fp4 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    for raw in rows.reshape(-1, width).tolist():
        if width == 528:
            scales = [math.ldexp(1.0, b - 127) if b != 255 else float("nan") for b in raw[512:]]
            decoded.append([_diagnostic_e4m3(raw[d]) * scales[d // 32] for d in range(512)])
        elif width == 288:
            scales = [_diagnostic_e4m3(b) for b in raw[256:]]
            values = []
            for d in range(512):
                code = (raw[d // 2] >> (4 * (d % 2))) & 15
                values.append((-1.0 if code & 8 else 1.0) * fp4[code & 7] * scales[d // 16])
            decoded.append(values)
        else:
            raise ValueError("unexpected packed diagnostic row width")
    return torch.tensor(decoded, dtype=torch.float64, device="cpu")


def _diagnostic_fsum_point(query, keys, sink, scale, dimension):
    logits = [math.fsum(a * b for a, b in zip(query, key)) * scale for key in keys]
    maximum = max([sink] + logits)
    unnormalized = [math.exp(value - maximum) for value in logits]
    denominator = math.fsum(unnormalized + [math.exp(sink - maximum)])
    weights = [value / denominator for value in unnormalized]
    output = math.fsum(weight * key[dimension] for weight, key in zip(weights, keys))
    return {"logits": logits, "weights": weights, "pv": output}


def _diagnostic_case(directory, case_id, *, prefill, with_global, q, bank, indices,
                     lengths, sink, plan, actual, reference, independent, assertion_errors, identity,
                     cpu64, keys_cpu64):
    rows = q.shape[0]
    if (rows not in (1, 32) or tuple(q.shape[1:]) != (10, 512)
            or tuple(indices.shape) != (rows, 1, 640) or case_id not in range(4)):
        raise ValueError("diagnostics are bounded to the four original cases")
    q_cpu, bank_cpu = q.detach().cpu(), bank.detach().cpu()
    indices_cpu, sink_cpu = indices.detach().cpu(), sink.detach().cpu()
    cpu_outputs = {"actual": actual.detach().cpu(), "accepted_bf16": reference.detach().cpu(),
                   "gpu_fp32_reference": independent.detach().cpu()}
    scale = 512 ** -0.5
    scale32 = struct.unpack("=f", struct.pack("=f", scale))[0]
    cpu64_rounded = cpu64["pv"].to(q_cpu.dtype)
    # Keep the old rounded-reference assertion outcome visible, but do not
    # confuse it with the corrected unrounded mathematical qualification gate.
    historical_rounded = {"status": "PASS"}
    try:
        torch.testing.assert_close(actual, independent, rtol=0.005, atol=0.005)
    except AssertionError as error:
        historical_rounded = {"status": "FAIL", "error": str(error)[:8192]}
    safe = indices[:, 0].clamp_min(0).long()
    keys_gpu = bank[:, 0][safe].float()
    gpu_stages = _diagnostic_stages(q.float(), keys_gpu, indices, scale, sink.float())
    stages_cpu = {name: value.detach().cpu() for name, value in gpu_stages.items()}
    # Each stage is also compared with a CPU double calculation receiving that
    # stage's exact GPU operands, to separate propagated QK/softmax/PV errors.
    exact_gpu_logits = stages_cpu["logits"].double()
    norm = torch.logsumexp(torch.cat((exact_gpu_logits,
        sink_cpu.double()[None, :, None].expand(rows, -1, 1)), -1), -1, keepdim=True)
    cpu_weights_from_gpu_logits = torch.exp(exact_gpu_logits - norm)
    cpu_pv_from_gpu_weights = torch.einsum("bhk,bkd->bhd", stages_cpu["weights"].double(), keys_cpu64)
    live = indices_cpu[:, 0, None] >= 0
    swa_bytes = plan.swa_cache.detach().cpu()
    global_bytes = plan.global_cache.detach().cpu() if with_global else None
    manual = _diagnostic_decode_rows(swa_bytes, 528)
    if with_global:
        manual = torch.cat((_diagnostic_decode_rows(global_bytes, 288), manual))
    manual = manual[:, None]
    focus = (2 if rows == 32 else 0, 4, 41)
    row, head, dimension = focus
    live_slots = indices_cpu[row, 0][indices_cpu[row, 0] >= 0].long()
    point_keys = bank_cpu[live_slots, 0].double().tolist()
    point = _diagnostic_fsum_point(q_cpu[row, head].double().tolist(), point_keys,
                                  float(sink_cpu[head]), scale32, dimension)
    # No BLAS/TF32 path for this one scalar; global backend policy is untouched.
    scalar_pv_gpu = (gpu_stages["weights"][row, head] * keys_gpu[row, :, dimension]).sum().cpu()
    focus_values = {name: float(value[focus]) for name, value in cpu_outputs.items()}
    focus_values.update(cpu_fp64=float(cpu64["pv"][focus]), cpu_fp64_rounded=float(cpu64_rounded[focus]),
                        gpu_pv_before_cast=float(stages_cpu["pv"][focus]),
                        gpu_pv_elementwise=float(scalar_pv_gpu), cpu_math_fsum=point["pv"])
    gpu_qk_elementwise = (q[row, head].float()[None] * keys_gpu[row]).sum(-1) * scale
    tensors = {
        "q": q_cpu, "decoded_bank": bank_cpu, "indices": indices_cpu, "lengths": lengths.cpu(),
        "sink": sink_cpu, **cpu_outputs, "swa_bytes": swa_bytes, "global_bytes": global_bytes,
        "swa_indices": plan.swa_indices.cpu(), "swa_lengths": plan.swa_lengths.cpu(),
        "global_indices": plan.global_indices.cpu() if with_global else None,
        "global_lengths": plan.global_lengths.cpu() if with_global else None,
        "gpu_stages": stages_cpu, "cpu_fp64_stages": cpu64, "cpu_fp64_rounded": cpu64_rounded,
        "cpu_weights_from_gpu_logits": cpu_weights_from_gpu_logits,
        "cpu_pv_from_gpu_weights": cpu_pv_from_gpu_weights, "manual_decoded_bank": manual,
        "gpu_qk_elementwise_focus": gpu_qk_elementwise.cpu(),
    }
    buffer = io.BytesIO()
    torch.save(tensors, buffer)
    payload = buffer.getvalue()
    if len(payload) > 64 * 1024 * 1024:
        raise ValueError("diagnostic tensor file exceeds 64 MiB")
    basename = "case%d-%s-%s" % (case_id, "prefill" if prefill else "decode",
                                 "dual" if with_global else "swa")
    with (directory / (basename + ".pt")).open("xb") as output:
        output.write(payload)
    comparisons = {
        "actual_vs_accepted": _diagnostic_metrics(cpu_outputs["actual"], cpu_outputs["accepted_bf16"]),
        "actual_vs_gpu_reference": _diagnostic_metrics(cpu_outputs["actual"], cpu_outputs["gpu_fp32_reference"]),
        "accepted_vs_gpu_reference": _diagnostic_metrics(cpu_outputs["accepted_bf16"], cpu_outputs["gpu_fp32_reference"]),
        "decoded_bank_vs_manual_bytes": _diagnostic_metrics(bank_cpu, manual),
        "logits_gpu_vs_cpu64": _diagnostic_metrics(stages_cpu["logits"], cpu64["logits"], live),
        "weights_gpu_vs_cpu64": _diagnostic_metrics(stages_cpu["weights"], cpu64["weights"]),
        "softmax_with_exact_gpu_logits": _diagnostic_metrics(stages_cpu["weights"], cpu_weights_from_gpu_logits),
        "pv_with_exact_gpu_weights": _diagnostic_metrics(stages_cpu["pv"], cpu_pv_from_gpu_weights),
    }
    for name, value in cpu_outputs.items():
        comparisons[name + "_vs_cpu64"] = _diagnostic_metrics(value, cpu64["pv"])
        comparisons[name + "_vs_cpu64_rounded"] = _diagnostic_metrics(value, cpu64_rounded)
    record = {
        "case": case_id, "prefill": prefill, "with_global": with_global, "seed": 5107,
        "qualification_assertion_errors": assertion_errors,
        "historical_rounded_assertion": historical_rounded,
        "reference_correction": "Unrounded CPU FP64 gate; historical BF16-rounded reference is diagnostic only",
        "original_failure_coordinate": [2, 4, 41],
        "original_coordinate_present": rows == 32, "focus_coordinate": list(focus),
        "focus_values": focus_values, "focus_math_fsum_stages": point,
        "focus_bf16_bits": {name: hex(int(value.view(torch.int16)[focus]) & 65535)
                            for name, value in cpu_outputs.items()},
        "focus_indices": live_slots.tolist(), "focus_sink": float(sink_cpu[head]),
        "manual_decoded_bank_exact": bool(torch.equal(bank_cpu.double(), manual)),
        "staged_gpu_reproduces_original": bool(torch.equal(stages_cpu["pv"].to(q_cpu.dtype),
                                                            cpu_outputs["gpu_fp32_reference"])),
        "focus_elementwise_qk_vs_einsum": _diagnostic_metrics(gpu_qk_elementwise.cpu(),
                                                             stages_cpu["logits"][row, head], live[row, 0]),
        "comparisons": comparisons, "cpu_oracle_device": str(cpu64["pv"].device),
        "cpu_oracle_dtype": str(cpu64["pv"].dtype),
        "requested_scale": scale, "effective_native_fp32_scale": scale32,
        "backend_policy": {"float32_matmul_precision": torch.get_float32_matmul_precision(),
                           "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32)},
        "backend_policy_changed": False, "packed_dependency": identity,
        "input_strides": {"q": list(q.stride()), "swa_cache": list(plan.swa_cache.stride()),
                          "global_cache": list(plan.global_cache.stride()) if with_global else None},
        "route_scope": "BF16 Q, primary528/extra288; padded outer strides reject fixed packed_h64; generic V41 dispatch; no Q8/P8 contract",
        "native_intermediates": "NOT_OBSERVED; staged logits/softmax/PV are explicit references only",
        "tensor_file": basename + ".pt", "tensor_bytes": len(payload),
        "tensor_sha256": hashlib.sha256(payload).hexdigest(),
        "root_cause": "B03 oracle final BF16 rounding; any new qualification failure requires separate analysis",
    }
    _diagnostic_json(directory / (basename + ".json"), record)
    return historical_rounded["status"]


def _pool(tag, width, entries, span, table):
    values = torch.randn(3 * entries, 512, device="cuda", dtype=torch.bfloat16)
    packed = encode_kv(values, pool_tag=tag)
    tensor = torch.full((3, entries * width + 16), 0xA5, dtype=torch.uint8, device="cuda")
    pool = NS(tensor=tensor, entries=entries, span=span,
              table=torch.tensor([table], device="cuda", dtype=torch.int32))
    view = packed_pool_view(pool, width)
    view[:, :, 0].copy_(packed.reshape(3, entries, width))
    decoded = torch.empty_like(values)
    decode_kv(packed, pool_tag=tag, out=decoded)
    return pool, decoded


def _assert_csa_ring(test, device):
    from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import SharedNativeCache

    table = torch.tensor([[1, 3, 1], [2, 4, 5]], device=device, dtype=torch.int32)
    tensor = torch.full((6, 2 * 1024 + 16), -777.0, device=device, dtype=torch.float32)
    other = torch.full((6, 2 * 528 + 16), 0xA5, device=device, dtype=torch.uint8)
    pool = NS(tensor=tensor, table=table, entries=2, span=256, framework_tag="csa_state")
    cache = SharedNativeCache.__new__(SharedNativeCache)
    cache.device = device
    cache.geometry = NS(window=2)
    cache.logical = NS(geometry=cache.geometry)
    cache.lengths, cache.writes = (768, 768), []
    cache.pools = {(2, "csa_state"): pool,
                   (2, "swa_kv"): NS(tensor=other, table=table, entries=2, span=256, framework_tag="swa_kv")}
    pointers = tensor.data_ptr(), other.data_ptr()
    other_before = other.clone()
    host_table = [[1, 3, 1], [2, 4, 5]]

    def apply(events, expected):
        slots = torch.tensor(events, device=device, dtype=torch.long)
        rows = (slots[:, 1, None].float() * 2048 + slots[:, 0, None].float() * 4194304
                + torch.arange(1024, device=device, dtype=torch.float32)[None])
        latest = {}
        for index, (rid, position) in enumerate(events):
            key = host_table[rid][position // 256], position % 2
            if key not in latest or position > events[latest[key]][1]:
                latest[key] = index
        for (block, offset), index in latest.items():
            expected[block, offset * 1024:(offset + 1) * 1024] = rows[index]
        cache.writes = []
        cache.insert_rows(2, "csa_state", slots, rows)
        test.assertTrue(torch.equal(tensor.view(torch.uint8), expected.view(torch.uint8)))
        test.assertTrue(torch.equal(other, other_before))
        test.assertEqual((tensor.data_ptr(), other.data_ptr()), pointers)
        for record in cache.writes:
            rid = record["request"]
            test.assertEqual(record["rows"], sum(r == rid for r, _ in events))
            test.assertEqual(record["physical_rows"], sum(events[i][0] == rid for i in latest.values()))

    cases = (
        [(0, p) for p in range(32)],
        [(0, p) for p in range(240, 272)],
        [(0, p) for p in (0, 1, 512, 513, 514, 515)],  # two logical columns alias one physical page
        [(0, p) for p in range(32)] + [(1, p) for p in range(250, 272)],
    )
    for events in cases:
        for ordered in (events, events[::-1], events[::2] + events[1::2][::-1]):
            tensor.fill_(-777)
            expected = tensor.clone()
            apply(ordered, expected)
    # Odd fresh Prefill retains raw30; partial appends retain the even boundary
    # needed by the next odd-position compressor. Compare every FP32 word.
    tensor.fill_(-777)
    expected = tensor.clone()
    apply([(0, p) for p in range(31)], expected)
    expected30 = (30 * 2048 + torch.arange(1024, device=device, dtype=torch.float32))[None]
    test.assertTrue(torch.equal(cache.read_csa(2, 0, torch.tensor([30], device=device)), expected30))
    apply([(0, 31)], expected)
    test.assertTrue(torch.equal(cache.read_csa(2, 0, torch.tensor([30], device=device)), expected30))
    apply([(0, 32)], expected)
    expected32 = (32 * 2048 + torch.arange(1024, device=device, dtype=torch.float32))[None]
    test.assertTrue(torch.equal(cache.read_csa(2, 0, torch.tensor([32], device=device)), expected32))
    # Other tags retain their existing write/log path and native alias.
    slots = torch.tensor([[0, 0], [0, 1], [1, 0]], device=device, dtype=torch.long)
    rows = torch.arange(3 * 528, device=device, dtype=torch.int32).reshape(3, 528).to(torch.uint8)
    expected_other = other.clone()
    for i, (rid, position) in enumerate(slots.tolist()):
        block = host_table[rid][position // 256]
        expected_other[block, (position % 2) * 528:(position % 2 + 1) * 528] = rows[i]
    csa_before = tensor.clone()
    cache.writes = []
    cache.insert_rows(2, "swa_kv", slots, rows)
    test.assertTrue(torch.equal(other, expected_other))
    test.assertTrue(torch.equal(tensor.view(torch.uint8), csa_before.view(torch.uint8)))
    test.assertEqual(cache.writes, [{"layer": 2, "tag": "swa_kv", "request": 0, "rows": 2},
                                   {"layer": 2, "tag": "swa_kv", "request": 1, "rows": 1}])
    test.assertEqual((tensor.data_ptr(), other.data_ptr()), pointers)


class SharedPackedAttentionDevice(unittest.TestCase):
    @torch.inference_mode()
    def test_eager_decode_defaults_to_packed_with_evicted_swa_pages(self):
        swa, _ = _pool("swa_kv", 528, 128, 256, [-1, 1, 2])
        state = NS(
            idle=False,
            prefill=False,
            lengths=(1,),
            attn=NS(is_target_verify=False),
            runtime=NS(
                policy=NS(mode="full", max_rollback_tokens=0),
                model=NS(_is_speculative=False),
            ),
            cache=NS(pools={(0, "swa_kv"): swa}, lengths=(513,), policy=NS()),
        )
        attn = NS(
            layer_id=0,
            kv_source_layer_id=None,
            compress_ratio=0,
            geometry=NS(head_dim=512, window=128, n_heads=64),
            attn_sink=torch.zeros(64, device="cuda"),
        )
        context = NS(
            positions=torch.tensor([512], device="cuda"),
            request_ids=torch.tensor([0], device="cuda"),
            valid_mask=torch.ones(1, device="cuda", dtype=torch.bool),
        )
        x = torch.zeros((1, 512), device="cuda", dtype=torch.bfloat16)

        with patch.dict(os.environ):
            os.environ.pop(PACKED_ENV, None)
            plan, reason = prepare_packed_attention(attn, state, context, x)
            self.assertEqual(reason, "eligible")
            self.assertEqual(plan.swa_cache.data_ptr(), swa.tensor.data_ptr())
            plan.bind_selection(None)
            self.assertEqual(plan.swa_lengths.tolist(), [128])

            swa.table[0, 1] = -1
            _, reason = prepare_packed_attention(attn, state, context, x)
            self.assertEqual(reason, "unallocated_physical_page")

            state.prefill = True
            _, reason = prepare_packed_attention(attn, state, context, x)
            self.assertEqual(reason, "disabled")
            state.prefill = False
            os.environ[PACKED_ENV] = "0"
            _, reason = prepare_packed_attention(attn, state, context, x)
            self.assertEqual(reason, "disabled")

    @torch.inference_mode()
    def test_decode_graph_reads_live_packed_pool_and_lengths(self):
        from rtp_llm.models_py.modules.dsv41.ppu.graph_cache_kernels import position_slots
        from rtp_llm.models_py.modules.dsv41.ppu.graph_packed_attention import GraphPackedAttention

        torch.manual_seed(5107)
        device = torch.device("cuda:0")
        geometry = NS(window=128, head_dim=512, n_heads=64, sm_scale=lambda: 512 ** -0.5)
        stream = torch.cuda.Stream(device=device)
        for ratio in (0, 1, 2):
            with self.subTest(ratio=ratio):
                pools = {}
                decoded = {}
                layouts = [("swa_kv", 528, 130, 256)]
                if ratio:
                    layouts.append((f"global_kv_{ratio}", 288, 128 // ratio, 128))
                for tag, width, entries, span in layouts:
                    pool, values = _pool(tag, width, entries, span, [2, 1] * 5)
                    pool.table = torch.cat((pool.table, pool.table.flip(-1)))
                    pools[tag], decoded[tag] = pool, values

                def slots(owner, tag, positions):
                    pool = pools[tag]
                    return position_slots(
                        positions, pool.table, span=pool.span, entries=pool.entries,
                        ratio=ratio if tag != "swa_kv" else 1, full=tag != "swa_kv",
                        pool_blocks=pool.tensor.shape[0],
                    )

                cache = NS(
                    device=device, geometry=NS(pool_keys=lambda: [(2, tag) for tag in pools]),
                    pool=lambda owner, tag: pools[tag].tensor, slots=slots,
                    _table_tag=lambda owner, tag: tag,
                    entries={tag: pool.entries for tag, pool in pools.items()},
                )
                op = GraphPackedAttention(cache, geometry)
                attn = NS(
                    layer_id=2, index_source_layer_id=2, kv_source_layer_id=2,
                    compress_ratio=ratio, geometry=geometry,
                    attn_sink=torch.zeros(64, device=device),
                )
                q = torch.randn(2, 64, 512, device=device, dtype=torch.bfloat16)
                metadata = NS(
                    positions=torch.tensor([0, 127], device=device),
                    valid_mask=torch.tensor([True, False], device=device),
                )
                selected = torch.arange(512, device=device, dtype=torch.int32).repeat(2, 1) if ratio else None
                graph = torch.cuda.CUDAGraph()
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    op.forward(attn, q, selected, metadata, {})
                torch.cuda.synchronize(device)
                with torch.cuda.graph(graph, stream=stream):
                    actual = op.forward(attn, q, selected, metadata, {})

                before = {tag: pool.tensor.clone() for tag, pool in pools.items()}
                for positions, valid in (([129, 257], [True, True]), ([1025, 0], [True, False]), ([1, 128], [False, True])):
                    q.mul_(-0.75)
                    metadata.positions.copy_(torch.tensor(positions, device=device))
                    metadata.valid_mask.copy_(torch.tensor(valid, device=device))
                    if selected is not None:
                        selected[:, 1] = -1
                    for pool in pools.values():
                        pool.table.copy_(pool.table.flip(-1))
                    default_eager = op.forward(attn, q, selected, metadata, {})
                    stream.wait_stream(torch.cuda.current_stream(device))
                    with torch.cuda.stream(stream):
                        eager = op.forward(attn, q, selected, metadata, {})
                        repeated = op.forward(attn, q, selected, metadata, {})
                        graph.replay()
                        replay = actual.clone()
                        graph.replay()
                    stream.synchronize()
                    # The pinned split-KV provider can vary in the last BF16 bits
                    # across repeated eager calls. Report this separately from
                    # the independent numerical check for every output below.
                    comparisons = {
                        "eager_stream": (default_eager, eager),
                        "eager_repeat": (eager, repeated),
                        "replay_repeat": (replay, actual),
                        "replay_eager": (actual, eager),
                    }
                    for name, (left, right) in comparisons.items():
                        print("packed-replay", ratio, positions, name,
                              int((left != right).sum()), float((left.float() - right.float()).abs().max()),
                              flush=True)

                    bank = torch.cat(tuple(decoded.values()))[:, None]
                    indices = torch.full((2, 1, 640), -1, device=device, dtype=torch.int32)
                    for row, position in enumerate(positions):
                        if not valid[row]:
                            continue
                        swa = pools["swa_kv"]
                        live = [int(swa.table[row, raw // swa.span]) * swa.entries + raw % swa.entries
                                for raw in range(max(0, position - 127), position + 1)]
                        if ratio:
                            extra = pools[f"global_kv_{ratio}"]
                            for key in range(min(512, (position + 1) // ratio)):
                                if key != 1:
                                    raw = key * ratio
                                    live.append(decoded["swa_kv"].shape[0] +
                                                int(extra.table[row, raw // extra.span]) * extra.entries +
                                                raw % extra.span // ratio)
                        indices[row, 0, :len(live)] = torch.tensor(live, device=device, dtype=torch.int32)
                    reference, _ = _attention_cpu64(
                        q.cpu(), bank.cpu(), indices.cpu(), geometry.sm_scale(), attn.attn_sink.cpu(),
                    )
                    outputs = {"default_eager": default_eager, "eager": eager, "repeated": repeated,
                               "first_replay": replay, "second_replay": actual}
                    for name, output in outputs.items():
                        with self.subTest(positions=positions, output=name):
                            torch.testing.assert_close(output.cpu().double(), reference["pv"], rtol=0.005, atol=0.005)
                            self.assertEqual(int(torch.count_nonzero(output[~metadata.valid_mask])), 0)
                for tag, pool in pools.items():
                    self.assertTrue(torch.equal(pool.tensor, before[tag]))

    @torch.inference_mode()
    def test_prefill_decode_and_physical_boundary(self):
        diagnostic_directory = _diagnostic_directory()
        numerical_failures = []
        historical_rounded_failures = 0
        # Missing/old dependency is a hard test error, never a skipped packed
        # branch. This loads the L0-qualified fixed native export family.
        module, identity = load_packed_backend()
        flash_mla_with_kvcache = module.flash_mla_with_kvcache
        get_mla_metadata = module.get_mla_metadata
        self.assertEqual(identity["native_entry"], "sparse_decode_fwd")
        self.assertTrue(torch.cuda.is_available(), "PPU execution required")
        torch.manual_seed(5107)
        swa, srows = _pool("swa_kv", 528, 128, 256, [2, 1])
        glob, grows = _pool("global_kv_2", 288, 64, 128, [2, 1])
        before = (swa.tensor.clone(), glob.tensor.clone())
        # Preflight refusals happen before production writes. No native call
        # is used to discover unsupported modes and no failure is retried.
        state = NS(idle=False, prefill=True, query_prefixes=(0,), lengths=(129,),
                   cache=NS(pools={(2, "swa_kv"): swa, (2, "global_kv_2"): glob},
                            lengths=(129,), policy=NS(ced=False), writes=[]),
                   attn=NS(is_target_verify=False),
                   runtime=NS(policy=NS(mode="full", max_rollback_tokens=0),
                              model=NS(_is_speculative=False)))
        attn = NS(layer_id=2, kv_source_layer_id=2, compress_ratio=2,
                  geometry=NS(head_dim=512, window=128, n_heads=10),
                  attn_sink=torch.zeros(10, device="cuda"))
        context = NS(valid_mask=torch.ones(1, device="cuda", dtype=torch.bool))
        saved_env = os.environ.get(PACKED_ENV)
        try:
            os.environ[PACKED_ENV] = "1"
            candidate, reason = prepare_packed_attention(
                attn, state, context, torch.zeros(1, 512, device="cuda", dtype=torch.bfloat16)
            )
            self.assertIsNone(candidate)
            self.assertEqual(reason, "prefill_swa_not_resident")
            self.assertEqual(state.cache.writes, [])
            state.attn.is_target_verify = True
            _, reason = prepare_packed_attention(attn, state, context, torch.empty(0))
            self.assertEqual(reason, "verify_deferred_n10")
        finally:
            if saved_env is None:
                os.environ.pop(PACKED_ENV, None)
            else:
                os.environ[PACKED_ENV] = saved_env
        for case_id, (prefill, with_global, positions) in enumerate((
            (True, True, torch.arange(32, device="cuda")),
            (False, True, torch.tensor([131], device="cuda")),
            (True, False, torch.arange(32, device="cuda")),
            (False, False, torch.tensor([131], device="cuda")),
        )):
            bank = (torch.cat((grows, srows)) if with_global else srows)[:, None]
            global_count = grows.shape[0] if with_global else 0
            rows = positions.numel()
            requests = torch.zeros(rows, device="cuda", dtype=torch.long)
            valid = torch.ones(rows, device="cuda", dtype=torch.bool)
            q = torch.randn(rows, 10, 512, device="cuda", dtype=torch.bfloat16)
            sink = torch.linspace(-1, 1, 10, device="cuda", dtype=torch.float32)
            plan = PackedAttentionPlan(
                packed_pool_view(swa, 528), packed_pool_view(glob, 288) if with_global else None,
                swa, glob if with_global else None,
                positions, requests, valid, 2 if with_global else 0, 128,
                flash_mla_with_kvcache, get_mla_metadata, backend_identity=identity,
            )
            chosen = torch.arange(512, device="cuda", dtype=torch.int32)[None].expand(rows, -1).clone()
            # A real interior hole tests live-prefix compaction independently
            # of padding width; the direct physical page boundary differs by tag.
            chosen[:, 1] = -1
            plan.bind_selection(chosen if with_global else None)
            expected_s, expected_g = [], []
            for row, position in enumerate(positions.tolist()):
                raw = list(range(max(0, position - 127), position + 1))
                expected_s.append([int(swa.table[0, p // swa.span]) * swa.entries + p % swa.entries for p in raw])
                raw_global = [2 * p for p in range((position + 1) // 2) if p != 1] if with_global else []
                expected_g.append([int(glob.table[0, p // glob.span]) * glob.entries + (p % glob.span) // 2 for p in raw_global])
            indices = torch.full((rows, 1, 640), -1, device="cuda", dtype=torch.int32)
            lengths = torch.empty(rows, device="cuda", dtype=torch.int32)
            for row in range(rows):
                ns, ng = len(expected_s[row]), len(expected_g[row])
                self.assertEqual(plan.swa_indices[row, 0, :ns].tolist(), expected_s[row])
                if with_global:
                    self.assertEqual(plan.global_indices[row, 0, :ng].tolist(), expected_g[row])
                    self.assertEqual(int(plan.global_lengths[row]), ng)
                self.assertEqual(int(plan.swa_lengths[row]), ns)
                reference = expected_g[row] + [p + global_count for p in expected_s[row]]
                indices[row, 0, :len(reference)] = torch.tensor(reference, device="cuda", dtype=torch.int32)
                lengths[row] = len(reference)
            adapter = packed_prefill if prefill else packed_decode
            actual = adapter(q, plan, softmax_scale=512 ** -0.5, attn_sink=sink)
            reference, _, _ = flash_mla_sparse_fwd_ppu(
                q, bank, indices, 512 ** -0.5, attn_sink=sink, topk_length=lengths,
            )
            independent = _attention_torch(q, bank, indices, 512 ** -0.5, sink)
            torch.cuda.synchronize()
            # Reference correction after immutable b03 device diagnosis: the
            # mathematical error budget applies to the unrounded result, not
            # an oracle with an extra BF16 quantization. Native values and
            # .005/.005 parameters are unchanged; old FAILs remain diagnostic.
            case_errors = []
            try:
                torch.testing.assert_close(actual, reference, rtol=0.005, atol=0.005)
            except AssertionError as error:
                if diagnostic_directory is None:
                    raise
                numerical_failures.append((error, error.__traceback__))
                case_errors.append({"comparison": "actual_vs_accepted", "error": str(error)[:8192]})
            scale32 = struct.unpack("=f", struct.pack("=f", 512 ** -0.5))[0]
            mathematical, mathematical_keys = _attention_cpu64(
                q.detach().cpu(), bank.detach().cpu(), indices.detach().cpu(),
                scale32, sink.detach().cpu(),
            )
            try:
                torch.testing.assert_close(actual.cpu().double(), mathematical["pv"], rtol=0.005, atol=0.005)
            except AssertionError as error:
                if diagnostic_directory is None:
                    raise
                numerical_failures.append((error, error.__traceback__))
                case_errors.append({"comparison": "actual_vs_unrounded_cpu64", "error": str(error)[:8192]})
            if diagnostic_directory is not None:
                historical_status = _diagnostic_case(
                    diagnostic_directory, case_id, prefill=prefill, with_global=with_global,
                    q=q, bank=bank, indices=indices, lengths=lengths, sink=sink, plan=plan,
                    actual=actual, reference=reference, independent=independent,
                    assertion_errors=case_errors, identity=identity,
                    cpu64=mathematical, keys_cpu64=mathematical_keys,
                )
                historical_rounded_failures += historical_status == "FAIL"
            self.assertEqual(plan.swa_cache.data_ptr(), swa.tensor.data_ptr())
            self.assertEqual(plan.swa_cache.stride(0), swa.tensor.stride(0))
            if with_global:
                self.assertEqual(plan.global_cache.data_ptr(), glob.tensor.data_ptr())
            else:
                self.assertIsNone(plan.global_cache)
                self.assertIsNone(plan.global_indices)
                self.assertIsNone(plan.global_lengths)
            self.assertTrue(torch.equal(before[0], swa.tensor))
            self.assertTrue(torch.equal(before[1], glob.tensor))
        raw = torch.tensor([[0, 1]], device="cuda")
        skipped = physical_slots(swa, torch.tensor([0], device="cuda"), raw,
                                 ratio=1, full=False, valid=torch.zeros_like(raw, dtype=torch.bool))
        compact, lengths = compact_prefix(skipped)
        self.assertEqual(lengths.tolist(), [0])
        self.assertEqual(compact.tolist(), [[[-1, -1]]])
        # Identical pre-existing packed inputs must produce exactly the same
        # physical store, including scale bytes, when the unused staging copy
        # is removed. This checks the new producer branch independently of
        # packed Attention reduction-order differences in later model layers.
        from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import SharedNativeCache

        swa.framework_tag = "swa_kv"
        cache = SharedNativeCache.__new__(SharedNativeCache)
        cache.device = swa.tensor.device
        cache.geometry = NS(window=128)
        cache.logical = NS(geometry=cache.geometry)
        cache.pools = {(2, "swa_kv"): swa}
        cache.lengths = (132,)
        cache._fresh_swa, cache.writes = {}, []
        fresh = encode_kv(torch.randn(1, 512, device="cuda", dtype=torch.bfloat16), pool_tag="swa_kv")
        position = torch.tensor([131], device="cuda")
        request = torch.zeros_like(position)
        cache.stage_swa(2, position, request, fresh, (131,))
        serial_store, serial_writes = swa.tensor.clone(), list(cache.writes)
        swa.tensor.copy_(before[0])
        cache._fresh_swa, cache.writes = {}, []
        cache.stage_swa(2, position, request, fresh, (131,), retain_rows=False)
        self.assertTrue(torch.equal(swa.tensor, serial_store))
        self.assertEqual(cache.writes, serial_writes)
        self.assertEqual(cache._fresh_swa, {})
        _assert_csa_ring(self, swa.tensor.device)
        if diagnostic_directory is not None:
            _diagnostic_json(diagnostic_directory / "summary.json", {
                "cases_collected": 4, "qualification_numeric_assertion_failures": len(numerical_failures),
                "historical_rounded_assertion_failures": historical_rounded_failures,
                "reference_correction": "CPU FP64 unrounded oracle; original rounded FAIL is not reclassified as PASS",
                "thresholds": {"rtol": 0.005, "atol": 0.005},
                "diagnostics_env": DIAGNOSTICS_ENV, "production_change": False,
                "qualification_failures_rethrown": bool(numerical_failures),
            })
        if numerical_failures:
            error, traceback = numerical_failures[0]
            raise error.with_traceback(traceback)


if __name__ == "__main__":
    unittest.main()
