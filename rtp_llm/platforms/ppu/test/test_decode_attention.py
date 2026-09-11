"""Real-checkpoint Attention Decode/Graph integration on one M890P.

Set RTP_PPU_DSV4_CHECKPOINT to a read-only Flash checkpoint. The cache fixture
implements the framework tensor-view API and native block geometry; this does
not test the C++ allocator, the full transformer, EP8, or SGLang accuracy.
"""

import hashlib
import inspect
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_fmha_impl import (
    DSv4DecodeFmhaImplConfigFP8,
    DSv4DecodeFmhaImplFP8,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.output_proj import decode_output_proj
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
    CSA_KV,
    CSA_STATE,
    HCA_KV,
    HCA_STATE,
    INDEXER_KV,
    INDEXER_STATE,
    SWA_KV,
)
from rtp_llm.platforms.ppu.models.dsv4.ppu_decode_provider import PpuDecodeProvider
from rtp_llm.platforms.ppu.models.dsv4.manifest import DECODE_EXECUTION_OPTIONS
from rtp_llm.platforms.ppu.models.dsv4.ppu_fp4_indexer import PpuFP4Attention
from rtp_llm.utils.model_weight import W


class TensorCache:
    """CP1, MTP-off native-compatible views with a reserved null block."""

    def __init__(self, layer, ratio, batch, max_seq_len=16384):
        self.layer = layer
        # One kernel block covers 256 raw tokens, including 64 FP4 entries.
        self.specs = {SWA_KV: (128, 16384, 1)}
        if ratio:
            kv, state = (CSA_KV, CSA_STATE) if ratio == 4 else (HCA_KV, HCA_STATE)
            self.specs[kv] = (256 // ratio, 256, max_seq_len // 256)
            self.specs[state] = (8 if ratio == 4 else 128, 16384, 1)
        if ratio == 4:
            self.specs[INDEXER_KV] = (64, 256, max_seq_len // 256)
            self.specs[INDEXER_STATE] = (8, 16384, 1)
        self.group_tags = list(self.specs)
        self.tables, self.pools = {}, {}
        state_dims = {CSA_STATE: 2048, HCA_STATE: 1024, INDEXER_STATE: 512}
        for tag, (entries, _, blocks) in self.specs.items():
            count = 1 + batch * blocks
            # Permuted physical blocks catch accidental use of local indices.
            self.tables[tag] = torch.arange(
                batch * blocks, 0, -1, device="cuda", dtype=torch.int32
            ).reshape(batch, blocks)
            if tag in state_dims:
                base = torch.zeros(
                    (count, entries * state_dims[tag]),
                    device="cuda",
                    dtype=torch.float32,
                )
                base.view(count, entries, state_dims[tag])[
                    ..., state_dims[tag] // 2 :
                ].fill_(-float("inf"))
            else:
                stride = entries * (68 if tag == INDEXER_KV else 584)
                if tag != INDEXER_KV:
                    stride = (stride + 575) // 576 * 576
                base = torch.zeros((count, stride), device="cuda", dtype=torch.uint8)
            self.pools[tag] = SimpleNamespace(kv_cache_base=base)

    def get_layer_cache(self, layer, tag):
        if layer != self.layer or tag not in self.pools:
            raise RuntimeError("layer does not own this cache tag")
        return self.pools[tag]

    def get_seq_size_per_block(self, tag):
        return self.specs[tag][1]

    def get_kernel_seq_size_per_block(self, tag):
        return self.specs[tag][1]

    def inputs(self, positions, active, *, stable=False):
        if stable and not hasattr(self, "_input_tables"):
            self._input_tables = {
                tag: torch.empty_like(table) for tag, table in self.tables.items()
            }
        result = {}
        for tag, table in self.tables.items():
            table = self._input_tables[tag].copy_(table) if stable else table.clone()
            table[active:].fill_(-1)
            result[tag] = SimpleNamespace(
                sequence_lengths=positions,
                is_prefill=False,
                is_target_verify=False,
                kv_cache_kernel_block_id_device=table,
            )
        return result

    def clone_contents(self):
        return {tag: view.kv_cache_base.clone() for tag, view in self.pools.items()}

    def restore(self, contents):
        for tag, tensor in contents.items():
            self.pools[tag].kv_cache_base.copy_(tensor)


def inject_long_history(cache, reference):
    """Seed valid nonzero packed history for the address-only long probes.

    Leaving skipped history zero creates thousands of tied Indexer scores, so
    unordered SG TopK can legitimately choose different sets. Distinct payloads
    also make wrong physical-block addresses observable. Block zero is reserved.
    """
    for tag in (CSA_KV, HCA_KV, INDEXER_KV):
        if tag not in cache.pools:
            continue
        raw = cache.pools[tag].kv_cache_base[1:]
        entries = cache.specs[tag][0]
        if tag == INDEXER_KV:
            # A page stores all FP4 payloads, followed by all UE8M0 scales.
            # The [page, entry, 68] public shape does not interleave them.
            raw[:, : entries * 64].random_(0, 256)
            raw[:, entries * 64 : entries * 68].fill_(127)
        else:
            rows = raw[:, : entries * 576].view(-1, entries, 576)
            nope = torch.randn(
                (*rows.shape[:-1], 448), device=raw.device, dtype=torch.bfloat16
            ).to(torch.float8_e4m3fn)
            rope = torch.randn(
                (*rows.shape[:-1], 64), device=raw.device, dtype=torch.bfloat16
            )
            rows[..., :448].copy_(nope.view(torch.uint8))
            rows[..., 448:].copy_(rope.view(torch.uint8))
            raw[:, entries * 576 : entries * 584].fill_(127)
        reference.pools[tag].kv_cache_base.copy_(cache.pools[tag].kv_cache_base)


def load_attention(checkpoint, layer, max_batch):
    from safetensors import safe_open

    config = json.loads((checkpoint / "config.json").read_text())
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    ratio = config["compress_ratios"][layer]
    names = {}
    for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
        names[f"attn.{name}.weight"] = getattr(W, f"v4_attn_{name}_w")
        names[f"attn.{name}.scale"] = getattr(W, f"v4_attn_{name}_s")
    for name in ("q_norm", "kv_norm"):
        names[f"attn.{name}.weight"] = getattr(W, f"v4_attn_{name}")
    names["attn.attn_sink"] = W.v4_attn_sink
    for inner in (False, True) if ratio == 4 else ((False,) if ratio else ()):
        prefix = "indexer_" if inner else ""
        ckpt_prefix = "attn.indexer.compressor" if inner else "attn.compressor"
        for name in ("wkv", "wgate", "norm", "ape"):
            names[f"{ckpt_prefix}.{name}" + ("" if name == "ape" else ".weight")] = (
                getattr(W, f"v4_{prefix}compressor_{name}")
            )
    if ratio == 4:
        names.update(
            {
                "attn.indexer.wq_b.weight": W.v4_indexer_wq_b_w,
                "attn.indexer.wq_b.scale": W.v4_indexer_wq_b_s,
                "attn.indexer.weights_proj.weight": W.v4_indexer_weights_proj_w,
            }
        )
    weights, hashes = {}, {}
    for suffix, tag in names.items():
        key = f"layers.{layer}.{suffix}"
        with safe_open(checkpoint / index[key], framework="pt", device="cpu") as handle:
            value = handle.get_tensor(key)
        hashes[key] = hashlib.sha256(
            value.view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        if tag == W.v4_indexer_compressor_norm:
            value = value.float()
        weights[tag] = value.cuda()
    rope = config["rope_scaling"]
    provider = PpuDecodeProvider(DECODE_EXECUTION_OPTIONS)
    attn = provider.build_attention(
        PpuFP4Attention,
        layer_id=layer,
        dim=config["hidden_size"],
        n_heads=config["num_attention_heads"],
        q_lora_rank=config["q_lora_rank"],
        head_dim=config["head_dim"],
        rope_head_dim=config["qk_rope_head_dim"],
        o_lora_rank=config["o_lora_rank"],
        o_groups=config["o_groups"],
        window_size=config["sliding_window"],
        compress_ratio=ratio,
        compress_rope_theta=config["compress_rope_theta"],
        rope_theta=config["rope_theta"],
        rope_factor=rope["factor"],
        beta_fast=rope["beta_fast"],
        beta_slow=rope["beta_slow"],
        original_seq_len=rope["original_max_position_embeddings"],
        max_batch_size=max_batch,
        max_seq_len=16384,
        index_n_heads=config["index_n_heads"],
        index_head_dim=config["index_head_dim"],
        index_topk=config["index_topk"],
        norm_eps=config["rms_norm_eps"],
        layer_weights=weights,
    )
    attn.reset_rope_cache(torch.device("cuda"))
    return attn, hashes


class AttentionTrace:
    """Observe actual MLA inputs and output before in-place inverse RoPE."""

    def __init__(self, attn):
        self.values = {}
        self.arguments = {}
        self.preparation_streams = {}
        for name, role in (
            ("_decode_write_swa_fp8", "kv"),
            ("_decode_update_compressor", "compressor"),
            ("_decode_update_indexer", "indexer"),
        ):
            method = getattr(attn, name)

            def observe(*args, _method=method, _role=role, **kwargs):
                self.preparation_streams[_role] = (
                    torch.cuda.current_stream().cuda_stream,
                    torch.cuda.is_current_stream_capturing(),
                )
                return _method(*args, **kwargs)

            setattr(attn, name, observe)
        if attn.indexer is not None:
            for owner, name, role in (
                (attn.indexer, "_compute_indexer_q", "indexer_q"),
                (
                    attn.indexer.compressor,
                    "forward_decode_vectorized",
                    "indexer_compressor",
                ),
            ):
                method = getattr(owner, name)

                def observe_inner(*args, _method=method, _role=role, **kwargs):
                    self.preparation_streams[_role] = (
                        torch.cuda.current_stream().cuda_stream,
                        torch.cuda.is_current_stream_capturing(),
                    )
                    return _method(*args, **kwargs)

                setattr(owner, name, observe_inner)
        op = attn._get_fp8_decode_op()
        original = op.forward
        self.original = original
        signature = inspect.signature(original)

        def traced(*args, **kwargs):
            arguments = signature.bind(*args, **kwargs).arguments
            self.arguments.update(arguments)
            for name in (
                "q",
                "attn_sink",
                "topk_idxs",
                "topk_length",
                "extra_topk_idxs",
                "extra_topk_length",
            ):
                value = arguments.get(name)
                if value is not None:
                    self.values[name] = value.clone()
            output = original(*args, **kwargs)
            self.values["output"] = output.clone()
            return output

        op.forward = traced

    def begin(self):
        self.values = {}
        self.arguments = {}
        return self.values


def bf16_ulp_distance(a, b):
    """Distance between finite BF16 values, treating signed zero equally."""
    assert a.dtype == b.dtype == torch.bfloat16
    ai = a.contiguous().view(torch.int16).int()
    bi = b.contiguous().view(torch.int16).int()
    ao = torch.where(ai < 0, -32768 - ai, ai)
    bo = torch.where(bi < 0, -32768 - bi, bi)
    return (ao - bo).abs()


@unittest.skipUnless(
    os.environ.get("RTP_PPU_DSV4_CHECKPOINT")
    and torch.cuda.is_available()
    and torch.cuda.get_device_name() == "ZW-M890P",
    "requires a read-only Flash checkpoint and a PPU M890P",
)
class DecodeAttentionTest(unittest.TestCase):
    @torch.inference_mode()
    def test_checkpoint_attention_graph_and_state(self):
        self._check_checkpoint_attention(shared_rope=False)

    @torch.inference_mode()
    def test_checkpoint_attention_with_shared_rope(self):
        self._check_checkpoint_attention(shared_rope=True)

    def _check_checkpoint_attention(self, *, shared_rope):
        torch.manual_seed(890412)
        batches = tuple(
            int(n)
            for n in os.environ.get("RTP_PPU_ATTN_BATCHES", "1,3,8,32,128").split(",")
        )
        reports = []
        overlap = True
        for layer in (0, 2, 3):
            attn, hashes = load_attention(
                Path(os.environ["RTP_PPU_DSV4_CHECKPOINT"]), layer, max(batches)
            )
            trace = AttentionTrace(attn)
            for batch in batches:
                with self.subTest(layer=layer, batch=batch):
                    print(f"ATTENTION_START layer={layer} batch={batch}", flush=True)
                    cache = TensorCache(layer, attn.compress_ratio, batch)
                    reference = TensorCache(layer, attn.compress_ratio, batch)
                    initial = reference.clone_contents()
                    positions = torch.full(
                        (batch,), 127, device="cuda", dtype=torch.int32
                    )
                    inputs = torch.randn(
                        (batch, 1, 4096), device="cuda", dtype=torch.bfloat16
                    )
                    config = DSv4DecodeFmhaImplConfigFP8(
                        max_batch_size=batch,
                        q_len=1,
                        window_size=128,
                        head_dim=512,
                        max_seq_len=16384,
                        compress_ratios=[attn.compress_ratio],
                        index_topk=512,
                        paged_pool_specs=cache.specs,
                        group_tags=cache.group_tags,
                    )
                    tagged = cache.inputs(positions, batch, stable=shared_rope)
                    impl = (
                        attn._platform_provider.build_decode_metadata(
                            DSv4DecodeFmhaImplFP8, config, torch.device("cuda"), tagged
                        )
                        if shared_rope
                        else DSv4DecodeFmhaImplFP8(
                            config, torch.device("cuda"), tagged[SWA_KV]
                        )
                    )
                    if shared_rope:
                        self.assertEqual(len(impl.metadata.rope_freqs_by_source), 1)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            attn.forward_decode(inputs, impl.metadata, cache)
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    graph_trace = trace.begin()
                    with torch.cuda.graph(graph, stream=stream):
                        actual = attn.forward_decode(inputs, impl.metadata, cache)
                    if overlap:
                        roles = ["kv"]
                        if attn.compress_ratio:
                            roles.append("compressor")
                        if attn.indexer is not None:
                            roles.append("indexer")
                        for role in roles:
                            self.assertEqual(
                                trace.preparation_streams[role],
                                (attn._decode_streams[role].cuda_stream, True),
                            )
                            self.assertNotEqual(
                                attn._decode_streams[role].cuda_stream,
                                stream.cuda_stream,
                            )
                    if attn._decode_indexer_streams is not None:
                        self.assertEqual(
                            trace.preparation_streams["indexer_q"],
                            (attn._decode_indexer_streams["q"].cuda_stream, True),
                        )
                        self.assertEqual(
                            trace.preparation_streams["indexer_compressor"],
                            (attn._decode_streams["indexer"].cuda_stream, True),
                        )
                    graph_arguments = trace.arguments
                    torch.cuda.current_stream().wait_stream(stream)
                    cache.restore(initial)
                    # Contiguous history starts empty and crosses both C4 and
                    # C128 boundaries. Long-position probes use injected cache
                    # state and only qualify address/schedule replay, not logits.
                    cases = [(p, batch) for p in range(132)]
                    cases += [
                        (p, active)
                        for p, active in (
                            (8190, batch),
                            (8191, batch),
                            (8192, batch),
                            (1022, max(0, batch - 1)),
                            (1023, max(0, batch - 1)),
                            (1024, batch),
                            (127, 0),
                            (128, max(1, batch // 2)),
                        )
                    ]
                    max_mla_ulp = 0
                    max_output_abs = 0.0
                    different_steps = 0
                    kernel_repeat_checks = 0
                    max_mla_abs = 0.0
                    topk_permutations = 0
                    for step, (position, active) in enumerate(cases):
                        if step == 132:
                            inject_long_history(cache, reference)
                        positions.fill_(position)
                        positions[active:].zero_()
                        inputs.normal_()
                        tagged = cache.inputs(positions, active, stable=shared_rope)
                        impl.prepare_cuda_graph(tagged)
                        if shared_rope:
                            self.assertTrue(
                                torch.equal(
                                    impl.metadata.rope_freqs_by_source[
                                        id(attn.freqs_cis)
                                    ],
                                    attn.freqs_cis.index_select(
                                        0, impl.metadata.position_ids_long
                                    ),
                                )
                            )
                        eager = DSv4DecodeFmhaImplFP8(
                            config, torch.device("cuda"), tagged
                        )
                        eager.prepare_cuda_graph(tagged)
                        eager_trace = trace.begin()
                        # The reference always uses the original sequential
                        # schedule, including when testing overlapping replay.
                        with patch.object(attn, "_decode_streams", None):
                            expected = attn.forward_decode(
                                inputs, eager.metadata, reference
                            )
                        actual.fill_(float("nan"))
                        graph.replay()
                        self.assertTrue(
                            bool(torch.isfinite(actual).all()), (layer, batch, step)
                        )
                        self.assertEqual(graph_trace.keys(), eager_trace.keys())
                        for name in graph_trace.keys() - {"output"}:
                            observed, wanted = graph_trace[name], eager_trace[name]
                            if name.endswith("topk_idxs"):
                                # SG TopK is unordered; differing permutations
                                # still select exactly the same KV entries.
                                topk_permutations += not torch.equal(observed, wanted)
                                observed = observed.sort(dim=-1).values
                                wanted = wanted.sort(dim=-1).values
                            torch.testing.assert_close(
                                observed,
                                wanted,
                                rtol=0,
                                atol=0,
                                msg=f"MLA {name} L{layer} B{batch} step{step}",
                            )
                        self.assertTrue(
                            bool(torch.isfinite(graph_trace["output"]).all())
                        )
                        self.assertTrue(
                            bool(torch.isfinite(eager_trace["output"]).all())
                        )
                        ulp = int(
                            bf16_ulp_distance(
                                graph_trace["output"], eager_trace["output"]
                            ).max()
                        )
                        max_mla_ulp = max(max_mla_ulp, ulp)
                        max_mla_abs = max(
                            max_mla_abs,
                            float(
                                (
                                    graph_trace["output"].float()
                                    - eager_trace["output"].float()
                                )
                                .abs()
                                .max()
                            ),
                        )
                        diff = (actual.float() - expected.float()).abs()
                        max_output_abs = max(max_output_abs, float(diff.max()))
                        different_steps += not torch.equal(actual, expected)
                        outside_repeat_ulp = 0
                        if ulp > 1 or not torch.equal(
                            graph_trace.get(
                                "extra_topk_idxs", graph_trace["topk_idxs"]
                            ),
                            eager_trace.get(
                                "extra_topk_idxs", eager_trace["topk_idxs"]
                            ),
                        ):
                            # Near-zero reductions can span many BF16 ULPs. Run
                            # the same real kernel with the captured Q, indices,
                            # cache and planner, without changing graph inputs.
                            # This tests replay consistency, not math accuracy.
                            kernel_repeat_checks += 1
                            low = high = trace.original(**graph_arguments).clone()
                            for _ in range(15):
                                repeated = trace.original(**graph_arguments)
                                self.assertTrue(bool(torch.isfinite(repeated).all()))
                                low = torch.minimum(low, repeated)
                                high = torch.maximum(high, repeated)
                            nearest = torch.maximum(
                                low, torch.minimum(high, graph_trace["output"])
                            )
                            outside_repeat_ulp = int(
                                bf16_ulp_distance(graph_trace["output"], nearest).max()
                            )
                        if outside_repeat_ulp > 1:
                            diagnostics = os.environ.get("RTP_PPU_ATTN_DIAGNOSTICS")
                            if diagnostics:
                                target = (
                                    Path(diagnostics)
                                    / f"attention-L{layer}-B{batch}-step{step}.pt"
                                )
                                torch.save(
                                    {
                                        "input": inputs.cpu(),
                                        "positions": positions.cpu(),
                                        "actual": actual.cpu(),
                                        "expected": expected.cpu(),
                                        "cache": {
                                            k: v.kv_cache_base.cpu()
                                            for k, v in cache.pools.items()
                                        },
                                        "reference": {
                                            k: v.kv_cache_base.cpu()
                                            for k, v in reference.pools.items()
                                        },
                                        "active": active,
                                        "graph_trace": {
                                            k: v.cpu() for k, v in graph_trace.items()
                                        },
                                        "eager_trace": {
                                            k: v.cpu() for k, v in eager_trace.items()
                                        },
                                    },
                                    target,
                                )
                            print(
                                "ATTENTION_DIFF "
                                + json.dumps(
                                    {
                                        "layer": layer,
                                        "batch": batch,
                                        "step": step,
                                        "mla_max_bf16_ulp": ulp,
                                        "outside_repeat_ulp": outside_repeat_ulp,
                                        "max_abs": diff.max().item(),
                                        "mean_abs": diff.mean().item(),
                                        "different": (actual != expected).sum().item(),
                                        "total": actual.numel(),
                                    }
                                ),
                                flush=True,
                            )
                        # Bound against observed same-kernel variation instead
                        # of increasing the final-output tolerance. One adjacent
                        # BF16 value at a sampled interval edge allows rounding.
                        self.assertLessEqual(
                            outside_repeat_ulp, 1, (layer, batch, step, ulp)
                        )
                        freqs = attn.freqs_cis.index_select(
                            0, impl.metadata.position_ids[:batch].long()
                        )
                        projected = decode_output_proj(
                            attn, graph_trace["output"].clone(), freqs, batch, 1
                        )
                        torch.testing.assert_close(actual, projected, rtol=0, atol=0)
                        for tag, view in cache.pools.items():
                            torch.testing.assert_close(
                                view.kv_cache_base,
                                reference.pools[tag].kv_cache_base,
                                rtol=0,
                                atol=0,
                                msg=f"cache {tag} L{layer} B{batch} step{step}",
                            )
                            # Unallocated rows must never write reserved block 0.
                            torch.testing.assert_close(
                                view.kv_cache_base[0], initial[tag][0], rtol=0, atol=0
                            )
                        self.assertIsNone(attn._kv_cache)
                        if attn.indexer is not None:
                            self.assertIsNone(attn.indexer._kv_pool_view)
                            self.assertIsNone(attn.indexer.compressor._state_pool_3d)
                    reports.append(
                        {
                            "layer": layer,
                            "batch": batch,
                            "overlap": overlap,
                            "steps": len(cases),
                            "max_mla_bf16_ulp": max_mla_ulp,
                            "max_mla_abs": max_mla_abs,
                            "kernel_repeat_checks": kernel_repeat_checks,
                            "topk_permutations": topk_permutations,
                            "max_output_abs": max_output_abs,
                            "different_output_steps": different_steps,
                            "checkpoint_tensor_sha256": hashes,
                        }
                    )
                    print(
                        f"ATTENTION_PASS layer={layer} batch={batch} steps={len(cases)}",
                        flush=True,
                    )
                    del graph, actual, expected, eager, impl, cache, reference, initial
            del attn
        print("ATTENTION_RESULT " + json.dumps(reports, sort_keys=True), flush=True)


if __name__ == "__main__":
    unittest.main()
