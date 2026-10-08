"""Upstream V4.1 owner: eager short prefill and native tagged cache views."""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from rtp_llm.models.dsv41_kv_cache import dsv41_page_layouts
from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_rope_triton import fused_rmsnorm_rope
from rtp_llm.models_py.modules.dsv4.fp8._v41_prefill_attention_sm120 import (
    sparse_prefill_attention,
)
from rtp_llm.models_py.modules.dsv4.fp8.attention import bind_attn_cache
from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import (
    AttentionV41FP8,
    rms_norm,
    rope_only,
)
from rtp_llm.models_py.modules.dsv4.prefill_workspace import PrefillWorkspace
from rtp_llm.models_py.modules.dsv4.rope import apply_rotary_emb
from rtp_llm.models_py.modules.dsv4.v41_cache import V41PagedCache
from rtp_llm.ops.compute_ops import LayerKVCache
from rtp_llm.utils.model_weight import W


class NativeTaggedCache:
    """Small native LayerKVCache allocations, without an engine scheduler."""

    def __init__(self, layers, gen=1, padding=128, pages=4):
        self.groups = {}
        self.storage = {}
        self.layouts = dsv41_page_layouts(gen)
        for layer in layers:
            tags = ["swa_kv"]
            if layer.is_kv_source:
                tags += [layer._global_region(), "indexer_kv"]
                if layer.compress_ratio == 2:
                    tags += ["csa_state"]
            for tag in tags:
                layout = self.layouts[tag]
                gap = 0 if tag == "csa_state" else padding
                raw = torch.full(
                    (pages, layout.stride_bytes + gap),
                    173,
                    device="cuda",
                    dtype=torch.uint8,
                )
                base = raw[:, : layout.stride_bytes]
                base.zero_()
                key = layer.cache_layer_id, tag
                self.storage[key] = raw
                self.groups[key] = LayerKVCache(
                    base, 256, layer.cache_layer_id, len(self.groups), tag
                )

    def get_layer_cache(self, layer, tag):
        return self.groups[layer, tag]

    def get_seq_size_per_block(self, tag):
        return 256

    def get_kernel_seq_size_per_block(self, tag):
        return 256


def make_layer(layer_id=0, ratio=0, shared=None, source=None, gen=1):
    shared = {} if shared is None else shared
    source = layer_id if source is None else source
    weights = {}
    for name, n, k in (
        ("wq_a", 256, 512),
        ("wq_b", 2048, 256),
        ("wkv", 512, 512),
        ("wo_a", 512, 1024),
        ("wo_b", 512, 512),
    ):
        weights[getattr(W, f"v4_attn_{name}_w")] = (
            torch.randn(n, k, device="cuda") * 0.025
        ).to(torch.float8_e4m3fn)
        weights[getattr(W, f"v4_attn_{name}_s")] = torch.ones(
            n // 32, k // 32, device="cuda"
        ).to(torch.float8_e8m0fnu)
    weights[W.v4_attn_q_norm] = torch.ones(256, device="cuda", dtype=torch.bfloat16)
    weights[W.v4_attn_kv_norm] = torch.ones(512, device="cuda", dtype=torch.bfloat16)
    weights[W.v4_attn_sink] = torch.linspace(-1, 1, 4, device="cuda")
    if ratio and layer_id == source:
        for key in (W.v4_compressor_wkv, W.v4_compressor_wgate):
            weights[key] = (torch.randn(512, 512, device="cuda") * 0.025).bfloat16()
        weights[W.v4_compressor_norm] = torch.ones(
            512, device="cuda", dtype=torch.bfloat16
        )
        weights[W.v41_indexer_wk] = (
            torch.randn(128, 512, device="cuda") * 0.025
        ).bfloat16()
        weights[W.v41_indexer_k_norm] = torch.ones(
            128, device="cuda", dtype=torch.bfloat16
        )
        weights[W.v4_indexer_wq_b_w] = torch.randn(4096, 256, device="cuda").to(
            torch.float8_e4m3fn
        )
        weights[W.v4_indexer_wq_b_s] = torch.ones(128, 8, device="cuda").to(
            torch.float8_e8m0fnu
        )
        weights[W.v4_indexer_weights_proj_w] = torch.ones(
            32, 512, device="cuda", dtype=torch.bfloat16
        )
    layer = AttentionV41FP8(
        layer_id=layer_id,
        cache_layer_id=layer_id - source,
        dim=512,
        n_heads=4,
        q_lora_rank=256,
        head_dim=512,
        rope_head_dim=64,
        o_lora_rank=256,
        o_groups=2,
        window_size=128,
        compress_ratio=ratio,
        compress_rope_theta=160000,
        rope_theta=10000,
        rope_factor=16,
        beta_fast=32,
        beta_slow=1,
        original_seq_len=65536,
        max_batch_size=2,
        max_seq_len=512,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=512,
        norm_eps=1e-20,
        layer_weights=weights,
        v41_config={
            "kv_source_layer_ids": [source] if ratio else [],
            "index_source_layer_ids": [source] if ratio else [],
        },
        shared_attention=shared,
        gen_num_per_cycle=gen,
    )
    layer.reset_rope_cache(torch.device("cuda"))
    return layer


def prepare(layer, x, cache, lengths, prefixes=None, tables=None):
    prefixes = [0] * len(lengths) if prefixes is None else prefixes
    device = x.device
    lens = torch.tensor(lengths, device=device, dtype=torch.int32)
    pref = torch.tensor(prefixes, device=device, dtype=torch.int32)
    positions = torch.cat(
        [torch.arange(p, p + n, device=device) for p, n in zip(prefixes, lengths)]
    )
    ids = torch.repeat_interleave(
        torch.arange(len(lengths), device=device, dtype=torch.int32), lens
    )
    cu = torch.cat(
        (torch.zeros(1, device=device, dtype=torch.int32), lens.cumsum(0).int())
    )
    if tables is None:
        tables = {
            tag: torch.tensor(
                [[2], [1]][: len(lengths)], device=device, dtype=torch.int32
            )
            for tag in cache.layouts
        }
    with bind_attn_cache(layer, cache, tables):
        meta = layer._build_shared_prefill_meta(
            x,
            positions,
            sp_per_req=pref.long(),
            cu_seqlens=cu,
            batch_size=len(lengths),
            input_lengths=lens,
            prefix_lengths=pref,
            position_ids=positions,
            req_id_per_token=ids,
            max_seqlen_q=max(lengths),
        )
    workspace = PrefillWorkspace(
        device,
        q_rows=len(x),
        q_dim=layer.n_heads * 512,
        reserve_cp=False,
        align_bytes=512,
    )
    meta = meta._replace(workspace=workspace)
    layer._set_prefill_meta_shared(meta)
    return positions, tables, meta


def dense_oracle(q, kv, indices, lengths, sinks):
    q, kv, indices, lengths, sinks = (t.cpu() for t in (q, kv, indices, lengths, sinks))
    result = torch.empty_like(q)
    for row in range(len(q)):
        ids = indices[row].flatten()[: int(lengths[row])]
        values = kv[ids[ids >= 0], 0].double()
        logits = q[row].double() @ values.T * (512**-0.5)
        probability = torch.cat((logits, sinks.double()[:, None]), -1).softmax(-1)
        result[row] = (probability[:, :-1] @ values).bfloat16()
    return result


def contraction_oracle(q, kv, indices, lengths, sinks):
    """CPU online softmax with the provider's BF16 probability boundary."""
    q, kv, indices, lengths, sinks = (t.cpu() for t in (q, kv, indices, lengths, sinks))
    result = torch.empty_like(q)
    for row in range(len(q)):
        ids = indices[row].flatten()
        maximum = torch.full_like(sinks, -torch.inf)
        total = torch.zeros_like(sinks)
        acc = torch.zeros_like(q[row], dtype=torch.float32)
        for begin in range(0, len(ids), 32):
            slots = ids[begin : begin + 32]
            valid = (slots >= 0) & (
                torch.arange(begin, begin + len(slots)) < lengths[row]
            )
            values = kv[slots.clamp_min(0), 0].float()
            scores = q[row].float() @ values.T * (512**-0.5)
            scores.masked_fill_(~valid[None], -torch.inf)
            next_max = torch.maximum(maximum, scores.amax(-1))
            finite = torch.where(torch.isneginf(next_max), 0, next_max)
            alpha = (maximum - finite).exp()
            probabilities = (scores - finite[:, None]).exp()
            acc = acc * alpha[:, None] + probabilities.bfloat16().float() @ values
            total = total * alpha + probabilities.sum(-1)
            maximum = next_max
        next_max = torch.maximum(maximum, sinks)
        alpha = (maximum - next_max).exp()
        denominator = total * alpha + (sinks - next_max).exp()
        result[row] = (acc * (alpha / denominator)[:, None]).bfloat16()
    return result


def output_composition(layer, attn, freqs):
    apply_rotary_emb(attn[..., -64:].unsqueeze(0), freqs, inverse=True)
    grouped = attn.reshape(len(attn), layer.n_groups, -1)
    projection = torch.stack(
        [op(grouped[:, g].contiguous()) for g, op in enumerate(layer._wo_a_groups_v41)],
        1,
    )
    return layer.wo_b(projection.flatten(1))


class AttentionOwnerSM120Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4112)
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", previous)

    @torch.no_grad()
    def test_bf16_provider_against_double_oracle(self):
        for heads, width in ((1, 17), (4, 128), (64, 640)):
            q = torch.randn(7, heads, 512, device="cuda", dtype=torch.bfloat16)
            kv = torch.randn(769, 1, 512, device="cuda", dtype=torch.bfloat16)
            ids = torch.randint(0, 769, (7, 1, width), device="cuda", dtype=torch.int32)
            ids[:, :, 3::5] = -1
            lengths = torch.tensor(
                [0, 1, 7, width // 2, width - 1, width, width],
                device="cuda",
                dtype=torch.int32,
            )
            sinks = torch.linspace(-2, 3, heads, device="cuda")
            actual = sparse_prefill_attention(q, kv, ids, lengths, sinks)
            expected = dense_oracle(q, kv, ids, lengths, sinks)
            torch.testing.assert_close(actual.cpu(), expected, rtol=0.01, atol=0.004)
            self.assertTrue(bool(torch.isfinite(actual).all()))

    @torch.no_grad()
    def test_swa_owner_full_forward_and_native_cache_bytes(self):
        layer = make_layer()
        cache = NativeTaggedCache([layer])
        for lengths in ([7], [5, 3], [137]):
            x = torch.randn(
                sum(lengths), layer.dim, device="cuda", dtype=torch.bfloat16
            )
            positions, tables, meta = prepare(layer, x, cache, lengths)
            qr = rms_norm(layer.wq_a(x), layer.q_norm, layer.eps)
            q = rope_only(
                layer.wq_b(qr).view(-1, layer.n_heads, 512), meta.freqs_cis, 64
            )
            kv = fused_rmsnorm_rope(
                layer.wkv(x).unsqueeze(0),
                layer.kv_norm,
                meta.freqs_cis,
                64,
                eps=layer.eps,
            ).squeeze(0)
            topk = meta.topk_idxs.reshape(len(x), 1, -1).int()
            attn = contraction_oracle(
                q,
                kv.unsqueeze(1),
                topk,
                meta.swa_meta.topk_length_kv_full,
                layer.attn_sink,
            ).cuda()
            expected = output_composition(layer, attn, meta.freqs_cis)
            expected_cache = V41PagedCache(
                cache.groups[0, "swa_kv"].kv_cache_base.clone(), "swa_kv"
            )
            expected_cache.write(kv, meta.swa_meta.slot_mapping)
            actual = layer(x, positions, cache, tables)
            torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.003)
            torch.testing.assert_close(
                cache.groups[0, "swa_kv"].kv_cache_base,
                expected_cache.raw,
                rtol=0,
                atol=0,
            )
            self.assertIsNone(layer._kv_cache)
            self.assertTrue(
                bool(
                    (
                        cache.storage[0, "swa_kv"][
                            :, cache.layouts["swa_kv"].stride_bytes :
                        ]
                        == 173
                    ).all()
                )
            )

    @torch.no_grad()
    def test_global_producer_consumer_and_partial_pair(self):
        for ratio, source in ((2, 2), (1, 20)):
            shared = {}
            producer = make_layer(source, ratio, shared)
            consumer = make_layer(source + 1, ratio, shared, source=source)
            cache = NativeTaggedCache([producer, consumer])
            history = []
            prefix = 0
            for count in (3, 4, 130):
                x = torch.randn(count, 512, device="cuda", dtype=torch.bfloat16)
                history.append(x)
                positions, tables, _ = prepare(producer, x, cache, [count], [prefix])
                output = producer(x, positions, cache, tables)
                self.assertTrue(bool(torch.isfinite(output).all()))
                keys = shared["global"][source]
                selected = shared["topk"][source]
                self.assertEqual(keys[0][0].shape, ((prefix + count) // ratio, 512))
                expected_ids = torch.arange(512, device="cuda")[None, :].expand(
                    count, -1
                )
                expected_ids = torch.where(
                    expected_ids < ((positions + 1) // ratio)[:, None], expected_ids, -1
                ).int()
                torch.testing.assert_close(selected, expected_ids, rtol=0, atol=0)
                saved = {
                    key: value.kv_cache_base.clone()
                    for key, value in cache.groups.items()
                    if key[0] == 0
                }
                cpos, ctables, _ = prepare(consumer, x, cache, [count], [prefix])
                result = consumer(x, cpos, cache, ctables)
                self.assertTrue(bool(torch.isfinite(result).all()))
                self.assertIs(shared["global"][source], keys)
                self.assertIs(shared["topk"][source], selected)
                for key, raw in saved.items():
                    torch.testing.assert_close(
                        cache.groups[key].kv_cache_base, raw, rtol=0, atol=0
                    )
                full = torch.cat(history)
                values = full.float() @ producer.global_wkv.float().T
                if ratio == 2:
                    scores = full.float() @ producer.global_wgate.float().T
                    paired = values[: len(full) // 2 * 2].reshape(-1, 2, 512)
                    weights = (
                        scores[: len(full) // 2 * 2].reshape(-1, 2, 512).softmax(1)
                    )
                    latent = rms_norm(
                        (paired * weights).sum(1), producer.global_norm, producer.eps
                    ).bfloat16()
                    with bind_attn_cache(producer, cache, tables):
                        state = producer._read_state(
                            positions[-1:],
                            torch.zeros(1, device="cuda", dtype=torch.long),
                        )
                    torch.testing.assert_close(
                        state[0],
                        torch.cat((values[-1], scores[-1])),
                        rtol=1e-5,
                        atol=1e-5,
                    )
                else:
                    latent = rms_norm(
                        values, producer.global_norm, producer.eps
                    ).bfloat16()
                freq = producer.freqs_cis[
                    torch.arange(len(latent), device="cuda") * ratio
                ]
                global_kv = rope_only(latent.clone(), freq, 64)
                reference_cache = V41PagedCache(
                    torch.zeros_like(
                        cache.groups[0, producer._global_region()].kv_cache_base
                    ),
                    producer._global_region(),
                )
                slots = 2 * reference_cache.layout.entries + torch.arange(
                    len(latent), device="cuda"
                )
                reference_cache.write(global_kv, slots)
                expected_kv = reference_cache.read(slots)
                torch.testing.assert_close(keys[0][0], expected_kv, rtol=0, atol=0)
                prefix += count

    @torch.no_grad()
    def test_compressed_owner_numerics_and_canonical_fallback(self):
        for ratio, source in ((2, 2), (1, 20)):
            layer = make_layer(source, ratio)
            cache = NativeTaggedCache([layer])
            for lengths, prefixes in (([5, 3], [0, 0]), ([8, 7], [5, 3])):
                x = torch.randn(sum(lengths), 512, device="cuda", dtype=torch.bfloat16)
                positions, tables, _ = prepare(layer, x, cache, lengths, prefixes)
                before = {
                    key: group.kv_cache_base.clone()
                    for key, group in cache.groups.items()
                }
                select = layer._select_indices
                permutation = torch.randperm(512, device="cuda")

                def shuffled_select(*args, **kwargs):
                    return select(*args, **kwargs)[:, permutation].contiguous()

                with patch.object(
                    layer, "_select_indices", side_effect=shuffled_select
                ):
                    actual = layer(x, positions, cache, tables)
                actual_plan = layer._shared_attention["prefill_index_plan"][1].clone()
                for row, position in enumerate(positions.tolist()):
                    count = (position + 1) // ratio
                    if not count:
                        continue
                    ids = actual_plan[row, :count]
                    torch.testing.assert_close(
                        ids,
                        ids[0] + torch.arange(count, device="cuda"),
                        rtol=0,
                        atol=0,
                        check_dtype=False,
                    )
                after = {
                    key: group.kv_cache_base.clone()
                    for key, group in cache.groups.items()
                }
                for key, value in before.items():
                    cache.groups[key].kv_cache_base.copy_(value)
                with patch(
                    "rtp_llm.models_py.modules.dsv4.fp8._v41_prefill_attention_sm120.sparse_prefill_attention",
                    side_effect=lambda *args: contraction_oracle(*args).cuda(),
                ), patch(
                    "rtp_llm.models_py.modules.dsv4.fp8._v41_prefill_index_plan.try_build_index_plan",
                    return_value=None,
                ), patch.object(
                    layer,
                    "_select_indices",
                    side_effect=shuffled_select,
                ):
                    expected = layer(x, positions, cache, tables)
                torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.003)
                torch.testing.assert_close(
                    actual_plan,
                    layer._shared_attention["prefill_index_plan"][1],
                    rtol=0,
                    atol=0,
                )
                for key, value in after.items():
                    torch.testing.assert_close(
                        cache.groups[key].kv_cache_base, value, rtol=0, atol=0
                    )

    @torch.no_grad()
    def test_nas_owner_prefill(self):
        from safetensors import safe_open

        root = Path(os.environ["DSV41_TEST_CHECKPOINT"])
        config = json.loads((root / "config.json").read_text())["text_config"]
        index = json.loads((root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        for layer_id in (0, 2, 20):
            ratio = config["compress_ratios"][layer_id]
            suffixes = {
                W.v4_attn_q_norm: "q_norm.weight",
                W.v4_attn_kv_norm: "kv_norm.weight",
                W.v4_attn_sink: "attn_sink",
            }
            for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
                suffixes[getattr(W, f"v4_attn_{name}_w")] = name + ".weight"
                suffixes[getattr(W, f"v4_attn_{name}_s")] = name + ".scale"
            if ratio:
                suffixes.update(
                    {
                        W.v4_compressor_wkv: "compressor.wkv.weight",
                        W.v4_compressor_norm: "compressor.norm.weight",
                        W.v41_indexer_wk: "indexer.wk.weight",
                        W.v41_indexer_k_norm: "indexer.k_norm.weight",
                        W.v4_indexer_wq_b_w: "indexer.wq_b.weight",
                        W.v4_indexer_wq_b_s: "indexer.wq_b.scale",
                        W.v4_indexer_weights_proj_w: "indexer.weights_proj.weight",
                    }
                )
                if ratio == 2:
                    suffixes[W.v4_compressor_wgate] = "compressor.wgate.weight"
            weights = {}
            for key, suffix in suffixes.items():
                name = f"layers.{layer_id}.attn.{suffix}"
                with safe_open(
                    root / index[name], framework="pt", device="cpu"
                ) as file:
                    value = file.get_tensor(name).cuda()
                if (
                    value.dtype in (torch.float32, torch.float16)
                    and key != W.v4_attn_sink
                ):
                    value = value.bfloat16()
                weights[key] = value
            layer = AttentionV41FP8(
                layer_id=layer_id,
                cache_layer_id=0,
                dim=config["hidden_size"],
                n_heads=config["num_attention_heads"],
                q_lora_rank=config["q_lora_rank"],
                head_dim=512,
                rope_head_dim=64,
                o_lora_rank=config["o_lora_rank"],
                o_groups=config["o_groups"],
                window_size=128,
                compress_ratio=ratio,
                compress_rope_theta=160000,
                rope_theta=10000,
                rope_factor=16,
                beta_fast=32,
                beta_slow=1,
                original_seq_len=65536,
                max_batch_size=2,
                max_seq_len=512,
                index_n_heads=32,
                index_head_dim=128,
                index_topk=512,
                norm_eps=config["rms_norm_eps"],
                layer_weights=weights,
                v41_config=config,
                shared_attention={},
            )
            layer.reset_rope_cache(torch.device("cuda"))
            cache = NativeTaggedCache([layer])
            x = torch.randn(5, layer.dim, device="cuda", dtype=torch.bfloat16)
            positions, tables, meta = prepare(layer, x, cache, [5])
            recorded = []

            def checked_attention(*args):
                value = sparse_prefill_attention(*args)
                torch.testing.assert_close(
                    value.cpu(), dense_oracle(*args), rtol=0.01, atol=0.004
                )
                torch.testing.assert_close(
                    value.cpu(), contraction_oracle(*args), rtol=0.01, atol=0.004
                )
                recorded.append(value.clone())
                return value

            with patch(
                "rtp_llm.models_py.modules.dsv4.fp8._v41_prefill_attention_sm120.sparse_prefill_attention",
                side_effect=checked_attention,
            ):
                actual = layer(x, positions, cache, tables)
            self.assertEqual(actual.shape, x.shape)
            self.assertTrue(bool(torch.isfinite(actual).all()))
            # FP8 projections amplify a one-ULP BF16 attention difference.
            # Check attention against independent oracles above, then check
            # the projection wiring exactly with identical attention input.
            expected = output_composition(layer, torch.cat(recorded), meta.freqs_cis)
            with self.subTest(layer=layer_id):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @torch.no_grad()
    def test_source_views_rebind_and_preserve_physical_stride(self):
        layer = make_layer(2, 2, gen=3)
        first = NativeTaggedCache([layer], gen=3, padding=256)
        second = NativeTaggedCache([layer], gen=3, padding=512)
        old = {}
        for cache in (first, second):
            with bind_attn_cache(layer, cache, {}):
                for _, tag in cache.groups:
                    base = cache.groups[0, tag].kv_cache_base
                    view = layer._source_pool(tag)
                    self.assertIs(view, layer._source_pool(tag))
                    self.assertEqual(view.data_ptr(), base.data_ptr())
                    if tag != "csa_state":
                        self.assertEqual(view.stride(0), base.stride(0))
                    if tag in old:
                        self.assertIsNot(view, old[tag])
                        self.assertNotEqual(view.data_ptr(), old[tag].data_ptr())
                    old[tag] = view
        raw = second.groups[0, "csa_state"].kv_cache_base
        padded = torch.zeros(
            raw.shape[0], raw.shape[1] + 128, device="cuda", dtype=torch.uint8
        )
        second.groups[0, "csa_state"] = LayerKVCache(
            padded[:, : raw.shape[1]], 256, 0, 0, "csa_state"
        )
        with bind_attn_cache(layer, second, {}):
            with self.assertRaisesRegex(ValueError, "contiguous physical pages"):
                layer._source_pool("csa_state")

    @torch.no_grad()
    def test_no_speculative_slack_matches_odd_continuation_and_swa_wrap(self):
        torch.manual_seed(41129)
        zero = make_layer(2, 2, gen=0)
        torch.manual_seed(41129)
        one = make_layer(2, 2, gen=1)
        caches = [NativeTaggedCache([zero], gen=0), NativeTaggedCache([one], gen=1)]
        x = torch.randn(135, 512, device="cuda", dtype=torch.bfloat16)
        prefix = 0
        for length in (3, 124, 3, 5):
            outputs = []
            for layer, cache in zip((zero, one), caches):
                chunk = x[prefix : prefix + length]
                positions, tables, _ = prepare(layer, chunk, cache, [length], [prefix])
                outputs.append(layer(chunk, positions, cache, tables).clone())
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
            for tag in ("csa_kv", "indexer_kv"):
                torch.testing.assert_close(
                    caches[0].groups[0, tag].kv_cache_base,
                    caches[1].groups[0, tag].kv_cache_base,
                    rtol=0,
                    atol=0,
                )
            prefix += length

    @torch.no_grad()
    def test_native_fp32_state_matches_opaque_bytes(self):
        layer = make_layer(2, 2)
        byte_cache = NativeTaggedCache([layer])
        native_cache = NativeTaggedCache([layer])
        state = native_cache.groups[0, "csa_state"].kv_cache_base
        native_cache.groups[0, "csa_state"].kv_cache_base = state.view(torch.float32)
        x = torch.randn(3, 512, device="cuda", dtype=torch.bfloat16)
        outputs = []
        for cache in (byte_cache, native_cache):
            positions, tables, _ = prepare(layer, x, cache, [3])
            outputs.append(layer(x, positions, cache, tables).clone())
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        for key, group in byte_cache.groups.items():
            torch.testing.assert_close(
                group.kv_cache_base,
                native_cache.groups[key].kv_cache_base.view(torch.uint8),
                rtol=0,
                atol=0,
            )
        with bind_attn_cache(layer, native_cache, {}):
            pool = layer._source_pool("csa_state")
            self.assertEqual(pool.dtype, torch.float32)
            self.assertEqual(pool.data_ptr(), state.data_ptr())
            self.assertIs(pool, layer._source_pool("csa_state"))
            native_cache.groups[0, "csa_state"].kv_cache_base = state.view(
                torch.float16
            )
            with self.assertRaisesRegex(ValueError, "checkpoint geometry"):
                layer._source_pool("csa_state")

    @torch.no_grad()
    def test_preflight_rejects_unqualified_work_before_writes(self):
        layer = make_layer(2, 2)
        cache = NativeTaggedCache([layer])
        x = torch.randn(3, 512, device="cuda", dtype=torch.bfloat16)
        positions, tables, meta = prepare(layer, x, cache, [3])
        original = {key: val.kv_cache_base.clone() for key, val in cache.groups.items()}
        for bad in (
            meta._replace(cp_on=True),
            meta._replace(input_lengths=torch.tensor([513], device="cuda")),
        ):
            layer._set_prefill_meta_shared(bad)
            with self.assertRaises(NotImplementedError):
                layer(x, positions, cache, tables)
        layer._set_prefill_meta_shared(meta)
        with self.assertRaises(NotImplementedError):
            prepare(layer, x, cache, [513])
        with self.assertRaisesRegex(ValueError, "nonempty requests"):
            prepare(layer, x, cache, [3], [-1])
        for value in (
            None,
            torch.tensor([[-1]], device="cuda", dtype=torch.int32),
            torch.tensor([[4]], device="cuda", dtype=torch.int32),
            torch.tensor([[2]], dtype=torch.int32),
        ):
            bad_tables = {**tables, "indexer_kv": value}
            with self.assertRaisesRegex(ValueError, "block table"):
                prepare(layer, x, cache, [3], tables=bad_tables)
            with self.assertRaisesRegex(ValueError, "block table"):
                layer(x, positions, cache, bad_tables)
        with self.assertRaisesRegex(ValueError, "decode"):
            layer.forward_decode(x.unsqueeze(1), None, cache)
        cache.groups[0, "indexer_kv"] = LayerKVCache(
            original[0, "indexer_kv"], 256, 99, 0, "indexer_kv"
        )
        with self.assertRaisesRegex(ValueError, "source binding"):
            layer(x, positions, cache, tables)
        for key, raw in original.items():
            torch.testing.assert_close(
                cache.groups[key].kv_cache_base, raw, rtol=0, atol=0
            )
        self.assertNotIn("global", layer._shared_attention)


def load_tests(loader, tests, pattern):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("V4.1 attention owner tests require real SM120 hardware")
    return tests


if __name__ == "__main__":
    unittest.main()
