"""Replay exactly the production decoder layers, excluding request preparation."""

from __future__ import annotations

import torch


class _Prepared(Exception):
    pass


class LayerBlock:
    def __init__(self, model, inputs):
        self.model = model
        self.inputs = inputs
        self.deferred = not inputs.attention_inputs.is_prefill
        first = model.layers[0]
        name = "forward_hc_deferred" if self.deferred else "forward"
        original = getattr(first, name)
        captured = {}

        def stop(*args, **kwargs):
            captured.update(args=args, kwargs=kwargs)
            raise _Prepared()

        setattr(first, name, stop)
        try:
            model(inputs)
        except _Prepared:
            pass
        finally:
            setattr(first, name, original)
        if not captured:
            raise RuntimeError("production forward did not reach the first layer")
        args, kwargs = captured["args"], captured["kwargs"]
        self.hidden = args[0]
        if self.deferred:
            _, self.fmha, _, self.attention, self.meta, _, self.residual, _, _ = args
        else:
            self.residual, self.fmha = args[1:3]
            self.attention, self.meta = kwargs["attention_inputs"], kwargs["attn_meta"]
        # Production mHC post writes into its residual buffer. A replay must
        # restore this input as well as recurrent/KV state, outside timing.
        self.initial_hidden = self.hidden.clone()
        self.initial_residual = (
            self.residual.clone()
            if self.residual.data_ptr() != self.hidden.data_ptr()
            else None
        )
        rows = self.hidden.shape[0]
        positions = (
            list(range(rows))
            if self.deferred
            else [
                0,
                1,
                2,
                127,
                128,
                4095,
                4096,
                32767,
                65535,
                65536,
                98303,
                130943,
                131068,
                131069,
                131070,
                131071,
            ]
        )
        self.positions = torch.tensor([p for p in positions if p < rows], device="cuda")

    def restore_inputs(self):
        self.hidden.copy_(self.initial_hidden)
        if self.initial_residual is not None:
            self.residual.copy_(self.initial_residual)

    def __call__(self, audit=False):
        from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer

        hidden, residual = self.hidden, self.residual
        post = comb = None
        samples = {}
        for index, layer in enumerate(self.model.layers):
            select_block_map_for_layer(self.attention, index)
            if self.meta.mla_cp_layout is not None:
                select_block_map_for_layer(self.fmha.attn_inputs, index)
            cache = self.model.kv_cache.get_layer_cache(index)
            with torch.profiler.record_function(
                f"glm53_smoke.layer{index}.{'kda' if index < 3 else 'mla'}"
            ):
                if self.deferred:
                    hidden, residual, post, comb = layer.forward_hc_deferred(
                        hidden,
                        self.fmha,
                        cache,
                        self.attention,
                        self.meta,
                        self.model.kv_cache,
                        residual,
                        post,
                        comb,
                    )
                    if audit:
                        # Resolve deferred mHC on sampled rows only. This audit
                        # pass is never used as the timed replay or its graph.
                        p = self.positions
                        resolved = layer.ffn_hc.post(
                            hidden[p], residual[p], post[p], comb[p]
                        )
                        samples[index] = resolved.cpu()
                else:
                    out = layer(
                        hidden,
                        residual,
                        self.fmha,
                        kv_cache=cache,
                        attention_inputs=self.attention,
                        attn_meta=self.meta,
                        global_kv_cache=self.model.kv_cache,
                    )
                    hidden, residual = out.hidden_states, out.residual
                    if audit:
                        # Preserve every token's full mHC output for future
                        # regressions; only vocabulary readouts are sampled.
                        samples[index] = hidden.cpu()
        if self.deferred:
            hidden = self.model.layers[-1].ffn_hc.post(hidden, residual, post, comb)
        return samples if audit else hidden


class CacheSnapshot:
    def __init__(self, cache):
        self.live = []
        seen = set()
        for layers in (
            cache.kv_cache_base_by_layer_region,
            cache.kv_scale_base_by_layer_region,
        ):
            for regions in layers:
                for value in regions:
                    if (
                        value is not None
                        and value.numel()
                        and value.data_ptr() not in seen
                    ):
                        seen.add(value.data_ptr())
                        self.live.append(value)
        self.saved = [x.clone() for x in self.live]

    def restore(self):
        for live, saved in zip(self.live, self.saved):
            live.copy_(saved)


def seed_decode_cache(model, seed):
    """Finite synthetic 128K history; no dependence on uninitialized pages."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    cache = model.kv_cache
    for index in range(3):
        base = cache.kv_cache_base_by_layer[index]
        converter = model.layers[index].self_attn.decode_kda.linear_cache_converter
        converter.get_ssm_state_tensor(base).normal_(0, 0.05, generator=generator)
        converter.get_conv_state_tensor(base).normal_(0, 0.1, generator=generator)
        base[0].zero_()
    kv = cache.kv_cache_base_by_layer[3]
    kv.normal_(0, 0.2, generator=generator)
    kv[0].zero_()
    raw = cache.kv_cache_base_by_layer_region[3][3]
    # Pool blocks are planar: 32*128 FP8 bytes, then 32 FP32 scales.
    for start in range(0, raw.shape[0], 1024):
        part = raw[start : start + 1024]
        values = (
            torch.randn((part.shape[0], 4096), device="cuda", generator=generator) * 16
        )
        part[:, :4096].copy_(values.to(torch.float8_e4m3fn).view(torch.uint8))
        part[:, 4096:].view(torch.float32).fill_(1 / 64)
    raw[0].zero_()
    cache.kv_cache_base_by_layer_region[3][4].normal_(0, 0.1, generator=generator)
