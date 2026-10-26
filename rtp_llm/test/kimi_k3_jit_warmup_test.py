"""The directed replay compilation must leave live request state untouched."""

import importlib.util
import sys
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.kimi_k3.kernel_jit_warmup import (
    _inactive_replay_inputs,
    _warmup_replay,
)
from rtp_llm.models_py.triton_kernels.linear_replay import (
    finalize_linear_replay,
    linear_serial_replay,
)


class KimiK3JitWarmupTest(unittest.TestCase):

    def test_replay_warmup_covers_active_multi_request_batches(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        import triton

        from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3Model
        from rtp_llm.models_py.modules.hybrid.kda_attention import KdaAttention
        from rtp_llm.models_py.modules.kimi_k3.attention import KimiK3KDA

        torch.manual_seed(67)
        device = torch.device("cuda:0")
        dim, channels, capacity, slots, blocks = 128, 384, 4, 7, 11

        def zeros(shape, dtype=torch.float32):
            return torch.zeros(shape, device=device, dtype=dtype)

        state = torch.ones((blocks, 1, dim, dim), device=device)
        conv = torch.ones((blocks, 3, channels), device=device, dtype=torch.bfloat16)
        replay = SimpleNamespace(
            k=zeros((slots, capacity, 1, dim)),
            u=zeros((slots, capacity, 1, dim)),
            g=zeros((slots, capacity, 1, dim)),
            conv_inputs=zeros((slots, capacity, channels), torch.bfloat16),
            slot_generations=zeros((slots,), torch.int64),
            log_epochs=zeros((slots,), torch.int64),
            valid_counts=zeros((slots,), torch.int32),
            error_flags=zeros((slots,), torch.int32),
        )
        # Bind the production attention through its shared KDA constructor so
        # warmup follows the same projection/decode attributes as the model.
        attention = object.__new__(KimiK3KDA)
        decoder = SimpleNamespace(
            _get_ssm_states=lambda _: state,
            _get_conv_states=lambda _: conv,
            parallelism_config=SimpleNamespace(tp_size=1),
            local_num_k_heads=1,
            local_num_v_heads=1,
            head_k_dim=dim,
            head_v_dim=dim,
            conv_weights=torch.ones((channels, 4), device=device, dtype=torch.bfloat16),
            alog=zeros((1,)),
            dt_bias=zeros((dim,)),
            gate_lower_bound=-5.0,
        )
        KdaAttention.__init__(
            attention,
            input_projection=torch.nn.Linear(
                dim, 4 * dim + dim + 1, device=device, dtype=torch.bfloat16
            ),
            forget_projection=torch.nn.Linear(
                dim, dim, device=device, dtype=torch.bfloat16
            ),
            output_projection=torch.nn.Identity(),
            fa_width=dim,
            norm=torch.nn.Identity(),
            prefill=decoder,
            decode=decoder,
        )
        model = object.__new__(KimiK3Model)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(gen_num_per_cycle=capacity - 1, hidden_size=dim)
        model.embed_tokens = torch.nn.Embedding(
            1, dim, device=device, dtype=torch.bfloat16
        )
        model.layers = [SimpleNamespace(index=0, attention=attention)]
        groups = tuple(
            SimpleNamespace(group_id=group, kv_cache_base=state, linear_replay=replay)
            for group in (0, 2)
        )
        model.kv_cache = SimpleNamespace(get_layer_cache_groups=lambda _: groups)
        tracked = (state, conv, *vars(replay).values())
        before = [tensor.clone() for tensor in tracked]
        prepared = []

        def observe_preparation(*, cache_hit, src, **_):
            prepared.append((src.fn.__name__, str(src.attrs), str(src.constants)))

        with triton.knobs.compilation.scope():
            triton.knobs.compilation.listener = observe_preparation
            _warmup_replay(model, SimpleNamespace(is_decode_role=True))
        for tensor, expected in zip(tracked, before):
            torch.testing.assert_close(tensor, expected, rtol=0, atol=0)

        compilations = []
        with torch.inference_mode(), triton.knobs.compilation.scope():

            def observe_request(*, cache_hit, src, **_):
                if not cache_hit:
                    compilations.append(
                        (src.fn.__name__, str(src.attrs), str(src.constants))
                    )

            triton.knobs.compilation.listener = observe_request
            for batch in (2, 3):
                inputs = SimpleNamespace(
                    slot_ids=torch.arange(batch, device=device, dtype=torch.int32),
                    slot_generations=torch.ones(
                        batch, device=device, dtype=torch.int64
                    ),
                    state_read_block_ids=torch.ones(
                        (3, batch), device=device, dtype=torch.int32
                    ),
                    active_block_ids=torch.full(
                        (3, batch), 4, device=device, dtype=torch.int32
                    ),
                    prev_accept_lengths=zeros((batch,), torch.int32),
                    history_valid_lengths=zeros((batch,), torch.int32),
                    history_epochs=zeros((batch,), torch.int64),
                    verify_epochs=torch.ones(batch, device=device, dtype=torch.int64),
                    init_kinds=torch.ones(batch, device=device, dtype=torch.int32),
                )
                inputs.active_block_ids[2] = torch.arange(
                    4, 4 + batch, device=device, dtype=torch.int32
                )
                for steps in (1, capacity):
                    fused = attention.input(zeros((batch * steps, dim), torch.bfloat16))
                    qkv, _, latent, beta = fused.split((channels, dim, dim, 1), dim=-1)
                    q, k, v = qkv.contiguous().split(dim, dim=-1)
                    result = linear_serial_replay(
                        q,
                        k,
                        v,
                        attention._project_forget(latent),
                        beta,
                        attention.decode.conv_weights,
                        attention.decode.alog,
                        attention.decode.dt_bias,
                        state,
                        conv,
                        replay,
                        inputs,
                        group_id=2,
                        vector_gate=True,
                        state_v_first=False,
                        lower_bound=-5.0,
                    )
                    finalize_linear_replay(replay, inputs, steps)
                    self.assertTrue(torch.isfinite(result).all().item())
                    self.assertGreater(torch.count_nonzero(result).item(), 0)
                    torch.testing.assert_close(
                        replay.valid_counts[:batch],
                        torch.full((batch,), steps, device=device, dtype=torch.int32),
                    )
        self.assertEqual(
            compilations,
            [],
            f"active batches must reuse prepared replay kernels; prepared={prepared}",
        )
        self.assertEqual(torch.count_nonzero(replay.error_flags).item(), 0)

    def test_inactive_production_shape_replay_does_not_change_pool(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        device = torch.device("cuda:0")
        slots, blocks, capacity, dim, width = 7, 11, 4, 128, 4
        channels = 3 * dim

        def tensor(shape, dtype=torch.float32):
            return torch.ones(shape, device=device, dtype=dtype)

        cache = SimpleNamespace(
            k=tensor((slots, capacity, 1, dim)),
            u=tensor((slots, capacity, 1, dim)),
            g=tensor((slots, capacity, 1, dim)),
            conv_inputs=tensor((slots, capacity, channels), torch.bfloat16),
            slot_generations=tensor((slots,), torch.int64),
            log_epochs=tensor((slots,), torch.int64),
            valid_counts=tensor((slots,), torch.int32),
            error_flags=torch.zeros((slots,), device=device, dtype=torch.int32),
        )
        state = tensor((blocks, 1, dim, dim))
        conv = tensor((blocks, width - 1, channels), torch.bfloat16)
        inputs = _inactive_replay_inputs(2, device)
        tracked = (state, conv, *(getattr(cache, name) for name in vars(cache)))
        before = tuple(value.clone() for value in tracked)

        for steps in range(1, capacity + 1):
            qkv = tensor((steps, channels), torch.bfloat16)
            q, k, v = qkv.split((dim, dim, dim), dim=-1)
            result = linear_serial_replay(
                q,
                k,
                v,
                tensor((steps, dim), torch.bfloat16),
                tensor((steps, 1), torch.bfloat16),
                tensor((channels, width), torch.bfloat16),
                tensor((1,)),
                tensor((dim,)),
                state,
                conv,
                cache,
                inputs,
                group_id=2,
                vector_gate=True,
                state_v_first=False,
                lower_bound=-5.0,
            )
            finalize_linear_replay(cache, inputs, steps)
            self.assertEqual(tuple(result.shape), (steps, 1, dim))
            self.assertEqual(torch.count_nonzero(result).item(), 0)
        torch.cuda.synchronize(device)
        for actual, expected in zip(tracked, before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def load_warmup():
    from rtp_llm.models_py.modules.kimi_k3 import mla_jit_warmup

    spec = importlib.util.spec_from_file_location(
        "k3_mla_warmup_under_test", mla_jit_warmup.__file__
    )
    module = importlib.util.module_from_spec(spec)
    missing = object()
    previous = sys.modules.get(spec.name, missing)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is missing:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
    return module


def model_fixture(dtype=torch.bfloat16):
    attention = type("KimiK3MLA", (), {})()
    attention.heads, attention.kv_rank = 3, 16
    attention.suffix_dim, attention.q_dim, attention.v_dim = 8, 192, 128
    storage = torch.ones((7, 64, 48), dtype=dtype)
    live = storage[:, :, :24]
    cache = SimpleNamespace(kv_cache_base=live, seq_size_per_block=64, tag="mla")
    model = SimpleNamespace(
        layers=[SimpleNamespace(index=i, attention=attention) for i in range(2)],
        kv_cache=SimpleNamespace(
            get_layer_cache_groups=lambda _index: [cache],
            get_seq_size_per_block=lambda _tag: 256,
        ),
        parallelism_config=SimpleNamespace(
            tp_size=2,
            tp_rank=1,
            prefill_cp_config=SimpleNamespace(kv_cache_sharded=True),
        ),
        config=SimpleNamespace(attn_config=SimpleNamespace(softmax_extra_scale=1.25)),
    )
    return model, live


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class MlaJitWarmupCudaTest(unittest.TestCase):
    def test_scratch_prefix_preparation_preserves_live_cache(self):
        warmup = load_warmup()
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                model, live = model_fixture(dtype)
                live_cuda = torch.empty_strided(
                    live.shape, live.stride(), device="cuda", dtype=dtype
                )
                live_cuda.copy_(live)
                model.kv_cache = SimpleNamespace(
                    get_layer_cache_groups=lambda _index: [
                        SimpleNamespace(
                            kv_cache_base=live_cuda,
                            seq_size_per_block=64,
                            tag="mla",
                        )
                    ],
                    get_seq_size_per_block=lambda _tag: 256,
                )
                before = live_cuda.to(torch.float32).clone()
                spec = next(warmup._mla_warmup_specs(model))
                # Exercise movement/merge independently of optional TokenSpeed.
                with patch.object(warmup, "_warmup_tokenspeed"):
                    warmup._warmup_spec(spec)
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    live_cuda.to(torch.float32), before, rtol=0, atol=0
                )

    def test_page_rr_metadata_changes_preserve_prefix_output(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_page_rr_cache_gather import (
            MlaPageRRCacheGather,
            _restore_mla_page_rr_prefix,
        )

        def launch(requests, blocks, local_pages):
            page, features = 64, 6
            cache = (
                torch.arange(blocks * page * features, device="cuda")
                .remainder(113)
                .reshape(blocks, page, features)
                .to(torch.bfloat16)
            )
            table = (
                torch.arange(requests * local_pages, device="cuda", dtype=torch.int32)
                .reshape(requests, local_pages)
                .remainder(blocks - 1)
                + 1
            )
            adapter = MlaPageRRCacheGather(page, 1, 0)
            payload = adapter._pack_prefix(
                cache, table, (page * local_pages,) * requests
            )
            output = _restore_mla_page_rr_prefix(
                payload.unsqueeze(0), adapter._prefix_descriptor
            )
            expected = torch.cat(
                [torch.cat([cache[int(block)] for block in row]) for row in table.cpu()]
            )
            torch.testing.assert_close(output, expected, rtol=0, atol=0)

        launch(1, 2, 1)
        launch(3, 7, 3)

    def test_tokenspeed_partial_shapes_reuse_prepared_causal_lse_keys(self):
        if importlib.util.find_spec("tokenspeed_mla") is None:
            self.skipTest("selected TokenSpeed dependency is not installed")
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("TokenSpeed Prefill requires Blackwell")
        from rtp_llm.models_py.modules.kimi_k3.tokenspeed_prefill import (
            TokenspeedMlaPrefill,
        )

        warmup = load_warmup()
        model, _ = model_fixture(torch.float8_e4m3fn)
        spec = replace(
            next(warmup._mla_warmup_specs(model)),
            device=torch.device("cuda:0"),
            heads=8,
        )
        warmup._warmup_tokenspeed(spec)
        from tokenspeed_mla import mla_prefill as backend

        before = len(backend._compiled_kernels)
        wrapper = TokenspeedMlaPrefill(spec.heads, spec.softmax_scale)
        for causal, kv_tokens in ((True, 7), (False, 129)):
            q = torch.zeros((7, 8, 192), device="cuda", dtype=torch.float8_e4m3fn)
            k = torch.zeros((kv_tokens, 8, 192), device="cuda", dtype=q.dtype)
            v = torch.zeros((kv_tokens, 8, 128), device="cuda", dtype=q.dtype)
            output, lse = wrapper.run_partial(
                q,
                k,
                v,
                qo_indptr=torch.tensor((0, 7), device="cuda", dtype=torch.int32),
                kv_indptr=torch.tensor(
                    (0, kv_tokens), device="cuda", dtype=torch.int32
                ),
                max_q=7,
                max_k=kv_tokens,
                causal=causal,
            )
            torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
            counts = (
                torch.arange(1, 8, device="cuda", dtype=torch.float32)
                if causal
                else torch.full((7,), kv_tokens, device="cuda")
            )
            expected_lse = counts.log()[:, None].expand(7, 8)
            torch.testing.assert_close(lse, expected_lse, rtol=1e-3, atol=1e-3)
        self.assertEqual(len(backend._compiled_kernels), before)


if __name__ == "__main__":
    unittest.main()
