"""Verify K3 attention wiring and private JIT warmup resource contracts."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from rtp_llm.models_py.model_desc import qwen3_next
from rtp_llm.models_py.modules.factory.attention import kda_impl
from rtp_llm.models_py.modules.kimi_k3 import mla_jit_warmup
from rtp_llm.models_py.modules.kimi_k3.kernel_jit_warmup import _inactive_replay_inputs


class KimiK3LinearReplayWiringTest(unittest.TestCase):
    def test_target_verify_forwards_checkpoint_gate_bound(self):
        with FakeTensorMode():

            def tensor(shape):
                return torch.empty(shape, device="cuda:0", dtype=torch.bfloat16)

            decode = kda_impl.KdaDecode.__new__(kda_impl.KdaDecode)
            torch.nn.Module.__init__(decode)
            decode.local_num_k_heads = 1
            decode.local_num_v_heads = 1
            decode.head_k_dim = 128
            decode.head_v_dim = 128
            decode.gate_lower_bound = -5.0
            decode.conv_weights = tensor((384, 4))
            decode.alog = tensor((1,))
            decode.dt_bias = tensor((1,))

            cache_base = tensor((2, 512))
            replay_cache = object()
            replay_inputs = object()
            kv_cache = SimpleNamespace(
                kv_cache_base=cache_base,
                linear_replay=replay_cache,
                group_id=3,
            )
            attn_inputs = SimpleNamespace(linear_replay=replay_inputs)
            metadata = kda_impl.KdaMetadata(is_target_verify=True)
            mixed_qkv = tensor((4, 384))
            forget_gate = tensor((4, 128))
            beta = tensor((4, 1))
            output = tensor((4, 1, 128))

            with (
                patch.object(
                    decode, "_get_ssm_states", return_value=tensor((2, 128, 128))
                ),
                patch.object(
                    decode, "_get_conv_states", return_value=tensor((2, 3, 384))
                ),
                patch.object(kda_impl, "is_cuda", return_value=True),
                patch.object(
                    kda_impl, "linear_serial_replay", return_value=output
                ) as replay,
            ):
                actual = decode.forward(
                    mixed_qkv, forget_gate, beta, attn_inputs, kv_cache, metadata
                )

            self.assertIs(actual, output)
            replay.assert_called_once()
            args, kwargs = replay.call_args
            self.assertIs(args[10], replay_cache)
            self.assertIs(args[11], replay_inputs)
            self.assertEqual(kwargs["group_id"], 3)
            self.assertTrue(kwargs["vector_gate"])
            self.assertEqual(kwargs["lower_bound"], -5.0)


class LinearReplayDispatchTest(unittest.TestCase):
    def test_verification_without_replay_metadata_keeps_existing_decode_path(self):
        for module, decode_cls, metadata_cls in (
            (
                kda_impl,
                kda_impl.KdaDecode,
                kda_impl.KdaMetadata,
            ),
            (
                qwen3_next,
                qwen3_next.Qwen3NextGatedDeltaNetDecode,
                qwen3_next.Qwen3NextMetadata,
            ),
        ):
            with self.subTest(decoder=decode_cls.__name__), FakeTensorMode():

                def tensor(shape):
                    return torch.empty(shape, device="cuda:0", dtype=torch.bfloat16)

                decode = decode_cls.__new__(decode_cls)
                torch.nn.Module.__init__(decode)
                inputs = SimpleNamespace(linear_replay=None)
                cache = SimpleNamespace(
                    kv_cache_base=tensor((2, 512)),
                    seq_size_per_block=128,
                    linear_replay=None,
                )
                metadata = metadata_cls(is_target_verify=True)
                projected = tensor((4, 384))
                gate = tensor((4, 1))
                convolved = tensor((4, 384))
                output = tensor((4, 1, 128))
                with (
                    patch.object(module, "is_cuda", return_value=True),
                    patch("torch.version.hip", None),
                    patch.object(module, "linear_serial_replay") as replay,
                    patch.object(decode, "_conv1d", return_value=convolved) as conv,
                    patch.object(decode, "_fla", return_value=output) as fla,
                ):
                    actual = decode.forward(
                        projected, gate, gate, inputs, cache, metadata
                    )
                self.assertIs(actual, output)
                replay.assert_not_called()
                self.assertIs(conv.call_args.args[0], projected)
                self.assertTrue(conv.call_args.args[-1])
                self.assertIs(fla.call_args.args[0], convolved)
                self.assertIs(fla.call_args.args[-2], inputs)
                if module is kda_impl:
                    self.assertIs(fla.call_args.args[-1], True)
                else:
                    self.assertIs(fla.call_args.args[-1], metadata)


class KimiK3ReplayWarmupMetadataTest(unittest.TestCase):
    def test_inactive_metadata_prepares_both_alignments_for_every_group(self):
        for group in (0, 1, 2, 5):
            for batch in (1, 2):
                for unaligned in (False, True):
                    inputs = _inactive_replay_inputs(
                        group, torch.device("cpu"), batch, unaligned=unaligned
                    )
                    self.assertEqual(inputs.slot_ids.tolist(), [-1] * batch)
                    for metadata in (
                        inputs.state_read_block_ids,
                        inputs.active_block_ids,
                    ):
                        self.assertEqual(
                            metadata[group].data_ptr() % 16, 4 if unaligned else 0
                        )
                        self.assertEqual(metadata[group].tolist(), [-1] * batch)


class KimiK3MlaWarmupContractTest(unittest.TestCase):
    @staticmethod
    def _model(dtype=torch.bfloat16, *, page_rr=True):
        attention = type("KimiK3MLA", (), {})()
        attention.heads, attention.kv_rank = 3, 16
        attention.suffix_dim, attention.q_dim, attention.v_dim = 8, 192, 128
        live = torch.ones((7, 64, 48), dtype=dtype)[:, :, :24]
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
                prefill_cp_config=SimpleNamespace(kv_cache_sharded=page_rr),
            ),
            config=SimpleNamespace(
                attn_config=SimpleNamespace(softmax_extra_scale=1.25)
            ),
        )
        return model, live

    def test_mla_geometry_and_duplicate_layers_prepare_once(self):
        model, live = self._model(torch.float8_e4m3fn)
        with patch.object(mla_jit_warmup, "_warmup_spec") as prepare:
            mla_jit_warmup.warmup_kimi_k3_mla_jit(
                model, SimpleNamespace(is_decode_role=False)
            )
        prepare.assert_called_once()
        spec = prepare.call_args.args[0]
        self.assertEqual(spec.cache_dtype, torch.float8_e4m3fn)
        self.assertEqual(spec.cache_stride, live.stride())
        self.assertEqual((spec.page_tokens, spec.physical_page_tokens), (64, 256))
        self.assertEqual((spec.shards, spec.rank), (2, 1))
        self.assertEqual((spec.heads, spec.latent, spec.rope), (3, 16, 8))
        self.assertEqual(spec.softmax_scale, 1.25 / 192**0.5)

    def test_mla_preparation_skips_ineligible_models(self):
        for case in ("non_page_rr", "decode", "unbound"):
            with self.subTest(case=case):
                model, _ = self._model(page_rr=case != "non_page_rr")
                if case == "unbound":
                    model.kv_cache = None
                with patch.object(mla_jit_warmup, "_warmup_spec") as prepare:
                    mla_jit_warmup.warmup_kimi_k3_mla_jit(
                        model, SimpleNamespace(is_decode_role=case == "decode")
                    )
                prepare.assert_not_called()

    def test_mla_prefix_uses_private_scratch_without_collectives(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
            mla_page_rr_cache_gather as gather,
        )

        model, live = self._model()
        spec = next(mla_jit_warmup._mla_warmup_specs(model))
        before = live.clone()
        payload = torch.ones(
            (1, spec.physical_page_tokens, spec.latent + spec.rope),
            dtype=spec.cache_dtype,
        )
        with (
            patch.object(
                gather.MlaPageRRCacheGather,
                "_pack_prefix_chunk",
                autospec=True,
                return_value=payload,
            ) as pack,
            patch.object(gather, "_restore_mla_page_rr_prefix") as restore,
            patch.object(
                gather.collective_torch,
                "all_gather_into",
                side_effect=AssertionError("warmup must not enter a collective"),
            ),
        ):
            mla_jit_warmup._warmup_page_rr_prefix(spec)

        pack.assert_called_once()
        restore.assert_called_once()
        adapter, scratch, table, descriptor = pack.call_args.args
        self.assertEqual(scratch.stride(), live.stride())
        self.assertNotEqual(scratch.data_ptr(), live.data_ptr())
        self.assertEqual(adapter.page_tokens, spec.physical_page_tokens)
        self.assertEqual(adapter.kernel_page_tokens, spec.page_tokens)
        self.assertEqual(
            (adapter.shard_size, adapter.shard_rank), (spec.shards, spec.rank)
        )
        self.assertEqual(descriptor.prefix_starts, (spec.rank * 256,))
        self.assertEqual(descriptor.prefix_lens, (1,))
        self.assertEqual(table.tolist(), [[1, 2, 3, 4]])
        gathered, restored_descriptor = restore.call_args.args
        self.assertIs(restored_descriptor, descriptor)
        torch.testing.assert_close(gathered[spec.rank], payload, rtol=0, atol=0)
        torch.testing.assert_close(
            gathered[0], torch.zeros_like(payload), rtol=0, atol=0
        )
        torch.testing.assert_close(live, before, rtol=0, atol=0)

    def test_mla_fp8_prepares_both_causal_modes_with_lse(self):
        from rtp_llm.models_py.modules.kimi_k3 import tokenspeed_prefill

        model, _ = self._model(torch.float8_e4m3fn)
        spec = next(mla_jit_warmup._mla_warmup_specs(model))
        backend = Mock()
        with patch.object(
            tokenspeed_prefill,
            "_load_backend",
            return_value=(backend, "test"),
        ):
            mla_jit_warmup._warmup_tokenspeed(spec)
        self.assertEqual(
            [call.kwargs["is_causal"] for call in backend.call_args_list], [True, False]
        )
        for call in backend.call_args_list:
            self.assertTrue(call.kwargs["return_lse"])
            self.assertEqual(call.kwargs["query"].shape, (2, spec.heads, spec.q_dim))
            self.assertEqual(call.kwargs["value"].shape, (2, spec.heads, spec.v_dim))
            self.assertEqual(call.kwargs["query"].dtype, torch.float8_e4m3fn)
            self.assertEqual(call.kwargs["softmax_scale"], 1.25 / spec.q_dim**0.5)


if __name__ == "__main__":
    unittest.main()
