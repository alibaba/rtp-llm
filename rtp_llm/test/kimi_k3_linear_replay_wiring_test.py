"""Verify the LINEAR replay and fallback contracts of KDA and gated delta attention."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from rtp_llm.models_py.model_desc import qwen3_next
from rtp_llm.models_py.modules.factory.attention import kda_impl


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
            decode._linear_replay = None
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
            replay = Mock(return_value=output)

            with (
                patch.object(
                    decode, "_get_ssm_states", return_value=tensor((2, 128, 128))
                ),
                patch.object(
                    decode, "_get_conv_states", return_value=tensor((2, 3, 384))
                ),
                patch.object(
                    kda_impl.LinearReplay, "bind", return_value=replay
                ) as bind,
            ):
                actual = decode.forward(
                    mixed_qkv, forget_gate, beta, attn_inputs, kv_cache, metadata
                )

            self.assertIs(actual, output)
            bind.assert_called_once_with(
                None,
                decode,
                kv_cache,
                vector_gate=True,
                state_v_first=False,
                lower_bound=-5.0,
            )
            replay.assert_called_once_with(mixed_qkv, forget_gate, beta, replay_inputs)


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
                    patch("torch.version.hip", None),
                    patch.object(module.LinearReplay, "bind") as replay,
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


if __name__ == "__main__":
    unittest.main()
