"""Verify that K3 target verification uses the checkpoint's bounded gate."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from rtp_llm.models_py.model_desc import kimi_linear, qwen3_next


class KimiK3LinearReplayWiringTest(unittest.TestCase):
    def test_target_verify_forwards_checkpoint_gate_bound(self):
        with FakeTensorMode():

            def tensor(shape):
                return torch.empty(shape, device="cuda:0", dtype=torch.bfloat16)

            decode = kimi_linear.KimiLinearKDADecode.__new__(
                kimi_linear.KimiLinearKDADecode
            )
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
            metadata = kimi_linear.KimiLinearMetadata(is_target_verify=True)
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
                patch.object(kimi_linear, "is_cuda", return_value=True),
                patch.object(
                    kimi_linear, "linear_serial_replay", return_value=output
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
                kimi_linear,
                kimi_linear.KimiLinearKDADecode,
                kimi_linear.KimiLinearMetadata,
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
                self.assertTrue(fla.call_args.args[-1].is_target_verify)


if __name__ == "__main__":
    unittest.main()
