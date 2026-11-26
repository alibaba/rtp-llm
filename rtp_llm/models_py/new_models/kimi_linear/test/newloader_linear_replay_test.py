"""Exercise main's replay binding on actual NewLoader Kimi and Qwen decoders."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.new_models.kimi_linear.language import KimiLinearKDA
from rtp_llm.models_py.new_models.qwen3_next.gdn import Qwen3NextGatedDeltaNet
from rtp_llm.models_py.triton_kernels import linear_replay


class NewLoaderLinearReplayTest(unittest.TestCase):
    def _owner(self, kimi):
        if kimi:
            return KimiLinearKDA(
                dict(
                    hidden_size=16,
                    kda_num_heads=4,
                    kda_head_dim=4,
                    conv_kernel=4,
                    attn_tp_size=2,
                    attn_tp_rank=1,
                    rms_norm_eps=1e-5,
                    quant_config=None,
                    params_dtype=torch.bfloat16,
                    ssm_state_dtype=torch.float32,
                    conv_state_dtype=torch.bfloat16,
                ),
                "layers.0.self_attn",
            )
        owner = Qwen3NextGatedDeltaNet(
            parallelism_config=SimpleNamespace(get_attn_tp_size=lambda: 2),
            linear_attn_config=SimpleNamespace(
                linear_key_head_dim=4,
                linear_value_head_dim=4,
                linear_num_key_heads=2,
                linear_num_value_heads=4,
                linear_conv_kernel_dim=4,
                ssm_state_dtype=torch.float32,
                conv_state_dtype=torch.bfloat16,
            ),
            hidden_size=16,
            rms_norm_eps=1e-5,
            attn_tp_size=2,
            attn_tp_rank=1,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix="layers.0.linear_attn",
        )
        owner.process_weights_after_loading()
        return owner

    def test_decode_binds_live_parameters_and_cache_pages_after_migration(self):
        for kimi in (True, False):
            with self.subTest(kimi=kimi):
                owner = self._owner(kimi).eval()
                get_decode = lambda: owner.decode_kda if kimi else owner.decode_gdn
                h, hv, dim = (2 if kimi else 1), 2, 4
                channels = (2 * h + hv) * dim
                replay_cache = SimpleNamespace(
                    k=torch.empty(2, 2, h, dim),
                    u=torch.empty(2, 2, hv, dim),
                    g=torch.empty(2, 2, hv, dim if kimi else 1),
                    conv_inputs=torch.empty(2, 2, channels, dtype=torch.bfloat16),
                )
                block_nbytes = get_decode().linear_cache_converter.block_size_bytes
                cache = SimpleNamespace(
                    kv_cache_base=torch.empty(2, block_nbytes // 4),
                    linear_replay=replay_cache,
                    group_id=3,
                )
                inputs = SimpleNamespace(
                    linear_replay=object(), prefix_lengths=torch.ones(1)
                )
                metadata = SimpleNamespace(is_target_verify=True)

                def bind():
                    projected = torch.zeros(1, channels)
                    gate = torch.zeros(1, hv * (dim if kimi else 1))
                    beta = torch.zeros(1, hv)
                    gates = (gate, beta) if kimi else (beta, gate)
                    # Only the GPU launch is stubbed. The selected decode, page
                    # converter, geometry checks, parameter references and cache
                    # identity checks are all production code running on CPU.
                    with patch.object(linear_replay, "_linear_serial_replay") as launch:
                        result = get_decode()(
                            projected, *gates, inputs, cache, metadata
                        )
                        self.assertIs(result, launch.return_value)
                        self.assertIs(launch.call_args.args[3], gate)
                        self.assertIs(launch.call_args.args[4], beta)
                    return get_decode()._linear_replay

                first = bind()
                # The partial captures the launch function, so call bind directly
                # when checking a cache hit (it must not recreate the partial).
                self.assertIs(
                    linear_replay.LinearReplay.bind(
                        first,
                        get_decode(),
                        cache,
                        vector_gate=kimi,
                        state_v_first=not kimi,
                    ),
                    first,
                )
                old_ptr = first.projected.keywords["conv_weights"].data_ptr()
                decoder = get_decode()
                if not kimi:
                    norm, prefill, projection = (
                        owner.norm,
                        owner.prefill_gdn,
                        owner.out_proj,
                    )
                    projection_dtype = projection.weight.dtype
                    owner._apply(lambda tensor: tensor.float(), recurse=False)
                    self.assertIs(owner.norm, norm)
                    self.assertIs(owner.prefill_gdn, prefill)
                    self.assertIs(owner.out_proj, projection)
                    self.assertEqual(projection.weight.dtype, projection_dtype)
                    self.assertEqual(norm.weight.data_ptr(), owner.norm_w.data_ptr())
                else:
                    owner.to(dtype=torch.float32)
                self.assertIs(get_decode(), decoder)
                self.assertFalse(decoder.training)
                self.assertIsNone(get_decode()._linear_replay)
                rebound = bind()
                weight = owner.conv1d.weight if kimi else owner.conv1d_w
                self.assertEqual(
                    rebound.projected.keywords["conv_weights"].data_ptr(),
                    weight.data_ptr(),
                )
                self.assertNotEqual(weight.data_ptr(), old_ptr)
                self.assertEqual(rebound.projected.keywords["a_log"].shape, (hv,))
                self.assertEqual(
                    rebound.projected.keywords["dt_bias"].numel(),
                    hv * (dim if kimi else 1),
                )
                cache.kv_cache_base = cache.kv_cache_base.clone()
                self.assertIsNot(bind(), rebound)


if __name__ == "__main__":
    unittest.main()
