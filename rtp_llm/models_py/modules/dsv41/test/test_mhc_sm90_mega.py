"""CPU/host contracts for the default-off SM90 materialized-mixes seam."""

import contextlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv41.hc import v41_mega_mhc as mega


def _cpu_args():
    return dict(
        attn_out=torch.zeros((1, 8), dtype=torch.bfloat16),
        residual=torch.zeros((1, 4, 8), dtype=torch.bfloat16),
        post=torch.zeros((1, 4, 1), dtype=torch.float32),
        comb=torch.zeros((1, 4, 4), dtype=torch.float32),
        previous=object(),
        next_hc=object(),
        norm=object(),
    )


class MhcSm90MegaHostTest(unittest.TestCase):
    def setUp(self):
        mega._get_sm90_mixes_backend.cache_clear()
        mega._SM90_WARMED_STREAMS.clear()
        self.addCleanup(mega._get_sm90_mixes_backend.cache_clear)
        self.addCleanup(mega._SM90_WARMED_STREAMS.clear)

    def test_switch_defaults_off_and_accepts_only_explicit_truthy_values(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DSV41_MEGA_MHC_SM90", None)
            self.assertFalse(mega._sm90_requested())
        for value in ("1", "true", "YES", "on"):
            with self.subTest(value=value), patch.dict(
                os.environ, {"DSV41_MEGA_MHC_SM90": value}
            ):
                self.assertTrue(mega._sm90_requested())
        for value in ("0", "", "false", "off", "unexpected"):
            with self.subTest(value=value), patch.dict(
                os.environ, {"DSV41_MEGA_MHC_SM90": value}
            ):
                self.assertFalse(mega._sm90_requested())

    def test_default_off_and_cpu_never_load_tilelang_or_sm100_backend(self):
        args = _cpu_args()
        with (
            patch.dict(os.environ, {}, clear=False),
            patch.object(mega, "_get_sm90_mixes_backend") as sm90_backend,
            patch.object(mega, "_get_mega_mhc") as sm100_backend,
        ):
            os.environ.pop("DSV41_MEGA_MHC_SM90", None)
            self.assertFalse(mega.is_sm90_supported(**args))
            self.assertIsNone(mega.try_fused_post_pre(**args))
        sm90_backend.assert_not_called()
        sm100_backend.assert_not_called()

        with patch.dict(os.environ, {"DSV41_MEGA_MHC_SM90": "1"}), patch.object(
            mega, "_get_sm90_mixes_backend"
        ) as sm90_backend:
            self.assertFalse(mega.is_sm90_supported(**args))
        sm90_backend.assert_not_called()

    def test_dispatch_is_independent_and_rejects_extended_sm100_modes(self):
        args = _cpu_args()
        sentinel = object()
        with (
            patch.dict(os.environ, {"DSV41_MEGA_MHC_SM90": "1"}),
            patch.object(mega, "is_sm90_supported", return_value=True) as gate,
            patch.object(mega, "_run_sm90_post_pre", return_value=sentinel) as run,
        ):
            self.assertIs(mega.try_fused_post_pre(**args), sentinel)
        gate.assert_called_once()
        run.assert_called_once()

        for kwargs in (
            {"original_tokens": 16},
            {"reuse_residual": True},
            {"output_fp8": True},
        ):
            with self.subTest(kwargs=kwargs), patch.dict(
                os.environ, {"DSV41_MEGA_MHC_SM90": "1"}
            ), patch.object(
                mega, "is_sm90_supported", return_value=True
            ) as gate, patch.object(
                mega, "_run_sm90_post_pre"
            ) as run, patch.object(
                mega, "is_supported", return_value=False
            ):
                self.assertIsNone(mega.try_fused_post_pre(**args, **kwargs))
            gate.assert_not_called()
            run.assert_not_called()

    def test_runner_keeps_materialized_prenorm_and_rmsnorm_boundaries(self):
        tokens = 1
        attn = torch.full((tokens, 5120), 2, dtype=torch.bfloat16)
        residual = torch.full((tokens, 4, 5120), 3, dtype=torch.bfloat16)
        post = torch.full((tokens, 4, 1), 0.25, dtype=torch.float32)
        comb = torch.eye(4, dtype=torch.float32).expand(tokens, 4, 4).contiguous()
        previous = SimpleNamespace(pre_mix_out=torch.full((tokens, 4), 0.5))
        next_hc = SimpleNamespace(
            fn=torch.empty((24, 20480)),
            scale=torch.ones(3),
            base=torch.zeros(24),
            norm_eps=1e-6,
            hc_eps=1e-6,
            hc_sinkhorn_iters=20,
            pre_mix_out=None,
        )
        mixes = torch.arange(24, dtype=torch.float32).view(tokens, 24)
        new_pre = torch.full((tokens, 4), 0.6)
        new_post = torch.full((tokens, 4), 0.7)
        new_comb = torch.full((tokens, 16), 0.8)
        layer_input = torch.full((tokens, 5120), 4, dtype=torch.bfloat16)
        backend = Mock()
        backend.pre_mixes_fwd.return_value = (
            new_pre,
            new_post,
            new_comb,
            layer_input,
        )
        norm = Mock(return_value=layer_input)

        def post_fwd(_x, _residual, _post, _comb, *, out):
            out.fill_(5)
            return out

        with (
            patch.object(mega, "_get_sm90_mixes_backend", return_value=backend),
            patch.object(
                mega.torch, "empty_like", wraps=torch.empty_like
            ) as empty_like,
            patch(
                "rtp_llm.models_py.modules.dsv41.hc.mhc_sm90_post.post_fwd",
                side_effect=post_fwd,
            ) as post_kernel,
            patch(
                "rtp_llm.models_py.modules.dsv41.hc.v41_prenorm.prenorm",
                return_value=mixes,
            ) as prenorm,
            patch.object(
                mega.torch.cuda, "device", return_value=contextlib.nullcontext()
            ),
            patch.object(
                mega.torch.cuda, "is_current_stream_capturing", return_value=False
            ),
            patch.object(mega, "_stream_key", return_value=(0, 7)),
        ):
            actual = mega._run_sm90_post_pre(
                attn, residual, post, comb, previous, next_hc, norm
            )

        post_kernel.assert_called_once()
        produced_residual = post_kernel.call_args.kwargs["out"]
        empty_like.assert_not_called()
        self.assertEqual(produced_residual.data_ptr(), residual.data_ptr())
        self.assertIs(prenorm.call_args.args[0], produced_residual)
        self.assertIs(prenorm.call_args.args[1], next_hc.fn)
        self.assertEqual(prenorm.call_args.args[2], next_hc.norm_eps)
        self.assertEqual(
            backend.pre_mixes_fwd.call_args.args[0].data_ptr(), mixes.data_ptr()
        )
        self.assertIs(backend.pre_mixes_fwd.call_args.args[1], produced_residual)
        self.assertEqual(
            backend.pre_mixes_fwd.call_args.args[2].data_ptr(),
            previous.pre_mix_out.data_ptr(),
        )
        norm.assert_called_once_with(layer_input)
        self.assertEqual(next_hc.pre_mix_out.data_ptr(), new_pre.data_ptr())
        self.assertEqual((0, 7) in mega._SM90_WARMED_STREAMS, True)
        self.assertTrue(torch.equal(actual[0], torch.full_like(actual[0], 5)))
        self.assertEqual(actual[0].data_ptr(), residual.data_ptr())
        self.assertEqual(actual[1].data_ptr(), layer_input.data_ptr())
        self.assertTrue(torch.equal(actual[2].reshape(tokens, 4), new_post))
        self.assertTrue(torch.equal(actual[3].reshape(tokens, 16), new_comb))


if __name__ == "__main__":
    unittest.main()
