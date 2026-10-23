"""Execute projection and warmup dispatch with CPU tensors and provider doubles."""

from __future__ import annotations

import ast
import logging
import os
import sys
import time
import types
import unittest
from functools import partial
from pathlib import Path
from unittest.mock import Mock, patch

import torch

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "rtp_llm/models_py/modules/dsv4/fp8/attention.py").is_file()
)
FP8 = "rtp_llm/models_py/modules/dsv4/fp8/"


def source_fn(path, name, ns, cls=None):
    """Compile the actual function, injecting only its dependencies."""
    nodes = ast.parse((ROOT / path).read_text()).body
    if cls:
        nodes = next(
            n for n in nodes if isinstance(n, ast.ClassDef) and n.name == cls
        ).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    mod = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(mod), str(ROOT / path), "exec"), ns)
    return ns[name]


class IntegrationPolicyTest(unittest.TestCase):
    def test_woa_and_warmup_choose_same_arm_at_invocation(self):
        for sm120 in (False, True):
            for flag in (None, "0", "1"):
                with self.subTest(sm120=sm120, flag=flag):
                    self._check_dispatch(sm120, flag)

    def _check_dispatch(self, sm120, flag):
        m, groups, width, rank = 3, 2, 512, 128
        activation = torch.ones(m, groups, width, dtype=torch.float8_e4m3fn)
        scales = torch.full((m, groups, 1), 0x7F7F7F7F, dtype=torch.int32)
        owner = types.SimpleNamespace(
            o_lora_rank=rank,
            _wo_a_stk_w=torch.ones(groups, rank, width, dtype=torch.float8_e4m3fn),
            _wo_a_stk_s=object(),
            wo_a_s=torch.ones(groups, rank // 128, width // 128),
        )

        def groupwise(a, b, a_scale, b_scale, **kwargs):
            self.assertEqual(a.shape, (4, width))
            self.assertEqual(b.shape, (rank, width))
            self.assertTrue(torch.equal(a[:m].float(), torch.ones(m, width)))
            self.assertEqual(a[m:].float().count_nonzero(), 0)
            self.assertTrue(torch.equal(a_scale, torch.ones(4, width // 128)))
            self.assertTrue(torch.equal(b_scale, torch.ones(1, width // 128)))
            self.assertEqual(kwargs["scale_granularity_mnk"], (1, 128, 128))
            self.assertEqual(kwargs["scale_major_mode"], "K")
            return torch.full((4, rank), 7, dtype=kwargs["out_dtype"])

        def einsum(equation, inputs, weights, out, **kwargs):
            self.assertEqual(equation, "bhr,hdr->bhd")
            self.assertIs(inputs[0], activation)
            self.assertIs(inputs[1], scales)
            self.assertIs(weights[0], owner._wo_a_stk_w)
            self.assertIs(weights[1], owner._wo_a_stk_s)
            self.assertEqual(kwargs["recipe"], (1, 1, 128))
            out.fill_(7)

        flashinfer = types.ModuleType("flashinfer.gemm")
        flashinfer.gemm_fp8_nt_groupwise = Mock(side_effect=groupwise)
        deep_gemm = types.SimpleNamespace(fp8_einsum=Mock(side_effect=einsum))
        launch = Mock()
        ns = dict(
            torch=torch,
            os=os,
            is_sm120=lambda device: sm120,
            deep_gemm=deep_gemm,
            logging=logging,
            time=time,
            partial=partial,
            model_warm_up_enabled=lambda: True,
            _is_cuda_device=lambda device: True,
            _assert_not_capturing=Mock(),
            _get_deep_gemm_num_sms=lambda device: 128,
            _generate_dense_gemm_warmup_m_grid=lambda **kwargs: (1, m),
            _BATCHED_FP8_EINSUM_JIT_WARMED_KEYS=set(),
            _dist_rank=lambda: 0,
            _run_deepgemm_warmup_launch_with_retry=lambda name, label, fn, **kw: fn(),
            _run_deepgemm_warmup_launches_serialized=lambda name, fn: fn(),
            _launch_dummy_batched_fp8_einsum=launch,
            _sync_cuda=Mock(),
            _release_cuda_cache=Mock(),
        )
        project = source_fn(
            FP8 + "attention.py", "_wo_a_einsum_from_fp8", ns, "AttentionFP8"
        )
        warmup = source_fn(
            "rtp_llm/models_py/modules/dsv4/dsv4_kernel_jit_warmup.py",
            "warmup_batched_fp8_einsum_jit",
            ns,
        )
        env = {} if flag is None else {"DSV4_SM120_WOA_EINSUM": flag}
        key, info = (groups, rank, width), {"weights": owner._wo_a_stk_w}
        with patch.dict(os.environ, env, clear=True), patch.dict(
            sys.modules, {"flashinfer.gemm": flashinfer}
        ):
            out = project(owner, activation, scales, 1, m)
            warmup({key: info}, max_m=m, device=torch.device("cpu"))
            warmup({key: info}, max_m=m, device=torch.device("cpu"))
        self.assertTrue(
            torch.equal(out, torch.full((1, m, groups, rank), 7, dtype=torch.bfloat16))
        )
        use_groupwise = sm120 and flag != "1"
        self.assertEqual(
            flashinfer.gemm_fp8_nt_groupwise.call_count, groups if use_groupwise else 0
        )
        self.assertEqual(deep_gemm.fp8_einsum.call_count, 0 if use_groupwise else 1)
        self.assertEqual(launch.call_count, 0 if use_groupwise else 2)
        if not use_groupwise:
            self.assertEqual(
                [call.kwargs["m_value"] for call in launch.call_args_list], [1, m]
            )
            for call in launch.call_args_list:
                self.assertEqual(call.kwargs["key"], key)
                self.assertIs(call.kwargs["info"], info)


if __name__ == "__main__":
    unittest.main(verbosity=2)
