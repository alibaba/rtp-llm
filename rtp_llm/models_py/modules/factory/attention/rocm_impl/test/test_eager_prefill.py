"""Regression specifications for eager launch plumbing (not performance tests)."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.factory.attention.rocm_impl.eager_prefill import (
    CKPrefillLauncher,
    allocate_pa_workspace,
    short_query_backend,
)


class EagerPrefillWorkspaceTest(unittest.TestCase):
    def test_geometry_and_batch_isolation(self):
        query = torch.empty(3, 8, 128, dtype=torch.bfloat16)
        first = allocate_pa_workspace(query, 1, 3, 2, 4)
        second = allocate_pa_workspace(query, 1, 3, 2, 4)
        self.assertEqual(first["output"].shape, (3, 8, 128))
        self.assertEqual(first["exp_sums"].shape, (1, 2, 4, 12))
        self.assertEqual(first["temporary_output"].shape, (1, 2, 4, 12, 128))
        for name in first:
            self.assertNotEqual(first[name].data_ptr(), second[name].data_ptr())

    def test_override_is_explicit_and_validated(self):
        self.assertEqual(short_query_backend("auto"), "auto")
        self.assertEqual(short_query_backend(" CK "), "ck")
        with self.assertRaises(ValueError):
            short_query_backend("typo")


class CKPrefillLauncherTest(unittest.TestCase):
    @staticmethod
    def tensor(shape, *, cuda=True):
        return SimpleNamespace(
            shape=shape,
            dtype=torch.bfloat16,
            is_cuda=cuda,
            dim=lambda: len(shape),
            element_size=lambda: 2,
            stride=lambda _: 1,
        )

    def test_low_level_arguments_match_public_wrapper_defaults(self):
        launcher = CKPrefillLauncher.__new__(CKPrefillLauncher)
        launcher.head_dim = 128
        launcher.scale = 128**-0.5
        result = object()
        launcher.entrypoint = Mock(return_value=(result, None, None, None))
        q = self.tensor((3, 8, 128))
        k = self.tensor((64, 2, 16, 16, 8))
        v = self.tensor((64, 2, 2, 128, 8))
        metadata = [object() for _ in range(5)]
        output = object()
        actual = launcher(
            q,
            k,
            v,
            *metadata[:3],
            3,
            835,
            causal=True,
            block_table=metadata[3],
            seqlen_k=metadata[4],
            out=output,
        )
        self.assertIs(actual, result)
        args = launcher.entrypoint.call_args.args
        self.assertEqual(len(args), 30)
        self.assertEqual(
            args[8:18], (0.0, 128**-0.5, 0.0, False, True, -1, -1, 0, False, False)
        )
        self.assertIs(args[18], output)
        self.assertEqual(args[19:26], (None,) * 7)
        self.assertIs(args[26], metadata[3])
        self.assertIs(args[27], metadata[4])

    def test_unknown_abi_and_cpu_keep_public_entrypoint(self):
        launcher = CKPrefillLauncher.__new__(CKPrefillLauncher)
        launcher.head_dim = 128
        launcher.entrypoint = None
        public = Mock(return_value=object())
        with patch.dict(
            "sys.modules", {"aiter": SimpleNamespace(mha_batch_prefill_func=public)}
        ):
            result = launcher(
                None,
                None,
                None,
                None,
                None,
                None,
                1,
                833,
                causal=True,
                block_table=None,
                seqlen_k=None,
            )
        self.assertIs(result, public.return_value)
        public.assert_called_once()
