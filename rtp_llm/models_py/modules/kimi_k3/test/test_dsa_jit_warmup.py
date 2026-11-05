"""CPU tensor-layout/dispatch regressions, not CUDA compilation coverage."""

import ast
import logging
import time
import types
import unittest
from itertools import product
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parent.parent


def load_functions(path, *, class_name=None):
    # Execute the real function bodies without importing their CUDA-only
    # dependencies or the engine extension. Tensor views still use real torch.
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name is not None:
        body = next(
            node.body for node in body if getattr(node, "name", "") == class_name
        )
    tree.body = [node for node in body if isinstance(node, ast.FunctionDef)]
    module = types.ModuleType(path.stem)
    module.__dict__.update(torch=torch, logging=logging, time=time)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


class DsaJitWarmupTest(unittest.TestCase):
    def setUp(self):
        self.load_warmup()

    def load_warmup(self):
        self.warmup = load_functions(ROOT / "dsa_jit_warmup.py")
        for name in (
            "_warmup_grouped_norm",
            "_warmup_decode_metadata",
            "_warmup_prefix",
            "_warmup_native64",
            "_warmup_native_score",
        ):
            setattr(self.warmup, name, mock.Mock())
        self.warmup._run_deepgemm_warmup_launches_serialized = (
            lambda label, launch: launch()
        )

    def indexer_op(self, fp8):
        name = "fp8_indexer_op.py" if fp8 else "bf16_indexer_op.py"
        class_name = "Fp8IndexerOp" if fp8 else "Bf16DraftIndexerOp"
        implementation = load_functions(
            ROOT.parent / "base" / "cuda" / name, class_name=class_name
        )
        op = NS(dim=128)
        op.cache_view = types.MethodType(implementation.cache_view, op)
        if fp8:
            op.native_score = object()
        return op

    def model(self, dtype, fp8_indexer, page, *, layers=1):
        features = 656 if dtype in (torch.uint8, torch.float8_e4m3fn) else 576
        sfa = torch.empty((3, page, features), dtype=dtype)
        sfa.view(torch.uint8).fill_(0xA5)
        indexer = torch.full(
            (3, page * (132 if fp8_indexer else 256)), 0x5A, dtype=torch.uint8
        )
        caches = [
            NS(tag="default", seq_size_per_block=page, kv_cache_base=sfa),
            NS(tag="indexer_kv", seq_size_per_block=page, kv_cache_base=indexer),
        ]
        attention = NS(
            kv_rank=512,
            suffix_dim=64,
            indexer=NS(indexer_op=self.indexer_op(fp8_indexer)),
        )
        model = NS(
            layers=[NS(index=index, attention=attention) for index in range(layers)],
            kv_cache=NS(
                get_layer_cache_groups=lambda _: caches,
                get_seq_size_per_block=lambda _: 4096,
            ),
            config=NS(gen_num_per_cycle=3),
            parallelism_config=NS(tp_size=8, tp_rank=1),
        )
        return model, caches

    def test_sfa_and_indexer_use_their_own_codec_in_both_roles(self):
        for dtype, fp8_indexer, page, decode in product(
            (torch.float8_e4m3fn, torch.uint8, torch.bfloat16),
            (True, False),
            (64, 128),
            (False, True),
        ):
            with self.subTest(
                dtype=dtype, fp8_indexer=fp8_indexer, page=page, decode=decode
            ):
                self.load_warmup()
                model, caches = self.model(dtype, fp8_indexer, page)
                before = [
                    cache.kv_cache_base.view(torch.uint8).clone() for cache in caches
                ]
                with mock.patch.object(torch.cuda, "synchronize"):
                    self.warmup.warmup_kimi_k31_dsa_jit(
                        model, NS(is_decode_role=decode)
                    )
                self.assertEqual(self.warmup._warmup_decode_metadata.call_count, 2)
                calls = self.warmup._warmup_prefix.call_args_list
                self.assertEqual(len(calls), 0 if decode else 2)
                if not decode:
                    sfa, indexer = [call.args[0] for call in calls]
                    packed_sfa = dtype != torch.bfloat16
                    self.assertEqual(sfa.shape, (3, page, 656 if packed_sfa else 576))
                    self.assertEqual(
                        sfa.dtype, torch.uint8 if packed_sfa else torch.bfloat16
                    )
                    self.assertFalse(calls[0].kwargs["indexer_fp8"])
                    self.assertEqual(
                        indexer.shape, (3, page, 132 if fp8_indexer else 128)
                    )
                    self.assertEqual(
                        indexer.dtype, torch.uint8 if fp8_indexer else torch.bfloat16
                    )
                    self.assertEqual(calls[1].kwargs["indexer_fp8"], fp8_indexer)
                    for view, cache in zip((sfa, indexer), caches):
                        self.assertEqual(
                            view.data_ptr(), cache.kv_cache_base.data_ptr()
                        )
                self.assertEqual(
                    self.warmup._warmup_native64.call_count, int(fp8_indexer)
                )
                self.assertEqual(
                    self.warmup._warmup_native_score.call_count, int(fp8_indexer)
                )
                for cache, original in zip(caches, before):
                    self.assertTrue(
                        torch.equal(cache.kv_cache_base.view(torch.uint8), original)
                    )

    def test_original_failed_allocation_geometry_without_allocating_gigabytes(self):
        model, caches = self.model(torch.float8_e4m3fn, True, 128)
        caches[0].kv_cache_base = torch.empty(
            (84592, 128, 656), device="meta", dtype=torch.float8_e4m3fn
        )
        self.assertEqual(caches[0].kv_cache_base.numel(), 7103021056)
        with mock.patch.object(torch.cuda, "synchronize"):
            self.warmup.warmup_kimi_k31_dsa_jit(model, NS(is_decode_role=False))
        view = self.warmup._warmup_prefix.call_args_list[0].args[0]
        self.assertEqual(view.shape, (84592, 128, 656))
        self.assertEqual(view.dtype, torch.uint8)

    def test_cache_geometry_is_only_warmed_once_across_layers(self):
        model, _ = self.model(torch.float8_e4m3fn, True, 128, layers=4)
        with mock.patch.object(torch.cuda, "synchronize"):
            self.warmup.warmup_kimi_k31_dsa_jit(model, NS(is_decode_role=False))
        self.assertEqual(self.warmup._warmup_prefix.call_count, 2)
        self.assertEqual(self.warmup._warmup_native64.call_count, 1)
        self.assertEqual(self.warmup._warmup_native_score.call_count, 1)

    def test_non_dsa_layers_and_absent_cache_are_skipped(self):
        self.warmup.warmup_kimi_k31_dsa_jit(NS(kv_cache=None), NS(is_decode_role=True))
        model, _ = self.model(torch.float8_e4m3fn, True, 128)
        model.layers[0].attention.indexer = None
        self.warmup.warmup_kimi_k31_dsa_jit(model, NS(is_decode_role=True))
        self.warmup._warmup_decode_metadata.assert_not_called()
        self.warmup._warmup_native_score.assert_not_called()


if __name__ == "__main__":
    unittest.main()
