# -*- coding: utf-8 -*-

import os
import unittest
from unittest import mock

import torch

from rtp_llm.models_py.triton_kernels.fla import chunk

EXPECTED_QWEN35_36_RUNTIME_SHAPES = frozenset(
    {
        (16, 16, 128, 128),
        (8, 8, 128, 128),
        (16, 32, 128, 128),
        (8, 16, 128, 128),
        (16, 48, 128, 128),
        (8, 24, 128, 128),
        (16, 64, 128, 128),
        (8, 32, 128, 128),
        (4, 16, 128, 128),
        (2, 8, 128, 128),
    }
)


def _inputs_for_shape(shape, dtype=torch.bfloat16):
    hg, h, k_dim, v_dim = shape
    q = torch.empty(1, 1, hg, k_dim, dtype=dtype)
    k = torch.empty(1, 1, hg, k_dim, dtype=dtype)
    v = torch.empty(1, 1, h, v_dim, dtype=dtype)
    beta = torch.empty(1, 1, h, dtype=dtype)
    return q, k, v, beta


class FlyDSLChunkGDNShapeGateTest(unittest.TestCase):
    def test_all_qwen35_36_target_shapes_recognized(self):
        self.assertEqual(
            chunk.FLYDSL_CHUNK_GDN_TARGET_SHAPES, EXPECTED_QWEN35_36_RUNTIME_SHAPES
        )
        # ENABLED is a strict subset: (8,8,128,128) excluded due to perf regression
        self.assertTrue(chunk.FLYDSL_CHUNK_GDN_ENABLED_SHAPES < EXPECTED_QWEN35_36_RUNTIME_SHAPES)
        self.assertEqual(
            EXPECTED_QWEN35_36_RUNTIME_SHAPES - chunk.FLYDSL_CHUNK_GDN_ENABLED_SHAPES,
            {(8, 8, 128, 128)},
        )

    def test_use_flydsl_accepts_every_enabled_shape_on_amd_bf16(self):
        with mock.patch.dict(os.environ, {"USE_FLYDSL": "1"}, clear=False):
            chunk._use_flydsl_chunk_gdn.cache_clear()
            with mock.patch.object(chunk, "is_amd", True):
                for shape in sorted(chunk.FLYDSL_CHUNK_GDN_ENABLED_SHAPES):
                    q, k, v, beta = _inputs_for_shape(shape)
                    with self.subTest(shape=shape):
                        self.assertTrue(
                            chunk.is_flydsl_chunk_gdn_shape_supported(
                                q=q,
                                k=k,
                                v=v,
                                beta=beta,
                            )
                        )

    def test_perf_excluded_shape_is_rejected(self):
        """(8,8,128,128) is a valid target but excluded due to perf regression."""
        excluded = (8, 8, 128, 128)
        self.assertIn(excluded, chunk.FLYDSL_CHUNK_GDN_TARGET_SHAPES)
        self.assertNotIn(excluded, chunk.FLYDSL_CHUNK_GDN_ENABLED_SHAPES)
        q, k, v, beta = _inputs_for_shape(excluded)
        with mock.patch.dict(os.environ, {"USE_FLYDSL": "1"}, clear=False):
            chunk._use_flydsl_chunk_gdn.cache_clear()
            with mock.patch.object(chunk, "is_amd", True):
                self.assertFalse(
                    chunk.is_flydsl_chunk_gdn_shape_supported(q, k, v, beta)
                )

    def test_full_gate_rejects_env_off(self):
        """The complete gate expression must reject when USE_FLYDSL=0."""
        supported = (8, 32, 128, 128)
        q, k, v, beta = _inputs_for_shape(supported)
        with mock.patch.object(chunk, "is_amd", True):
            with mock.patch.dict(os.environ, {"USE_FLYDSL": "0"}, clear=False):
                chunk._use_flydsl_chunk_gdn.cache_clear()
                gate_result = (
                    chunk.is_flydsl_chunk_gdn_enabled()
                    and chunk.is_flydsl_chunk_gdn_shape_supported(q, k, v, beta)
                    and chunk.is_flydsl_chunk_gdn_length_supported(q)
                )
                self.assertFalse(gate_result)

    def test_gate_rejects_unsupported_shape_and_wrong_dtype(self):
        supported = (8, 32, 128, 128)
        with mock.patch.object(chunk, "is_amd", True):
            with mock.patch.dict(os.environ, {"USE_FLYDSL": "1"}, clear=False):
                chunk._use_flydsl_chunk_gdn.cache_clear()

                bad_q, bad_k, bad_v, bad_beta = _inputs_for_shape((3, 12, 128, 128))
                self.assertFalse(
                    chunk.is_flydsl_chunk_gdn_shape_supported(
                        bad_q,
                        bad_k,
                        bad_v,
                        bad_beta,
                    )
                )

                fp32_q, fp32_k, fp32_v, fp32_beta = _inputs_for_shape(
                    supported, dtype=torch.float32
                )
                self.assertFalse(
                    chunk.is_flydsl_chunk_gdn_shape_supported(
                        fp32_q,
                        fp32_k,
                        fp32_v,
                        fp32_beta,
                    )
                )


if __name__ == "__main__":
    unittest.main()
