"""Tests for exact MLA decode backend selection."""

import os
from unittest import TestCase, main, mock

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.paged_mla_decode import (
    MLA_DECODE_KERNEL_ENV,
    get_mla_decode_kernel,
)


class PagedMlaDecodeSelectionTest(TestCase):
    def test_defaults_to_auto(self):
        with mock.patch.dict(os.environ):
            os.environ.pop(MLA_DECODE_KERNEL_ENV, None)
            self.assertEqual(get_mla_decode_kernel(), "auto")

    def test_accepts_only_declared_backend_names(self):
        for backend in ("auto", "flashinfer", "tokenspeed_mla", "trtllm_gen"):
            with self.subTest(backend=backend), mock.patch.dict(
                os.environ, {MLA_DECODE_KERNEL_ENV: backend}
            ):
                self.assertEqual(get_mla_decode_kernel(), backend)

    def test_rejects_misspellings_and_implicit_normalization(self):
        for backend in ("tokenspeed", "trtllm", "TRTLLM_GEN", " trtllm_gen"):
            with self.subTest(backend=backend), mock.patch.dict(
                os.environ, {MLA_DECODE_KERNEL_ENV: backend}
            ):
                with self.assertRaisesRegex(RuntimeError, "invalid RTP_MLA"):
                    get_mla_decode_kernel()


if __name__ == "__main__":
    main()
