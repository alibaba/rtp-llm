"""Import split wheels in Bazel's production runfiles, without a backend fallback."""

import unittest


class Cuda13DependencyImportTest(unittest.TestCase):
    def test_cutlass_and_fa4(self):
        import cutlass
        from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

        self.assertEqual(cutlass.__version__, "4.7.1")
        self.assertTrue(callable(flash_attn_func))
        self.assertTrue(callable(flash_attn_varlen_func))

    def test_flashinfer_split_cache(self):
        import flashinfer
        import flashinfer_jit_cache

        self.assertTrue(flashinfer.__version__.startswith("0.7.2"))
        self.assertTrue(flashinfer_jit_cache.__version__.startswith("0.7.2"))
        providers = flashinfer_jit_cache.get_jit_cache_providers()
        self.assertTrue(
            any("sm103a" in provider.cuda_architectures for provider in providers)
        )


if __name__ == "__main__":
    unittest.main()
