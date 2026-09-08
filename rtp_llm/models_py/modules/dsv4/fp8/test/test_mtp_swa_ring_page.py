import unittest

from rtp_llm.models_py.modules.dsv4.fp8.decode.attention_kernels import (
    _validate_rich_pool_page,
)


class MtpSwaRingPageTest(unittest.TestCase):
    def test_normal_page_is_accepted(self):
        _validate_rich_pool_page(
            actual_page=128,
            expected_page=128,
            allow_ring_extension=True,
            name="swa",
        )

    def test_gamma_one_even_aligned_page_is_accepted(self):
        _validate_rich_pool_page(
            actual_page=130,
            expected_page=128,
            allow_ring_extension=True,
            name="swa",
        )

    def test_odd_or_compressed_pool_extension_is_rejected(self):
        for actual_page, allow_ring_extension in ((129, True), (130, False)):
            with self.subTest(
                actual_page=actual_page,
                allow_ring_extension=allow_ring_extension,
            ):
                with self.assertRaisesRegex(ValueError, "unsupported MODEL1 page"):
                    _validate_rich_pool_page(
                        actual_page=actual_page,
                        expected_page=128,
                        allow_ring_extension=allow_ring_extension,
                        name="swa",
                    )


if __name__ == "__main__":
    unittest.main()
