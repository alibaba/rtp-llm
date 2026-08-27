import os
import unittest
from unittest import mock

import kimi_k3_full_model_two_host_pd_smoke_driver as driver


class ForwardedOptionalEnvironmentTest(unittest.TestCase):
    def test_forwards_explicit_rdma_hca_allowlist_to_both_roles(self) -> None:
        value = "mlx5_bond_0,mlx5_bond_1"
        with mock.patch.dict(os.environ, {"SMOKE_ACCL_USE_NICS": value}, clear=True):
            self.assertEqual(
                driver.forwarded_optional_environment("prefill")[
                    "SMOKE_ACCL_USE_NICS"
                ],
                value,
            )
            self.assertEqual(
                driver.forwarded_optional_environment("decode")[
                    "SMOKE_ACCL_USE_NICS"
                ],
                value,
            )

    def test_does_not_invent_driver_override_when_unset(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertNotIn(
                "SMOKE_ACCL_USE_NICS",
                driver.forwarded_optional_environment("prefill"),
            )


if __name__ == "__main__":
    unittest.main()
