import unittest

from kimi_k3_full_model_pd_nic_selection import (
    DEFAULT_BOND_HCAS,
    HcaValidationError,
    select_default_allowlist,
    validate_allowlist,
)


def inventory(*nics: str) -> tuple[str, str]:
    devices = "device                 node GUID\n" + "\n".join(
        f"    {nic}              0000" for nic in nics
    )
    links = "\n".join(
        f"link {nic}/1 state ACTIVE physical_state LINK_UP netdev reth{index * 2}"
        for index, nic in enumerate(nics)
    )
    return devices, links


class NicSelectionTest(unittest.TestCase):
    def test_selects_default_bonds_when_all_are_active(self) -> None:
        devices, links = inventory(*DEFAULT_BOND_HCAS)

        selected, reason = select_default_allowlist(devices, links)

        self.assertEqual(selected, ",".join(DEFAULT_BOND_HCAS))
        self.assertIsNone(reason)

    def test_falls_back_when_machine_has_only_raw_hcas(self) -> None:
        devices, links = inventory("mlx5_2", "mlx5_3", "mlx5_4", "mlx5_5")

        selected, reason = select_default_allowlist(devices, links)

        self.assertIsNone(selected)
        self.assertIn("absent from ibv_devices", reason)

    def test_falls_back_when_default_bond_set_is_partial(self) -> None:
        devices, links = inventory(*DEFAULT_BOND_HCAS[:-1])

        selected, reason = select_default_allowlist(devices, links)

        self.assertIsNone(selected)
        self.assertIn("mlx5_bond_7", reason)

    def test_explicit_active_subset_is_valid(self) -> None:
        devices, links = inventory(*DEFAULT_BOND_HCAS)

        selected = validate_allowlist(
            "mlx5_bond_0,mlx5_bond_1", devices, links
        )

        self.assertEqual(selected, ("mlx5_bond_0", "mlx5_bond_1"))

    def test_explicit_missing_hca_remains_an_error(self) -> None:
        devices, links = inventory("mlx5_2", "mlx5_3")

        with self.assertRaisesRegex(HcaValidationError, "mlx5_bond_0"):
            validate_allowlist("mlx5_bond_0", devices, links)

    def test_explicit_inactive_hca_remains_an_error(self) -> None:
        devices, _ = inventory("mlx5_bond_0")
        links = "link mlx5_bond_0/1 state DOWN physical_state DISABLED"

        with self.assertRaisesRegex(HcaValidationError, "not ACTIVE/LINK_UP"):
            validate_allowlist("mlx5_bond_0", devices, links)


if __name__ == "__main__":
    unittest.main()
