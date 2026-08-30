import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]


class ShellFlexlbConfigTest(unittest.TestCase):
    def test_schema_v2_scripts_do_not_use_removed_routing_fields(self) -> None:
        offenders = []
        for script in sorted(SCRIPT_DIR.glob("*.sh")):
            content = script.read_text(encoding="utf-8")
            if "schemaVersion" in content and "maxWaitVsAverageMultiplier" in content:
                offenders.append(script.name)

        self.assertEqual([], offenders)

    def test_matrix_passes_the_same_config_to_master_and_clients(self) -> None:
        content = (SCRIPT_DIR / "run_matrix_smoke.sh").read_text(encoding="utf-8")

        self.assertIn('"FLEXLB_CONFIG=${FLEXLB_CONFIG}"', content)
        self.assertIn('FLEXLB_CONFIG="${FLEXLB_CONFIG}" PYTHONDONTWRITEBYTECODE=1', content)
        self.assertIn('cmd_args+=(--schedule-mode "${group}")', content)
        self.assertIn('cmd_args+=(--mock-http-port "${MOCK_HTTP_PORT}")', content)


if __name__ == "__main__":
    unittest.main()
