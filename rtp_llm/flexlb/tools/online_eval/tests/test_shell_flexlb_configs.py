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


if __name__ == "__main__":
    unittest.main()
