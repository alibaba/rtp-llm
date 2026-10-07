import json
import socket
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

from flexlb_ft import harness
from flexlb_ft.harness import (
    EnvManager,
    EnvSpec,
    FlexEnv,
    port_in_use,
)


class PortProbeTest(unittest.TestCase):

    def test_detects_local_listener_and_released_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)

            port = listener.getsockname()[1]
            self.assertTrue(port_in_use(port))
        self.assertFalse(port_in_use(port))


class MasterEnvironmentTest(unittest.TestCase):

    def test_passes_request_lifecycle_budget_in_config_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = FlexEnv(EnvSpec(discovery="none"), root / "env", 44999)
            master_env = EnvManager(root, verbose=False)._master_env(env)
            config = json.loads(master_env["FLEXLB_CONFIG"])
            self.assertEqual(3, config["schemaVersion"])
            self.assertEqual(60_000, config["requestLifecycle"]["request"]["timeoutMs"])
            self.assertEqual(2.0, config["requestLifecycle"]["decision"]["lifetime"])
            self.assertNotIn("FLEXLB_MONITOR_PROVIDER", master_env)
            self.assertNotIn("FLEXLB_MONITOR_METRIC_WHITELIST", master_env)

    def test_passes_spring_logging_paths_to_master_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "env"
            log_directory = run_dir / "master_logs"
            log_properties = [f"--flexlb.log.path={log_directory}",
                              f"--flexlb.log.app-path={log_directory}"]
            env = FlexEnv(
                EnvSpec(master_profile="none", discovery="none",
                        master_extra_args=log_properties),
                run_dir,
                44999,
            )
            manager = EnvManager(Path(tmp), verbose=False)
            jar = Path(tmp) / "flexlb-api.jar"
            jar.touch()
            with (
                patch.object(harness, "API_JAR", jar),
                patch.object(harness, "resolve_java21", return_value="/fixture/java"),
                patch.object(manager, "_master_ports_in_use", return_value=[]),
                patch.object(Path, "home", return_value=Path(tmp)),
                patch.object(harness.ProcessOps, "start",
                             side_effect=RuntimeError("launch captured")) as start_process,
                self.assertRaisesRegex(RuntimeError, "launch captured"),
            ):
                manager.start_master(env)
            argv = start_process.call_args.args[0]
            for property_argument in log_properties:
                self.assertIn(property_argument, argv)

            logback = ET.parse(harness.FLEXLB_DIR / "flexlb-api/src/main/resources/logback-spring.xml")
            sources = {element.attrib["name"]: element.attrib["source"]
                       for element in logback.findall("springProperty")}
            self.assertEqual("flexlb.log.path", sources["LOG_PATH"])
            self.assertEqual("flexlb.log.app-path", sources["APP_LOG_PATH"])


if __name__ == "__main__":
    unittest.main()
