import multiprocessing
import os
import time
import unittest
from unittest.mock import patch

from rtp_llm.config.pdfusion_coordination import initialize_pdfusion_coordination
from rtp_llm.config.py_config_modules import PyEnvConfigs


def report_identity(config, writer):
    try:
        fifo = config.runtime_config.fifo_scheduler_config
        writer.send(
            (
                fifo.pdfusion_coord_group_id,
                fifo.pdfusion_schedule_trace,
                fifo.pdfusion_trace_run_id,
            )
        )
    finally:
        writer.close()


class PDFusionCoordinationStartupTest(unittest.TestCase):
    def setUp(self):
        scope = patch.dict(os.environ, {"PDFUSION_CADENCE_ADAPTIVE": "0"})
        scope.start()
        self.addCleanup(scope.stop)
        os.environ.pop("PDFUSION_SCHEDULE_TRACE", None)
        os.environ.pop("PDFUSION_TRACE_RUN_ID", None)

    def test_disabled_mode_does_not_create_group(self):
        config = PyEnvConfigs()
        initialize_pdfusion_coordination(config)
        self.assertEqual(
            config.runtime_config.fifo_scheduler_config.pdfusion_coord_group_id, ""
        )

    def test_restart_and_trace_labels_do_not_reuse_group(self):
        config = PyEnvConfigs()
        fifo = config.runtime_config.fifo_scheduler_config
        fifo.pdfusion_coord_mode = "cadence"
        fifo.pdfusion_trace_run_id = "same-trace-label"
        initialize_pdfusion_coordination(config)
        first = fifo.pdfusion_coord_group_id
        initialize_pdfusion_coordination(config)
        self.assertNotEqual(first, fifo.pdfusion_coord_group_id)
        self.assertEqual(fifo.pdfusion_trace_run_id, "same-trace-label")
        self.assertFalse(fifo.pdfusion_schedule_trace)

    def test_adaptive_validation_matches_native_configuration(self):
        config = PyEnvConfigs()
        for value in ("", "-1", "2", "true"):
            with self.subTest(value=value), patch.dict(
                os.environ, PDFUSION_CADENCE_ADAPTIVE=value
            ):
                with self.assertRaises(ValueError):
                    initialize_pdfusion_coordination(config)
        with patch.dict(os.environ, PDFUSION_CADENCE_ADAPTIVE="01"):
            initialize_pdfusion_coordination(config)
        self.assertRegex(
            config.runtime_config.fifo_scheduler_config.pdfusion_coord_group_id,
            r"^[0-9a-f]{32}$",
        )

    def test_spawned_dp2_dp4_share_internal_identity_without_trace_env(self):
        ctx = multiprocessing.get_context("spawn")
        previous = set()
        for adaptive in (False, True):
            os.environ["PDFUSION_CADENCE_ADAPTIVE"] = "1" if adaptive else "0"
            for ranks in (2, 4):
                with self.subTest(adaptive=adaptive, ranks=ranks):
                    config = PyEnvConfigs()
                    fifo = config.runtime_config.fifo_scheduler_config
                    fifo.pdfusion_coord_mode = "off" if adaptive else "cadence"
                    initialize_pdfusion_coordination(config)
                    group_id = fifo.pdfusion_coord_group_id
                    self.assertRegex(group_id, r"^[0-9a-f]{32}$")
                    self.assertNotIn(group_id, previous)
                    previous.add(group_id)
                    children = []
                    deadline = time.monotonic() + 60
                    try:
                        for _ in range(ranks):
                            reader, writer = ctx.Pipe(duplex=False)
                            process = ctx.Process(
                                target=report_identity, args=(config, writer)
                            )
                            process.start()
                            writer.close()
                            children.append((process, reader))
                        for process, reader in children:
                            self.assertTrue(
                                reader.poll(max(0, deadline - time.monotonic()))
                            )
                            self.assertEqual(reader.recv(), (group_id, False, "unset"))
                            process.join(max(0, deadline - time.monotonic()))
                            self.assertEqual(process.exitcode, 0)
                    finally:
                        for process, reader in children:
                            reader.close()
                            if process.is_alive():
                                process.kill()
                            process.join()


if __name__ == "__main__":
    unittest.main()
