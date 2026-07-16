import logging
import logging.config
import multiprocessing
import os
import tempfile
from unittest import TestCase, main, mock

from rtp_llm.config.log_config import get_logging_config
from rtp_llm.test.utils.maga_server_manager import MagaServerManager


def _write_application_log(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    logging.config.dictConfig(get_logging_config(log_dir, world_rank=3))
    logging.info(
        "MMScheduler: forward batch=2 requests, images=2 "
        "(first cross-request merge)"
    )
    logging.shutdown()


class MagaServerManagerTest(TestCase):
    def test_application_logs_follow_server_logging_layout(self):
        manager = MagaServerManager(port="12345", role_name="vit")

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ,
                {
                    "MAGA_SERVER_WORK_DIR": tmp,
                    "TEST_UNDECLARED_OUTPUTS_DIR": tmp,
                },
            ), mock.patch(
                "rtp_llm.test.utils.maga_server_manager.subprocess.Popen"
            ) as popen, mock.patch.object(
                manager, "wait_sever_done", return_value=True
            ):
                self.assertTrue(
                    manager.start_server(
                        model_path="/tmp/model",
                        model_type="fake_model",
                        log_to_file=False,
                    )
                )
                child_args = popen.call_args.kwargs
                self.assertEqual(child_args["cwd"], tmp)
                self.assertEqual(child_args["env"]["LOG_PATH"], "vit_logs")
            manager._server_process = None

            # process.log is only stdout/stderr and deliberately has no scheduler
            # signal. The configured Python application logger writes main_3.log.
            assert manager.log_file_path is not None
            os.makedirs(os.path.dirname(manager.log_file_path), exist_ok=True)
            with open(manager.log_file_path, "w"):
                pass

            app_log_dir = os.path.join(tmp, "vit_logs")
            writer = multiprocessing.Process(
                target=_write_application_log, args=(app_log_dir,)
            )
            writer.start()
            writer.join(timeout=5.0)
            self.assertEqual(writer.exitcode, 0)

            self.assertEqual(
                manager.application_log_file_paths,
                [os.path.join(app_log_dir, "main_3.log")],
            )
            with open(manager.application_log_file_paths[0]) as app_log:
                self.assertIn("MMScheduler: forward batch=2", app_log.read())
            with open(manager.log_file_path) as process_log:
                self.assertNotIn("MMScheduler: forward batch=2", process_log.read())

if __name__ == "__main__":
    main()
