import os
import random
import sys
import threading
import time
import traceback
import unittest

import requests

from rtp_llm.start_frontend_server import start_frontend_server


class FrontendAppTest(unittest.TestCase):

    def test_frontend_app_start(self):
        from rtp_llm.config.server_config_setup import setup_and_configure_server
        from rtp_llm.server.server_args.server_args import setup_args
        from rtp_llm.utils.concurrency_controller import init_controller

        """Test that FrontendApp can start successfully."""
        # Setup args and configure server (same as start_server.py main())
        # Keep only script name
        py_env_configs = setup_args()

        # Override with test-specific settings
        py_env_configs.server_config.start_port = 36000
        py_env_configs.server_config.frontend_server_count = 1
        py_env_configs.server_config.rank_id = 0
        py_env_configs.server_config.frontend_server_id = 0
        py_env_configs.server_config.worker_info_port_num = 8
        # Enable fake process mode to avoid loading actual model
        py_env_configs.profiling_debug_logging_config.debug_start_fake_process = 1

        # Setup and configure server (same as start_server.py main())
        setup_and_configure_server(py_env_configs)

        # Initialize concurrency controller (same as start_server.py)
        global_controller = init_controller(
            py_env_configs.concurrency_config, dp_size=1
        )

        # Start frontend server with same parameters as start_server.py (config already has rank_id=0)
        # Use thread to run start_frontend_server since app.start() blocks (server.run() is blocking)
        start_error = [None]
        error_traceback = [None]

        def run_start():
            try:
                start_frontend_server(0, 0, global_controller, py_env_configs)
            except BaseException as e:
                start_error[0] = e
                error_traceback[0] = traceback.format_exc()

        frontend_thread = threading.Thread(target=run_start, daemon=True)
        frontend_thread.start()

        health_check_url = f"http://localhost:{py_env_configs.server_config.server_port}/frontend_health"

        # Wait for initialization and check for errors periodically
        max_wait_time = 30
        check_interval = 1  # Check every 1 second for faster detection
        waited = 0
        health_check_passed = False
        last_error = None
        while waited < max_wait_time:
            if start_error[0]:
                # Error occurred, print detailed error information
                error_msg = f"FrontendApp start failed:\n"
                error_msg += f"Error: {start_error[0]}\n"
                error_msg += f"Type: {type(start_error[0]).__name__}\n"
                if error_traceback[0]:
                    error_msg += f"Traceback:\n{error_traceback[0]}"
                self.fail(error_msg)

            if not frontend_thread.is_alive():
                # Thread died unexpectedly
                error_msg = "FrontendApp thread died unexpectedly"
                if start_error[0]:
                    error_msg += f"\nError: {start_error[0]}"
                    if error_traceback[0]:
                        error_msg += f"\nTraceback:\n{error_traceback[0]}"
                self.fail(error_msg)

            # Check frontend health endpoint
            if not health_check_passed:
                try:
                    response = requests.post(health_check_url, timeout=2)
                    if response.status_code == 200:
                        # FastAPI may return JSON string or plain text, handle both cases
                        response_text = response.text.strip().strip('"').strip("'")
                        if response_text == "ok":
                            health_check_passed = True
                            break
                        else:
                            last_error = f"Unexpected response text: status_code={response.status_code}, text={response.text!r}, stripped={response_text!r}"
                    else:
                        last_error = f"Unexpected status code: status_code={response.status_code}, text={response.text}"
                except requests.exceptions.ConnectionError as e:
                    last_error = f"Connection error: {e}"
                except requests.exceptions.Timeout as e:
                    last_error = f"Timeout error: {e}"
                except Exception as e:
                    last_error = f"Unexpected error: {type(e).__name__}: {e}"

            time.sleep(check_interval)
            waited += check_interval

        # Verify thread is still alive after waiting
        if not frontend_thread.is_alive():
            error_msg = "FrontendApp thread is not alive after initialization"
            if start_error[0]:
                error_msg += f"\nError: {start_error[0]}"
                if error_traceback[0]:
                    error_msg += f"\nTraceback:\n{error_traceback[0]}"
            self.fail(error_msg)

        # Verify health check passed
        if not health_check_passed:
            error_msg = (
                f"Frontend health check failed: unable to reach {health_check_url} "
                f"after {max_wait_time} seconds"
            )
            if last_error:
                error_msg += f"\nLast error: {last_error}"
            self.fail(error_msg)


if __name__ == "__main__":
    print(f"LD_LIBRARY_PATH: {os.environ['LD_LIBRARY_PATH']}")
    unittest.main()
