"""Bazel ownership adapter for the shared traditional smoke worker."""

import os
import subprocess
from pathlib import Path

from rtp_llm.test.smoke.kimi_k3.worker import Worker as SharedWorker
from rtp_llm.test.smoke.kimi_k3.worker import pd_endpoints
from rtp_llm.test.smoke.kimi_k3.two_host_launcher import prefill_workspace_environment
from rtp_llm.test.smoke.kimi_k31_bazel.pd_config import validate_config


class Worker(SharedWorker):
    run_id_prefix = "k31-pd-"

    def prepare_native(self, config, role, run_id):
        if self.config is not None or os.geteuid() == 0:
            raise ValueError("Worker must be new and run as a non-root user")
        return self._prepare_runtime(validate_config(config, local_role=role), role, run_id)

    def launcher_path(self):
        return Path(__file__).with_name("server_launcher.py")

    def launcher_environment(self):
        return prefill_workspace_environment(self.config) if self.role == "prefill" else {}
