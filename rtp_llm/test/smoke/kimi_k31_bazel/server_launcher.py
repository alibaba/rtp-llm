"""Bazel imports for the shared K3.1 server launcher."""

from rtp_llm.test.smoke.kimi_k3.k31_server_launcher import *

if __name__ == "__main__":
    shared.main(configure=launch_config, extend_parser=extend_parser)
