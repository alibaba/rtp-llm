"""RTP server entrypoint that also installs overrides in spawn children."""
import runpy
import sys
import json

sys.dont_write_bytecode = True

from qwen36_overlay import install, preflight_required_kernels

install()

if __name__ == "__main__":
    print("RTP_RUNTIME_PREFLIGHT " + json.dumps(preflight_required_kernels()), flush=True)
    runpy.run_module("rtp_llm.start_server", run_name="__main__")
