import os
import sys

from rtp_llm.test.smoke.scr_checkpoint_probe import BlockingTestEpsilon, PinnedProbe

# Spawn reimports this module, so every application participant subscribes before loading CUDA.
if "SCR_TEST_BARRIER_SOCKET" in os.environ:
    sys.modules["epsilon"] = BlockingTestEpsilon(PinnedProbe())


if __name__ == "__main__":
    from rtp_llm.start_server import main

    main()
