import subprocess
import sys

import torch

from rtp_llm.test.utils.device_resource import DeviceResource

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_runner.py <path_to_cc_test>")
        sys.exit(1)

    if torch.cuda.is_available() and torch.cuda.device_count() >= 4:
        with DeviceResource(4) as device_resource:
            test_path = sys.argv[1]
            subprocess.run([test_path], check=True)
    else:
        print("DistributedTest skipped due to insufficient devices\n")
