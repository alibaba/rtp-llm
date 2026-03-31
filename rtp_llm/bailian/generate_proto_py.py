"""
Generate Python gRPC stubs for predict_v2 and model_config (pre-generated, no runtime generation).
Same as model_rpc: run once before use; do not generate at server startup.

  python -m rtp_llm.bailian.generate_proto_py
  # or: python rtp_llm/bailian/proto/create_grpc_proto.py [output_dir]
"""

import os
import subprocess
import sys


def main():
    bailian_dir = os.path.dirname(os.path.abspath(__file__))
    proto_dir = os.path.join(bailian_dir, "proto")
    script = os.path.join(proto_dir, "create_grpc_proto.py")
    subprocess.run(
        [sys.executable, script, proto_dir],
        check=True,
    )


if __name__ == "__main__":
    main()
