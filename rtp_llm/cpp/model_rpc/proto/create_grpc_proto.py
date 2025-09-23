# create_grpc_proto.py
import subprocess
import sys


def main():
    if len(sys.argv) != 3:
        print("Usage: create_grpc_proto.py <proto_file>")
        sys.exit(1)

    proto_file = sys.argv[1]
    output_dir = sys.argv[2]

    subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            "-I.",
            f"--python_out={output_dir}",
            f"--grpc_python_out={output_dir}",
            proto_file,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
