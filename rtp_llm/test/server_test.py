import argparse
import os
import shlex
import sys
from unittest import TestCase, main
from unittest.mock import patch

print(os.getcwd())
print(
    "PYTHONPATH="
    + os.environ["PYTHONPATH"]
    + " LD_LIBRARY_PATH="
    + os.environ["LD_LIBRARY_PATH"]
    + " "
    + sys.executable
    + " "
)

from rtp_llm.start_server import main as server_main


def _split_unittest_and_server_args(argv):
    r"""
    Split the argv into two parts: one for unittest and one for server.

    This allows us to run the server_test with the server_args directly.

    Usage:
    ```
    --server_args="--model_type qwen_2
        --checkpoint_path <model_checkpoint_path>
        --start_port 8088"

    or

    -- \
    --model_type=qwen_2 \
    --checkpoint_path=<model_checkpoint_path> \
    --start_port=8088
    ```

    For bazel
    ```
    --test_arg=--server_args="--model_type qwen_2
        --checkpoint_path <model_checkpoint_path>
        --start_port 8088"

    or

    --test_arg=-- \
    --test_arg=--model_type=qwen_2 \
    --test_arg=--checkpoint_path=<model_checkpoint_path> \
    --test_arg=--start_port=8088
    ```
    """
    argv = list(argv)
    try:
        separator_index = argv.index("--")
    except ValueError:
        unittest_argv = argv
        passthrough_argv = []
    else:
        unittest_argv = argv[:separator_index]
        passthrough_argv = argv[separator_index + 1 :]

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--server_args", default="")
    wrapper_args, filtered_unittest_args = parser.parse_known_args(unittest_argv[1:])
    server_args = shlex.split(wrapper_args.server_args)

    return [argv[0]] + filtered_unittest_args, [
        argv[0]
    ] + server_args + passthrough_argv


_UNITTEST_ARGV, _SERVER_ARGV = _split_unittest_and_server_args(sys.argv)


class ServerTest(TestCase):
    def test_simple(self):
        with patch.object(sys, "argv", _SERVER_ARGV):
            server_main()


if __name__ == "__main__":
    main(argv=_UNITTEST_ARGV)
