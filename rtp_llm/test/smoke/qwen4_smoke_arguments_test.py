import ast
import shlex
import unittest
from pathlib import Path

from rtp_llm.config.qwen4_collective_config import Qwen4CollectiveOptions
from rtp_llm.config.server_config_setup import set_parallelism_config
from rtp_llm.server.server_args import server_args


class Qwen4SmokeArgumentsTest(unittest.TestCase):
    def test_public_smoke_arguments_parse_with_current_native_topology(self):
        tree = ast.parse((Path(__file__).parent / "suites_sm120.bzl").read_text())
        args = next(
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "_QWEN4_SMOKE_ARGS"
                for t in node.targets
            )
        )
        self.assertNotIn("--ffn_sp_size", args)
        for backend in ("nccl", "sm120_ipc"):
            configs = server_args.setup_args(
                shlex.split(args) + ["--qwen4_all_reduce_backend", backend]
            )
            pc = configs.parallelism_config
            set_parallelism_config(pc)
            self.assertEqual((pc.tp_size, pc.ep_size), (2, 2))
            Qwen4CollectiveOptions(backend).validate_topology(pc)


if __name__ == "__main__":
    unittest.main()
