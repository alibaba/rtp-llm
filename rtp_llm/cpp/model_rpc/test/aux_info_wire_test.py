import os
import subprocess
import unittest

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import GenerateOutputsPB


LONG_DURATION_US = (1 << 31) + 12345


class AuxInfoWireTest(unittest.TestCase):

    def test_cpp_serialized_long_duration_response(self):
        fixture_path = os.path.join(
            os.environ["TEST_SRCDIR"],
            os.environ["TEST_WORKSPACE"],
            "rtp_llm/cpp/model_rpc/test/aux_info_wire_fixture",
        )
        serialized = subprocess.run(
            [fixture_path], check=True, stdout=subprocess.PIPE
        ).stdout

        outputs_pb = GenerateOutputsPB()
        outputs_pb.ParseFromString(serialized)
        aux_info = outputs_pb.flatten_output.aux_info[0]

        self.assertEqual(aux_info.cost_time_us, LONG_DURATION_US)
        self.assertEqual(aux_info.first_token_cost_time_us, LONG_DURATION_US - 1)
        self.assertEqual(aux_info.wait_time_us, LONG_DURATION_US - 2)


if __name__ == "__main__":
    unittest.main()
