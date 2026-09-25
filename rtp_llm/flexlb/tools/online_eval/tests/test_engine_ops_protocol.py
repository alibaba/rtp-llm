import unittest
from unittest.mock import Mock, patch

from flexlb_ft.engine_ops import EngineOps


class EngineOpsProtocolTest(unittest.TestCase):

    def setUp(self):
        self.ops = EngineOps("127.0.0.1", 28080, 35150)

    def tearDown(self):
        self.ops.close()

    def test_master_control_plane_uses_string_request_ids(self):
        schedule = self.ops.build_schedule_request(123)
        self.assertEqual("123", schedule.request_id)

        stub = Mock()
        with patch.object(
            self.ops.schedule_pb2_grpc,
            "FlexlbServiceStub",
            return_value=stub,
        ):
            self.ops.cancel(123)
        cancel = stub.Cancel.call_args.args[0]
        self.assertEqual("123", cancel.request_id)

    def test_role_addresses_dual_write_legacy_enum_and_string(self):
        response = self.ops.schedule_pb2.FlexlbScheduleResponsePB()
        response.server_status.add(
            role="PREFILL",
            server_ip="127.0.0.1",
            http_port=35150,
            grpc_port=35151,
        )
        input_pb = self.ops.build_generate_input(123)

        self.ops._copy_role_addrs(input_pb, response)

        role_addr = input_pb.generate_config.role_addrs[0]
        self.assertEqual(self.ops.pb2.RoleAddrPB.PREFILL, role_addr.role)
        self.assertEqual("PREFILL", role_addr.role_str)


if __name__ == "__main__":
    unittest.main()
