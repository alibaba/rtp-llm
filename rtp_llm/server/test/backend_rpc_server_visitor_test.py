import unittest
from unittest.mock import MagicMock

from rtp_llm.config.generate_config import RoleAddr, RoleType
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor


class BackendRPCServerVisitorHealthTest(unittest.TestCase):
    def _make_visitor(self):
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor.host_service = MagicMock()
        return visitor

    def test_master_only_route_is_ready_when_master_is_cached(self):
        visitor = self._make_visitor()
        visitor.backend_role_list = []
        visitor.host_service.get_master_addr.return_value = "127.0.0.1:12345"

        self.assertTrue(visitor.is_backend_service_ready())
        visitor.host_service.get_backend_role_addrs.assert_not_called()

    def test_master_only_route_is_not_ready_without_master(self):
        visitor = self._make_visitor()
        visitor.backend_role_list = []
        visitor.host_service.get_master_addr.return_value = None

        self.assertFalse(visitor.is_backend_service_ready())
        visitor.host_service.get_backend_role_addrs.assert_not_called()

    def test_direct_role_route_still_checks_required_roles(self):
        visitor = self._make_visitor()
        visitor.backend_role_list = [RoleType.PDFUSION]
        visitor.host_service.get_master_addr.return_value = None
        visitor.host_service.get_backend_role_addrs.return_value = [
            RoleAddr(
                role=RoleType.PDFUSION,
                ip="127.0.0.1",
                grpc_port=12346,
                http_port=12345,
            )
        ]

        self.assertTrue(visitor.is_backend_service_ready(refresh=True))
        visitor.host_service.get_backend_role_addrs.assert_called_once_with(
            [RoleType.PDFUSION], True
        )


if __name__ == "__main__":
    unittest.main()
