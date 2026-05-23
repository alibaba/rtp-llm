import unittest

from rtp_llm.config.generate_config import RoleType
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.server.host_service import HostServiceArgs


class FakePDSepConfig:
    def __init__(self, role_type: RoleType, decode_entrance: bool = False):
        self.role_type = role_type
        self.decode_entrance = decode_entrance

    def to_string(self) -> str:
        return f"role_type: {self.role_type.name}"


class BackendRPCServerVisitorTest(unittest.TestCase):
    def test_pdfusion_routes_to_pdfusion_domain(self):
        host_args = HostServiceArgs(pdfusion_domain="127.0.0.1:12345")
        pd_sep_config = FakePDSepConfig(RoleType.PDFUSION)

        roles = BackendRPCServerVisitor.get_backend_role_list(
            pd_sep_config, host_args
        )

        self.assertEqual([RoleType.PDFUSION], roles)

    def test_pdfusion_without_domain_has_no_backend_role(self):
        host_args = HostServiceArgs()
        pd_sep_config = FakePDSepConfig(RoleType.PDFUSION)

        roles = BackendRPCServerVisitor.get_backend_role_list(
            pd_sep_config, host_args
        )

        self.assertEqual([], roles)

    def test_frontend_keeps_all_configured_domains(self):
        host_args = HostServiceArgs(
            pdfusion_domain="127.0.0.1:12345",
            prefill_domain="127.0.0.1:12346",
            decode_domain="127.0.0.1:12347",
        )
        pd_sep_config = FakePDSepConfig(RoleType.FRONTEND)

        roles = BackendRPCServerVisitor.get_backend_role_list(
            pd_sep_config, host_args
        )

        self.assertEqual(
            [RoleType.DECODE, RoleType.PREFILL, RoleType.PDFUSION], roles
        )


if __name__ == "__main__":
    unittest.main()
