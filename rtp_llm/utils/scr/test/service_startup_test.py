"""Keep discovery out of imports performed before the template barrier."""

import subprocess
import sys
import textwrap
import unittest


class ScrServiceStartupTest(unittest.TestCase):
    def test_service_imports_leave_discovery_until_first_lookup(self):
        # A fresh interpreter prevents another test's imports from hiding an
        # eager VIPServer dependency in the pre-barrier service import graph.
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent(
                    """
                    import sys
                    import threading
                    from unittest.mock import Mock, patch

                    import rtp_llm.server.backend_manager
                    import rtp_llm.frontend.frontend_app
                    import rtp_llm.dash_sc.app
                    from rtp_llm.server.host_service import VipServerWrapper

                    assert not any(
                        name == "rtp_llm.vipserver" or name.startswith("rtp_llm.vipserver.")
                        for name in sys.modules
                    ), "pre-barrier service imports loaded VIPServer"

                    # This lookup is reached by service initialization after
                    # prebind. Exercise the original eager client with no real
                    # network or background threads in this unit test.
                    response = Mock(
                        text="192.0.2.1",
                        json=Mock(return_value={"hosts": [
                            {"valid": True, "ip": "192.0.2.10", "port": "9000"}
                        ]}),
                    )
                    with patch("requests.get", return_value=response) as request, patch.object(
                        threading.Thread, "start"
                    ) as start:
                        service = VipServerWrapper("decode.vip")
                        assert [(host.ip, host.port) for host in service.hosts] == [
                            ("192.0.2.10", "9000")
                        ]
                        assert start.call_count == 2, "VIPServer refresh threads did not start"
                        assert request.call_count == 2, "first lookup did not query VIPServer"
                        service.get_hosts()
                        assert start.call_count == 2
                        assert request.call_count == 2
                    """
                ),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
