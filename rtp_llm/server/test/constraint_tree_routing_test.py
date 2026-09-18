"""Tree bootstrap must not put Master scheduling on the local inference path."""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from rtp_llm.config.generate_config import RoleType
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.server.constraint_tree_bootstrap import ConstraintTreeBootstrap


class ConstraintTreeRoutingTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            max_seq_len=128,
            role_type=RoleType.PDFUSION,
            vit_separation=0,
            decode_entrance=False,
        )
        self.input = SimpleNamespace(
            request_id=1,
            prompt_length=3,
            generate_config=SimpleNamespace(
                max_new_tokens=3, force_disable_sp_run=True
            ),
        )

    async def test_local_inference_never_schedules_even_when_tree_master_is_down(self):
        with patch.dict(
            os.environ,
            {
                "MODEL_SERVICE_CONFIG": '{"service_id":"service"}',
                "CONSTRAINT_TREE_MASTER_ENDPOINT": "tree.master.vip",
                "CONSTRAINT_TREE_REQUIRED": "true",
            },
        ), patch("rtp_llm.server.host_service.MasterService") as master, patch(
            "rtp_llm.server.backend_rpc_server_visitor.ModelRpcClient"
        ) as rpc, patch.object(
            ConstraintTreeBootstrap, "_master_from_vip", return_value=None
        ):
            visitor = BackendRPCServerVisitor(self.config)
            visitor.route_ips = AsyncMock(
                side_effect=AssertionError("inference must stay local")
            )
            self.assertFalse(visitor.host_service.service_available)
            bootstrap = ConstraintTreeBootstrap.from_env(
                visitor.host_service, 23495, "PDFUSION"
            )
            with self.assertRaisesRegex(RuntimeError, "not yet discoverable"):
                bootstrap._register_once(Mock())
            for _ in range(10):
                self.assertIs(
                    rpc.return_value.enqueue.return_value,
                    await visitor.enqueue(self.input),
                )
            self.assertEqual(10, rpc.return_value.enqueue.call_count)
            visitor.route_ips.assert_not_awaited()
            master.return_value.get_master_addr.assert_not_called()
            self.assertEqual([], visitor.backend_role_list)

    async def test_explicit_flexlb_routing_is_not_disabled_by_tree_endpoint(self):
        with patch.dict(
            os.environ,
            {
                "MODEL_SERVICE_CONFIG": json.dumps(
                    {
                        "service_id": "service",
                        "use_local": True,
                        "master_endpoint": {
                            "type": "VipServer",
                            "address": "127.0.0.1:7001",
                            "protocol": "http",
                            "path": "/",
                        },
                    }
                ),
                "CONSTRAINT_TREE_MASTER_ENDPOINT": "tree.master.vip",
            },
        ), patch("rtp_llm.server.host_service.MasterService"), patch(
            "rtp_llm.server.backend_rpc_server_visitor.ModelRpcClient"
        ):
            visitor = BackendRPCServerVisitor(self.config)
            visitor.route_ips = AsyncMock()
            self.assertTrue(visitor.host_service.service_available)
            await visitor.enqueue(self.input)
            visitor.route_ips.assert_awaited_once_with(self.input)


if __name__ == "__main__":
    unittest.main()
