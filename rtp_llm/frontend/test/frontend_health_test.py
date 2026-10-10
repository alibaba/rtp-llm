import ast
import asyncio
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import HTTPException


class FrontendHealthTest(unittest.IsolatedAsyncioTestCase):
    def checker(self, *, required=True, separated=False, embedding=False):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "frontend_app.py").read_text())
        app_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FrontendApp")
        create = next(node for node in app_class.body if isinstance(node, ast.FunctionDef) and node.name == "create_app")
        checker = next(node for node in create.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "check_constraint_tree_ready")
        request = AsyncMock(return_value="ok")
        namespace = {
            "asyncio": asyncio, "HTTPException": HTTPException, "async_request_server": request,
            "tree_required": required,
            "self": SimpleNamespace(separated_frontend=separated,
                frontend_server=SimpleNamespace(is_embedding=embedding),
                server_config=SimpleNamespace(http_port=23495)),
        }
        exec(compile(ast.fix_missing_locations(ast.Module(body=[checker], type_ignores=[])), "frontend_app.py", "exec"), namespace)
        return namespace["check_constraint_tree_ready"], request

    async def test_native_readiness_is_required_even_when_grpc_is_alive(self):
        checker, request = self.checker()
        for response in (None, {"error": "HTTP Error 503"}, {"error": "Connection failed"}):
            request.return_value = response
            with self.assertRaises(HTTPException) as error:
                await checker()
            self.assertEqual(503, error.exception.status_code)
        request.return_value = "ok"
        await checker()
        request.assert_awaited_with("get", 23495, "health", {})

    async def test_timeout_is_unready(self):
        checker, request = self.checker()
        request.side_effect = asyncio.TimeoutError
        with self.assertRaises(HTTPException) as error:
            await checker()
        self.assertEqual(503, error.exception.status_code)

    async def test_disabled_embedding_and_separated_frontends_keep_existing_checks(self):
        for settings in ({"required": False}, {"separated": True}, {"embedding": True}):
            checker, request = self.checker(**settings)
            await checker()
            request.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
