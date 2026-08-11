import asyncio
import json
from unittest import TestCase, main

from rtp_llm.frontend.frontend_server import FrontendServer
from rtp_llm.utils.concurrency_controller import ConcurrencyException


class _RaisingController:
    def increment(self):
        raise ConcurrencyException("Concurrency limit 128 reached")

    def decrement(self):
        pass


class ChatCompletionConcurrencyLimitTest(TestCase):
    """Concurrency-limit overflow on /v1/chat/completions must map to HTTP 429
    (409_CONCURRENCY_LIMIT_ERROR), never leak through ASGI as a 500."""

    def test_chat_completion_returns_429_when_concurrency_limit_reached(self):
        server = FrontendServer.__new__(FrontendServer)
        server._global_controller = _RaisingController()

        rep = asyncio.get_event_loop().run_until_complete(
            FrontendServer.chat_completion(server, request=None, raw_request=None)
        )

        self.assertEqual(rep.status_code, 429)
        body = json.loads(rep.body)
        self.assertEqual(body.get("error_code"), 409)
        self.assertEqual(body.get("error_code_str"), "409_CONCURRENCY_LIMIT_ERROR")

    def test_track_business_request_has_concurrency_safety_net(self):
        # Source-level guard for the frontend_app safety net (closure is not
        # directly constructible without a full app): the except must exist
        # between the call() await and the finally block.
        import inspect

        import rtp_llm.frontend.frontend_app as app_mod

        src = inspect.getsource(app_mod)
        self.assertIn("except ConcurrencyException", src)
        self.assertIn("status_code=429", src)


if __name__ == "__main__":
    main()
