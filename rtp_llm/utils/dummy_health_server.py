"""Zero-GPU dummy health server used to hold a Whale pod as a dev container.

Set ``RTP_DUMMY_SERVE=1`` to make :mod:`rtp_llm.start_server` serve a tiny
OpenAI-compatible HTTP surface instead of starting the engine. The server
answers the platform health probes (``GET /health`` on ``START_PORT``) while
importing nothing but the standard library, so the process never initializes
CUDA and the container's GPUs stay free for interactive development.

This module must stay stdlib-only: it deliberately imports neither torch nor
any engine module.
"""

import json
import logging
import os
import signal
import socketserver
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

DEFAULT_START_PORT = 12233
# Keep in sync with the RTP_DUMMY_SERVE check in rtp_llm/start_server.py.
ENABLE_VALUES = ("1", "true")
DUMMY_MODEL_ID = "dummy-model"
MAX_BODY_BYTES = 8 * 1024 * 1024


def is_enabled(env=None):
    source = os.environ if env is None else env
    return source.get("RTP_DUMMY_SERVE", "").strip().lower() in ENABLE_VALUES


def _resolve_start_port(env=None):
    source = os.environ if env is None else env
    raw = source.get("START_PORT", "").strip() or str(DEFAULT_START_PORT)
    try:
        port = int(raw)
    except ValueError:
        return None, "START_PORT is not an integer: %r" % raw
    if not 1 <= port <= 65535:
        return None, "START_PORT out of range: %d" % port
    return port, None


def _log_gpu_info():
    """Best-effort one-shot nvidia-smi probe for the startup log.

    The dummy server never touches the GPU itself; this only makes the startup
    log look like a normal GPU service by listing the visible devices.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        logger.info(
            "nvidia-smi unavailable; the dummy server never touches the GPU anyway"
        )
        return
    if result.returncode != 0:
        logger.info(
            "nvidia-smi failed with code %d; skipping GPU info", result.returncode
        )
        return
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if line:
            logger.info("dummy server sees GPU: %s", line)


class _Stats:
    def __init__(self):
        self._lock = threading.Lock()
        self._requests = 0
        self._start_time = time.time()

    def record(self):
        with self._lock:
            self._requests += 1

    @property
    def requests(self):
        with self._lock:
            return self._requests

    @property
    def uptime_seconds(self):
        return max(0.0, time.time() - self._start_time)


class _DummyHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler_class, stats):
        super().__init__(address, handler_class)
        self.stats = stats

    def server_bind(self):
        # Skip HTTPServer.server_bind: its socket.getfqdn() reverse-DNS lookup
        # can stall bind for tens of seconds on hosts without a PTR record for
        # the bind address, and the dummy server never needs server_name.
        socketserver.TCPServer.server_bind(self)
        self.server_name = "dummy"
        self.server_port = self.server_address[1]


class _DummyHandler(BaseHTTPRequestHandler):
    server_version = "rtp-llm-dummy"
    sys_version = ""

    # Suppress the default per-request stderr logging: platform probes hit
    # /health and /metrics continuously and must not flood the log files.
    def log_message(self, format, *args):  # noqa: A002
        return

    def _send_bytes(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status, payload):
        self._send_bytes(
            status, "application/json", json.dumps(payload).encode("utf-8")
        )

    def _send_error_json(self, status, message):
        self._send_json(
            status,
            {
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "code": None,
                }
            },
        )

    def _read_json_body(self):
        header = self.headers.get("Content-Length")
        if header is None:
            return None, "missing Content-Length"
        try:
            length = int(header)
        except ValueError:
            return None, "invalid Content-Length"
        if not 0 <= length <= MAX_BODY_BYTES:
            return None, "request body too large or invalid"
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8")), None
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None, "request body is not valid JSON"

    def _send_metrics(self):
        stats = self.server.stats
        lines = [
            "# HELP rtp_llm_dummy_up Whether the rtp-llm dummy health server is running.",
            "# TYPE rtp_llm_dummy_up gauge",
            "rtp_llm_dummy_up 1",
            "# HELP rtp_llm_dummy_uptime_seconds Seconds since the dummy server started.",
            "# TYPE rtp_llm_dummy_uptime_seconds gauge",
            "rtp_llm_dummy_uptime_seconds %.3f" % stats.uptime_seconds,
            "# HELP rtp_llm_dummy_requests_total HTTP requests served by the dummy server.",
            "# TYPE rtp_llm_dummy_requests_total counter",
            "rtp_llm_dummy_requests_total %d" % stats.requests,
        ]
        self._send_bytes(
            200,
            "text/plain; version=0.0.4; charset=utf-8",
            ("\n".join(lines) + "\n").encode("utf-8"),
        )

    def _chat_reply(self, request):
        messages = request.get("messages") if isinstance(request, dict) else None
        user_text = None
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict) and message.get("role") == "user":
                    user_text = str(message.get("content"))
                    break
        suffix = (
            "last user message: %r" % user_text
            if user_text is not None
            else "no user message provided"
        )
        return (
            "[rtp-llm dummy health server] no engine is running in this pod; %s"
            % suffix
        )

    def _send_chat_completion(self, request):
        model = request.get("model") if isinstance(request, dict) else None
        if not isinstance(model, str) or not model:
            model = DUMMY_MODEL_ID
        reply = self._chat_reply(request)
        created = int(time.time())
        if request.get("stream") is True:
            chunks = [
                {
                    "id": "chatcmpl-dummy",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chatcmpl-dummy",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {"index": 0, "delta": {"content": reply}, "finish_reason": None}
                    ],
                },
                {
                    "id": "chatcmpl-dummy",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            ]
            events = ["data: " + json.dumps(chunk) for chunk in chunks]
            events.append("data: [DONE]")
            body = ("\n\n".join(events) + "\n\n").encode("utf-8")
            self._send_bytes(200, "text/event-stream", body)
            logger.info("chat completion (stream) model=%s", model)
            return
        payload = {
            "id": "chatcmpl-dummy",
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        self._send_json(200, payload)
        logger.info("chat completion model=%s", model)

    def _dispatch(self, method):
        try:
            self.server.stats.record()
            path = urlparse(self.path).path
            if path == "/health":
                self._send_json(200, {"status": "ok"})
            elif path == "/metrics":
                self._send_metrics()
            elif path == "/":
                self._send_json(
                    200,
                    {
                        "service": "rtp-llm-dummy-health-server",
                        "mode": "dummy",
                        "port": self.server.server_address[1],
                        "uptime_seconds": round(self.server.stats.uptime_seconds, 3),
                        "pid": os.getpid(),
                    },
                )
            elif path == "/v1/models":
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": DUMMY_MODEL_ID,
                                "object": "model",
                                "created": 0,
                                "owned_by": "rtp-llm-dummy",
                            }
                        ],
                    },
                )
            elif path == "/v1/chat/completions":
                if method != "POST":
                    self._send_error_json(405, "method not allowed: use POST")
                    return
                request, error = self._read_json_body()
                if error is not None:
                    self._send_error_json(400, error)
                    return
                self._send_chat_completion(request)
            else:
                self._send_error_json(404, "unknown path: %s" % path)
        except (BrokenPipeError, ConnectionResetError):
            # The client is gone; nothing left to answer.
            pass
        except Exception:
            logger.exception("dummy server handler error on %s %s", method, self.path)
            try:
                self._send_error_json(500, "dummy server internal error")
            except Exception:
                pass

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_HEAD(self):
        path = urlparse(self.path).path
        status = 200 if path == "/health" else 404
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "0")
        self.end_headers()


def _shutdown_from_signal(server, signum, _frame):
    logger.info("received signal %d; shutting down the dummy health server", signum)
    # shutdown() blocks until serve_forever() exits, so it must run in a
    # separate thread when triggered from a signal handler.
    threading.Thread(
        target=server.shutdown, name="dummy-server-shutdown", daemon=True
    ).start()


def run_dummy_server():
    """Serve the dummy HTTP surface; returns a process exit code."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
        )
    port, error = _resolve_start_port()
    if error is not None:
        logger.error("%s", error)
        return 2
    logger.info(
        "RTP_DUMMY_SERVE enabled: running the zero-GPU dummy health server on port %d",
        port,
    )
    _log_gpu_info()
    stats = _Stats()
    try:
        server = _DummyHTTPServer(("0.0.0.0", port), _DummyHandler, stats)
    except OSError as e:
        logger.error("failed to bind 0.0.0.0:%d: %s", port, e)
        return 2
    previous_handlers = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            previous_handlers[signum] = signal.signal(
                signum,
                lambda s, f, _server=server: _shutdown_from_signal(_server, s, f),
            )
        except ValueError:
            # Signal registration only works in the main thread (unit tests
            # run the server in a helper thread); tests drive shutdown directly.
            pass
    logger.info("dummy health server ready on 0.0.0.0:%d (no engine, no GPU)", port)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        for signum, handler in previous_handlers.items():
            try:
                signal.signal(signum, handler)
            except ValueError:
                pass
    logger.info("dummy health server stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_dummy_server())
