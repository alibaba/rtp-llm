#!/usr/bin/env python3
"""Tiny HTTP gate for the D-side decode slow-down: POST /hold creates the hold file, POST /release removes it.
GET /status reports whether the hold file exists. usage: gate_agent.py [--port 31999] [--file /dev/shm/dsv4_decode_hold]
"""
import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=31999)
    ap.add_argument(
        "--file",
        default=os.environ.get("DSV4_DECODE_HOLD_FILE", "/dev/shm/dsv4_decode_hold"),
    )
    args = ap.parse_args()
    path = args.file

    class H(BaseHTTPRequestHandler):
        def _reply(self, body):
            b = body.encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_POST(self):
            if self.path == "/hold":
                with open(path, "w") as f:
                    f.write(str(time.time()))
                self._reply("hold")
            elif self.path == "/release":
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
                self._reply("release")
            elif self.path == "/ts":
                trig = os.environ.get("DSV4_SM120_RAIL_TS_DUMP", "/tmp/rr2ts")
                with open(trig, "w") as f:
                    f.write(str(time.time()))
                self._reply("ts")
            else:
                self.send_error(404)

        def do_GET(self):
            if self.path == "/queued":
                d = os.path.dirname(path) or "."
                base = os.path.basename(path) + ".q"
                q = {}
                for f in os.listdir(d):
                    if f.startswith(base):
                        try:
                            q[f[len(base) :]] = int(
                                open(os.path.join(d, f)).read().strip() or -1
                            )
                        except (OSError, ValueError):
                            pass
                self._reply(json.dumps(q))
            else:
                self._reply("hold" if os.path.exists(path) else "free")

        def log_message(self, *a):
            pass

    ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
