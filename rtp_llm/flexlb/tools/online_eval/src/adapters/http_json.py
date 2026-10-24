"""Bounded protocol reads for on-demand standard exporters.

Adapters never schedule, spawn threads or write artifacts. Prometheus owns
scrapes; protocol projection distinguishes outages from malformed responses.
"""

import json
import urllib.request



class HttpJsonAdapter:
    """One bounded read; transport outages and malformed payloads are distinct.

    Only protocols where unavailability is an observation provide unavailable().
    Otherwise a failed request aborts collection. Parsing errors always fail.
    """

    def __init__(self, url, project, unavailable=None, *, max_response_bytes=32 * 1024 * 1024):
        if type(max_response_bytes) is not int or max_response_bytes <= 0:
            raise ValueError("HTTP response byte budget must be positive")
        self.url, self.project, self.unavailable = url, project, unavailable
        self.max_response_bytes = max_response_bytes

    def __call__(self, *, timeout):
        try:
            with urllib.request.urlopen(self.url, timeout=timeout) as response:
                body = response.read(self.max_response_bytes + 1)
                if len(body) > self.max_response_bytes:
                    raise ValueError("HTTP evidence response exceeds byte budget")
                data = json.loads(body)
        except OSError:
            if self.unavailable is None:
                raise
            return self.unavailable()
        return self.project(data)
