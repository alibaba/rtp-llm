"""One endpoint owner. Workload metrics are views of Prometheus scrape samples.

Unowned reads exist only for bounded functional checks; they do not archive
samples or generate performance curves. Registered failures never fall back.
"""

import threading
import urllib.request

_REGISTRY = {}
_REGISTRY_LOCK = threading.Lock()


def http_text(url, timeout=5.0):
    with _REGISTRY_LOCK:
        owner = _REGISTRY.get(url)
    if owner is not None:
        return owner.read(timeout)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def shared_samples_since(url, sequence):
    with _REGISTRY_LOCK:
        owner = _REGISTRY.get(url)
    return None if owner is None else owner.samples_since(sequence)
