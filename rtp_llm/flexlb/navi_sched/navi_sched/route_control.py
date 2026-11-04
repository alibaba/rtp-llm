import json
import os
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


ROUTE_STATE_DIR_ENV = "RTP_LLM_NAVI_SCHED_ROUTE_STATE_DIR"


def validate_master_address(address: str) -> str:
    value = str(address or "").strip()
    if not value:
        return ""
    if value.startswith("["):
        bracket = value.find("]")
        if bracket <= 1 or value[bracket + 1:bracket + 2] != ":":
            raise ValueError(f"invalid master address: {address}")
        port = value[bracket + 2:]
    else:
        host, separator, port = value.rpartition(":")
        if not separator or not host or ":" in host:
            raise ValueError(f"invalid master address: {address}")
    if not port.isdigit() or not 0 < int(port) <= 65533:
        raise ValueError(f"invalid master address: {address}")
    return value


def make_control_address(navi_master_address: str) -> str:
    address = str(navi_master_address).split(",", 1)[0].strip()
    if not address:
        raise ValueError("NaviSched Master address is required")
    if address.startswith("["):
        bracket = address.find("]")
        if bracket <= 1 or address[bracket + 1:bracket + 2] != ":":
            raise ValueError(f"invalid NaviSched Master address: {address}")
        host = address[:bracket + 1]
        port = address[bracket + 2:]
    else:
        host, separator, port = address.rpartition(":")
        if not separator or not host or ":" in host:
            raise ValueError(f"invalid NaviSched Master address: {address}")
    if not port.isdigit() or not 0 < int(port) < 65535:
        raise ValueError(f"invalid NaviSched Master address: {address}")
    return f"http://{host}:{int(port) + 1}/route-mode"


class RouteAddressSubscriber:
    def __init__(
        self,
        navi_master_address: str,
        reconnect_delay_s: float = 0.2,
        watch_timeout_ms: int = 1000,
    ):
        self._watch_url = make_control_address(navi_master_address)
        self._master_address = ""
        self._generation = 0
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._reconnect_delay_s = reconnect_delay_s
        self._watch_timeout_ms = max(1, min(int(watch_timeout_ms), 30000))
        state_dir = os.environ.get(ROUTE_STATE_DIR_ENV, "").strip()
        self._state_file = (
            Path(state_dir) / "route_state.json"
            if state_dir
            else None
        )
        self._thread = threading.Thread(
            target=self._run,
            name="navi-sched-route-subscriber",
            daemon=True,
        )
        self._thread.start()

    @property
    def master_address(self) -> str:
        with self._lock:
            return self._master_address

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def _apply(self, payload) -> None:
        address = validate_master_address(
            payload.get("masterAddress", payload.get("master_address", ""))
        )
        generation = int(payload["generation"])
        with self._lock:
            if generation >= self._generation:
                self._master_address = address
                self._generation = generation
                self._write_state(address, generation)

    def _write_state(self, master_address: str, generation: int) -> None:
        if self._state_file is None:
            return
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._state_file.with_suffix(".tmp")
            temporary.write_text(json.dumps({
                "pid": os.getpid(),
                "masterAddress": master_address,
                "generation": generation,
            }), encoding="utf-8")
            os.replace(temporary, self._state_file)
        except OSError:
            # Route observation is test-only and must not affect request routing.
            pass

    def _watch_once(self) -> None:
        body = json.dumps({
            "watch": True,
            "generation": self.generation,
            "timeoutMs": self._watch_timeout_ms,
        }).encode()
        request = Request(
            self._watch_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        timeout_s = self._watch_timeout_ms / 1000.0 + 2.0
        with urlopen(request, timeout=timeout_s) as response:
            self._apply(json.loads(response.read().decode()))

    def _run(self) -> None:
        while not self._stopped.is_set():
            try:
                self._watch_once()
            except (OSError, URLError, ValueError, KeyError, json.JSONDecodeError):
                if self._stopped.wait(self._reconnect_delay_s):
                    return

    def wait_for(self, master_address: str, timeout: float = 5.0) -> bool:
        expected = validate_master_address(master_address)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.master_address == expected:
                return True
            time.sleep(0.01)
        return self.master_address == expected

    def stop(self) -> None:
        self._stopped.set()
        self._thread.join(timeout=2)
