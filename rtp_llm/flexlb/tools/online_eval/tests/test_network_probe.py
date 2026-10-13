"""Socket occupancy probes must preserve address coverage without serving traffic."""

import socket
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from runtime.network import PROBE_BIND_HOST, port_accepting, port_in_use


@pytest.mark.parametrize("occupied", [False, True])
def test_wildcard_probe_closes_socket_without_listening(occupied):
    probe = MagicMock()
    if occupied:
        probe.bind.side_effect = OSError("address occupied")
    context = MagicMock()
    context.__enter__.return_value = probe
    with patch("runtime.network.socket.socket", return_value=context) as create:
        assert port_in_use(12345, PROBE_BIND_HOST) is occupied
    create.assert_called_once_with(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind.assert_called_once_with(("0.0.0.0", 12345))
    probe.listen.assert_not_called()
    probe.accept.assert_not_called()
    probe.connect.assert_not_called()
    probe.recv.assert_not_called()
    probe.send.assert_not_called()
    context.__exit__.assert_called_once()


def test_probe_detects_listener_and_releases_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((PROBE_BIND_HOST, 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        assert port_in_use(port, PROBE_BIND_HOST)
    assert not port_in_use(port, PROBE_BIND_HOST)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as successor:
        successor.bind((PROBE_BIND_HOST, port))


def test_post_start_probe_detects_loopback_listener():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        assert port_accepting(port)
