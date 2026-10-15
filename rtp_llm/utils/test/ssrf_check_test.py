"""Exercise the real downloader using local HTTP/TLS servers.

DNS fixtures return explicit public addresses so production SSRF checks still
run. Only socket.connect translates enumerated public-IP/server-port pairs to
loopback; every other connection is rejected by the fixture. No external DNS or
network, Native stand-in, disabled certificate verification, or transport mock
is used. Loading this pure utility by file keeps its tests independent of GPU
package initialization.
"""

import importlib.util
import ipaddress
import socket
import ssl
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import requests

_SPEC = importlib.util.spec_from_file_location(
    "ssrf_check_under_test", Path(__file__).resolve().parents[1] / "ssrf_check.py"
)
ssrf_check = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ssrf_check)

pytestmark = pytest.mark.cpu
SENTINEL = "Bearer non-secret-download-regression-sentinel"
PUBLIC_HOSTS = {
    "origin.download.test": "8.8.8.8",
    "other.download.test": "1.1.1.1",
    "wrong.download.test": "9.9.9.9",
    "private.download.test": "127.0.0.1",
}


@pytest.fixture(scope="session")
def certificates(tmp_path_factory):
    directory = tmp_path_factory.mktemp("download-tls")
    ca_key, ca = directory / "ca.key", directory / "ca.pem"
    key, csr, cert = (
        directory / "server.key",
        directory / "server.csr",
        directory / "server.pem",
    )
    ext = directory / "server.ext"
    ext.write_text(
        "basicConstraints=CA:FALSE\nkeyUsage=digitalSignature,keyEncipherment\n"
        "extendedKeyUsage=serverAuth\n"
        "subjectAltName=DNS:origin.download.test,DNS:other.download.test\n"
    )
    commands = [
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-subj",
            "/CN=Download regression temporary CA",
            "-keyout",
            str(ca_key),
            "-out",
            str(ca),
        ],
        [
            "openssl",
            "req",
            "-new",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-subj",
            "/CN=origin.download.test",
            "-keyout",
            str(key),
            "-out",
            str(csr),
        ],
        [
            "openssl",
            "x509",
            "-req",
            "-in",
            str(csr),
            "-CA",
            str(ca),
            "-CAkey",
            str(ca_key),
            "-CAcreateserial",
            "-days",
            "1",
            "-extfile",
            str(ext),
            "-out",
            str(cert),
        ],
    ]
    for command in commands:
        subprocess.run(command, check=True, capture_output=True, text=True)
    return ca, cert, key


@pytest.fixture
def network(monkeypatch, certificates, tmp_path):
    """Return real local servers plus auditable DNS and socket records."""
    ca, cert, key = certificates
    for name in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(ca))
    monkeypatch.delenv("CURL_CA_BUNDLE", raising=False)
    netrc = tmp_path / "empty-netrc"
    netrc.write_text("")
    monkeypatch.setenv("NETRC", str(netrc))
    original_getaddrinfo = socket.getaddrinfo
    original_connect = socket.socket.connect
    routes, servers, connects, resolutions = {}, [], [], []
    dns_answers = {}

    def getaddrinfo(host, port, *args, **kwargs):
        text = host.decode() if isinstance(host, bytes) else host
        resolutions.append(text)
        if text in PUBLIC_HOSTS:
            answers = dns_answers.get(text)
            address = answers.pop(0) if answers else PUBLIC_HOSTS[text]
            return original_getaddrinfo(address, port, *args, **kwargs)
        try:
            ipaddress.ip_address(text)
        except ValueError:
            raise AssertionError(f"Unexpected DNS query: {text!r}")
        return original_getaddrinfo(text, port, *args, **kwargs)

    def connect(sock, address):
        key = (address[0], address[1])
        assert key in routes, f"Unexpected physical connection: {key!r}"
        connects.append(key)
        return original_connect(sock, ("127.0.0.1", routes[key]))

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", connect)

    def start(host="origin.download.test", tls=False, required_sni=None):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.server.requests.append(
                    {
                        "path": self.path,
                        "host": self.headers.get("Host"),
                        "authorization": self.headers.get("Authorization"),
                        "cookie": self.headers.get("Cookie"),
                        "proxy_authorization": self.headers.get("Proxy-Authorization"),
                    }
                )
                status, response_headers, body = self.server.responses.get(
                    self.path, (200, {}, b"download-ok")
                )
                self.send_response(status)
                for name, value in response_headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        server.requests, server.responses, server.sni = [], {}, []
        if tls:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(str(cert), str(key))

            def observe_sni(_socket, hostname, _context):
                server.sni.append(hostname)
                if required_sni is not None and hostname != required_sni:
                    return ssl.ALERT_DESCRIPTION_UNRECOGNIZED_NAME

            context.set_servername_callback(observe_sni)
            server.socket = context.wrap_socket(server.socket, server_side=True)
        port = server.server_address[1]
        routes[(PUBLIC_HOSTS[host], port)] = port
        server.url = f"{'https' if tls else 'http'}://{host}:{port}"
        server.expected_host = f"{host}:{port}"
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()
        servers.append((server, thread))
        return server

    yield start, connects, resolutions, dns_answers
    for server, thread in reversed(servers):
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive(), "HTTP fixture thread did not stop"


def fetch(url, headers=None):
    with ssrf_check.safe_request_get(url, headers or {}, timeout=3) as response:
        assert response.status_code == 200
        assert response.content == b"download-ok"


def sensitive_headers():
    return {
        "aUtHoRiZaTiOn": SENTINEL,
        "Cookie": "nonsecret=sentinel",
        "Proxy-Authorization": SENTINEL,
    }


@pytest.mark.parametrize("absolute", [False, True])
def test_same_origin_redirect_preserves_credentials_and_caller_headers(
    network, absolute
):
    start, connects, _, _ = network
    server = start()
    server.responses["/redirect"] = (
        302,
        {"Location": server.url + "/final" if absolute else "/final"},
        b"",
    )
    headers = sensitive_headers()
    original = dict(headers)
    fetch(server.url + "/redirect", headers)
    assert headers == original
    assert len(server.requests) == 2
    assert [r["authorization"] for r in server.requests] == [SENTINEL, SENTINEL]
    assert [r["host"] for r in server.requests] == [server.expected_host] * 2
    assert all(ip == "8.8.8.8" for ip, _ in connects)


@pytest.mark.parametrize("second_host", ["origin.download.test", "other.download.test"])
def test_cross_origin_redirect_strips_credentials(network, second_host):
    start, _, _, _ = network
    first, second = start(), start(second_host)
    first.responses["/redirect"] = (302, {"Location": second.url + "/final"}, b"")
    headers = sensitive_headers()
    original = dict(headers)
    fetch(first.url + "/redirect", headers)
    assert headers == original
    assert first.requests[0]["authorization"] == SENTINEL
    assert len(second.requests) == 1
    assert second.requests[0]["authorization"] is None
    assert second.requests[0]["cookie"] is None
    assert second.requests[0]["proxy_authorization"] is None
    assert second.requests[0]["host"] == second.expected_host


def test_tls_downgrade_strips_credentials(network):
    start, _, _, _ = network
    first, second = start(tls=True), start()
    first.responses["/redirect"] = (302, {"Location": second.url + "/final"}, b"")
    fetch(first.url + "/redirect", sensitive_headers())
    assert first.requests[0]["authorization"] == SENTINEL
    assert second.requests[0]["authorization"] is None
    assert second.requests[0]["cookie"] is None
    assert second.requests[0]["proxy_authorization"] is None


def test_tls_passes_actual_sni_and_original_host(network):
    start, connects, _, _ = network
    server = start(tls=True, required_sni="origin.download.test")
    fetch(server.url + "/file")
    assert server.sni == ["origin.download.test"]
    assert server.requests[0]["host"] == server.expected_host
    assert connects == [("8.8.8.8", server.server_address[1])]


def test_tls_rejects_trusted_ca_certificate_for_wrong_name(network):
    start, _, _, _ = network
    server = start("wrong.download.test", tls=True)
    with pytest.raises(requests.exceptions.SSLError):
        fetch(server.url + "/file")
    assert server.sni == ["wrong.download.test"]
    assert server.requests == []


def test_tls_rejects_untrusted_certificate(network, monkeypatch):
    start, _, _, _ = network
    server = start(tls=True)
    monkeypatch.delenv("REQUESTS_CA_BUNDLE")
    with pytest.raises(requests.exceptions.SSLError):
        fetch(server.url + "/file")
    assert server.requests == []


@pytest.mark.parametrize(
    "destination", ["http://127.0.0.1/private", "http://private.download.test/private"]
)
def test_redirect_to_private_address_is_rejected_before_connect(network, destination):
    start, connects, _, _ = network
    server = start()
    server.responses["/redirect"] = (302, {"Location": destination}, b"")
    with pytest.raises(ValueError, match="private/internal"):
        fetch(server.url + "/redirect", sensitive_headers())
    assert len(server.requests) == 1
    assert len(connects) == 1


def test_initial_private_address_is_rejected_before_connect(network):
    _, connects, _, _ = network
    with pytest.raises(ValueError, match="private/internal"):
        fetch("http://127.0.0.1/private")
    assert connects == []


def test_connection_uses_validated_address_without_hostname_reresolution(network):
    start, connects, resolutions, dns_answers = network
    server = start()
    dns_answers["origin.download.test"] = ["8.8.8.8", "8.8.8.8", "127.0.0.1"]
    fetch(server.url + "/file")
    assert resolutions.count("origin.download.test") == 2
    assert dns_answers["origin.download.test"] == ["127.0.0.1"]
    assert connects == [("8.8.8.8", server.server_address[1])]


def test_each_redirect_is_validated_and_credentials_are_not_restored(network):
    start, _, _, _ = network
    first, second = start(), start("other.download.test")
    first.responses["/redirect"] = (302, {"Location": second.url + "/return"}, b"")
    second.responses["/return"] = (302, {"Location": first.url + "/final"}, b"")
    fetch(first.url + "/redirect", sensitive_headers())
    assert first.requests[0]["authorization"] == SENTINEL
    assert second.requests[0]["authorization"] is None
    assert first.requests[1]["authorization"] is None


def test_private_target_on_later_redirect_is_rejected(network):
    start, connects, _, _ = network
    first, second = start(), start("other.download.test")
    first.responses["/redirect"] = (302, {"Location": second.url + "/next"}, b"")
    second.responses["/next"] = (302, {"Location": "http://127.0.0.1/private"}, b"")
    with pytest.raises(ValueError, match="private/internal"):
        fetch(first.url + "/redirect", sensitive_headers())
    assert len(connects) == 2
    assert second.requests[0]["authorization"] is None


@pytest.mark.parametrize("scheme", ["file", "ftp"])
def test_non_http_scheme_is_rejected_without_connect(network, scheme):
    _, connects, _, _ = network
    with pytest.raises(ValueError, match="scheme"):
        fetch(f"{scheme}://origin.download.test/file")
    assert connects == []
