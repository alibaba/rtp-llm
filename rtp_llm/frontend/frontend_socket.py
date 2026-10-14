import socket


def create_frontend_socket(host: str, port: int) -> socket.socket:
    """Bind without listening until Uvicorn completes application startup."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.bind((host, port))
        # A listening SO_REUSEPORT socket receives connections even while its
        # worker is still waiting for backend health. Let Uvicorn/asyncio call
        # listen only after lifespan startup, when it can actually serve them.
        return sock
    except BaseException:
        sock.close()
        raise
