import logging

from rtp_llm.config.py_config_modules import (
    MM_TRANSPORT_MODE_GRPC,
    MM_TRANSPORT_MODE_RDMA,
    MM_TRANSPORT_MODES,
)
from rtp_llm.multimodal.transport.base import MMOutputTransport, MMTransportBackend
from rtp_llm.multimodal.transport.grpc.backend import GrpcInlineOutputBackend


class NegotiatedOutputBackend(MMTransportBackend):
    name = "negotiated"

    def __init__(self, rdma_backend: MMTransportBackend):
        self._rdma = rdma_backend
        self._grpc = GrpcInlineOutputBackend()

    def transfer(self, request, res):
        backend = self._rdma if request.support_rdma else self._grpc
        return backend.transfer(request, res)

    def offer(self, handles):
        return self._rdma.offer(handles)

    def claim(self, handles):
        return self._rdma.claim(handles)

    def release(self, handles):
        self._rdma.release(handles)

    def close(self):
        self._rdma.close()


def create_mm_output_transport(
    transport_config=None, local_device_id: int = 0
) -> MMOutputTransport:
    if transport_config is None or transport_config.mode == MM_TRANSPORT_MODE_GRPC:
        logging.info("[VIT] mm transport mode grpc: RDMA backend disabled")
        backend = GrpcInlineOutputBackend()
    elif transport_config.mode == MM_TRANSPORT_MODE_RDMA:
        from rtp_llm.multimodal.transport.rdma.backend import RdmaOutputBackend

        backend = RdmaOutputBackend.create(transport_config.rdma, local_device_id)
        if transport_config.allow_inline_clients:
            backend = NegotiatedOutputBackend(backend)
        logging.info("[VIT] mm output backend: %s", backend.name)
    else:
        raise ValueError(
            f"invalid mm_transport_mode: {transport_config.mode!r}; "
            f"expected one of {MM_TRANSPORT_MODES}"
        )

    return MMOutputTransport(backend)
