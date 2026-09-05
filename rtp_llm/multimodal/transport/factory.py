import logging

from rtp_llm.config.py_config_modules import (
    MM_TRANSPORT_MODE_GRPC,
    MM_TRANSPORT_MODE_RDMA,
    MM_TRANSPORT_MODES,
)
from rtp_llm.multimodal.transport.base import MMOutputTransport
from rtp_llm.multimodal.transport.grpc.backend import GrpcInlineOutputBackend


def create_mm_output_transport(
    transport_config=None, local_device_id: int = 0
) -> MMOutputTransport:
    if transport_config is None or transport_config.mode == MM_TRANSPORT_MODE_GRPC:
        logging.info("[VIT] mm transport mode grpc: RDMA backend disabled")
        backend = GrpcInlineOutputBackend()
    elif transport_config.mode == MM_TRANSPORT_MODE_RDMA:
        from rtp_llm.multimodal.transport.rdma.backend import RdmaOutputBackend

        backend = RdmaOutputBackend.create(transport_config.rdma, local_device_id)
        logging.info("[VIT] mm transport mode rdma: output backend enabled")
    else:
        raise ValueError(
            f"invalid mm_transport_mode: {transport_config.mode!r}; "
            f"expected one of {MM_TRANSPORT_MODES}"
        )

    return MMOutputTransport(backend)
