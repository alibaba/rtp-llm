import logging
from typing import List

from rtp_llm.config.py_config_modules import (
    MM_TRANSPORT_MODE_AUTO,
    MM_TRANSPORT_MODE_GRPC,
    MM_TRANSPORT_MODES,
)
from rtp_llm.multimodal.mm_output_transport.base import MMTransportBackend
from rtp_llm.multimodal.mm_output_transport.grpc.backend import GrpcInlineTerminal
from rtp_llm.multimodal.mm_output_transport.manager import MMOutputTransport
from rtp_llm.multimodal.mm_output_transport.rdma.backend import RdmaTransportBackend


def create_mm_output_transport(transport_config=None) -> MMOutputTransport:
    terminal = GrpcInlineTerminal()
    backends: List[MMTransportBackend] = []

    if transport_config is None or transport_config.mode == MM_TRANSPORT_MODE_GRPC:
        logging.info("[VIT] mm transport mode grpc: RDMA backend disabled")
    elif transport_config.mode == MM_TRANSPORT_MODE_AUTO:
        rdma = RdmaTransportBackend.try_create(transport_config.rdma)
        if rdma is None:
            logging.warning(
                "[VIT] mm rdma requested but unavailable, fall back to bytes"
            )
        else:
            logging.info("[VIT] mm rdma encoder enabled")
            backends.append(rdma)
    else:
        raise ValueError(
            f"invalid mm_transport_mode: {transport_config.mode!r}; "
            f"expected one of {MM_TRANSPORT_MODES}"
        )

    return MMOutputTransport(backends=backends, terminal=terminal)
