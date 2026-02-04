"""XQA implementation factory.

This module provides a factory function to select the appropriate XQA implementation
based on CUDA version and flashinfer availability.
"""
import logging
from typing import Type

import torch

from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.models_py.modules.factory.attention.cuda_impl.xqa_impl import XQAImpl
from rtp_llm.models_py.modules.factory.attention.cuda_impl.xqa_decode_impl import XQADecodeImpl


def get_xqa_impl() -> Type[FMHAImplBase]:
    """Select the appropriate XQA implementation based on CUDA version and flashinfer availability.

    Returns:
        XQADecodeImpl if CUDA >= 12.8 and flashinfer.xqa is available,
        otherwise falls back to XQAImpl.
    """
    try:
        major, minor = map(int, torch.version.cuda.split(".")[:2])
        if (major, minor) >= (12, 8):
            try:
                from flashinfer.xqa import xqa

                logging.info(
                    "CUDA >= 12.8 and flashinfer.xqa available, using XQADecodeImpl"
                )
                return XQADecodeImpl
            except (ImportError, AttributeError) as e:
                logging.info(
                    f"CUDA >= 12.8 but flashinfer.xqa not available ({e}), falling back to XQAImpl"
                )
                return XQAImpl
        else:
            logging.info(f"CUDA version {major}.{minor} < 12.8, using XQAImpl")
            return XQAImpl
    except Exception as e:
        logging.warning(f"Failed to check CUDA version ({e}), using XQAImpl")
        return XQAImpl
