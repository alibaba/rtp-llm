"""DeepEP initialization manager

Manages singleton initialization of DeepEP environment, ensuring thread safety.
"""

import logging
import threading
from typing import Optional

import torch
import torch.distributed

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters

try:
    import rtp_llm.models_py.distributed.deepep_wrapper as deepep_wrapper_module
except Exception as e:
    logging.error(f"DeepEP is not supported on this device: {e}")
    deepep_wrapper_module = None


class DeepEpInitializer:
    """Singleton class for managing DeepEP initialization state"""

    _initialized: bool = False
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def supported(cls) -> bool:
        return deepep_wrapper_module is not None

    @classmethod
    def ensure_initialized(
        cls, config: GptInitModelParameters, timeout: Optional[int] = None
    ) -> None:
        """Ensure DeepEP environment is initialized (thread-safe)

        Args:
            config: Model initialization parameters
        """
        if cls._initialized:
            return

        if not cls.supported():
            raise RuntimeError("DeepEP is not supported on this device")

        with cls._lock:
            if cls._initialized:
                return
            cls._do_initialization(config, timeout)
            cls._initialized = True

    @classmethod
    def get_deepep_wrapper(cls, config: GptInitModelParameters):
        cls.ensure_initialized(config)
        assert deepep_wrapper_module is not None
        return deepep_wrapper_module.get_deepep_wrapper()

    @classmethod
    def _do_initialization(
        cls, config: GptInitModelParameters, timeout: Optional[int]
    ) -> None:
        """Perform actual initialization logic

        Args:
            config: Model initialization parameters
        """
        assert (
            torch.distributed.is_initialized()
        ), "Distributed environment is not initialized"
        assert deepep_wrapper_module is not None, "deepep_wrapper is not imported"
        default_group = torch.distributed.group.WORLD
        assert default_group is not None, "Default process group is not initialized"
        deepep_wrapper_module.init_deepep_wrapper(group=default_group, params=config)

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if initialized

        Returns:
            Whether initialized
        """
        return cls._initialized

    @classmethod
    def reset(cls) -> None:
        """Reset initialization state (for testing only)"""
        with cls._lock:
            if cls._initialized and deepep_wrapper_module is not None:
                deepep_wrapper_module.destroy_deepep_wrapper()
            cls._initialized = False
