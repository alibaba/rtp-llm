from abc import abstractmethod
from typing import Any, AsyncGenerator, Dict

from rtp_llm.ops import EngineScheduleInfo, KVCacheInfo, WorkerStatusInfo


class BaseEngine:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def decode(self, input: Any) -> AsyncGenerator[Any, None]:
        raise NotImplementedError()

    @abstractmethod
    def get_worker_status_info(self, latest_finished_version: int) -> WorkerStatusInfo:
        raise NotImplementedError()

    @abstractmethod
    def get_cache_status_info(self, latest_cache_version: int) -> KVCacheInfo:
        raise NotImplementedError()

    @abstractmethod
    def get_engine_schedule_info(
        self, latest_finised_version: int
    ) -> EngineScheduleInfo:
        raise NotImplementedError()

    @abstractmethod
    def update_scheduler_info(self, scheduler_info: str) -> None:
        raise NotImplementedError()

    @abstractmethod
    def update_eplb_config(self, req: Dict[str, str]) -> bool:
        raise NotImplementedError()

    def pause(self) -> None:
        """Pauses the engine's execution.

        When called, this method sets the `pause_` flag to true. The engine's
        `step` method checks this flag and sleeps when it's true, effectively
        pausing execution. This is necessary for tasks like updating model weights
        or clearing GPU memory, which require the engine to be inactive. The `pause_`
        parameter is modified only by this interface, so it doesn't need to be
        thread-safe.
        """
        raise NotImplementedError()

    def restart(self) -> None:
        """Restarts the engine's execution."""
        raise NotImplementedError()

    @abstractmethod
    def detach_physical_memory(self) -> bool:
        """
        Release physical GPU memory while retaining the virtual address space.
        This method is intended for engines that support virtual memory. It
        immediately unmaps and frees all **physical** backing memory without
        releasing the reserved **virtual** addresses.  If any requests are still
        in flight, the engine **must** wait for them to complete before
        performing the detach operation.
        Returns
        -------
        bool
            ``True``  – physical memory was successfully released.
            ``False`` – the engine does not support virtual memory **or** the
            detach operation failed.
        Notes
        -----
        After a successful detach, the virtual addresses remain valid but
        accessing them will raise a device page-fault until
        :meth:`attach_physical_memory` is called.
        """
        raise NotImplementedError()

    @abstractmethod
    def attach_physical_memory(self) -> bool:
        """
        Re-attach / map physical memory to previously reserved virtual addresses.
        For every virtual address range that was **reserved but not mapped**
        (e.g., after :meth:`detach_physical_memory`), this method allocates
        physical GPU memory and binds it to those ranges.  Virtual addresses that
        already have physical backing are **not** re-allocated.
        Returns
        -------
        bool
            ``True``  – physical memory was successfully (re-)mapped.
            ``False`` – the engine lacks virtual-memory support **or** the
            mapping operation failed.
        """
        raise NotImplementedError()

    @abstractmethod
    def rebuild_rope(self, rescale_factor: float) -> None:
        """
        Re-generate the RoPE (Rotary Position Embedding) cache with a new rescale
        factor.  This is typically used by YaRN-style length-extension algorithms
        to adapt the frequency scaling when the context window is enlarged or
        shrunk.
        Parameters
        ----------
        rescale_factor : float
            Multiplicative factor applied to the original base frequencies
            (e.g. 1.0 keeps the original scale, 2.0 doubles the effective
            wavelength, 0.5 halves it).
        Returns
        -------
        None
        Raises
        ------
        NotImplementedError
            If the backend does not support dynamic RoPE re-scaling.
        """
        raise NotImplementedError()