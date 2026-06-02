import logging
from typing import Dict, List, Optional

import torch
from torch.distributed import ProcessGroup

from rtp_llm.ops.compute_ops import (
    allocate_shared_buffer,
    dispose_communicator,
    init_communicator,
    open_ipc_handle,
    register_buffer_to_communicator,
    userbuffers_recv,
    userbuffers_ring_all_gather,
    userbuffers_send,
)


class UserBufferCommunicator:

    def __init__(
        self,
        group: ProcessGroup,
        local_rank: int,
        world_size: int,
        buffer_size: int = 1024 * 1024,
    ):
        """
        Initialize UserBufferCommunicator.
            group: ProcessGroup
            local_rank: The local rank (GPU index) for this process
            world_size: Total number of GPUs in the communication group
            buffer_size: Size of communication buffers in bytes (default: 1MB)

        Raises:
            RuntimeError: If CUDA is not available or P2P access cannot be enabled
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"local_rank {local_rank} exceeds available GPU count {torch.cuda.device_count()}"
            )
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{local_rank}")
        self.buffer_size = buffer_size
        self.per_rank_buffer_size = buffer_size // world_size
        self.group = group
        torch.cuda.set_device(self.device)
        # check p2p access between all pairs of GPUs
        self._enable_p2p_access()

        # Create CUDA streams for async operations
        self._send_streams: Dict[int, torch.cuda.Stream] = {}
        self._send_stream_ids: list[int] = []
        self._rank_offsets: Dict[int, int] = {}
        self._rank_offset_lists: list[int] = []
        for rank in range(world_size):
            self._send_streams[rank] = torch.cuda.Stream(device=self.device)
            self._send_stream_ids.append(self._send_streams[rank].cuda_stream)
            self._rank_offsets[rank] = rank * (buffer_size // world_size)
            self._rank_offset_lists.append(rank * (buffer_size // world_size))

        self._recv_stream = torch.cuda.Stream(device=self.device)
        self._current_stream = torch.cuda.current_stream(self.device)

        # Signals for buffer synchronization, buffer size is equal to world_size * sizeof(int)
        self._gpu_ptrs, self._gpu_ptr_handles = self._create_buffers(
            32 * 4, group
        )  # ub_handle_0
        # Create communication buffer
        self._buffer_ptrs, self._ipc_handles = self._create_buffers(buffer_size, group)

        self._communicator_ptr = init_communicator(local_rank, world_size)

        self._gpu_ptr_handle = register_buffer_to_communicator(
            self._communicator_ptr, self._gpu_ptrs
        )
        self._ub_handle = register_buffer_to_communicator(
            self._communicator_ptr, self._buffer_ptrs
        )

        logging.info(
            f"[Rank {local_rank}] Initialized UserBufferCommunicator with world_size={world_size}, "
            f"buffer_size={buffer_size} bytes"
        )

    def _create_buffers(
        self, size_in_bytes: int, group: ProcessGroup
    ) -> tuple[int, torch.Tensor]:
        """
        Create shared CUDA buffer and get IPC handle using PyBind11 interface.

        Args:
            size_in_bytes: Size of buffer to allocate in bytes
            group: PyTorch ProcessGroup

        Returns:
            Tuple of (buffer_ptrs: list[int], ipc_handles: list[torch.Tensor])
                - buffer_address: GPU device pointer as int64_t
                - ipc_handles: IPC memory handle tensor (64 bytes, uint8)
        """
        buffer_addr, ipc_handle = allocate_shared_buffer(size_in_bytes)
        handles = [None] * self.world_size
        # TODO: Serialize object needed?
        torch.distributed.all_gather_object(handles, ipc_handle, group=group)

        buffer_ptrs: list[int] = []
        for i, h in enumerate(handles):
            if i == self.local_rank:
                buffer_ptrs.append(buffer_addr)
            else:
                buffer_ptrs.append(open_ipc_handle(h))
        return buffer_ptrs, handles

    def _enable_p2p_access(self):
        """Check if enable P2P access between all GPU pairs."""
        current_device = self.local_rank

        for other_rank in range(self.world_size):
            if other_rank == current_device:
                continue
            try:
                can_access = torch.cuda.can_device_access_peer(
                    current_device, other_rank
                )
                if can_access:
                    logging.info(
                        f"[Rank {current_device}] P2P access available to rank {other_rank}"
                    )
                else:
                    logging.warning(
                        f"[Rank {current_device}] P2P access NOT available to rank {other_rank}"
                    )
            except Exception as e:
                logging.warning(
                    f"[Rank {current_device}] Error checking P2P access to rank {other_rank}: {e}"
                )

    def can_handle_tensor(self, tensor: torch.Tensor) -> bool:
        """
        Check if tensor size is within the per-rank buffer limit.

        Args:
            tensor: Tensor to check

        Returns:
            bool: True if tensor can be handled by user buffers, False otherwise
        """
        data_bytes = tensor.numel() * tensor.element_size()
        if data_bytes > self.per_rank_buffer_size:
            logging.debug(
                f"Tensor size {data_bytes} bytes exceeds per-rank buffer limit "
                f"{self.per_rank_buffer_size} bytes"
            )
            return False
        return True

    def send(
        self,
        tensor: torch.Tensor,
        dst: int,
    ) -> bool:
        """
        Send data to destination GPU using CUDA IPC shared buffers.
        Copies data to local shared buffer using cudaMemcpyAsync.
        The receiver will access this shared buffer via IPC handles.

        Args:
            tensor: Tensor to send (must be on local CUDA device)
            dst: Destination GPU rank (same node, different GPU)

        Returns:
            bool: True if data was sent successfully, False if tensor size exceeds per-rank buffer limit
        """
        if not self.can_handle_tensor(tensor):
            return False

        data_bytes = tensor.numel() * tensor.element_size()
        userbuffers_send(
            tensor,
            self._ub_handle,
            self._rank_offsets[self.local_rank],
            self._rank_offsets[self.local_rank],
            data_bytes,
            self._communicator_ptr,
            dst,
            self._send_streams[dst].cuda_stream,
        )
        torch.cuda.current_stream().wait_stream(self._send_streams[dst])
        return True

    def recv(
        self,
        tensor: torch.Tensor,
        src: int,
    ) -> bool:
        """
        Receive data from source GPU using CUDA IPC shared buffers.

        Accesses remote shared buffer via IPC handle and copies data to local device.
        Uses cudaMemcpyAsync for device-to-device copy from shared buffer.

        Args:
            tensor: Destination tensor to store received data
            src: Source GPU rank (same node, different GPU)

        Returns:
            bool: True if data was received successfully, False if tensor size exceeds per-rank buffer limit
        """
        if not self.can_handle_tensor(tensor):
            return False

        userbuffers_recv(
            tensor,
            self._ub_handle,
            self._rank_offsets[src],
            self._rank_offsets[src],
            self._communicator_ptr,
            src,
            self._recv_stream.cuda_stream,
        )
        torch.cuda.current_stream().wait_stream(self._recv_stream)
        return True

    def all_gather(
        self, tensor: torch.Tensor, output_tensor: Optional[torch.Tensor] = None
    ) -> tuple[bool, Optional[torch.Tensor]]:
        """
        Perform all-gather operation using user buffers.

        Args:
            tensor: Input tensor to gather
            output_tensor: Optional output tensor to store result

        Returns:
            tuple[None, Optional[torch.Tensor]]: (None, output_tensor)
                - output_tensor: Gathered tensor if successful, None otherwise
        """
        if not self.can_handle_tensor(tensor):
            return None

        if output_tensor is None:
            output_tensor = torch.empty(
                [self.world_size * tensor.shape[0]] + list(tensor.shape)[1:],
                device=tensor.device,
                dtype=tensor.dtype,
            )

        userbuffers_ring_all_gather(
            output_tensor,
            tensor,
            self._ub_handle,
            self._rank_offset_lists,
            self._communicator_ptr,
            self._send_stream_ids,
            self._recv_stream.cuda_stream,
        )
        torch.cuda.current_stream().wait_stream(self._recv_stream)
        return output_tensor

    def synchronize(self):
        """
        Synchronize ALL CUDA streams created here.
        """
        for stream in self._send_streams.values():
            stream.synchronize()
        self._current_stream.synchronize()
        self._recv_stream.synchronize()

    def cleanup(self):
        """Clean up resources."""
        self.synchronize()
        dispose_communicator(self._communicator_ptr)
        self._send_streams.clear()

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception as e:
            logging.warning(
                f"Error happend during destructing UbCommunicator, may causing resource leak."
                f"Error: {type(e).__name__}: {str(e)}"
            )


# Global communicator instance
_global_communicator: Optional[UserBufferCommunicator] = None


def init_user_buffers_communicator(
    group: ProcessGroup,
    world_rank: int,
    world_size: int,
    buffer_size: int,
) -> UserBufferCommunicator:
    """
    Initialize the global CUDA P2P communicator.

    Args:
        world_rank: World rank of this process
        world_size: Total number of processes

    Returns:
        Initialized communicator instance
    """
    global _global_communicator

    if _global_communicator is not None:
        logging.warning("CUDA P2P communicator already initialized")
        return _global_communicator

    _global_communicator = UserBufferCommunicator(
        group, world_rank, world_size, buffer_size
    )
    return _global_communicator


def get_user_buffers_communicator() -> Optional[UserBufferCommunicator]:
    """
    Get the global CUDA P2P communicator.

    Returns:
        The global communicator instance, or None if not initialized
    """
    return _global_communicator


def destroy_user_buffers_communicator() -> None:
    global _global_communicator
    if _global_communicator is not None:
        del _global_communicator


__all__ = [
    "UserBufferCommunicator",
    "init_user_buffers_communicator",
    "get_user_buffers_communicator",
    "destroy_user_buffers_communicator",
]
