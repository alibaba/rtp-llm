import base64
import json
import math
from dataclasses import asdict, dataclass
from multiprocessing import shared_memory

import numpy as np
import torch

from .ffi import CUDA

COMMON_PREFIX = "TIPC_TRANSPROTING"


def torch_dtype_to_str(dtype: torch.dtype) -> str:
    """Converts a PyTorch dtype to its string representation."""
    return str(dtype).replace("torch.", "")


def str_to_torch_dtype(dtype: str) -> torch.dtype:
    """Converts a string representation of a PyTorch dtype back to its corresponding dtype object."""
    return getattr(torch, dtype)


def np_dtype_to_str(dtype: np.dtype) -> str:
    """Converts a NumPy dtype to its string representation."""
    return str(dtype)


def str_to_np_dtype(dtype_str: str) -> np.dtype:
    """Converts a string representation of a NumPy dtype back to its corresponding dtype object."""
    return np.dtype(dtype_str)


@dataclass
class SharedMemIpcMeta:
    """
    Data class representing the metadata required to rebuild a torch.Tensor
    that is backed by a `multiprocessing.shared_memory` segment.
    """

    shm_name: str  # The unique name of the shared memory block
    shape: torch.Size
    dtype: torch.dtype  # Logical tensor dtype; shared memory stores its raw bytes.
    stride: tuple[int, ...]  # Stride is essential for non-contiguous views
    offset_bytes: int  # Offset within the shared memory block, in bytes
    size_bytes: (
        int  # Total size of the tensor data within the shared memory block, in bytes
    )

    @classmethod
    def decode(cls, encoded: str) -> "SharedMemIpcMeta":
        """Decodes a base64 string back into a SharedMemIpcMeta instance."""
        decoded_bytes = base64.b64decode(encoded)
        serialized_dict = json.loads(decoded_bytes.decode("utf-8"))

        # Convert string representations back to original types
        serialized_dict["shape"] = torch.Size(serialized_dict["shape"])
        serialized_dict["dtype"] = str_to_torch_dtype(serialized_dict["dtype"])
        # Stride is already tuple, no conversion needed.

        return cls(**serialized_dict)

    def encode(self) -> str:
        """Encodes this SharedMemIpcMeta instance into a base64 string."""
        metadata_dict = asdict(self)
        # Convert specific types to serializable formats
        metadata_dict["shape"] = tuple(self.shape)
        metadata_dict["dtype"] = torch_dtype_to_str(self.dtype)
        metadata_dict["stride"] = tuple(self.stride)  # Ensure stride is tuple

        json_string = json.dumps(metadata_dict)
        return base64.b64encode(json_string.encode("utf-8")).decode("utf-8")


@dataclass
class CuIpcTensorMeta:
    """
    Data class representing the metadata required to rebuild a torch.Tensor,
    including considerations for sharing CUDA tensors.
    """

    dtype: torch.dtype
    shape: torch.Size
    stride: tuple[int, ...]
    storage_device: int
    storage_handle: bytes
    storage_size_bytes: int
    storage_offset_bytes: int
    requires_grad: bool
    ref_counter_handle: bytes
    ref_counter_offset: int
    event_handle: bytes
    event_sync_required: bool

    @classmethod
    def decode(cls, encoded: str) -> "CuIpcTensorMeta":
        """Decodes a hex string back into a CuIpcTensorMeta instance."""
        decoded_bytes = bytes.fromhex(encoded)
        return cls(raw=decoded_bytes)

    def encode(self) -> str:
        """Encodes this CuIpcTensorMeta instance into a base64 string."""
        # Take the raw bytes, encode them using base64, and decode the result into a string
        return base64.b64encode(self.raw).decode("utf-8")


class SharedMemoryIPCHelper:
    """
    Helper for creating and managing shared memory segments for tensor transfer.
    """

    def __init__(self):
        pass

    def build_tensor_meta(
        self, t: torch.Tensor, shm: shared_memory.SharedMemory
    ) -> SharedMemIpcMeta:
        """
        Copies tensor data to a given shared memory (SHM) object, builds its meta data, and returns it.
        This function assumes the 'shm' object is already created and has sufficient size.

        Args:
            t (torch.Tensor): The PyTorch tensor whose data is to be copied.
            shm (shared_memory.SharedMemory): The pre-existing shared memory object
                                              where the tensor data will be copied.

        Returns:
            SharedMemIpcMeta: Metadata describing the tensor's location and properties
                              within the shared memory.

        Raises:
            RuntimeError: If data copying fails or the shared memory is too small.
        """
        if not isinstance(t, torch.Tensor) or t.layout != torch.strided or t.is_meta:
            raise TypeError(
                "Shared memory transport requires a materialized strided tensor"
            )
        t = t.detach().cpu().resolve_conj().resolve_neg().contiguous()

        # Calculate required size in bytes
        tensor_size_bytes = (
            t.numel() * t.itemsize
        )  # t.itemsize is size of one element in bytes

        # Validate if the provided shared memory block is large enough
        if shm.size < tensor_size_bytes:
            raise RuntimeError(
                f"Provided shared memory block '{shm.name}' (size: {shm.size} bytes) "
                f"is too small to hold tensor (required: {tensor_size_bytes} bytes)."
            )

        try:
            # Keep the existing wire dtype/shape/stride descriptor. NumPy has
            # no BF16/FP8 dtype, so copy their storage bytes, not numeric casts.
            # Existing FP16/FP32 descriptors remain byte-for-byte compatible.
            shared_np_array = np.ndarray(
                (tensor_size_bytes,),
                dtype=np.uint8,
                buffer=shm.buf,
            )
            shared_np_array[:] = t.reshape(-1).view(torch.uint8).numpy()

            # Get the name of the shared memory block from the provided object
            shm_name = shm.name

            meta = SharedMemIpcMeta(
                shm_name=shm_name,
                shape=t.size(),
                dtype=t.dtype,
                stride=t.stride(),
                offset_bytes=0,  # Assuming the tensor starts at the beginning of the SHM block
                size_bytes=tensor_size_bytes,
            )
            return meta

        except Exception as e:
            raise RuntimeError(
                f"Failed to copy tensor data to shared memory '{shm.name}': {e}"
            )

    def build_from_meta(self, m: SharedMemIpcMeta) -> torch.Tensor:
        """
        Copy a tensor out of shared memory without NumPy dtype conversion.

        The returned tensor owns its storage, so the sender can reuse/unlink
        the segment after the RPC acknowledgment. Existing strided descriptors
        remain accepted, but every addressed byte must lie within the segment.
        """
        if not isinstance(m, SharedMemIpcMeta):
            raise TypeError(
                "Expected SharedMemIpcMeta for rebuilding from shared memory."
            )

        if not isinstance(m.dtype, torch.dtype):
            raise TypeError("Shared memory descriptor dtype must be a torch.dtype")
        shape, strides = tuple(m.shape), tuple(m.stride)
        if len(shape) != len(strides) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (*shape, *strides, m.offset_bytes, m.size_bytes)
        ):
            raise ValueError("Invalid shared memory shape, stride, offset or size")
        element_size = torch.empty((), dtype=m.dtype, device="cpu").element_size()
        logical_bytes = math.prod(shape) * element_size
        span = (
            0
            if logical_bytes == 0
            else 1 + sum((size - 1) * stride for size, stride in zip(shape, strides))
        )
        span_bytes = span * element_size
        if m.offset_bytes % element_size or m.size_bytes < logical_bytes:
            raise ValueError("Unaligned or truncated shared memory tensor descriptor")

        shm = shared_np_array = storage = view = None
        try:
            # Attach to the shared memory block
            shm = shared_memory.SharedMemory(name=m.shm_name)

            if m.offset_bytes + max(span_bytes, m.size_bytes) > shm.size:
                raise ValueError(
                    "Shared memory tensor descriptor exceeds segment bounds"
                )
            if logical_bytes == 0:
                return torch.empty_strided(shape, strides, dtype=m.dtype, device="cpu")
            shared_np_array = np.ndarray(
                (span_bytes,),
                dtype=np.uint8,
                buffer=shm.buf,
                offset=m.offset_bytes,
            )
            storage = torch.from_numpy(shared_np_array).view(m.dtype)
            view = torch.as_strided(storage, shape, strides)
            return view.clone()

        except FileNotFoundError:
            raise RuntimeError(
                f"Shared memory block '{m.shm_name}' not found. "
                "It might have been unlinked or never created."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to rebuild tensor from shared memory: {e}")
        finally:
            # Release every buffer-backed view before closing the attachment,
            # including conversion/metadata failure paths.
            view = storage = shared_np_array = None
            if shm is not None:
                shm.close()


class CudaIpcHelper:
    """
    A helper class for building and rebuilding tensors, particularly focusing on
    sharing CUDA tensors.
    """

    def build_tensor_meta(self, t: torch.Tensor):
        """
        Extracts metadata from a CUDA torch.Tensor
        to enable its reconstruction elsewhere.
        """
        if not isinstance(t, torch.Tensor):
            raise TypeError(
                f"Unsupported type for sharing: {type(t)}. Expected torch.Tensor."
            )

        if not t.is_cuda:
            raise ValueError("CUDA IPC can only be used with CUDA tensors.")

        # For CUDA IPC, _share_cuda_ requires contiguous tensors
        if not t.is_contiguous():
            raise ValueError(
                "Only contiguous CUDA tensors can be shared directly with CUDA IPC method "
                "to ensure consistent memory layout. Consider calling .contiguous() first."
            )

        return CUDA.build_cuipc_meta(t)

    def build_from_meta(self, m) -> torch.Tensor:
        """
        Rebuilds a CUDA tensor from the provided metadata.
        """
        return CUDA.build_tensor_from_meta(m)
