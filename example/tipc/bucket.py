import torch
import requests
from tqdm import tqdm
from typing import List, Dict, Any
from core import CudaIpcHelper

# A custom type hint for gigabytes, improving code readability.
GB = int


class BufferedTransportClient:
    """
    A class for batching small PyTorch tensors into a larger buffer for efficient IPC.

    The primary purpose is to reduce the overhead of launching many small IPC transfers
    by packing tensors together into a single, larger transfer.
    """
    def __init__(self, url: str, device: str = "cuda", size: GB = 4):
        """
        Initializes a new TransportBucket with a specified buffer size.

        Args:
            url (str): The URL of the IPC receiver.
            device (str): The device for the buffer (e.g., "cuda"). Defaults to "cuda".
            size (GB): The size of the buffer in gigabytes. Defaults to 4.
        """
        # Note: 'url', 'device', and 'method' are currently unused in the provided code,
        # but they are kept here for potential future use.
        self.url: str = url
        
        # Current size of data in the buffer, in bytes.
        self._mounted_bytes: int = 0
        
        # List of metadata for each tensor in the buffer.
        self._metas: List[Dict[str, Any]] = []
        
        # Alignment size in bytes. Tensors are aligned to this boundary.
        self._align_size: int = 256
        
        # Total size of the buffer in bytes.
        self._size_in_bytes: int = size * 1024 * 1024 * 1024
        
        # The main buffer for packaging tensors. Initialized on the correct device.
        self._buffer = torch.empty(self._size_in_bytes, dtype=torch.uint8, device=device)

        self._cuipc = CudaIpcHelper()

    def _aligned_size(self, size: int) -> int:
        """
        Calculates the aligned size for a given size.

        This ensures that tensors are stored at addresses that are multiples of
        the alignment size, which can improve memory access efficiency.
        """
        return ((size + self._align_size - 1) // self._align_size) * self._align_size
    
    def _bucket_meta(self, t: torch.Tensor, name: str, offset: int) -> Dict[str, Any]:
        """
        Creates a metadata dictionary for a single tensor.

        Args:
            t (torch.Tensor): The tensor to be wrapped.
            name (str): The name of the tensor.
            offset (int): The starting byte offset within the buffer.

        Returns:
            Dict[str, Any]: A dictionary containing tensor metadata.
        """
        return {
            'name': name,
            'offset': offset,
            'dtype': str(t.dtype).split('.')[-1],  # Use a simple string for dtype
            'shape': list(t.shape)  # Convert tuple to list for serialization
        }

    def _transport(self, tensor: torch.Tensor, metas: List[Dict[str, Any]]) -> bool:
        """
        Performs the IPC transport of the batched tensors.
        """
        payload = {
            'root': self._cuipc.build_tensor_meta(tensor).hex(),
            'tensors': metas
        }
        response = requests.post(self.url, json=payload)
        return response.status_code == 200

    def _launch_copy(self, src: torch.Tensor, dst: torch.Tensor, offset: int):
        """
        Copies data from a source tensor to a destination tensor at a specific offset.
        
        Args:
            src (torch.Tensor): The source tensor.
            dst (torch.Tensor): The destination tensor (the buffer).
            offset (int): The byte offset to start copying into the destination.
        """
        # Correctly view the source tensor as a flat byte tensor for copying.
        src_bytes = src.view(dtype=torch.uint8)
        
        # Copy the data to the correct offset in the buffer.
        dst[offset : offset + src_bytes.numel()].copy_(src_bytes)

    def enqueue(self, name: str, t: torch.Tensor) -> bool:
        """
        Adds a tensor to the transfer queue.

        The tensor is either immediately transported if it's too large, or
        it's copied into the internal buffer for later batching.

        Args:
            name (str): The name of the tensor.
            t (torch.Tensor): The tensor to enqueue.

        Returns:
            bool: True if the tensor was successfully enqueued or sent, False otherwise.
        """
        size_in_bytes: int = t.numel() * t.element_size()
        
        # Tensors larger than half the buffer size are sent immediately.
        if size_in_bytes >= self._size_in_bytes // 2:
            return self._transport(t, [self._bucket_meta(t, name, 0)])
        else:
            aligned_size = self._aligned_size(size_in_bytes)
            
            # If the new tensor would overflow the buffer, flush the buffer first.
            if self._mounted_bytes + aligned_size > self._size_in_bytes:
                return self.flush()

            # Add tensor metadata to the queue.
            self._metas.append(self._bucket_meta(t, name, self._mounted_bytes))
            
            # Copy tensor data to the buffer at the current offset.
            self._launch_copy(t, self._buffer, self._mounted_bytes)
            
            # Update the current offset.
            self._mounted_bytes += aligned_size
            return True

    def flush(self) -> bool:
        """
        Transports all tensors currently in the buffer.
        """
        if self._mounted_bytes != 0:
            success = self._transport(self._buffer, self._metas)
            self._metas.clear()
            self._mounted_bytes = 0
            return success


if __name__ == '__main__':
    from random import randint
    print("Starting Client...")
    client = BufferedTransportClient(url="http://127.0.0.1:5000/bucket_transport_tensors", size=1)

    for _ in tqdm(range(128000)):
        t = torch.rand(size=[randint(a=1024, b=16384)], device="cuda")
        success = client.enqueue(name="tensor", t=t)

        if not success:
            raise Exception("Send Failed.")

    success = client.flush()
    if not success:
        raise Exception("Send Failed.")
