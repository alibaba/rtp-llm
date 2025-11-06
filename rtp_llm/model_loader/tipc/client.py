from time import time
import uuid

import requests  # Import the requests library for making HTTP requests
import torch

from multiprocessing import shared_memory
from .core import CuIpcTensorMeta, SharedMemIpcMeta, CudaIpcHelper, COMMON_PREFIX, SharedMemoryIPCHelper

class IPCTransportClient:

    def __init__(self, shm_size: int=8*1024*1024*1024):
        """
        Initializes the IPCTransportClient, creating a single, large
        shared memory block for efficient repeated transfers.

        Args:
            shm_size (int): The maximum size (in bytes) of the shared memory
                            block to pre-allocate. This should be large enough
                            to accommodate the largest tensor you plan to send.
        """
        self.cpu_ipc: SharedMemoryIPCHelper = SharedMemoryIPCHelper()
        self.cu_ipc: CudaIpcHelper = CudaIpcHelper()

        self.shm_size = shm_size
        self.shm_name = f"{COMMON_PREFIX}_persistent_{uuid.uuid4().__str__()}" # Unique name for this client's block
        try:
            # Create a single, persistent shared memory block
            self.shm = shared_memory.SharedMemory(create=True, size=self.shm_size, name=self.shm_name)
        except Exception as e:
            raise RuntimeError(f"Failed to create persistent shared memory block: {e}")

    def send(self, url: str, name: str, t: torch.Tensor, method: str="shm"):
        """
        Sends a tensor to a remote host using a POST request.

        When data is passed using a shared memory (SHM) method, 
        the server must send an acknowledgment signal back to the client after receiving the data. 
        All data is sent through a single, unified buffer. To prevent data conflicts, 
        do not call this function again to send more data 
        until the recipient has completely received the previous transmission.

        After receiving the data, please copy it to your own memory space.

        Args:
            url (str): The URL of the remote host.
            name (str): The name to associate with the tensor.
            t (torch.Tensor): The tensor to send.
            method (str): "cuda_ipc" or "shm" (shared memory based ipc)

        Raises:
            requests.exceptions.RequestException: If there's an error during the HTTP request.
            ValueError: If an unsupported IPC method is chosen or if the tensor is too large for SHM.
        """

        # make it a is_contiguous tensor first
        if not t.is_contiguous():
            t = t.contiguous()

        if method == "cuda_ipc":
            if not t.is_cuda: # Check if it's a CUDA tensor already
                t = t.cuda() # Move to CUDA if not
            m: CuIpcTensorMeta = self.cu_ipc.build_tensor_meta(t)
            payload: dict = {
                "name": name,
                "time": time(),
                "method": "cuda_ipc",
                "desc": m.hex()
            }
        elif method == "shm":
            if not t.is_cpu:
                t = t.cpu()

            tensor_size_bytes = t.numel() * t.itemsize
            if tensor_size_bytes > self.shm_size:
                raise ValueError(
                    f"Tensor of size {tensor_size_bytes} bytes exceeds pre-allocated "
                    f"shared memory size of {self.shm_size} bytes. "
                    "Initialize client with a larger shm_size or send a smaller tensor."
                )

            m: SharedMemIpcMeta = self.cpu_ipc.build_tensor_meta(t, self.shm)
            payload: dict = {
                "name": name,
                "time": time(),
                "method": "shm",
                "desc": m.encode()
            }
        else:
            raise ValueError(f"unsupported ipc method: {method}. Choose 'cuda_ipc' or 'shm'.")

        try:
            response = requests.post(url, json=payload)

            if response.status_code == 200:
                response_ = response.json()
                if "error" in response_:
                    raise Exception(f"IPC Transport failed, Server returns error: {response_}")
            else:
                raise IOError(f"IPC Tranport failed, Server returns code: {response.status_code}")

        except requests.exceptions.RequestException as e:
            raise e
        except Exception as e:
            raise e

    def __del__(self):
        """
        Ensures the persistent shared memory block is closed and unlinked
        when the IPCTransportClient object is garbage collected.
        """
        if hasattr(self, 'shm') and self.shm:
            try:
                self.shm.close()
                self.shm.unlink() # Unlink the shared memory block from the system
                print(f"Persistent shared memory block '{self.shm_name}' closed and unlinked.")
            except FileNotFoundError:
                print(f"Warning: Persistent shared memory block '{self.shm_name}' already unlinked.")
            except Exception as e:
                print(f"Error during __del__ of shared memory block '{self.shm_name}': {e}")
