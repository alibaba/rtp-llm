# test_server.py

from flask import Flask, request, jsonify
import torch
from core import CudaIpcHelper, CuIpcTensorMeta
from typing import Dict, List, Any

# A simple Flask application to act as the IPC receiver server.
app = Flask(__name__)

def try_rebuild_root(meta: CuIpcTensorMeta) -> torch.Tensor:
    helper: CudaIpcHelper = CudaIpcHelper()
    return helper.build_from_meta(meta)

def try_rebuild_tensor(
        root: torch.Tensor, shape: list, 
        dtype: str, offset: int
    ) -> torch.Tensor:
    """
    Rebuilds a specific tensor from a root tensor (buffer) and its metadata.

    Args:
        root (torch.Tensor): The main buffer tensor containing the data.
        shape (list): The shape of the original tensor.
        dtype (str): The string representation of the original tensor's data type.
        offset (int): The byte offset where the tensor's data begins in the root buffer.

    Returns:
        torch.Tensor: The rebuilt tensor.
    """
    dtype_map = {
        'float32': torch.float32,
        'float64': torch.float64,
        'int32': torch.int32,
        'int64': torch.int64,
        'uint8': torch.uint8,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float8_e4m3fn': torch.float8_e4m3fn
    }
    if not dtype in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}")
    _dtype = dtype_map[dtype]

    # Calculate the size of the tensor in bytes.
    element_size = torch.tensor([], dtype=_dtype).element_size()
    size_in_elements = torch.prod(torch.tensor(shape)).item()
    size_in_bytes = size_in_elements * element_size

    # Slice the buffer to get the specific tensor's data.
    tensor_bytes = root.view(torch.uint8)[offset : offset + size_in_bytes]

    # View the sliced byte tensor as the original tensor's shape and dtype.
    return tensor_bytes.view(_dtype).view(shape)

@app.route('/bucket_transport_tensors', methods=['POST'])
def bucket_transport_tensors() -> tuple[dict, int]:
    """
    Handles incoming POST requests from the TransportBucket client.

    This endpoint expects a JSON payload containing the metadata for the
    batched tensors and the root tensor's IPC handle. It simulates the
    reception and processing of this data.
    """
    
    # 1. Get the JSON payload from the request.
    payload: Dict[str, Any] = request.get_json()
    if not payload:
        return jsonify({'error': 'No JSON payload received'}), 400

    root_meta_hex: str = payload.get('root')

    root: torch.Tensor = try_rebuild_root(bytes.fromhex(root_meta_hex))
    tensors_meta: List[Dict[str, Any]] = payload.get('tensors')

    # 2. Validate the payload structure.
    if not all([root_meta_hex, isinstance(tensors_meta, list)]):
        return jsonify({'error': 'Invalid payload format'}), 400

    if tensors_meta:
        for meta in tensors_meta:
            if 'name' not in meta:
                return jsonify({'error': 'missing tensor name'}), 400
            name = meta["name"]

            if 'shape' not in meta:
                return jsonify({'error': 'missing tensor shape'}), 400
            shape = meta["shape"]

            if 'dtype' not in meta:
                return jsonify({'error': 'missing tensor dtype'}), 400
            dtype = meta["dtype"]

            if 'offset' not in meta:
                return jsonify({'error': 'missing tensor offset'}), 400
            offset = meta["offset"]
            
            tensor = try_rebuild_tensor(root, shape, dtype, offset)

    return jsonify({'message': 'Tensors received successfully'}), 200

if __name__ == '__main__':
    print("Starting IPC Test Server...")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)