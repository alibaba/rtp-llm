## Multimodal Part Debug and Separate Deployment

### Multimodal Development

A multimodal model refers to an LLM that incorporates multimodal inputs. Currently, the primary input format is URLs, which are distinguished by specific placeholders using OpenAI formatting. Each multimodal model has its unique preprocessing pipeline but must implement the following interfaces:

* The multimodal model must inherit from `MultimodalMixin` in `rtp_llm/models/multimodal/multimodal_mixin.py` and instantiate `mm_part` as the processing class for multimodal inputs.

* `mm_part` has various interface implementations based on input types, such as images, videos, or audio. The logic must be self-consistent, with the most critical interfaces being `mm_embedding`, `_mm_preprocess`, and `mm_process`:

    * `mm_embedding` has a default implementation that calls `_mm_preprocess` and `mm_process`, converting the multimodal input URL into an embedding tensor and other information (e.g., position IDs).

    * `_mm_preprocess` also has default implementations for specific modalities, preprocessing byte data from input url and preparing inputs for mm_process. This separation is necessary because preprocessing is CPU-bound, while subsequent processing is GPU-bound.

    * `mm_process` handles GPU-based transformation of preprocessed inputs into outputs.

* For model weights, the required weights must be registered in `GptInitModelParameters` under `mm_related_params.vit_weights`. Refer to `BaseVitWeights` for specific implementation logic.

### Debug

Start multimodal part.

``` bash
START_PORT=12345 \
VIT_SEPARATION=1 \
ACT_TYPE=bf16 \
MODEL_TYPE=qwen2_5_vl \
CHECKPOINT_PATH=/home/xieshui.yyx/Qwen2.5-VL-3B \
/opt/conda310/bin/python -m rtp_llm.start_server
```

Start a grpc client.

``` Python
from rtp_llm.cpp.proto.model_rpc_service_pb2_grpc import (
    MultimodalRpcServiceStub,
)

from rtp_llm.cpp.proto.model_rpc_service_pb2 import (
    MultimodalInputPB,
    MultimodalInputsPB
)

from rtp_llm.utils.grpc_util import trans_tensor

import grpc
import torch

def trans_multimodal_input(urls):
    input_pb = MultimodalInputsPB()
    for url in urls:
        mm_input_pb = MultimodalInputPB()
        mm_input_pb.multimodal_url = url
        mm_input_pb.multimodal_type = 1
        mm_input_pb.mm_preprocess_config.width = -1
        mm_input_pb.mm_preprocess_config.height = -1
        mm_input_pb.mm_preprocess_config.min_pixels = -1
        mm_input_pb.mm_preprocess_config.max_pixels = -1
        mm_input_pb.mm_preprocess_config.fps = -1
        mm_input_pb.mm_preprocess_config.min_frames = -1
        mm_input_pb.mm_preprocess_config.max_frames = -1

        input_pb.multimodal_inputs.append(mm_input_pb)

    return input_pb

def main():
    with grpc.insecure_channel('localhost:12346', options=[('grpc.max_receive_message_length', 1024 * 1024 * 1024),
                                                           ('grpc.max_send_message_length', 1024 * 1024 * 1024)]) as channel:
        stub = MultimodalRpcServiceStub(channel)
        response = stub.RemoteMultimodalEmbedding(trans_multimodal_input(['/mnt/nas1/hf/llava-v1.5-7b/1.jpg']))
    for res in response.multimodal_outputs:
        print(trans_tensor(res.multimodal_embedding))
        print(trans_tensor(res.multimodal_pos_id))

if __name__ == '__main__':
    main()
```

hints: Grpc port is START_PORT + 1.