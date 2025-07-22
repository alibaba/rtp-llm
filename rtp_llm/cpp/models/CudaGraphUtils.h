#pragma once
#include "ATen/core/TensorBody.h"
#include <ATen/cuda/CUDAGraph.h>
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include "rtp_llm/cpp/models/GptModel.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaFlashInfer.h"
using namespace torch_ext;

class CaptureMemoryHold {
public:
    CaptureMemoryHold() {}

    CaptureMemoryHold(at::Tensor hidden_states, PyModelInputs& inputs, int kv_cache_block_offset):
        decoder_layer_hidden_states(hidden_states) {
        py_model_inputs.attention_inputs.input_lengths            = inputs.attention_inputs.input_lengths;
        py_model_inputs.attention_inputs.sequence_lengths         = inputs.attention_inputs.sequence_lengths;
        py_model_inputs.attention_inputs.kv_cache_block_id_device = inputs.attention_inputs.kv_cache_block_id_device;
        py_model_inputs.attention_inputs.kv_cache_block_id_host   = inputs.attention_inputs.kv_cache_block_id_host;
        py_model_inputs.attention_inputs.prefix_lengths           = inputs.attention_inputs.prefix_lengths;
        py_model_inputs.input_ids                                 = inputs.input_ids;
        py_model_inputs.attention_inputs.cu_seqlens               = inputs.attention_inputs.cu_seqlens;
        py_model_inputs.attention_inputs.is_prefill               = false;
        py_model_inputs.attention_inputs.dtype                    = caffe2::TypeMeta::Make<c10::Half>();
        py_model_inputs.attention_inputs.kv_block_offset          = kv_cache_block_offset;
    }

public:
    rtp_llm::FlashInferAttnParams* params_{nullptr};
    // for output
    at::Tensor decoder_layer_hidden_states;
    // for input
    PyModelInputs py_model_inputs;
};

class GraphInstance {
public:
    at::cuda::CUDAGraph graph;
    CaptureMemoryHold   mem_hold;
};

class CudaGraphStreamLife {
public:
    // CudaDevice's `stream_` is torch default stream
    CudaGraphStreamLife(at::cuda::CUDAStream capture_stream, rtp_llm::DeviceBase* device):
        origin_stream(at::cuda::getCurrentCUDAStream(at::cuda::current_device())) {
        cuda_device = dynamic_cast<rtp_llm::CudaDevice*>(device);
        // Set `capture_stream` for capture. All kernels should use this stream while capturing.
        origin_cuda_device_stream = cuda_device->getStream();
        cuda_device->setStream(capture_stream.stream());
        RTP_LLM_LOG_INFO("Set Cuda Stream: capture_stream -> %d, set_stream -> %d, origin_cuda_device_stream-> %d",
                         capture_stream.stream(),
                         reinterpret_cast<int64_t>(cuda_device->getStream()),
                         origin_cuda_device_stream);
        at::cuda::setCurrentCUDAStream(capture_stream);
    }
    ~CudaGraphStreamLife() {
        at::cuda::setCurrentCUDAStream(origin_stream);
        cuda_device->setStream(origin_cuda_device_stream);
    }

private:
    at::cuda::CUDAStream origin_stream;
    cudaStream_t         origin_cuda_device_stream;
    rtp_llm::CudaDevice* cuda_device;
};
