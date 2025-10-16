#pragma once
#include "ATen/core/TensorBody.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/cpp/devices/cuda_impl/CudaFlashInfer.h"
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraph.h>
using namespace torch_ext;

class CaptureMemoryHold {
public:
    void setHiddenStates(at::Tensor hidden_states) {
        decoder_layer_hidden_states_ = hidden_states;
    };

    CaptureMemoryHold() {}

    CaptureMemoryHold(at::Tensor hidden_states, PyModelInputs& inputs, int kv_cache_block_offset, bool is_embedding):
        decoder_layer_hidden_states_(hidden_states) {
        py_model_inputs_.attention_inputs.input_lengths            = inputs.attention_inputs.input_lengths;
        py_model_inputs_.attention_inputs.sequence_lengths         = inputs.attention_inputs.sequence_lengths;
        py_model_inputs_.attention_inputs.kv_cache_block_id_device = inputs.attention_inputs.kv_cache_block_id_device;
        py_model_inputs_.attention_inputs.kv_cache_block_id_host   = inputs.attention_inputs.kv_cache_block_id_host;
        py_model_inputs_.attention_inputs.prefix_lengths           = inputs.attention_inputs.prefix_lengths;
        py_model_inputs_.input_ids                                 = inputs.input_ids;
        py_model_inputs_.attention_inputs.cu_seqlens               = inputs.attention_inputs.cu_seqlens;
        py_model_inputs_.attention_inputs.padding_offset           = inputs.attention_inputs.padding_offset;
        py_model_inputs_.attention_inputs.is_prefill               = is_embedding;
        py_model_inputs_.attention_inputs.dtype                    = inputs.attention_inputs.dtype;
        py_model_inputs_.attention_inputs.kv_block_offset          = kv_cache_block_offset;
        py_model_inputs_.attention_inputs.prefill_cuda_graph_copy_params =
            inputs.attention_inputs.prefill_cuda_graph_copy_params;
        py_model_inputs_.bert_embedding_inputs = inputs.bert_embedding_inputs;
    }

public:
    // for attention params
    rtp_llm::ParamsBasePtr params_ptr{nullptr};
    // for output
    at::Tensor decoder_layer_hidden_states_;
    // for input
    PyModelInputs py_model_inputs_;
};

class GraphInstance {
public:
    at::cuda::CUDAGraph graph_;
    CaptureMemoryHold   mem_hold_;
};

class CudaGraphStreamLife {
public:
    // CudaDevice's `stream_` is torch default stream
    CudaGraphStreamLife(at::cuda::CUDAStream capture_stream, rtp_llm::DeviceBase* device):
        origin_stream_(at::cuda::getCurrentCUDAStream(at::cuda::current_device())) {
        cuda_device_ = dynamic_cast<rtp_llm::CudaDevice*>(device);
        // Set `capture_stream` for capture. All kernels should use this stream while capturing.
        origin_cuda_device_stream_ = cuda_device_->getStream();
        cuda_device_->setStream(capture_stream.stream());
        RTP_LLM_LOG_INFO("Set Cuda Stream: capture_stream -> %d, set_stream -> %d, origin_cuda_device_stream_-> %d",
                         capture_stream.stream(),
                         reinterpret_cast<int64_t>(cuda_device_->getStream()),
                         origin_cuda_device_stream_);
        at::cuda::setCurrentCUDAStream(capture_stream);
    }
    ~CudaGraphStreamLife() {
        at::cuda::setCurrentCUDAStream(origin_stream_);
        cuda_device_->setStream(origin_cuda_device_stream_);
    }

private:
    at::cuda::CUDAStream origin_stream_;
    cudaStream_t         origin_cuda_device_stream_;
    rtp_llm::CudaDevice* cuda_device_;
};
