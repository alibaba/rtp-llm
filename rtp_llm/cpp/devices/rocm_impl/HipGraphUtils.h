#pragma once
#include "ATen/core/TensorBody.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include <ATen/hip/HIPGraph.h>
#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
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

class HipGraphStreamLife {
public:
    // CudaDevice's `stream_` is torch default stream
    HipGraphStreamLife(at::hip::HIPStream capture_stream, rtp_llm::DeviceBase* device);
    ~HipGraphStreamLife();

private:
    at::hip::HIPStream   origin_stream_;
    hipStream_t          origin_rocm_device_stream_;
    rtp_llm::ROCmDevice* rocm_device_;
};