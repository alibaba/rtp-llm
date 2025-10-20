#pragma once
#include "rtp_llm/cpp/models/GptModel.h"
#include <optional>
#include <string>
#include <mutex>
#include "rtp_llm/cpp/core/Types.h"
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include "rtp_llm/models_py/bindings/OpDefsUtils.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

namespace py = pybind11;

namespace rtp_llm {

class PyWrappedModel: public GptModel {
public:
    // py_instance is `py_model` indeedly.
    PyWrappedModel(const GptModelInitParams& params, py::object py_instance, bool is_prefill_cuda_graph_mode = false);
    ~PyWrappedModel();

    GptModelOutputs forward(const GptModelInputs& inputs) override;
    GptModelOutputs forwardMicroBatched(const GptModelInputs& inputs);

private:
    std::optional<PyCacheStoreInputs> prepareWriteCacheParams(const GptModelInputs& inputs);

private:
    // Helper functions to reduce code duplication
    torch_ext::PyAttentionInputs   buildPyAttentionInputs(const GptModelInputs& inputs);
    torch_ext::BertEmbeddingInputs buildBertEmbeddingInputs(const GptModelInputs& inputs);
    void                           setupKVCacheForAttentionInputs(torch_ext::PyAttentionInputs& py_attn_inputs,
                                                                  const GptModelInputs&         inputs,
                                                                  BufferPtr&                    kv_cache_block_id_device);
    GptModelOutputs
                  callForwardPostLayers(BufferPtr hidden_states, const GptModelInputs& inputs, bool is_forward_method);
    GraphBase*    graph_runner_{nullptr};
    py::object    py_model_;
    bool          enable_cuda_graph_{false};
    torch::Tensor k_cache_base_tensor_;
    torch::Tensor v_cache_base_tensor_;
    torch::Tensor k_scale_base_tensor_;
    torch::Tensor v_scale_base_tensor_;
};

// NOTE(wangyin): constructor can not be compiled correctly when placed in cc file.
inline PyWrappedModel::PyWrappedModel(const GptModelInitParams& params,
                                      py::object                py_instance,
                                      bool                      is_prefill_cuda_graph_mode):
    GptModel(params), enable_cuda_graph_(params.device->initParams().hw_kernel_config.enable_cuda_graph) {
    if (setenv("PYTHONUNBUFFERED", "TRUE", 1) != 0) {
        RTP_LLM_LOG_WARNING("Failed to set PYTHONUNBUFFERED environment variable on POSIX.");
    } else {
        RTP_LLM_LOG_INFO("Set PYTHONUNBUFFERED=TRUE for Python interpreter.");
    }
    if (k_cache_buffer_) {
        k_cache_base_tensor_ = Buffer2torchTensor(k_cache_buffer_, false);
        v_cache_base_tensor_ = Buffer2torchTensor(v_cache_buffer_, false);
    }
    if (k_scale_buffer_) {
        k_scale_base_tensor_ = Buffer2torchTensor(k_scale_buffer_, false);
        v_scale_base_tensor_ = Buffer2torchTensor(v_scale_buffer_, false);
    }

    py::gil_scoped_acquire          gil;
    torch_ext::PyModelInitResources init_resources;
    if (k_cache_buffer_) {
        torch_ext::KVCache kv_cache;
        kv_cache.k_cache_base = k_cache_base_tensor_;
        kv_cache.v_cache_base = v_cache_base_tensor_;
        if (k_scale_buffer_) {
            kv_cache.k_scale_base = k_scale_base_tensor_;
            kv_cache.v_scale_base = v_scale_base_tensor_;
        }
        init_resources.kv_cache = kv_cache;
    }
    py::object py_init_result;
    if (enable_cuda_graph_) {
        int kv_cache_offset =
            is_prefill_cuda_graph_mode ? 0 : k_cache_buffer_->shape()[0] * k_cache_buffer_->shape()[1];
        graph_runner_ = device_->getDeviceGraphRunner(
            params.device->initParams(), py_instance, kv_cache_offset, is_prefill_cuda_graph_mode);
        if (weights_.position_encoding) {
            graph_runner_->setPositionEncoding(Buffer2torchTensor(weights_.position_encoding->kernel, false).cuda());
        }
        if (weights_.token_type_embedding) {
            graph_runner_->setTokenTypeEmbedding(
                Buffer2torchTensor(weights_.token_type_embedding->kernel, false).cuda());
        }
        if (weights_.layers[0].self_attention_weights.qkv_weight->kernel) {
            graph_runner_->setQKVDim(weights_.layers[0].self_attention_weights.qkv_weight->kernel->shape()[1]);
        }
        graph_runner_->setInputEmbeddingScalar(description_.input_embedding_scalar);
        caffe2::TypeMeta dtype = torch::scalarTypeToTypeMeta(dataTypeToTorchType(description_.data_type));
        graph_runner_->setModelDataType(dtype);
        RTP_LLM_CHECK_WITH_INFO(graph_runner_ != nullptr, "graph_runner_ can't be null");
        auto py_initialize_method = py_instance.attr("initialize");
        py_init_result            = py_initialize_method(init_resources);
        graph_runner_->initCapture();
    } else {
        py_model_                 = std::move(py_instance);
        auto py_initialize_method = py_model_.attr("initialize");
        py_init_result            = py_initialize_method(init_resources);
    }

    auto py_init_success = py_init_result.cast<bool>();
    if (!py_init_success) {
        throw std::runtime_error("PyWrappedModel constructor: Python model initialization failed.");
    }
    RTP_LLM_LOG_INFO("PyWrappedModel initialized done.");
}

}  // namespace rtp_llm
