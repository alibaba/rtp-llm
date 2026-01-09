
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
#include "rtp_llm/cpp/devices/GraphBase.h"
#if USING_CUDA
#include <c10/cuda/CUDAStream.h>
#include "rtp_llm/cpp/devices/cuda_impl/CudaGraphRunner.h"
#endif

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
    bool          is_prefill_cuda_graph_mode_{false};
    torch::Tensor k_cache_base_tensor_;
    torch::Tensor v_cache_base_tensor_;
    torch::Tensor k_scale_base_tensor_;
    torch::Tensor v_scale_base_tensor_;
};

// NOTE(wangyin): constructor can not be compiled correctly when placed in cc file.
inline PyWrappedModel::PyWrappedModel(const GptModelInitParams& params,
                                      py::object                py_instance,
                                      bool                      is_prefill_cuda_graph_mode):
    GptModel(params),
    enable_cuda_graph_(params.device->initParams().hw_kernel_config.enable_cuda_graph),
    is_prefill_cuda_graph_mode_(is_prefill_cuda_graph_mode) {

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
        kv_cache.k_cache_base       = k_cache_base_tensor_;
        kv_cache.v_cache_base       = v_cache_base_tensor_;
        kv_cache.seq_size_per_block = params.description.attention_conf.tokens_per_block;
        if (k_scale_buffer_) {
            kv_cache.k_scale_base = k_scale_base_tensor_;
            kv_cache.v_scale_base = v_scale_base_tensor_;
        }
        init_resources.kv_cache = kv_cache;
    }
    py::object py_init_result;
    // Always initialize py_model_ so it can be used as fallback when CUDA graph cannot run
    py_model_                 = py_instance;
    auto py_initialize_method = py_model_.attr("initialize");
    py_init_result            = py_initialize_method(init_resources);
    if (enable_cuda_graph_) {
#if USING_CUDA
        at::cuda::CUDAStream capture_stream = at::cuda::getCurrentCUDAStream(at::cuda::current_device());
        c10::ScalarType      dtype          = dataTypeToTorchType(description_.data_type);

        int num_tokens_per_bs = 1;
        if (is_prefill_cuda_graph_mode) {
            // For embedding model (prefill-only), use max_seq_len
            num_tokens_per_bs = params.device->initParams().max_seq_len;
        } else if (params.device->initParams().sp_config.gen_num_per_cycle > 1 && !params.model_id) {
            // For speculative sampling
            // -- model_id == 0: target model
            // -- model_id == 1: draft model
            num_tokens_per_bs = params.device->initParams().sp_config.gen_num_per_cycle + 1;
        }

        graph_runner_ = new CudaGraphRunner(params.device->initParams(),
                                            py_instance,
                                            capture_stream,
                                            dtype,
                                            num_tokens_per_bs,
                                            is_prefill_cuda_graph_mode);
        RTP_LLM_CHECK_WITH_INFO(graph_runner_ != nullptr, "graph_runner_ can't be nullptr in PyWrapper");
#else
        RTP_LLM_CHECK_WITH_INFO(false, "CUDA Graph is only supported on CUDA platform for now");
#endif

        if (weights_.position_encoding) {
            graph_runner_->setPositionEncoding(Buffer2torchTensor(weights_.position_encoding->kernel, false).cuda());
        }
        if (weights_.token_type_embedding) {
            graph_runner_->setTokenTypeEmbedding(
                Buffer2torchTensor(weights_.token_type_embedding->kernel, false).cuda());
        }
        graph_runner_->setInputEmbeddingScalar(description_.input_embedding_scalar);
        auto py_get_factor          = py_instance.attr("get_position_id_len_factor");
        int  position_id_len_factor = py_get_factor().cast<int>();
        graph_runner_->setPositionIdLenFactor(position_id_len_factor);
        auto py_need_combo_position_ids = py_instance.attr("need_combo_position_ids");
        bool need_combo_position_ids    = py_need_combo_position_ids().cast<bool>();
        graph_runner_->setNeedComboPositionIds(need_combo_position_ids);
        RTP_LLM_CHECK_WITH_INFO(graph_runner_ != nullptr, "graph_runner_ can't be null");
        auto py_initialize_method = py_instance.attr("initialize");
        py_init_result            = py_initialize_method(init_resources);
        graph_runner_->initCapture();
    }

    auto py_init_success = py_init_result.cast<bool>();
    if (!py_init_success) {
        throw std::runtime_error("PyWrappedModel constructor: Python model initialization failed.");
    }
    RTP_LLM_LOG_INFO("PyWrappedModel initialized done.");
}

}  // namespace rtp_llm
