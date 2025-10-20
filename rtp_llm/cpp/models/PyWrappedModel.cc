#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/utils/utils.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include <cstdint>
#include <stdexcept>
#include <mutex>
#include <vector>
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include <cstdlib>
#include <iostream>
#include "rtp_llm/cpp/devices/utils/DevicePerfWrapper.h"
namespace rtp_llm {

PyWrappedModel::~PyWrappedModel() {
    try {
        py::gil_scoped_acquire gil;
        if (!device_->initParams().hw_kernel_config.enable_cuda_graph) {
            py_model_.release();  // Release the Python object
        } else {
            RTP_LLM_CHECK_WITH_INFO(graph_runner_ != nullptr, "graph_runner_ can not be nullptr");
            delete graph_runner_;
        }
        RTP_LLM_LOG_INFO("PyWrappedModel destroyed, Python object instance released.");
    } catch (const py::error_already_set& e) {
        RTP_LLM_LOG_ERROR("Python error during PyWrappedModel destruction: %s", e.what());
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("C++ error during PyWrappedModel destruction: %s", e.what());
    }
}

// Helper function to build PyAttentionInputs from GptModelInputs
torch_ext::PyAttentionInputs PyWrappedModel::buildPyAttentionInputs(const GptModelInputs& inputs) {
    torch_ext::PyAttentionInputs py_attn_inputs;
    py_attn_inputs.prefix_lengths   = Buffer2torchTensor(inputs.prefix_lengths);
    py_attn_inputs.sequence_lengths = Buffer2torchTensor(inputs.sequence_lengths, false);
    py_attn_inputs.input_lengths    = Buffer2torchTensor(inputs.input_lengths);

    if (k_cache_buffer_) {
        py_attn_inputs.kv_cache_block_id_host = Buffer2torchTensor(inputs.kv_cache_block_id);
        py_attn_inputs.kv_block_offset =
            k_cache_buffer_ ? k_cache_buffer_->shape()[0] * k_cache_buffer_->shape()[1] : 0;
    }

    py_attn_inputs.dtype      = dataTypeToTorchType(description_.data_type);
    py_attn_inputs.is_prefill = !py_attn_inputs.sequence_lengths.size(0);

    // Calculate cu_seqlens
    torch::Tensor cu_seqlens = torch::zeros({device_->initParams().concurrency_config.concurrency_limit + 1},
                                            torch::TensorOptions(torch::kInt32).device(torch::kCPU));
    int           batch_size = py_attn_inputs.input_lengths.size(0);
    cu_seqlens               = cu_seqlens.cuda();
    cu_seqlens.slice(0, 1, batch_size + 1) = py_attn_inputs.input_lengths.cumsum(0);
    py_attn_inputs.cu_seqlens              = cu_seqlens;
    py_attn_inputs.sequence_lengths.pin_memory();
    return py_attn_inputs;
}

// Helper function to setup KV cache for attention inputs
void PyWrappedModel::setupKVCacheForAttentionInputs(torch_ext::PyAttentionInputs& py_attn_inputs,
                                                    const GptModelInputs&         inputs,
                                                    BufferPtr&                    kv_cache_block_id_device) {
    if (k_cache_buffer_) {
        kv_cache_block_id_device =
            device_->clone({*inputs.kv_cache_block_id, AllocationType::DEVICE, {"kv_cache_block_id"}});
        py_attn_inputs.kv_cache_block_id_device = Buffer2torchTensor(kv_cache_block_id_device, false);
    }
}

// Helper function to build BertEmbeddingInputs from GptModelInputs
torch_ext::BertEmbeddingInputs PyWrappedModel::buildBertEmbeddingInputs(const GptModelInputs& inputs) {
    torch_ext::BertEmbeddingInputs bert_embedding_inputs;

    // Convert combo_position_ids from Buffer to torch::Tensor
    if (inputs.combo_position_ids) {
        bert_embedding_inputs.combo_position_ids = Buffer2torchTensor(inputs.combo_position_ids, false).cuda();
    }

    // Convert combo_tokens_type_ids from Buffer to torch::Tensor
    if (inputs.combo_tokens_type_ids) {
        bert_embedding_inputs.combo_tokens_type_ids = Buffer2torchTensor(inputs.combo_tokens_type_ids, false).cuda();
    }

    // Get position_encoding from model weights (no clone needed for weights)
    if (weights_.position_encoding) {
        bert_embedding_inputs.position_encoding = Buffer2torchTensor(weights_.position_encoding->kernel, false).cuda();
    }

    // Get token_type_embedding from model weights (no clone needed for weights)
    if (weights_.token_type_embedding) {
        bert_embedding_inputs.token_type_embedding =
            Buffer2torchTensor(weights_.token_type_embedding->kernel, false).cuda();
    }

    // Set input_embedding_scalar
    bert_embedding_inputs.input_embedding_scalar = description_.input_embedding_scalar;
    return bert_embedding_inputs;
}

// Helper function to call forwardPostLayers with common parameters
GptModelOutputs PyWrappedModel::callForwardPostLayers(BufferPtr             hidden_states,
                                                      const GptModelInputs& inputs,
                                                      bool                  skip_final_layernorm) {
    return forwardPostLayers(hidden_states,
                             inputs.input_lengths->shape()[0] != inputs.sequence_lengths->shape()[0],
                             false,
                             inputs.lm_output_indexes,
                             false,
                             inputs.combo_tokens->shape()[0],
                             inputs,
                             nullptr,
                             skip_final_layernorm);
}

std::optional<PyCacheStoreInputs> PyWrappedModel::prepareWriteCacheParams(const GptModelInputs& inputs) {
    std::optional<PyCacheStoreInputs> params;
    if (!inputs.warmup && inputs.pd_separation) {
        const auto           decoder_batch_size = inputs.sequence_lengths->shape()[0];
        const auto           context_batch_size = inputs.input_lengths->shape()[0] - decoder_batch_size;
        std::vector<int64_t> cache_keys_vec;
        if (inputs.cache_keys) {
            cache_keys_vec = rtp_llm::buffer2vector<int64_t>(*inputs.cache_keys);
        }
        PyCacheStoreInputs cache_store_inputs{context_batch_size,
                                              decoder_batch_size,
                                              Buffer2torchTensor(inputs.request_id),
                                              Buffer2torchTensor(inputs.request_pd_separation),
                                              transVectorToString(cache_keys_vec),
                                              inputs.seq_size_per_block,
                                              inputs.k_block_size,
                                              inputs.v_block_size,
                                              inputs.scale_block_size,
                                              inputs.pd_separation,
                                              model_id_,
                                              inputs.decode_entrance,
                                              inputs.warmup,
                                              description_.attention_conf.use_mla
                                                  && device_->mla_ops_type != rtp_llm::MlaOpsType::MHA};
        params = cache_store_inputs;
    }
    return params;
}

GptModelOutputs PyWrappedModel::forwardMicroBatched(const GptModelInputs& inputs) {
    py::object py_forward_method = py_model_.attr("forward_micro_batch");
    if (device_props_.ffn_as_service) {
        py::object py_outputs_obj = py_forward_method(std::vector<PyModelInputs>{});
        return GptModelOutputs({nullptr, nullptr, nullptr, nullptr, nullptr});
    }

    auto micro_batch_plan  = planMicroBatches(inputs);
    auto [split_inputs, _] = splitInputsIntoMicroBatches(inputs, micro_batch_plan);
    std::vector<PyModelInputs> input_list;
    input_list.reserve(split_inputs.size());
    std::vector<BufferPtr> kv_cache_block_ids_device(split_inputs.size());

    for (size_t i = 0; i < split_inputs.size(); ++i) {
        const auto& micro_inputs          = split_inputs[i].kv_cache_block_id ? split_inputs[i] : split_inputs[0];
        auto        py_attn_inputs        = buildPyAttentionInputs(micro_inputs);
        auto        bert_embedding_inputs = buildBertEmbeddingInputs(micro_inputs);
        setupKVCacheForAttentionInputs(py_attn_inputs, micro_inputs, kv_cache_block_ids_device[i]);
        calculatePaddingOffset(py_attn_inputs);
        torch::Tensor token_ids = Buffer2torchTensor(micro_inputs.combo_tokens).cuda();
        input_list.emplace_back(PyModelInputs{token_ids, py_attn_inputs, bert_embedding_inputs});
    }

    py::object py_outputs_obj   = py_forward_method(input_list);
    auto       py_model_outputs = py_outputs_obj.cast<std::vector<PyModelOutputs>>();
    RTP_LLM_CHECK_WITH_INFO(py_model_outputs.size() == input_list.size(),
                            "py_model_outputs.size:%d != micro_batch_inputs.size:%d",
                            py_model_outputs.size(),
                            input_list.size());

    // TODO: merge hidden states in one buffer
    BufferPtr hidden_states = nullptr;
    if (!micro_batch_plan.enable) {
        RTP_LLM_CHECK_WITH_INFO(py_model_outputs[0].hidden_states.size(0) == inputs.combo_tokens->shape()[0],
                                "py_model_outputs[0].hidden_states.size(0):%d != inputs.combo_tokens->shape()[0]:%d",
                                py_model_outputs[0].hidden_states.size(0),
                                inputs.combo_tokens->shape()[0]);
        hidden_states = torchTensor2Buffer(py_model_outputs[0].hidden_states);
    } else {
        hidden_states =
            device_->allocateBuffer({description_.data_type,
                                     {inputs.combo_tokens->shape()[0], description_.attention_conf.hidden_size},
                                     AllocationType::DEVICE});
        int offset = 0;
        for (int i = 0; i < py_model_outputs.size(); i++) {
            RTP_LLM_CHECK_WITH_INFO(
                offset + py_model_outputs[i].hidden_states.size(0) <= inputs.combo_tokens->shape()[0],
                "offset + py_model_outputs[i].hidden_states.size(0):%d > inputs.combo_tokens->shape()[0]:%d",
                offset + py_model_outputs[i].hidden_states.size(0),
                inputs.combo_tokens->shape()[0]);
            auto hidden_states_slice = hidden_states->slice(offset, offset + py_model_outputs[i].hidden_states.size(0));
            auto py_model_output     = py_model_outputs[i];
            device_->copy({*hidden_states_slice, *torchTensor2Buffer(py_model_output.hidden_states)});
            offset += py_model_outputs[i].hidden_states.size(0);
        }
        RTP_LLM_CHECK_WITH_INFO(offset == inputs.combo_tokens->shape()[0],
                                "total out hidden size:%d != inputs.combo_tokens->shape()[0]:%d",
                                offset,
                                inputs.combo_tokens->shape()[0]);
    }

    RTP_LLM_LOG_DEBUG("Python object instance forward method called successfully.");

    return callForwardPostLayers(hidden_states, inputs, false);
}

GptModelOutputs PyWrappedModel::forward(const GptModelInputs& inputs) {
    DevicePerfWrapper      wrapper(device_, "py model forward");
    py::gil_scoped_acquire gil;
    try {
        RTP_LLM_LOG_DEBUG("Calling forward method on Python object instance.");

        if (int(device_props_.enable_layer_micro_batch)) {
            return forwardMicroBatched(inputs);
        }

        torch::Tensor token_ids = Buffer2torchTensor(inputs.combo_tokens).cuda();

        auto      attention_inputs      = buildPyAttentionInputs(inputs);
        auto      bert_embedding_inputs = buildBertEmbeddingInputs(inputs);
        BufferPtr kv_cache_block_id_device;
        if (!inputs.warmup && inputs.pd_separation) {
            attention_inputs.cache_store_inputs = prepareWriteCacheParams(inputs);
        }
        setupKVCacheForAttentionInputs(attention_inputs, inputs, kv_cache_block_id_device);
        calculatePaddingOffset(attention_inputs);

        auto           py_model_inputs = PyModelInputs({token_ids, attention_inputs, bert_embedding_inputs});
        PyModelOutputs py_model_outputs;
        // Cast the Python object to PyModelOutputs and extract hidden states
        if (enable_cuda_graph_) {
            DevicePerfWrapper wrapper(device_, "cuda graph python forward");
            py_model_outputs = graph_runner_->forward(py_model_inputs);
        } else {
            DevicePerfWrapper wrapper(device_, "normal forward");
            auto              py_model_forward = py_model_.attr("forward");
            auto              outputs          = py_model_forward(py_model_inputs);
            py_model_outputs                   = outputs.cast<PyModelOutputs>();
        }
        auto hidden_states_tensor = py_model_outputs.hidden_states;
        auto hidden_states        = torchTensor2Buffer(hidden_states_tensor);

        RTP_LLM_LOG_DEBUG("Python object instance forward method called successfully.");

        return callForwardPostLayers(hidden_states, inputs, true);

    } catch (const py::error_already_set& e) {
        RTP_LLM_LOG_ERROR("Python error during forward call on Python instance: %s", e.what());
        throw std::runtime_error(std::string("pybind11 error during forward call on Python instance: ") + e.what());
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("C++ error during forward call on Python instance: %s", e.what());
        throw std::runtime_error(std::string("C++ error during forward call on Python instance: ") + e.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("An unknown error occurred during forward call on Python instance.");
        throw std::runtime_error("An unknown error occurred during forward call on Python instance.");
    }
}

}  // namespace rtp_llm
