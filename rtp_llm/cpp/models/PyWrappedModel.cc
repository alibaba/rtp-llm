#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/utils/utils.h"
#include "rtp_llm/cpp/utils/AttentionConfig.h"
#include <stdexcept>
#include <mutex>
#include "rtp_llm/cpp/utils/PyUtils.h"

#include <cstdlib>
#include <iostream>

using namespace torch_ext;

namespace rtp_llm {

void getPaddingOffset(
    int32_t* padding_offset, int32_t* prefill_length, int32_t* prefix_length, int32_t batch_size, int32_t max_seq_len) {
    // do cumulated sum
    int32_t cum_offset = 0;
    int32_t index      = 0;
    for (int32_t i = 0; i < batch_size; i++) {
        int32_t seq_len = prefill_length[i];
        if (prefix_length) {
            seq_len += prefix_length[i];
        }
        if (padding_offset) {
            for (int32_t j = 0; j < seq_len; j++) {
                padding_offset[index] = cum_offset;
                index++;
            }
        }
        cum_offset += max_seq_len - seq_len;
    }
}

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

GptModelOutputs PyWrappedModel::forwardMicroBatched(const GptModelInputs& inputs) {
    py::object py_forward_method = py_model_.attr("forward_micro_batch");
    if (device_props_.ffn_as_service) {
        py::object py_outputs_obj = py_forward_method(std::vector<PyModelInputs>{});
        return GptModelOutputs({nullptr, nullptr, nullptr, nullptr, nullptr});
    }

    auto micro_batch_plan = planMicroBatches(inputs);

    auto [split_inputs, _] = splitInputsIntoMicroBatches(inputs, micro_batch_plan);
    std::vector<PyModelInputs> input_list;
    input_list.reserve(split_inputs.size());
    std::vector<BufferPtr> kv_cache_block_ids_device(split_inputs.size());

    for (size_t i = 0; i < split_inputs.size(); ++i) {
        const auto& micro_inputs = split_inputs[i].kv_cache_block_id ? split_inputs[i] : split_inputs[0];
        torch_ext::PyAttentionInputs py_attn_inputs;
        py_attn_inputs.prefix_lengths = Buffer2torchTensor(micro_inputs.prefix_lengths);

        py_attn_inputs.sequence_lengths = Buffer2torchTensor(micro_inputs.sequence_lengths, false);
        py_attn_inputs.input_lengths    = Buffer2torchTensor(micro_inputs.input_lengths);

        BufferPtr kv_cache_block_id_device;
        if (k_cache_buffer_) {
            kv_cache_block_ids_device[i] =
                device_->clone({*micro_inputs.kv_cache_block_id, AllocationType::DEVICE, {"kv_cache_block_id"}});
            py_attn_inputs.kv_cache_block_id_host = Buffer2torchTensor(micro_inputs.kv_cache_block_id);

            py_attn_inputs.kv_cache_block_id_device = Buffer2torchTensor(kv_cache_block_ids_device[i], false);
            py_attn_inputs.kv_block_offset =
                k_cache_buffer_ ? k_cache_buffer_->shape()[0] * k_cache_buffer_->shape()[1] : 0;
            // py_attn_inputs.kv_block_offset += 2;
        }
        py_attn_inputs.dtype      = torch::kBFloat16;
        py_attn_inputs.is_prefill = !py_attn_inputs.sequence_lengths.size(0);
        torch::Tensor cu_seqlens  = torch::zeros({device_->initParams().concurrency_config.concurrency_limit + 1},
                                                torch::TensorOptions(torch::kInt32).device(torch::kCPU));
        int           batch_size  = py_attn_inputs.input_lengths.size(0);
        cu_seqlens                = cu_seqlens.cuda();
        cu_seqlens.slice(0, 1, batch_size + 1) = py_attn_inputs.input_lengths.cumsum(0);
        py_attn_inputs.cu_seqlens              = cu_seqlens;

        torch::Tensor token_ids = Buffer2torchTensor(micro_inputs.combo_tokens).cuda();

        input_list.emplace_back(PyModelInputs{token_ids, py_attn_inputs});
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

    return forwardPostLayers(hidden_states,
                             inputs.input_lengths->shape()[0] != inputs.sequence_lengths->shape()[0],
                             false,
                             inputs.lm_output_indexes,
                             false,
                             inputs.combo_tokens->shape()[0],
                             inputs,
                             nullptr);
}

GptModelOutputs PyWrappedModel::forward(const GptModelInputs& inputs) {

    py::gil_scoped_acquire gil;

    try {
        RTP_LLM_LOG_INFO("Calling forward method on Python object instance.");

        if (int(device_props_.enable_layer_micro_batch)) {
            return forwardMicroBatched(inputs);
        }

        torch::Tensor token_ids = Buffer2torchTensor(inputs.combo_tokens).cuda();

        PyAttentionInputs attention_inputs;
        attention_inputs.prefix_lengths = Buffer2torchTensor(inputs.prefix_lengths);
        // `sequence_lengths`: pinned memory
        attention_inputs.sequence_lengths = Buffer2torchTensor(inputs.sequence_lengths, false);
        attention_inputs.input_lengths    = Buffer2torchTensor(inputs.input_lengths);
        BufferPtr kv_cache_block_id_device;
        if (k_cache_buffer_) {
            kv_cache_block_id_device =
                device_->clone({*inputs.kv_cache_block_id, AllocationType::DEVICE, {"kv_cache_block_id"}});
            attention_inputs.kv_cache_block_id_host   = Buffer2torchTensor(inputs.kv_cache_block_id);
            attention_inputs.kv_cache_block_id_device = Buffer2torchTensor(kv_cache_block_id_device, false);
            attention_inputs.kv_block_offset =
                k_cache_buffer_ ? k_cache_buffer_->shape()[0] * k_cache_buffer_->shape()[1] : 0;
        }
        attention_inputs.dtype      = torch::kBFloat16;
        attention_inputs.is_prefill = !attention_inputs.sequence_lengths.size(0);

        torch::Tensor cu_seqlens = torch::zeros({device_->initParams().concurrency_config.concurrency_limit + 1},
                                                torch::TensorOptions(torch::kInt32).device(torch::kCPU));
        int           batch_size = attention_inputs.input_lengths.size(0);
        cu_seqlens.slice(0, 1, batch_size + 1) = attention_inputs.input_lengths.cumsum(0);
        int32_t total_tokens                   = cu_seqlens[batch_size].item<int32_t>();
        cu_seqlens                             = cu_seqlens.cuda();
        attention_inputs.cu_seqlens            = cu_seqlens;

        // inputs_length:  [1,2,1,1] ,total_tokens = 5
        // padding_offsets: [0,1,1,1,2]
        int  max_seq_len = attention_inputs.input_lengths.max().item<int32_t>();
        auto padding_offset_host =
            torch::zeros({total_tokens}, torch::TensorOptions(torch::kInt32).device(torch::kCPU));
        getPaddingOffset(padding_offset_host.data_ptr<int32_t>(),
                         attention_inputs.input_lengths.data_ptr<int32_t>(),
                         attention_inputs.prefix_lengths.data_ptr<int32_t>(),
                         batch_size,
                         max_seq_len);
        attention_inputs.padding_offset = padding_offset_host.cuda();

        auto           py_model_inputs = PyModelInputs({token_ids, attention_inputs});
        PyModelOutputs py_model_outputs;
        // Cast the Python object to PyModelOutputs and extract hidden states
        if (enable_cuda_graph_) {
            py_model_outputs = graph_runner_->forward(py_model_inputs);
        } else {
            auto py_model_forward = py_model_.attr("forward");
            auto outputs          = py_model_forward(py_model_inputs);
            py_model_outputs      = outputs.cast<PyModelOutputs>();
        }
        auto hidden_states_tensor = py_model_outputs.hidden_states;
        // std::cout << "hidden_states_tensor: " << hidden_states_tensor << std::endl;
        auto hidden_states = torchTensor2Buffer(hidden_states_tensor);

        RTP_LLM_LOG_INFO("Python object instance forward method called successfully.");

        return forwardPostLayers(hidden_states,
                                 inputs.input_lengths->shape()[0] != inputs.sequence_lengths->shape()[0],
                                 false,
                                 inputs.lm_output_indexes,
                                 false,
                                 inputs.combo_tokens->shape()[0],
                                 inputs,
                                 nullptr,
                                 true);

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
