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

PyWrappedModel::~PyWrappedModel() {
    try {
        py::gil_scoped_acquire gil;
        cuda_graph_runner_.py_instance_.release();  // Release the Python object
        RTP_LLM_LOG_INFO("PyWrappedModel destroyed, Python object instance released.");
    } catch (const py::error_already_set& e) {
        RTP_LLM_LOG_ERROR("Python error during PyWrappedModel destruction: %s", e.what());
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("C++ error during PyWrappedModel destruction: %s", e.what());
    }
}

GptModelOutputs PyWrappedModel::forward(const GptModelInputs& inputs) {

    py::gil_scoped_acquire gil;

    try {
        RTP_LLM_LOG_INFO("Calling forward method on Python object instance.");

        torch::Tensor token_ids = Buffer2torchTensor(inputs.combo_tokens).cuda();

        PyAttentionInputs attention_inputs;
        attention_inputs.prefix_lengths = Buffer2torchTensor(inputs.prefix_lengths);
        // `sequence_lengths`: pinned memory
        attention_inputs.sequence_lengths = Buffer2torchTensor(inputs.sequence_lengths, false);
        attention_inputs.input_lengths    = Buffer2torchTensor(inputs.input_lengths);
        auto kv_cache_block_id_device =
            device_->clone({*inputs.kv_cache_block_id, AllocationType::DEVICE, {"kv_cache_block_id"}});
        attention_inputs.kv_cache_block_id_host   = Buffer2torchTensor(inputs.kv_cache_block_id);
        attention_inputs.kv_cache_block_id_device = Buffer2torchTensor(kv_cache_block_id_device, false);
        attention_inputs.dtype                    = torch::kHalf;
        attention_inputs.is_prefill               = !attention_inputs.sequence_lengths.size(0);
        attention_inputs.kv_block_offset          = k_cache_buffer_->shape()[0] * k_cache_buffer_->shape()[1];
        auto py_model_inputs                      = PyModelInputs({token_ids, attention_inputs});
        // Cast the Python object to PyModelOutputs and extract hidden states
        auto py_model_outputs = cuda_graph_runner_.forward(py_model_inputs);
        // std::cout << "py_model_outputs:\n " << py_model_outputs.hidden_states << std::endl;
        auto hidden_states_tensor = py_model_outputs.hidden_states;
        auto hidden_states        = torchTensor2Buffer(hidden_states_tensor);

        RTP_LLM_LOG_INFO("Python object instance forward method called successfully.");

        return forwardPostLayers(hidden_states,
                                 inputs.input_lengths->shape()[0] != inputs.sequence_lengths->shape()[0],
                                 false,
                                 inputs.lm_output_indexes,
                                 false,
                                 inputs.combo_tokens->shape()[0],
                                 inputs,
                                 nullptr);

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
