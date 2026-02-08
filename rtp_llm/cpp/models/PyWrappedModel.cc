#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/utils/utils.h"
#include "rtp_llm/cpp/model_utils/AttentionConfig.h"
#include <cstdint>
#include <stdexcept>
#include <mutex>
#include <vector>
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include <cstdlib>
#include <iostream>
#include "rtp_llm/cpp/devices/utils/DevicePerfWrapper.h"

namespace rtp_llm {

torch::Tensor PyWrappedModel::tensorHoldHostAndToCuda(const torch::Tensor& tensor) {
    if (tensor.device().is_cuda()) {
        return tensor;
    }

    buffer_holder_.hold_host(tensor);
    return tensor.to(torch::kCUDA, /*non_blocking=*/true, /*copy=*/false);
}

PyWrappedModel::~PyWrappedModel() {
    try {
        py::gil_scoped_acquire gil;
        // Always release py_model_ since it's always initialized now
        py_model_.release();
        if (graph_runner_ != nullptr) {
            delete graph_runner_;
            graph_runner_ = nullptr;
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
    DevicePerfWrapper            wrapper(device_, "py model buildPyAttentionInputs");
    torch_ext::PyAttentionInputs py_attn_inputs;
    py_attn_inputs.prefix_lengths   = Buffer2torchTensor(inputs.prefix_lengths, false);
    py_attn_inputs.sequence_lengths = Buffer2torchTensor(inputs.sequence_lengths, false);
    py_attn_inputs.input_lengths    = Buffer2torchTensor(inputs.input_lengths, false);

    if (kv_cache_buffer_) {
        py_attn_inputs.kv_cache_block_id_host = Buffer2torchTensor(inputs.kv_cache_block_id);
    }

    // Calculate cu_seqlens
    int    batch_size         = py_attn_inputs.input_lengths.size(0);
    size_t context_batch_size = py_attn_inputs.prefix_lengths.size(0);
    size_t decode_batch_size  = py_attn_inputs.sequence_lengths.size(0);
    py_attn_inputs.dtype      = dataTypeToTorchType(description_.data_type);
    py_attn_inputs.is_prefill = !decode_batch_size;
    RTP_LLM_CHECK_WITH_INFO(
        context_batch_size + decode_batch_size == batch_size,
        "batch size check failed context_batch_size[%ld] decode_batch_size[%ld] total_batch_size[%ld]",
        context_batch_size,
        decode_batch_size,
        batch_size);

    if (context_batch_size > 0) {
        torch::Tensor cu_seqlens =
            torch::zeros({batch_size + 1}, torch::TensorOptions(torch::kInt32).device(torch::kCPU));
        torch::Tensor cu_kv_seqlens =
            torch::zeros({batch_size + 1}, torch::TensorOptions(torch::kInt32).device(torch::kCPU));

        cu_seqlens.slice(0, 1, context_batch_size + 1) = py_attn_inputs.input_lengths.cumsum(0);
        cu_kv_seqlens.slice(0, 1, context_batch_size + 1) =
            py_attn_inputs.input_lengths.add(py_attn_inputs.prefix_lengths).cumsum(0);

        py_attn_inputs.context_total_kv_length = cu_kv_seqlens[context_batch_size].item<int>();
        py_attn_inputs.total_tokens            = cu_seqlens[batch_size].item<int>();
        py_attn_inputs.cu_seqlens              = tensorHoldHostAndToCuda(cu_seqlens);
        py_attn_inputs.cu_kv_seqlens           = tensorHoldHostAndToCuda(cu_kv_seqlens);
    } else {
        py_attn_inputs.total_tokens = 0;
        py_attn_inputs.cu_seqlens =
            torch::zeros({batch_size + 1}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
        py_attn_inputs.cu_kv_seqlens =
            torch::zeros({batch_size + 1}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
        torch::Tensor decode_cu_seqlens = torch::arange(
            0, py_attn_inputs.sequence_lengths.size(0) + 1, 1, torch::TensorOptions(torch::kInt32).device(torch::kCPU));
        py_attn_inputs.decode_cu_seqlens_host = decode_cu_seqlens;
        py_attn_inputs.decode_cu_seqlens_d    = tensorHoldHostAndToCuda(decode_cu_seqlens);
    }

    // create device tensors
    py_attn_inputs.prefix_lengths_d          = tensorHoldHostAndToCuda(py_attn_inputs.prefix_lengths);
    py_attn_inputs.sequence_lengths_plus_1_d = tensorHoldHostAndToCuda(py_attn_inputs.sequence_lengths + 1);
    py_attn_inputs.input_lengths_d           = tensorHoldHostAndToCuda(py_attn_inputs.input_lengths);
    return py_attn_inputs;
}

void PyWrappedModel::handleContextParallelInputs(GptModelInputs& model_input, PyContextParallelParams& cp_params) {
    DevicePerfWrapper wrapper(device_, "py model handleContextParallelInputs");
    // CP reuses TP communication group: cp_rank = tp_rank, cp_size = tp_size
    int prefill_cp_rank = device_props_.tp_rank;
    int prefill_cp_size = device_props_.tp_size;
    int cp_align_size   = prefill_cp_size * 2;

    auto& total_input_tokens = model_input.combo_tokens;
    auto& input_lengths      = model_input.input_lengths;     // prefill + decode
    auto& sequence_lengths   = model_input.sequence_lengths;  // decode
    // auto& prefix_lengths = model_input.prefix_lengths; TODO
    auto input_lengths_cpu_tensor = Buffer2torchTensor(input_lengths, true);

    size_t num_decode_stream  = sequence_lengths->shape()[0];
    size_t num_prefill_stream = input_lengths->shape()[0] - num_decode_stream;

    auto prefill_cp_padding_lengths = CACHED_HOST_BUF(TYPE_INT32, {num_prefill_stream});
    auto prefill_cp_chunk_lengths   = CACHED_HOST_BUF(TYPE_INT32, {num_prefill_stream});
    int* padding_lengths            = (int*)prefill_cp_padding_lengths->data();
    int* chunk_lengths              = (int*)prefill_cp_chunk_lengths->data();

    size_t prefill_cp_split_tokens_size = 0;
    for (int p = 0; p < num_prefill_stream; ++p) {
        int num_prefill_token = input_lengths->data<int>()[num_decode_stream + p];

        int padded_seq_len = ((num_prefill_token + cp_align_size - 1) / cp_align_size) * cp_align_size;
        int padding_size   = padded_seq_len - num_prefill_token;
        int chunk_size     = padded_seq_len / prefill_cp_size;

        prefill_cp_split_tokens_size += chunk_size;
        padding_lengths[p] = padding_size;
        chunk_lengths[p]   = chunk_size;
    }

    auto cp_split_input_tokens   = CACHED_HOST_BUF(TYPE_INT32, {num_decode_stream + prefill_cp_split_tokens_size});
    auto prefill_shuffle_indices = CACHED_HOST_BUF(TYPE_INT32, {prefill_cp_split_tokens_size});

    int* input_token_ptr             = (int*)cp_split_input_tokens->data();
    int* input_length_ptr            = (int*)input_lengths->data();
    int* prefill_shuffle_indices_ptr = (int*)prefill_shuffle_indices->data();

    int input_token_idx       = 0;
    int total_input_token_idx = 0;

    // directly memcpy decode stream input tokens
    if (num_decode_stream > 0) {
        std::memcpy(input_token_ptr,
                    total_input_tokens->dataWithOffset<int>(total_input_token_idx),
                    num_decode_stream * sizeof(int));
        input_token_idx += num_decode_stream;
        total_input_token_idx += num_decode_stream;
    }

    // handle prefill stream
    for (int p = 0; p < num_prefill_stream; ++p) {
        int input_chunk_length   = prefill_cp_chunk_lengths->data<int>()[p];
        int input_padding_length = prefill_cp_padding_lengths->data<int>()[p];
        int input_length         = input_lengths->data<int>()[num_decode_stream + p];
        // Copy input tokens for this prefill stream
        int*             src_tokens = total_input_tokens->dataWithOffset<int>(total_input_token_idx);
        std::vector<int> total_input_token_vec(src_tokens, src_tokens + input_length);
        std::vector<int> chunk_input_token(input_chunk_length, 0);
        std::vector<int> shuffle_index(input_chunk_length, -1);

        RTP_LLM_CHECK_WITH_INFO(context_parallel_planner_ != nullptr, "Context parallel planner is not initialized");
        bool success = context_parallel_planner_->plan(total_input_token_vec,
                                                       chunk_input_token,
                                                       shuffle_index,
                                                       prefill_cp_rank,
                                                       prefill_cp_size,
                                                       input_chunk_length,
                                                       input_padding_length);
        RTP_LLM_CHECK_WITH_INFO(success, "Context parallel planning failed for prefill stream %d", p);

        std::memcpy(input_token_ptr + input_token_idx, chunk_input_token.data(), input_chunk_length * sizeof(int));
        std::memcpy(
            prefill_shuffle_indices_ptr + input_token_idx, shuffle_index.data(), input_chunk_length * sizeof(int));
        input_token_idx += input_chunk_length;
        total_input_token_idx += input_length;
        input_length_ptr[num_decode_stream + p] = input_chunk_length;
    }
    model_input.combo_tokens = std::move(cp_split_input_tokens);
    auto cp_padding_lengths  = Buffer2torchTensor(prefill_cp_padding_lengths);
    auto cp_chunk_lengths    = Buffer2torchTensor(prefill_cp_chunk_lengths);
    auto shuffle_indices     = Buffer2torchTensor(prefill_shuffle_indices);

    RTP_LLM_CHECK_WITH_INFO(context_parallel_planner_ != nullptr, "Context parallel planner is not initialized");
    auto qkv_restore_indice = context_parallel_planner_->generateQKVRestoreIndices(cp_chunk_lengths, prefill_cp_size);
    auto qkv_padding_mask =
        context_parallel_planner_->generateQKVPaddingMask(cp_chunk_lengths, cp_padding_lengths, prefill_cp_size);

    cp_params.prefill_cp_padding_lengths       = cp_padding_lengths.cuda();
    cp_params.prefill_cp_chunk_lengths         = cp_chunk_lengths.cuda();
    cp_params.prefill_shuffle_indices          = shuffle_indices.cuda();
    cp_params.prefill_qkv_restore_indice       = qkv_restore_indice.cuda();
    cp_params.prefill_qkv_padding_mask         = qkv_padding_mask.cuda();
    cp_params.prefill_actual_input_lengths_cpu = input_lengths_cpu_tensor;
}

size_t PyWrappedModel::handleContextParallelOutputs(BufferPtr&                     hidden_states,
                                                    const GptModelInputs&          inputs,
                                                    const PyContextParallelParams& cp_params) {
    DevicePerfWrapper wrapper(device_, "py model handleContextParallelOutputs");
    int               prefill_cp_size = device_->getDeviceProperties().tp_size;

    // AllGather hidden states from all CP ranks
    BufferPtr all_hidden_states = device_->allocateBuffer(
        {hidden_states->type(), {hidden_states->shape()[0] * prefill_cp_size, hidden_states->shape()[1]}},
        {"allgather_hidden_states"});
    device_->allGather({{all_hidden_states}, ParallelMode::TP, {hidden_states}, false});

    auto    all_hidden_states_tensor = Buffer2torchTensor(all_hidden_states, false);
    int64_t num_valid_tokens         = all_hidden_states_tensor.size(0);

    // do restore and unpadding
    if (context_parallel_planner_->getPlannerType() == PlannerType::ZIG_ZAG) {
        auto          prefill_qkv_restore_indice = cp_params.prefill_qkv_restore_indice;
        auto          prefill_qkv_padding_mask   = cp_params.prefill_qkv_padding_mask;
        torch::Tensor valid_indices              = torch::nonzero(prefill_qkv_padding_mask).squeeze(-1);
        num_valid_tokens                         = valid_indices.size(0);
        torch::Tensor combined_indices           = prefill_qkv_restore_indice.index_select(0, valid_indices);
        torch::Tensor valid_hidden_states        = all_hidden_states_tensor.index_select(0, combined_indices);
        hidden_states                            = torchTensor2Buffer(valid_hidden_states);
    } else {
        hidden_states = all_hidden_states;
    }
    return num_valid_tokens;
}

// Helper function to setup KV cache for attention inputs
void PyWrappedModel::setupKVCacheForAttentionInputs(torch_ext::PyAttentionInputs& py_attn_inputs,
                                                    const GptModelInputs&         inputs,
                                                    BufferPtr&                    kv_cache_block_id_device) {
    if (kv_cache_buffer_) {
        DevicePerfWrapper wrapper(device_, "py model setupKVCacheForAttentionInputs");
        kv_cache_block_id_device =
            device_->clone({*inputs.kv_cache_block_id, AllocationType::DEVICE, {"kv_cache_block_id"}});
        py_attn_inputs.kv_cache_block_id_device = Buffer2torchTensor(kv_cache_block_id_device, false);
    }
}

// Helper function to build BertEmbeddingInputs from GptModelInputs
torch_ext::BertEmbeddingInputs PyWrappedModel::buildBertEmbeddingInputs(const GptModelInputs& inputs) {
    DevicePerfWrapper              wrapper(device_, "py model buildBertEmbeddingInputs");
    torch_ext::BertEmbeddingInputs bert_embedding_inputs;

    // Convert combo_position_ids from Buffer to torch::Tensor
    if (inputs.combo_position_ids) {
        bert_embedding_inputs.combo_position_ids = Buffer2torchTensor(inputs.combo_position_ids, false).cuda();
    }

    // Convert combo_tokens_type_ids from Buffer to torch::Tensor
    if (inputs.combo_tokens_type_ids) {
        {
            DevicePerfWrapper wrapper(device_, "py model combo_tokens.cuda()");
            bert_embedding_inputs.combo_tokens_type_ids =
                Buffer2torchTensor(inputs.combo_tokens_type_ids, false).cuda();
        }
    }

    // Get position_encoding from model weights (no clone needed for weights)
    if (weights_.position_encoding) {
        DevicePerfWrapper wrapper(device_, "py model weights_.position_encoding->kernel");
        bert_embedding_inputs.position_encoding = Buffer2torchTensor(weights_.position_encoding->kernel, false);
    }

    // Get token_type_embedding from model weights (no clone needed for weights)
    if (weights_.token_type_embedding) {
        DevicePerfWrapper wrapper(device_, "py model weights_.token_type_embedding->kernel");
        bert_embedding_inputs.token_type_embedding = Buffer2torchTensor(weights_.token_type_embedding->kernel, false);
    }

    // Set input_embedding_scalar
    bert_embedding_inputs.input_embedding_scalar = description_.input_embedding_scalar;
    return bert_embedding_inputs;
}

// Helper function to call forwardPostLayers with common parameters
GptModelOutputs PyWrappedModel::callForwardPostLayers(BufferPtr             hidden_states,
                                                      const GptModelInputs& inputs,
                                                      bool                  skip_final_layernorm,
                                                      size_t                num_valid_tokens) {
    size_t num_input_tokens = num_valid_tokens != -1 ? num_valid_tokens : inputs.combo_tokens->shape()[0];
    return forwardPostLayers(hidden_states,
                             inputs.input_lengths->shape()[0] != inputs.sequence_lengths->shape()[0],
                             false,
                             inputs.lm_output_indexes,
                             false,
                             num_input_tokens,
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
                                              Buffer2torchTensor(inputs.request_id, false),
                                              Buffer2torchTensor(inputs.request_pd_separation, false),
                                              transVectorToString(cache_keys_vec),
                                              inputs.seq_size_per_block,
                                              inputs.kv_block_stride_bytes,
                                              inputs.kv_scale_stride_bytes,
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
        py_attn_inputs.padding_offset = tensorHoldHostAndToCuda(py_attn_inputs.padding_offset);

        torch::Tensor token_ids = Buffer2torchTensor(micro_inputs.combo_tokens).cuda();
        torch::Tensor input_hiddens =
            inputs.last_hidden_states ? Buffer2torchTensor(inputs.last_hidden_states, false) : torch::empty({0});
        input_list.emplace_back(PyModelInputs{token_ids, input_hiddens, py_attn_inputs, bert_embedding_inputs});
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
                                     {inputs.combo_tokens->shape()[0],
                                      description_.attention_conf.head_num * description_.attention_conf.size_per_head},
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
    DevicePerfWrapper wrapper(device_, "py model forward");
    holdInputsHostBuffers(inputs);
    py::gil_scoped_acquire gil;
    try {
        RTP_LLM_LOG_DEBUG("Calling forward method on Python object instance.");

        if (int(device_props_.enable_layer_micro_batch)) {
            return forwardMicroBatched(inputs);
        }
        // handle context parallel inputs
        PyContextParallelParams cp_params;
        if (device_props_.enable_prefill_cp) {
            handleContextParallelInputs(const_cast<GptModelInputs&>(inputs), cp_params);
        }

        torch::Tensor token_ids;
        token_ids = tensorHoldHostAndToCuda(Buffer2torchTensor(inputs.combo_tokens, false));

        torch::Tensor input_hiddens =
            inputs.last_hidden_states ? Buffer2torchTensor(inputs.last_hidden_states, false) : torch::empty({0});

        auto attention_inputs = buildPyAttentionInputs(inputs);

        if (device_props_.enable_prefill_cp) {
            attention_inputs.context_parallel_info = cp_params;
        }

        auto      bert_embedding_inputs = buildBertEmbeddingInputs(inputs);
        BufferPtr kv_cache_block_id_device;
        if (!inputs.warmup && inputs.pd_separation) {
            attention_inputs.cache_store_inputs = prepareWriteCacheParams(inputs);
        }

        setupKVCacheForAttentionInputs(attention_inputs, inputs, kv_cache_block_id_device);

        calculatePaddingOffset(attention_inputs);
        attention_inputs.padding_offset = tensorHoldHostAndToCuda(attention_inputs.padding_offset);

        auto py_model_inputs = PyModelInputs({token_ids, input_hiddens, attention_inputs, bert_embedding_inputs});
        PyModelOutputs py_model_outputs;
        BufferPtr      hidden_states;

        // Cast the Python object to PyModelOutputs and extract hidden states
        if (enable_cuda_graph_ && graph_runner_->canRun(py_model_inputs)) {
            DevicePerfWrapper wrapper(device_, "cuda graph python forward");
            py_model_inputs.attention_inputs.is_s_padded = true;
            py_model_outputs                             = graph_runner_->forward(py_model_inputs);
            hidden_states = device_->clone({*torchTensor2Buffer(py_model_outputs.hidden_states)});
        } else {
            DevicePerfWrapper wrapper(device_, "normal forward");
            auto              attn_pyobj = py_model_.attr("prepare_fmha_impl")(py_model_inputs, false);
            // attn_pyobj.attr("prepare")(py_model_inputs.attention_inputs);
            auto py_model_forward = py_model_.attr("forward");
            auto outputs          = py_model_forward(py_model_inputs, attn_pyobj);
            py_model_outputs      = outputs.cast<PyModelOutputs>();
            hidden_states         = device_->clone({*torchTensor2Buffer(py_model_outputs.hidden_states)});
        }

        RTP_LLM_LOG_DEBUG("Python object instance forward method called successfully.");
        if (device_props_.enable_prefill_cp) {
            size_t num_valid_tokens = handleContextParallelOutputs(hidden_states, inputs, cp_params);
            return callForwardPostLayers(hidden_states, inputs, true, num_valid_tokens);
        }
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
