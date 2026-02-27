#include "rtp_llm/cpp/models/context_parallel/ContextParallelProcessorBase.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace rtp_llm {

void IContextParallelProcessor::handleInputs(DeviceBase*                         device,
                                             GptModelInputs&                     model_input,
                                             torch_ext::PyContextParallelParams& cp_params) {
    auto device_props    = device->getDeviceProperties();
    int  prefill_cp_rank = device_props.tp_rank;
    int  prefill_cp_size = device_props.tp_size;
    int  cp_align_size   = prefill_cp_size * 2;

    auto& total_input_tokens       = model_input.combo_tokens;
    auto& input_lengths            = model_input.input_lengths;
    auto& sequence_lengths         = model_input.sequence_lengths;
    auto  input_lengths_cpu_tensor = Buffer2torchTensor(input_lengths, true);

    size_t num_decode_stream  = sequence_lengths->shape()[0];
    size_t num_prefill_stream = input_lengths->shape()[0] - num_decode_stream;

    // CACHED_HOST_BUF macro captures `device_` by name from the enclosing scope.
    auto* device_                    = device;
    auto  prefill_cp_padding_lengths = CACHED_HOST_BUF(TYPE_INT32, {num_prefill_stream});
    auto  prefill_cp_chunk_lengths   = CACHED_HOST_BUF(TYPE_INT32, {num_prefill_stream});
    int*  padding_lengths            = (int*)prefill_cp_padding_lengths->data();
    int*  chunk_lengths              = (int*)prefill_cp_chunk_lengths->data();

    size_t prefill_cp_split_tokens_size = 0;
    for (size_t p = 0; p < num_prefill_stream; ++p) {
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

    if (num_decode_stream > 0) {
        std::memcpy(input_token_ptr,
                    total_input_tokens->dataWithOffset<int>(total_input_token_idx),
                    num_decode_stream * sizeof(int));
        input_token_idx += num_decode_stream;
        total_input_token_idx += num_decode_stream;
    }

    for (size_t p = 0; p < num_prefill_stream; ++p) {
        int input_chunk_length   = prefill_cp_chunk_lengths->data<int>()[p];
        int input_padding_length = prefill_cp_padding_lengths->data<int>()[p];
        int input_length         = input_lengths->data<int>()[num_decode_stream + p];

        int*             src_tokens = total_input_tokens->dataWithOffset<int>(total_input_token_idx);
        std::vector<int> total_input_token_vec(src_tokens, src_tokens + input_length);
        std::vector<int> chunk_input_token(input_chunk_length, 0);
        std::vector<int> shuffle_index(input_chunk_length, -1);

        bool success = plan(total_input_token_vec,
                            chunk_input_token,
                            shuffle_index,
                            prefill_cp_rank,
                            prefill_cp_size,
                            input_chunk_length,
                            input_padding_length);
        RTP_LLM_CHECK_WITH_INFO(success, "Context parallel planning failed for prefill stream %zu", p);

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

    auto qkv_restore_indice = generateQKVRestoreIndices(cp_chunk_lengths, prefill_cp_size);
    auto qkv_padding_mask   = generateQKVPaddingMask(cp_chunk_lengths, cp_padding_lengths, prefill_cp_size);

    cp_params.prefill_cp_padding_lengths       = cp_padding_lengths.cuda();
    cp_params.prefill_cp_chunk_lengths         = cp_chunk_lengths.cuda();
    cp_params.prefill_shuffle_indices          = shuffle_indices.cuda();
    cp_params.prefill_qkv_restore_indice       = qkv_restore_indice.cuda();
    cp_params.prefill_qkv_padding_mask         = qkv_padding_mask.cuda();
    cp_params.prefill_actual_input_lengths_cpu = input_lengths_cpu_tensor;
}

size_t IContextParallelProcessor::handleOutputs(DeviceBase*                               device,
                                                BufferPtr&                                hidden_states,
                                                const GptModelInputs&                     inputs,
                                                const torch_ext::PyContextParallelParams& cp_params) {
    int prefill_cp_size = device->getDeviceProperties().tp_size;

    BufferPtr all_hidden_states = device->allocateBuffer(
        {hidden_states->type(), {hidden_states->shape()[0] * prefill_cp_size, hidden_states->shape()[1]}},
        {"allgather_hidden_states"});
    device->allGather({{all_hidden_states}, ParallelMode::TP, {hidden_states}, false});

    auto    all_hidden_states_tensor = Buffer2torchTensor(all_hidden_states, false);
    int64_t num_valid_tokens         = all_hidden_states_tensor.size(0);

    hidden_states = all_hidden_states;
    return num_valid_tokens;
}

}  // namespace rtp_llm
