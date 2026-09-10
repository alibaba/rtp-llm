#include "rtp_llm/cpp/normal_engine/speculative/MtpCompute.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include <cstring>
#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif

namespace rtp_llm {
namespace mtp {
namespace {
torch::Tensor toCudaInt32(const torch::Tensor& tensor, TensorHolder& host_holder) {
    if (!tensor.defined()) {
        return tensor;
    }
    if (tensor.is_cuda() && tensor.scalar_type() == torch::kInt32) {
        return tensor;
    }
    if (tensor.numel() == 0) {
        return torch::empty(tensor.sizes(), torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    }
    host_holder.hold_host(tensor);
    return tensor.to(torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA), /*non_blocking=*/true);
}

void applySpecLogitsAcceptLenCap(const SpecLogitsVerifyRunner::LaunchResult& verify_result,
                                torch::Tensor                              target_token_ids,
                                speculative::SpeculativeSamplerOutput&     accepted) {
    auto cap_src = verify_result.spec_cap_gpu;
    if (!cap_src.defined()) {
        cap_src = verify_result.spec_cap_cpu;
    }
    if (!cap_src.defined()) {
        return;
    }
    auto&      accept_tokens = accepted.accept_tokens;
    auto&      accept_len    = accepted.accept_len;
    const auto batch_size    = accept_tokens.size(0);
    const auto propose_step  = accept_tokens.size(1) - 1;
    RTP_LLM_CHECK_WITH_INFO(accept_len.defined() && accept_len.is_cuda(),
                            "spec logits cap requires CUDA accept_len");

    if (verify_result.ready_event) {
        verify_result.ready_event->block(cuda_graph::graphGetCurrentStream());
    }
    if (!cap_src.is_cuda()) {
        cap_src = cap_src.to(accept_len.device());
    }
#if USING_CUDA
    c10::cuda::CUDACachingAllocator::recordStream(cap_src.storage().data_ptr(),
                                                at::cuda::getCurrentCUDAStream(cap_src.device().index()));
#endif
    auto cap_gpu      = cap_src.to(accept_len.options());
    auto cap_plus_one = cap_gpu + 1;
    accept_len = torch::minimum(accept_len, cap_plus_one);

    RTP_LLM_CHECK_WITH_INFO(accept_tokens.defined() && accept_tokens.is_cuda(),
                            "spec logits cap requires CUDA accept_tokens");
    RTP_LLM_CHECK_WITH_INFO(target_token_ids.defined(),
                            "spec logits cap requires target sampler token_ids");
    if (!target_token_ids.is_cuda()) {
        target_token_ids = target_token_ids.to(accept_tokens.device(), /*non_blocking=*/true);
    }
    const int64_t token_stride  = target_token_ids.size(1);
    auto          target_tokens = target_token_ids.reshape({batch_size, propose_step + 1, token_stride})
                             .select(2, token_stride - 1)
                             .to(accept_tokens.options());
    auto cap_index   = cap_src.to(torch::TensorOptions().device(accept_tokens.device()).dtype(torch::kLong));
    auto replacement = target_tokens.gather(1, cap_index.unsqueeze(1));

    auto cols = torch::arange(propose_step + 1,
                              torch::TensorOptions().device(accept_tokens.device()).dtype(torch::kLong))
                    .unsqueeze(0)
                    .expand({batch_size, propose_step + 1});
    auto replace_mask = (cap_gpu < propose_step).unsqueeze(1) & (accept_len > cap_gpu).unsqueeze(1)
                        & (cols == cap_index.unsqueeze(1));
    accept_tokens =
        torch::where(replace_mask, replacement.expand({batch_size, propose_step + 1}), accept_tokens);
}

torch::Tensor compactAcceptedPositionIds(const torch::Tensor&     combo_position_ids,
                                        const std::vector<int>& accept_lens,
                                        size_t                  total_accept_len,
                                        size_t                  position_id_len_factor,
                                        int64_t                 tokens_per_request) {
    if (!combo_position_ids.defined()) {
        return torch::Tensor();
    }

    auto         compact_position_ids =
        torch::empty({(int64_t)(total_accept_len * position_id_len_factor)}, torch::kInt32).pin_memory();
    const int* src_position_ids = combo_position_ids.data_ptr<int>();
    int*       dst_position_ids = compact_position_ids.data_ptr<int>();

    int token_offset = 0;
    for (size_t i = 0; i < accept_lens.size(); ++i) {
        for (int step = 0; step < accept_lens[i]; ++step) {
            memcpy(dst_position_ids + (token_offset + step) * position_id_len_factor,
                   src_position_ids + (i * tokens_per_request + step) * position_id_len_factor,
                   position_id_len_factor * sizeof(int));
        }
        token_offset += accept_lens[i];
    }

    return compact_position_ids;
}

}  // namespace

void prepareDraftInputForPrefill(GptModelInputs&      draft_input,
                                 const torch::Tensor& target_hidden_states,
                                 const torch::Tensor& sampled_token_ids,
                                 const torch::Tensor& next_position_ids,
                                 size_t               position_id_len_factor,
                                 TensorHolder&        host_holder) {
    const auto batch_size           = sampled_token_ids.size(0);
    const auto sampler_token_stride = sampled_token_ids.size(1);
    const auto sampled_tokens_cpu   = sampled_token_ids.cpu().contiguous();
    const auto input_lengths_cpu    = draft_input.input_lengths.cpu().contiguous();
    const auto target_tokens_cpu    = draft_input.combo_tokens.cpu().contiguous();
    const auto host_options         = torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true);
    auto       combo_tokens         = torch::empty(target_tokens_cpu.sizes(), host_options);

    torch::Tensor target_position_ids_cpu;
    torch::Tensor next_position_ids_cpu;
    torch::Tensor combo_position_ids;
    if (draft_input.combo_position_ids.defined()) {
        target_position_ids_cpu = draft_input.combo_position_ids.cpu().contiguous();
        next_position_ids_cpu   = next_position_ids.cpu().contiguous();
        combo_position_ids      = torch::empty(target_position_ids_cpu.sizes(), host_options);
    }

    const auto* input_lengths_ptr  = input_lengths_cpu.data_ptr<int32_t>();
    const auto* target_tokens_ptr  = target_tokens_cpu.data_ptr<int32_t>();
    const auto* sampled_tokens_ptr = sampled_tokens_cpu.data_ptr<int32_t>();
    auto*       combo_tokens_ptr   = combo_tokens.data_ptr<int32_t>();
    int64_t     token_offset       = 0;
    for (int64_t i = 0; i < batch_size; ++i) {
        const auto input_length = input_lengths_ptr[i];
        memcpy(combo_tokens_ptr + token_offset,
               target_tokens_ptr + token_offset + 1,
               (input_length - 1) * sizeof(int32_t));
        combo_tokens_ptr[token_offset + input_length - 1] =
            sampled_tokens_ptr[i * sampler_token_stride + sampler_token_stride - 1];

        if (combo_position_ids.defined()) {
            auto* request_positions = combo_position_ids.data_ptr<int32_t>() + token_offset * position_id_len_factor;
            memcpy(request_positions,
                   target_position_ids_cpu.data_ptr<int32_t>() + (token_offset + 1) * position_id_len_factor,
                   (input_length - 1) * position_id_len_factor * sizeof(int32_t));
            memcpy(request_positions + (input_length - 1) * position_id_len_factor,
                   next_position_ids_cpu.data_ptr<int32_t>() + i * position_id_len_factor,
                   position_id_len_factor * sizeof(int32_t));
        }
        token_offset += input_length;
    }

    draft_input.is_target_verify   = false;
    draft_input.last_hidden_states = target_hidden_states;
    draft_input.input_lengths      = toCudaInt32(draft_input.input_lengths, host_holder);
    draft_input.combo_tokens       = toCudaInt32(combo_tokens, host_holder);
    draft_input.combo_position_ids = std::move(combo_position_ids);
}

void prepareDraftInputForDecode(GptModelInputs&      draft_input,
                               const torch::Tensor& target_hidden_states,
                               const torch::Tensor& accepted_token_ids,
                               const torch::Tensor& accepted_lengths,
                               DraftInputLayout     layout,
                               size_t               position_id_len_factor,
                               TensorHolder&        host_holder) {
    const size_t batch_size         = accepted_lengths.numel();
    const auto   tokens_per_request = accepted_token_ids.size(1);
    draft_input.is_target_verify    = false;
    if (layout == DraftInputLayout::FIXED_WIDTH) {
        const auto total_tokens = accepted_token_ids.numel();
        draft_input.combo_tokens = toCudaInt32(accepted_token_ids.reshape({total_tokens}), host_holder);
        auto accepted_lengths_gpu = toCudaInt32(accepted_lengths, host_holder);
        draft_input.lm_output_indexes =
            torch::arange(0, total_tokens, tokens_per_request, accepted_lengths_gpu.options()) + (accepted_lengths_gpu - 1);
        draft_input.last_hidden_states = target_hidden_states;
        return;
    }

    const auto      accepted_token_ids_cpu = accepted_token_ids.cpu().contiguous();
    const auto      accepted_lengths_cpu   = accepted_lengths.cpu().contiguous();
    const auto*     accepted_lengths_ptr   = accepted_lengths_cpu.data_ptr<int32_t>();
    std::vector<int> accepted_lengths_per_request(batch_size);
    size_t          total_accepted_tokens = 0;
    for (size_t i = 0; i < batch_size; ++i) {
        accepted_lengths_per_request[i] = accepted_lengths_ptr[i];
        total_accepted_tokens += accepted_lengths_per_request[i];
    }

    auto combo_tokens =
        torch::empty({static_cast<int64_t>(total_accepted_tokens)}, torch::TensorOptions(torch::kInt32).pinned_memory(true));
    auto input_lengths =
        torch::empty({static_cast<int64_t>(batch_size)}, torch::TensorOptions(torch::kInt32).pinned_memory(true));
    auto lm_output_indexes =
        torch::empty({static_cast<int64_t>(batch_size)}, torch::TensorOptions(torch::kInt32).pinned_memory(true));
    const auto* accepted_token_ids_ptr = accepted_token_ids_cpu.data_ptr<int32_t>();
    auto*       combo_tokens_ptr       = combo_tokens.data_ptr<int32_t>();
    auto*       input_lengths_ptr      = input_lengths.data_ptr<int32_t>();
    auto*       output_indexes_ptr     = lm_output_indexes.data_ptr<int32_t>();

    size_t                    token_offset = 0;
    std::vector<torch::Tensor> accepted_hidden_states;
    accepted_hidden_states.reserve(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        RTP_LLM_CHECK_WITH_INFO(accepted_lengths_per_request[i] > 0
                                    && accepted_lengths_per_request[i] <= tokens_per_request,
                                "invalid accept_len[%zu]=%d for token width=%ld",
                                i,
                                accepted_lengths_per_request[i],
                                tokens_per_request);
        memcpy(combo_tokens_ptr + token_offset,
               accepted_token_ids_ptr + i * tokens_per_request,
               accepted_lengths_per_request[i] * sizeof(int32_t));
        accepted_hidden_states.push_back(
            target_hidden_states.narrow(0, i * tokens_per_request, accepted_lengths_per_request[i]));
        input_lengths_ptr[i] = accepted_lengths_per_request[i];
        token_offset += accepted_lengths_per_request[i];
        output_indexes_ptr[i] = static_cast<int32_t>(token_offset - 1);
    }

    draft_input.combo_tokens       = std::move(combo_tokens);
    draft_input.input_lengths      = std::move(input_lengths);
    draft_input.lm_output_indexes  = std::move(lm_output_indexes);
    draft_input.last_hidden_states = torch::cat(accepted_hidden_states).contiguous();
    auto verify_position_ids = draft_input.combo_position_ids.defined() && draft_input.combo_position_ids.is_cuda() ?
                                   draft_input.combo_position_ids.cpu().contiguous() :
                                   draft_input.combo_position_ids;
    auto compact_position_ids = compactAcceptedPositionIds(verify_position_ids,
                                                           accepted_lengths_per_request,
                                                           total_accepted_tokens,
                                                           position_id_len_factor,
                                                           tokens_per_request);
    if (compact_position_ids.defined()) {
        draft_input.combo_position_ids = std::move(compact_position_ids);
    }
}

void advanceDraftInput(GptModelInputs&      draft_input,
                       const torch::Tensor& draft_hidden_states,
                       const torch::Tensor& draft_token_ids,
                       size_t               position_id_len_factor,
                       TensorHolder&        host_holder) {
    int batch_size                 = draft_input.combo_tokens.size(0);
    draft_input.last_hidden_states = draft_hidden_states.clone();
    draft_input.combo_tokens       = draft_token_ids.reshape({batch_size});

    if (draft_input.combo_position_ids.defined()) {
        auto         next_position_ids =
            torch::empty({(int64_t)(batch_size * position_id_len_factor)}, torch::kInt32).pin_memory();
        int*       next_position_ids_ptr = next_position_ids.data_ptr<int>();
        const auto position_ids_cpu      = draft_input.combo_position_ids.cpu().contiguous();
        const int* position_ids_ptr      = position_ids_cpu.data_ptr<int>();
        for (int64_t i = 0; i < next_position_ids.numel(); ++i) {
            next_position_ids_ptr[i] = position_ids_ptr[i] + 1;
        }
        draft_input.combo_position_ids = std::move(next_position_ids);
    }

    if (draft_input.sequence_lengths.is_cuda()) {
        draft_input.sequence_lengths = (draft_input.sequence_lengths + 1).to(torch::kInt32);
    } else {
        auto sequence_lengths_cpu = draft_input.sequence_lengths.cpu().clone().pin_memory();
        for (int i = 0; i < batch_size; i++) {
            sequence_lengths_cpu.data_ptr<int>()[i]++;
        }
        draft_input.sequence_lengths = toCudaInt32(sequence_lengths_cpu, host_holder);
    }
}

void runRejectionSampling(speculative::SpeculativeSampler&               sampler,
                          const speculative::SpeculativeSamplingParams& params,
                          SamplerOutput&                                draft_sampler_output,
                          SamplerOutput&                                target_sampler_output,
                          const SpecLogitsVerifyRunner::LaunchResult&    verify_result,
                          speculative::SpeculativeSamplerOutput&        output) {
    output                  = sampler.forward(params, draft_sampler_output, target_sampler_output);
    output.processor_errors = verify_result.processor_errors;
    applySpecLogitsAcceptLenCap(verify_result, target_sampler_output.token_ids, output);
    output.accept_tokens_cpu = output.accept_tokens.to(torch::kCPU, /*non_blocking=*/true);
    output.accept_len_cpu    = output.accept_len.to(torch::kCPU, /*non_blocking=*/true);
    output.transfer_done_event->record(cuda_graph::graphGetCurrentStream());
    if (verify_result.consumed_event) {
        verify_result.consumed_event->record(cuda_graph::graphGetCurrentStream());
    }
}

}  // namespace mtp
}  // namespace rtp_llm
