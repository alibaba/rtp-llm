#include "rtp_llm/cpp/normal_engine/speculative/MtpTargetVerifyLayout.h"
#include "rtp_llm/cpp/normal_engine/speculative/MtpCompute.h"
#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#if USING_CUDA
#include "rtp_llm/models_py/bindings/cuda/kernels/mtp_target_verify_prepare.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif

namespace rtp_llm {
namespace mtp_input {
torch::TensorOptions cudaInt32Options() {
    return torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
}

torch::Tensor emptyInt32OnCuda(std::initializer_list<int64_t> shape) {
    return torch::empty(shape, cudaInt32Options());
}

torch::Tensor fullInt32OnCuda(std::initializer_list<int64_t> shape, int64_t value) {
    return torch::full(shape, value, cudaInt32Options());
}

torch::Tensor toCudaInt32(const torch::Tensor& tensor, TensorHolder& host_holder) {
    if (!tensor.defined()) {
        return tensor;
    }
    if (tensor.is_cuda() && tensor.scalar_type() == torch::kInt32) {
        return tensor;
    }
    if (tensor.numel() == 0) {
        return torch::empty(tensor.sizes(), cudaInt32Options());
    }
    host_holder.hold_host(tensor);
    return tensor.to(cudaInt32Options(), /*non_blocking=*/true);
}

void setVerifyPairInputs(GptModelInputs& model_input,
                         torch::Tensor   combo_tokens,
                         size_t          batch_size,
                         size_t          score_len,
                         TensorHolder&   host_holder) {
    model_input.combo_tokens       = std::move(combo_tokens);
    model_input.sequence_lengths   = emptyInt32OnCuda({0});
    model_input.last_hidden_states = torch::Tensor();
    model_input.prefix_lengths     = toCudaInt32(model_input.prefix_lengths, host_holder).contiguous();
    model_input.input_lengths      = fullInt32OnCuda({static_cast<int64_t>(batch_size)}, score_len);
    model_input.lm_output_indexes  = torch::arange(static_cast<int64_t>(batch_size * score_len), cudaInt32Options());
}

}  // namespace mtp_input
using namespace mtp_input;

void prepareDSparkTargetVerifyModelInput(int                     propose_step,
                                         const DSparkRoundState& round_state,
                                         GptModelInputs&         model_input,
                                         const torch::Tensor&    proposals,
                                         TensorHolder&           host_holder,
                                         torch::Tensor*          owned_proposals) {
    if (!round_state.anchors.defined() || round_state.anchors.numel() == 0) {
        return;
    }
    const int64_t batch_size = round_state.anchors.numel();

    RTP_LLM_CHECK_WITH_INFO(proposals.defined() && proposals.dim() == 2 && proposals.size(0) == batch_size
                                && proposals.size(1) == propose_step,
                            "dspark verify: round-head proposals must be [%ld, %d]",
                            batch_size,
                            propose_step);

#if USING_CUDA
    auto fused = tryMtpDSparkVerifyPrepare(round_state.anchors,
                                           proposals,
                                           round_state.engram_windows,
                                           round_state.position_bases,
                                           at::cuda::getCurrentCUDAStream().stream(),
                                           owned_proposals != nullptr);
    if (fused.tokens.defined()) {
        model_input.combo_tokens      = std::move(fused.tokens);
        model_input.input_lengths     = std::move(fused.input_lengths);
        model_input.lm_output_indexes = std::move(fused.lm_indexes);
        if (round_state.engram_windows.defined()) {
            model_input.engram_token_windows = std::move(fused.history);
        }
        model_input.combo_position_ids = std::move(fused.positions);
        model_input.sequence_lengths   = emptyInt32OnCuda({0});
        model_input.last_hidden_states = torch::Tensor();
        model_input.prefix_lengths     = toCudaInt32(round_state.committed_ends, host_holder).contiguous();
        model_input.is_target_verify   = true;
        if (owned_proposals) {
            *owned_proposals = std::move(fused.proposals);
        }
        return;
    }
#endif
    auto anchor_col = round_state.anchors.reshape({batch_size, 1});
    auto verify     = torch::cat({anchor_col, proposals.to(torch::kInt32)}, 1).reshape({-1});
    if (round_state.engram_windows.defined()) {
        model_input.engram_token_windows = mtp::makeEngramVerifyWindows(
            toCudaInt32(round_state.engram_windows, host_holder), verify.reshape({batch_size, propose_step + 1}));
    }
    model_input.prefix_lengths = round_state.committed_ends;
    setVerifyPairInputs(model_input, std::move(verify), batch_size, propose_step + 1, host_holder);
    model_input.combo_position_ids = expandDSparkPositionIds(round_state.position_bases, propose_step + 1);
    model_input.is_target_verify   = true;
    if (owned_proposals) {
        *owned_proposals = proposals.clone();
    }
}

DSparkPackedTargetLayout prepareDSparkPackedTargetVerifyModelInput(int                     propose_step,
                                                                   const DSparkRoundState& round_state,
                                                                   GptModelInputs&         model_input,
                                                                   const torch::Tensor&    proposals,
                                                                   const torch::Tensor&    valid_draft_lengths,
                                                                   int64_t                 total_draft_budget,
                                                                   TensorHolder&           host_holder,
                                                                   torch::Tensor*          owned_proposals,
                                                                   bool                    force_packed_layout) {
    RTP_LLM_CHECK_WITH_INFO(round_state.anchors.defined() && round_state.anchors.numel() > 0,
                            "dspark packed verify: a nonempty round is required");
    const int64_t batch_size   = round_state.anchors.numel();
    const int64_t verify_width = propose_step + 1;
    RTP_LLM_CHECK_WITH_INFO(valid_draft_lengths.defined() && valid_draft_lengths.is_cuda()
                                && valid_draft_lengths.scalar_type() == torch::kInt32 && valid_draft_lengths.dim() == 1
                                && valid_draft_lengths.size(0) == batch_size && valid_draft_lengths.is_contiguous(),
                            "dspark packed verify: valid_draft_lengths must be contiguous CUDA int32 [batch]");
    RTP_LLM_CHECK_WITH_INFO(proposals.defined() && proposals.is_cuda()
                                && proposals.device() == valid_draft_lengths.device()
                                && round_state.anchors.device() == proposals.device(),
                            "dspark packed verify: lengths, proposals and anchors must share a CUDA device");
    RTP_LLM_CHECK_WITH_INFO(total_draft_budget >= 0 && total_draft_budget <= batch_size * propose_step,
                            "dspark packed verify: draft budget %ld is outside [0, %ld]",
                            total_draft_budget,
                            batch_size * propose_step);

    prepareDSparkTargetVerifyModelInput(
        propose_step, round_state, model_input, proposals, host_holder, owned_proposals);
    DSparkPackedTargetLayout result;
    result.dense_tokens       = model_input.combo_tokens.reshape({batch_size, verify_width});
    result.dense_position_ids = model_input.combo_position_ids;
    if (total_draft_budget == batch_size * propose_step && !force_packed_layout) {
        return result;
    }

    // All query lengths include the carried anchor. With a host-known budget,
    // searchsorted has a fixed-size output and never synchronizes to discover
    // the number of selected rows. The unique round-head planner owns the
    // value/sum invariant; bounds keep a broken caller from indexing outside
    // the dense source buffers without introducing a host validation copy.
    auto query_lengths = valid_draft_lengths + 1;
    auto query_ends    = query_lengths.cumsum(0, torch::kInt64);
    auto query_starts  = query_ends - query_lengths;
    auto packed_rows   = torch::arange(batch_size + total_draft_budget, proposals.options().dtype(torch::kInt64));
    auto owners =
        at::searchsorted(query_ends, packed_rows, /*out_int32=*/false, /*right=*/true).clamp(0, batch_size - 1);
    auto positions         = (packed_rows - query_starts.index_select(0, owners)).clamp(0, propose_step);
    result.packed_to_dense = owners * verify_width + positions;
    result.dense_to_packed = torch::full({batch_size * verify_width}, -1, proposals.options().dtype(torch::kInt32));
    auto packed_indexes    = packed_rows.to(torch::kInt32);
    result.dense_to_packed.index_copy_(0, result.packed_to_dense, packed_indexes);

    model_input.combo_tokens = model_input.combo_tokens.index_select(0, result.packed_to_dense);
    if (result.dense_position_ids.defined() && result.dense_position_ids.numel() > 0) {
        model_input.combo_position_ids = result.dense_position_ids.reshape({batch_size * verify_width, -1})
                                             .index_select(0, result.packed_to_dense)
                                             .reshape({-1});
    }
    if (round_state.engram_windows.defined()) {
        model_input.engram_token_windows = model_input.engram_token_windows.index_select(0, result.packed_to_dense);
    }
    model_input.input_lengths           = std::move(query_lengths);
    model_input.sequence_lengths_plus_1 = model_input.prefix_lengths + 1;
    model_input.lm_output_indexes       = packed_indexes;
    return result;
}

DSparkPackedTargetLayout prepareDSparkBoundedTargetVerifyModelInput(int                         propose_step,
                                                                    const DSparkRoundState&     round_state,
                                                                    GptModelInputs&             model_input,
                                                                    const torch::Tensor&        proposals,
                                                                    const torch::Tensor&        confidence_rates,
                                                                    const std::vector<int64_t>& stable_order,
                                                                    int64_t                     total_draft_budget,
                                                                    TensorHolder&               host_holder,
                                                                    torch::Tensor&              valid_draft_lengths,
                                                                    torch::Tensor*              owned_proposals,
                                                                    bool                        force_packed_layout) {
    RTP_LLM_PROFILE_SCOPE("mtp.prepare_dspark_bounded_target");
    RTP_LLM_CHECK_WITH_INFO(round_state.anchors.defined() && round_state.anchors.numel() > 0,
                            "dspark bounded verify: a nonempty round is required");
    const int64_t batch_size = round_state.anchors.numel();
    RTP_LLM_CHECK_WITH_INFO(confidence_rates.defined() && confidence_rates.is_cuda()
                                && confidence_rates.scalar_type() == torch::kFloat32 && confidence_rates.is_contiguous()
                                && confidence_rates.dim() == 2 && confidence_rates.size(0) == batch_size
                                && confidence_rates.size(1) == static_cast<int64_t>(propose_step),
                            "dspark bounded verify: confidence rates must be CUDA FP32 [batch, gamma]");
    RTP_LLM_CHECK_WITH_INFO(total_draft_budget >= 0 && total_draft_budget <= confidence_rates.numel(),
                            "dspark bounded verify: draft budget is outside [0, batch * gamma]");
    RTP_LLM_CHECK_WITH_INFO(stable_order.size() == static_cast<size_t>(batch_size),
                            "dspark bounded verify: stable identity order must contain every row");
    std::vector<bool> seen(static_cast<size_t>(batch_size), false);
    for (const int64_t row : stable_order) {
        RTP_LLM_CHECK_WITH_INFO(row >= 0 && row < batch_size && !seen[row],
                                "dspark bounded verify: stable identity order must be a row permutation");
        seen[row] = true;
    }

#if USING_CUDA
    // The target consumes contiguous prefixes. Strided inputs retain the
    // existing conversion path instead of introducing another copy launch.
    if ((total_draft_budget < confidence_rates.numel() || force_packed_layout) && round_state.committed_ends.defined()
        && round_state.committed_ends.is_contiguous()) {
        auto fused = tryMtpDSparkBudgetedVerifyPrepare(confidence_rates,
                                                       stable_order,
                                                       round_state.anchors,
                                                       proposals,
                                                       round_state.engram_windows,
                                                       round_state.position_bases,
                                                       round_state.committed_ends,
                                                       total_draft_budget,
                                                       at::cuda::getCurrentCUDAStream().stream(),
                                                       owned_proposals != nullptr,
                                                       force_packed_layout);
        if (fused.packed.tokens.defined()) {
            DSparkPackedTargetLayout result;
            result.packed_to_dense              = std::move(fused.packed_to_dense);
            result.dense_to_packed              = std::move(fused.dense_to_packed);
            result.dense_tokens                 = std::move(fused.dense_tokens);
            result.dense_position_ids           = std::move(fused.dense_positions);
            valid_draft_lengths                 = std::move(fused.valid_draft_lengths);
            model_input.combo_tokens            = std::move(fused.packed.tokens);
            model_input.input_lengths           = std::move(fused.packed.input_lengths);
            model_input.lm_output_indexes       = std::move(fused.packed.lm_indexes);
            model_input.combo_position_ids      = std::move(fused.packed.positions);
            model_input.sequence_lengths        = emptyInt32OnCuda({0});
            model_input.prefix_lengths          = round_state.committed_ends;
            model_input.sequence_lengths_plus_1 = std::move(fused.sequence_lengths_plus_1);
            model_input.last_hidden_states      = torch::Tensor();
            model_input.is_target_verify        = true;
            if (round_state.engram_windows.defined()) {
                model_input.engram_token_windows = std::move(fused.packed.history);
            }
            if (owned_proposals) {
                *owned_proposals = std::move(fused.packed.proposals);
            }
            return result;
        }
    }
#endif

    auto order_cpu = torch::tensor(stable_order, torch::kInt64).pin_memory();
    host_holder.hold_host(order_cpu);
    auto order = order_cpu.to(confidence_rates.device(), /*non_blocking=*/true);
    valid_draft_lengths =
        speculative::SpeculativeSampler::buildValidDraftLengths(confidence_rates, order, total_draft_budget);
    return prepareDSparkPackedTargetVerifyModelInput(propose_step,
                                                     round_state,
                                                     model_input,
                                                     proposals,
                                                     valid_draft_lengths,
                                                     total_draft_budget,
                                                     host_holder,
                                                     owned_proposals,
                                                     force_packed_layout);
}

}  // namespace rtp_llm
