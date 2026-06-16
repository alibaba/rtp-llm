#include "rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.h"

#include <algorithm>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {

namespace {

void fillAllAllowBitmask(const torch::Tensor& tensor) {
    if (tensor.defined() && tensor.numel() > 0) {
        std::fill_n(tensor.data_ptr<int32_t>(), tensor.numel(), SpecLogitsProcessor::kBitmaskAllowAll);
    }
}

void bitwiseAndBitmaskInplace(int32_t* dst, const int32_t* src, size_t words) {
    for (size_t i = 0; i < words; ++i) {
        dst[i] &= src[i];
    }
}

}  // namespace

void SpecLogitsVerifyRunner::ensureBuffersFit(size_t total_streams,
                                              int    propose_step,
                                              size_t vocab_size,
                                              size_t bitmask_words) {
    const int64_t B    = static_cast<int64_t>(total_streams);
    const int64_t P    = static_cast<int64_t>(propose_step);
    const int64_t rows = B * (P + 1);
    const int64_t W    = static_cast<int64_t>(bitmask_words);
    (void)vocab_size;

    auto cpu_i32    = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto pinned_i32 = cpu_i32.pinned_memory(true);

    if (!draft_tokens_cpu_.defined() || draft_tokens_cpu_.numel() < B * P) {
        draft_tokens_cpu_ = torch::empty({B, P}, pinned_i32);
    }
    // Pinned: merged_bitmask_cpu_ is the H2D source; pageable source strips non_blocking.
    if (!processor_bitmask_cpu_.defined() || processor_bitmask_cpu_.numel() < (P + 1) * W) {
        processor_bitmask_cpu_ = torch::empty({P + 1, W}, pinned_i32);
    }
    if (!merged_bitmask_cpu_.defined() || merged_bitmask_cpu_.numel() < rows * W) {
        merged_bitmask_cpu_ = torch::empty({rows, W}, pinned_i32);
        std::fill_n(merged_bitmask_cpu_.data_ptr<int32_t>(),
                    merged_bitmask_cpu_.numel(),
                    SpecLogitsProcessor::kBitmaskAllowAll);
        last_active_stream_rows_.clear();
    }
    if (!spec_cap_cpu_.defined() || spec_cap_cpu_.numel() < B) {
        spec_cap_cpu_ = torch::empty({B}, pinned_i32);
    }
    // torch::kCUDA aliases the HIP device on ROCm builds, so the same code
    // path allocates correctly on both platforms; only the apply-side differs
    // (CUDA kernel vs ROCm torch-op fallback).
    auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    if (!spec_cap_gpu_.defined() || spec_cap_gpu_.numel() < B) {
        spec_cap_gpu_ = torch::empty({B}, cuda_i32);
    }
    if (!merged_bitmask_gpu_.defined() || merged_bitmask_gpu_.numel() < rows * W) {
        merged_bitmask_gpu_ = torch::empty({rows, W}, cuda_i32);
        // Mirror the CPU-realloc branch above: post-realloc invariant is
        // (buffer == allow-all) ∧ (tracking == ∅). When CPU/GPU sizes diverge
        // after a prior CUDA OOM, the CPU buffer can hold dirty rows; clearing
        // tracking without re-filling would strand them on the new GPU.
        std::fill_n(merged_bitmask_cpu_.data_ptr<int32_t>(),
                    merged_bitmask_cpu_.numel(),
                    SpecLogitsProcessor::kBitmaskAllowAll);
        merged_bitmask_gpu_.copy_(merged_bitmask_cpu_.narrow(0, 0, rows).narrow(1, 0, W));
        last_active_stream_rows_.clear();
    }
}

void SpecLogitsVerifyRunner::materializeDraftTokensToCpu(const LaunchTask& task) {
    const int64_t B = static_cast<int64_t>(task.total_streams);
    const int64_t P = static_cast<int64_t>(task.propose_step);
    if (B == 0 || P == 0) {
        return;
    }

    const auto& draft_tokens = task.draft_tokens;
    RTP_LLM_CHECK_WITH_INFO(draft_tokens.defined(), "MTP spec logits verify requires draft tokens");
    RTP_LLM_CHECK_WITH_INFO(draft_tokens.numel() >= B * P && draft_tokens.numel() % B == 0,
                            "MTP spec logits verify draft token shape mismatch");
    const int64_t draft_cols   = draft_tokens.numel() / B;
    const int64_t draft_offset = draft_cols > P ? 1 : 0;
    RTP_LLM_CHECK_WITH_INFO(draft_cols >= draft_offset + P, "MTP spec logits verify draft token columns mismatch");
    auto draft     = draft_tokens.reshape({B, draft_cols}).narrow(1, draft_offset, P);
    auto dst       = draft_tokens_cpu_.narrow(0, 0, B).narrow(1, 0, P);
    auto draft_i32 = draft.scalar_type() == torch::kInt32 ? draft.contiguous() : draft.to(torch::kInt32).contiguous();
    dst.copy_(draft_i32);
}

SpecLogitsVerifyRunner::LaunchResult SpecLogitsVerifyRunner::buildInline(const LaunchTask& task) {
    RTP_LLM_PROFILE_SCOPE("spec_logits_verify_runner.build_inline");
    LaunchResult result;

    if (task.active.empty()) {
        return result;
    }

    const size_t B    = task.total_streams;
    const int    P    = task.propose_step;
    const size_t V    = task.vocab_size;
    const size_t W    = SpecLogitsProcessor::bitmaskWordCount(V);
    const size_t rows = B * static_cast<size_t>(P + 1);
    RTP_LLM_CHECK_WITH_INFO(B > 0 && P > 0 && V > 0, "invalid MTP spec logits verify task");

    ensureBuffersFit(B, P, V, W);
    // Defer the O(B*P) draft-token D2H until at least one processor is eligible.
    bool draft_tokens_materialized = false;

    // Sparse-row: keep untouched rows allow-all, reset prev-call rows + fill this-call rows.
    auto  merged = merged_bitmask_cpu_.narrow(0, 0, static_cast<int64_t>(rows)).narrow(1, 0, static_cast<int64_t>(W));
    auto* merged_base        = merged_bitmask_cpu_.data_ptr<int32_t>();
    const size_t row_words   = (P + 1) * W;
    const size_t buffer_rows = static_cast<size_t>(merged_bitmask_cpu_.size(0)) / static_cast<size_t>(P + 1);

    // Reset prev-call active rows; rows past B keep their allow-all state.
    std::vector<size_t> rows_to_reset;
    rows_to_reset.reserve(last_active_stream_rows_.size());
    for (size_t prev : last_active_stream_rows_) {
        if (prev < buffer_rows) {
            std::fill_n(merged_base + prev * row_words, row_words, SpecLogitsProcessor::kBitmaskAllowAll);
            rows_to_reset.push_back(prev);
        }
    }
    std::fill_n(spec_cap_cpu_.data_ptr<int32_t>(), B, P);

    auto                proc_mask = processor_bitmask_cpu_.narrow(0, 0, P + 1).narrow(1, 0, static_cast<int64_t>(W));
    std::vector<size_t> this_active_rows;
    this_active_rows.reserve(task.active.size());
    bool applied_processor = false;
    for (const auto& item : task.active) {
        if (!item.processor || !item.processor->isSpecVerifyEligible()) {
            continue;
        }
        applied_processor = true;
        if (!draft_tokens_materialized) {
            materializeDraftTokensToCpu(task);
            draft_tokens_materialized = true;
        }

        fillAllAllowBitmask(proc_mask);
        SpecLogitsProcessorRequest request;
        request.draft_tokens       = draft_tokens_cpu_.data_ptr<int32_t>() + item.stream_idx * P;
        request.propose_step       = P;
        request.bitmask_cpu_out    = proc_mask.data_ptr<int32_t>();
        request.bitmask_size_int32 = W;
        request.vocab_size         = V;

        int cap = item.processor->tryAcceptAndFillBitmask(request);
        cap     = std::max(0, std::min(cap, P));

        auto* merged_row = merged_base + item.stream_idx * row_words;
        bitwiseAndBitmaskInplace(merged_row, proc_mask.data_ptr<int32_t>(), row_words);
        auto* cap_ptr            = spec_cap_cpu_.data_ptr<int32_t>();
        cap_ptr[item.stream_idx] = std::min<int32_t>(cap_ptr[item.stream_idx], cap);
        result.applied_processors.push_back({item.stream_id, item.processor_idx});
        this_active_rows.push_back(item.stream_idx);
    }

    if (!applied_processor) {
        // Sync prev-call rows back to GPU allow-all before bailing.
        last_active_stream_rows_.clear();
        if (!rows_to_reset.empty()) {
            for (size_t row : rows_to_reset) {
                auto cpu_slice = merged_bitmask_cpu_.narrow(0, row * (P + 1), P + 1).narrow(1, 0, W);
                auto gpu_slice = merged_bitmask_gpu_.narrow(0, row * (P + 1), P + 1).narrow(1, 0, W);
                gpu_slice.copy_(cpu_slice, /*non_blocking=*/true);
            }
        }
        return {};
    }

    auto cap_cpu  = spec_cap_cpu_.narrow(0, 0, static_cast<int64_t>(B));
    auto mask_gpu = merged_bitmask_gpu_.narrow(0, 0, static_cast<int64_t>(rows)).narrow(1, 0, static_cast<int64_t>(W));
    auto cap_gpu  = spec_cap_gpu_.narrow(0, 0, static_cast<int64_t>(B));

    // Upload only changed rows; H2D stays O(active streams), not O(B*(P+1)).
    // torch::kCUDA aliases the HIP device on ROCm so this works on both
    // platforms; consumers (CUDA kernel vs ROCm torch-op fallback) decide how
    // to apply the packed bitmask.
    auto upload_row = [&](size_t row) {
        auto cpu_slice = merged_bitmask_cpu_.narrow(0, row * (P + 1), P + 1).narrow(1, 0, W);
        auto gpu_slice = merged_bitmask_gpu_.narrow(0, row * (P + 1), P + 1).narrow(1, 0, W);
        gpu_slice.copy_(cpu_slice, /*non_blocking=*/true);
    };
    for (size_t row : rows_to_reset) {
        upload_row(row);
    }
    for (size_t row : this_active_rows) {
        upload_row(row);
    }
    cap_gpu.copy_(cap_cpu, /*non_blocking=*/true);

    last_active_stream_rows_ = std::move(this_active_rows);

    result.spec_vocab_mask_gpu       = mask_gpu;
    result.spec_cap_gpu              = cap_gpu;
    result.has_active_processor      = true;
    result.spec_vocab_mask_cpu_owner = merged;
    result.spec_cap_cpu_owner        = cap_cpu;
    return result;
}

}  // namespace rtp_llm
