#include "rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.h"

#include <algorithm>
#include <limits>

#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {

namespace {

void fillAllAllowBitmask(const torch::Tensor& tensor) {
    if (tensor.defined() && tensor.numel() > 0) {
        std::fill_n(tensor.data_ptr<int32_t>(), tensor.numel(), SpecLogitsProcessorRequest::kBitmaskAllowAll);
    }
}

void bitwiseAndBitmaskInplace(int32_t* dst, const int32_t* src, size_t words) {
    for (size_t i = 0; i < words; ++i) {
        dst[i] &= src[i];
    }
}

bool has1DCapacity(const torch::Tensor& tensor, int64_t size) {
    return tensor.defined() && tensor.dim() == 1 && tensor.size(0) >= size;
}

bool has2DCapacity(const torch::Tensor& tensor, int64_t rows, int64_t cols) {
    return tensor.defined() && tensor.dim() == 2 && tensor.size(0) >= rows && tensor.size(1) == cols;
}

}  // namespace

void SpecLogitsVerifyRunner::applyMaskToLogits(torch::Tensor&       logits,
                                               const torch::Tensor& packed_allow_mask_gpu,
                                               const torch::Tensor& logits_row_indices_gpu,
                                               size_t               vocab_size) {
    if (!packed_allow_mask_gpu.defined()) {
        return;
    }
    runtimeApplyPackedMaskLogits(logits, packed_allow_mask_gpu, logits_row_indices_gpu, vocab_size);
}

SpecLogitsVerifyRunner::ActiveStreamLayout
SpecLogitsVerifyRunner::buildActiveStreamLayout(const LaunchTask& task) const {
    ActiveStreamLayout layout;
    layout.compact_slot_by_stream.assign(task.total_streams, -1);
    layout.stream_indices.reserve(task.active.size());

    for (const auto& item : task.active) {
        RTP_LLM_CHECK_WITH_INFO(item.processor != nullptr, "MTP spec logits verify active processor is null");
        RTP_LLM_CHECK_WITH_INFO(item.stream_idx < task.total_streams,
                                "MTP spec logits verify stream_idx=%zu out of range, total_streams=%zu",
                                item.stream_idx,
                                task.total_streams);
        if (layout.compact_slot_by_stream[item.stream_idx] >= 0) {
            continue;
        }
        layout.compact_slot_by_stream[item.stream_idx] = static_cast<int32_t>(layout.stream_indices.size());
        layout.stream_indices.push_back(item.stream_idx);
    }
    return layout;
}

void SpecLogitsVerifyRunner::ensureBuffersFit(const VerifyShape& shape) {
    const int64_t B    = static_cast<int64_t>(shape.batch_size);
    const int64_t P    = static_cast<int64_t>(shape.propose_step);
    const int64_t rows = static_cast<int64_t>(shape.compact_rows);
    const int64_t W    = static_cast<int64_t>(shape.bitmask_words);

    auto cpu_i32    = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto pinned_i32 = cpu_i32.pinned_memory(true);
    auto device_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

    if (!has2DCapacity(draft_tokens_cpu_, B, P)) {
        draft_tokens_cpu_ = torch::empty({B, P}, pinned_i32);
    }
    if (!has2DCapacity(processor_bitmask_cpu_, P + 1, W)) {
        processor_bitmask_cpu_ = torch::empty({P + 1, W}, cpu_i32);
    }
    if (!has2DCapacity(merged_bitmask_cpu_, rows, W)) {
        merged_bitmask_cpu_ = torch::empty({rows, W}, pinned_i32);
    }
    if (!has2DCapacity(merged_bitmask_gpu_, rows, W)) {
        merged_bitmask_gpu_ = torch::empty({rows, W}, device_i32);
    }
    if (!has1DCapacity(logits_row_indices_cpu_, rows)) {
        logits_row_indices_cpu_ = torch::empty({rows}, pinned_i32);
    }
    if (!has1DCapacity(logits_row_indices_gpu_, rows)) {
        logits_row_indices_gpu_ = torch::empty({rows}, device_i32);
    }
    if (!has1DCapacity(spec_cap_cpu_, B)) {
        spec_cap_cpu_ = torch::empty({B}, pinned_i32);
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
    RTP_LLM_CHECK_WITH_INFO(draft_tokens.numel() % B == 0, "MTP spec logits verify draft token shape mismatch");
    const int64_t draft_cols   = draft_tokens.numel() / B;
    const int64_t draft_offset = draft_cols == P + 1 ? 1 : 0;
    RTP_LLM_CHECK_WITH_INFO(draft_cols == P || draft_cols == P + 1,
                            "MTP spec logits verify draft token columns must be propose_step (%lld) "
                            "or propose_step+1 (%lld), got %lld",
                            static_cast<long long>(P),
                            static_cast<long long>(P + 1),
                            static_cast<long long>(draft_cols));
    auto draft     = draft_tokens.reshape({B, draft_cols}).narrow(1, draft_offset, P);
    auto dst       = draft_tokens_cpu_.narrow(0, 0, B).narrow(1, 0, P);
    auto draft_i32 = draft.scalar_type() == torch::kInt32 ? draft.contiguous() : draft.to(torch::kInt32).contiguous();
    dst.copy_(draft_i32);
}

void SpecLogitsVerifyRunner::initializeCompactRows(const ActiveStreamLayout& layout, const VerifyShape& shape) {
    auto compact_masks = merged_bitmask_cpu_.narrow(0, 0, static_cast<int64_t>(shape.compact_rows))
                             .narrow(1, 0, static_cast<int64_t>(shape.bitmask_words));
    fillAllAllowBitmask(compact_masks);

    auto* row_indices = logits_row_indices_cpu_.data_ptr<int32_t>();
    for (size_t compact_stream = 0; compact_stream < layout.stream_indices.size(); ++compact_stream) {
        const size_t stream_idx = layout.stream_indices[compact_stream];
        for (int offset = 0; offset <= shape.propose_step; ++offset) {
            const size_t compact_row =
                compact_stream * static_cast<size_t>(shape.propose_step + 1) + static_cast<size_t>(offset);
            row_indices[compact_row] =
                static_cast<int32_t>(stream_idx * static_cast<size_t>(shape.propose_step + 1) + offset);
        }
    }
}

SpecLogitsVerifyRunner::MergeProcessorMasksResult SpecLogitsVerifyRunner::mergeProcessorMasks(
    const LaunchTask& task, const ActiveStreamLayout& layout, const VerifyShape& shape) {
    auto proc_mask = processor_bitmask_cpu_.narrow(0, 0, shape.propose_step + 1)
                         .narrow(1, 0, static_cast<int64_t>(shape.bitmask_words));
    auto*                                 merged_base = merged_bitmask_cpu_.data_ptr<int32_t>();
    auto*                                 cap_ptr     = spec_cap_cpu_.data_ptr<int32_t>();
    std::vector<std::optional<ErrorInfo>> processor_errors(shape.batch_size);

    for (const auto& item : task.active) {
        fillAllAllowBitmask(proc_mask);
        SpecLogitsProcessorRequest request;
        request.draft_tokens       = draft_tokens_cpu_.data_ptr<int32_t>() + item.stream_idx * shape.propose_step;
        request.propose_step       = shape.propose_step;
        request.bitmask_cpu_out    = proc_mask.data_ptr<int32_t>();
        request.bitmask_size_int32 = shape.bitmask_words;
        request.vocab_size         = shape.vocab_size;

        int  cap    = 0;
        auto cap_or = item.processor->prepareSpeculative(request);
        if (!cap_or.ok()) {
            if (!processor_errors[item.stream_idx].has_value()) {
                processor_errors[item.stream_idx] = cap_or.status();
            }
        } else {
            cap = std::max(0, std::min(cap_or.value(), shape.propose_step));
        }

        const int32_t compact_slot = layout.compact_slot_by_stream[item.stream_idx];
        RTP_LLM_CHECK_WITH_INFO(compact_slot >= 0, "MTP spec logits verify compact stream mapping is missing");
        auto* merged_row = merged_base + static_cast<size_t>(compact_slot) * shape.words_per_stream;
        bitwiseAndBitmaskInplace(merged_row, proc_mask.data_ptr<int32_t>(), shape.words_per_stream);
        cap_ptr[item.stream_idx] = std::min<int32_t>(cap_ptr[item.stream_idx], cap);
    }
    return {std::move(processor_errors)};
}

SpecLogitsVerifyRunner::LaunchResult SpecLogitsVerifyRunner::makeResult(const VerifyShape& shape) {
    auto packed_mask_cpu = merged_bitmask_cpu_.narrow(0, 0, static_cast<int64_t>(shape.compact_rows))
                               .narrow(1, 0, static_cast<int64_t>(shape.bitmask_words));
    auto packed_mask_gpu = merged_bitmask_gpu_.narrow(0, 0, static_cast<int64_t>(shape.compact_rows))
                               .narrow(1, 0, static_cast<int64_t>(shape.bitmask_words));
    auto row_indices_cpu = logits_row_indices_cpu_.narrow(0, 0, static_cast<int64_t>(shape.compact_rows));
    auto row_indices_gpu = logits_row_indices_gpu_.narrow(0, 0, static_cast<int64_t>(shape.compact_rows));
    packed_mask_gpu.copy_(packed_mask_cpu, /*non_blocking=*/true);
    row_indices_gpu.copy_(row_indices_cpu, /*non_blocking=*/true);

    LaunchResult result;
    result.packed_allow_mask_gpu           = std::move(packed_mask_gpu);
    result.logits_row_indices_gpu          = std::move(row_indices_gpu);
    result.has_active_processor            = true;
    result.packed_allow_mask_cpu_lifetime  = std::move(packed_mask_cpu);
    result.logits_row_indices_cpu_lifetime = std::move(row_indices_cpu);
    result.spec_cap_cpu                    = spec_cap_cpu_.narrow(0, 0, static_cast<int64_t>(shape.batch_size));
    return result;
}

SpecLogitsVerifyRunner::LaunchResult SpecLogitsVerifyRunner::run(const LaunchTask& task) {
    RTP_LLM_PROFILE_SCOPE("spec_logits_verify_runner.run");

    if (task.active.empty()) {
        return LaunchResult{};
    }

    const size_t B = task.total_streams;
    const int    P = task.propose_step;
    const size_t V = task.vocab_size;
    RTP_LLM_CHECK_WITH_INFO(B > 0 && P > 0 && V > 0, "invalid MTP spec logits verify task");
    RTP_LLM_CHECK_WITH_INFO(P < std::numeric_limits<int32_t>::max(),
                            "MTP spec logits verify propose_step exceeds int32 row-stride capacity");
    RTP_LLM_CHECK_WITH_INFO(V <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                            "MTP spec logits verify vocab_size exceeds kernel int32 capacity");
    const size_t rows_per_stream = static_cast<size_t>(P) + 1;
    RTP_LLM_CHECK_WITH_INFO(B <= static_cast<size_t>(std::numeric_limits<int32_t>::max()) / rows_per_stream,
                            "MTP spec logits verify row count exceeds int32 row-index capacity");
    auto         layout = buildActiveStreamLayout(task);
    const size_t W      = SpecLogitsProcessorRequest::bitmaskWordCount(V);
    VerifyShape  shape{
        B,
        P,
        V,
        W,
        layout.stream_indices.size() * rows_per_stream,
        rows_per_stream * W,
    };

    ensureBuffersFit(shape);
    std::fill_n(spec_cap_cpu_.data_ptr<int32_t>(), B, P);
    materializeDraftTokensToCpu(task);
    initializeCompactRows(layout, shape);
    auto merge_result       = mergeProcessorMasks(task, layout, shape);
    auto result             = makeResult(shape);
    result.processor_errors = std::move(merge_result.processor_errors);
    return result;
}

}  // namespace rtp_llm
