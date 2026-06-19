#include "rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.h"

#include <algorithm>

#include "rtp_llm/cpp/models/logits_processor/BitmaskUtils.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorException.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/Logger.h"
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
    const int64_t V    = static_cast<int64_t>(vocab_size);
    const int64_t W    = static_cast<int64_t>(bitmask_words);

    auto cpu_i32     = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto pinned_i32  = cpu_i32.pinned_memory(true);
    auto pinned_bool = torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU).pinned_memory(true);

    if (!draft_tokens_cpu_.defined() || draft_tokens_cpu_.numel() < B * P) {
        draft_tokens_cpu_ = torch::empty({B, P}, pinned_i32);
    }
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
    auto cuda_i32  = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto cuda_bool = torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA);
    if (!spec_cap_gpu_.defined() || spec_cap_gpu_.numel() < B) {
        spec_cap_gpu_ = torch::empty({B}, cuda_i32);
    }
    if (!disallow_mask_cpu_.defined() || disallow_mask_cpu_.size(0) < rows || disallow_mask_cpu_.size(1) < V) {
        disallow_mask_cpu_ = torch::zeros({rows, V}, pinned_bool);
        last_active_stream_rows_.clear();
    }
    if (!disallow_mask_gpu_.defined() || disallow_mask_gpu_.size(0) < rows || disallow_mask_gpu_.size(1) < V) {
        disallow_mask_gpu_ = torch::zeros({rows, V}, cuda_bool);
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

void SpecLogitsVerifyRunner::unpackRowToBoolDisallow(size_t row, size_t vocab_size, size_t bitmask_words) {
    const auto* bits = merged_bitmask_cpu_.data_ptr<int32_t>() + row * bitmask_words;
    auto*       out  = disallow_mask_cpu_.data_ptr<bool>() + row * disallow_mask_cpu_.size(1);
    for (size_t token = 0; token < vocab_size; ++token) {
        out[token] = !bitmaskAllowsToken(bits, bitmask_words, static_cast<int32_t>(token));
    }
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
    bool draft_tokens_materialized = false;

    auto* merged_base        = merged_bitmask_cpu_.data_ptr<int32_t>();
    const size_t row_words   = static_cast<size_t>(P + 1) * W;
    const size_t buffer_rows = static_cast<size_t>(merged_bitmask_cpu_.size(0)) / static_cast<size_t>(P + 1);

    // Reset prev-call active rows (packed allow-all). Bool unpack happens after AND merge below.
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

        int cap = 0;
        try {
            cap = std::max(0, std::min(item.processor->tryAcceptAndFillBitmask(request), P));
        } catch (const LogitsProcessorException& e) {
            RTP_LLM_LOG_WARNING("spec verify: stream_id=%llu processor_idx=%zu reported %s",
                                static_cast<unsigned long long>(item.stream_id),
                                item.processor_idx,
                                e.what());
            if (item.error_sink) {
                item.error_sink(e.code(), e.what());
            }
            cap = 0;
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING("spec verify: stream_id=%llu processor_idx=%zu unexpected exception: %s",
                                static_cast<unsigned long long>(item.stream_id),
                                item.processor_idx,
                                e.what());
            if (item.error_sink) {
                item.error_sink(ErrorCode::EXECUTION_EXCEPTION,
                                std::string("spec verify exception: ") + e.what());
            }
            cap = 0;
        } catch (...) {
            RTP_LLM_LOG_WARNING("spec verify: stream_id=%llu processor_idx=%zu unknown exception",
                                static_cast<unsigned long long>(item.stream_id),
                                item.processor_idx);
            if (item.error_sink) {
                item.error_sink(ErrorCode::EXECUTION_EXCEPTION, "spec verify unknown exception");
            }
            cap = 0;
        }

        auto* merged_row = merged_base + item.stream_idx * row_words;
        bitwiseAndBitmaskInplace(merged_row, proc_mask.data_ptr<int32_t>(), row_words);
        auto* cap_ptr            = spec_cap_cpu_.data_ptr<int32_t>();
        cap_ptr[item.stream_idx] = std::min<int32_t>(cap_ptr[item.stream_idx], cap);
        result.applied_processors.push_back({item.stream_id, item.processor_idx});
        this_active_rows.push_back(item.stream_idx);
    }

    auto upload_row_bool = [&](size_t stream_row) {
        const size_t row_begin = stream_row * static_cast<size_t>(P + 1);
        for (size_t r = row_begin; r < row_begin + static_cast<size_t>(P + 1); ++r) {
            unpackRowToBoolDisallow(r, V, W);
        }
        auto cpu_slice = disallow_mask_cpu_.narrow(0, row_begin, P + 1).narrow(1, 0, V);
        auto gpu_slice = disallow_mask_gpu_.narrow(0, row_begin, P + 1).narrow(1, 0, V);
        gpu_slice.copy_(cpu_slice, /*non_blocking=*/true);
    };

    if (!applied_processor) {
        // Sync prev-call rows back to GPU allow-all (false) before bailing.
        last_active_stream_rows_.clear();
        for (size_t row : rows_to_reset) {
            upload_row_bool(row);
        }
        return {};
    }

    auto cap_cpu  = spec_cap_cpu_.narrow(0, 0, static_cast<int64_t>(B));
    auto mask_gpu = disallow_mask_gpu_.narrow(0, 0, static_cast<int64_t>(rows)).narrow(1, 0, static_cast<int64_t>(V));
    auto cap_gpu  = spec_cap_gpu_.narrow(0, 0, static_cast<int64_t>(B));

    for (size_t row : rows_to_reset) {
        upload_row_bool(row);
    }
    for (size_t row : this_active_rows) {
        upload_row_bool(row);
    }
    cap_gpu.copy_(cap_cpu, /*non_blocking=*/true);

    last_active_stream_rows_ = std::move(this_active_rows);

    result.spec_vocab_mask_gpu       = mask_gpu;
    result.spec_cap_gpu              = cap_gpu;
    result.has_active_processor      = true;
    result.spec_vocab_mask_cpu_owner = disallow_mask_cpu_.narrow(0, 0, static_cast<int64_t>(rows))
                                            .narrow(1, 0, static_cast<int64_t>(V));
    result.spec_cap_cpu_owner = cap_cpu;
    return result;
}

}  // namespace rtp_llm
