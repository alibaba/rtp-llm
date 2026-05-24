#include "rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.h"

#include <algorithm>

#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {

namespace {

void fillAllAllow(const torch::Tensor& tensor) {
    if (tensor.defined() && tensor.numel() > 0) {
        std::fill_n(tensor.data_ptr<int32_t>(), tensor.numel(), SpecLogitsProcessor::kBitmaskAllowAll);
    }
}

void bitwiseAndInplace(int32_t* dst, const int32_t* src, size_t words) {
    for (size_t i = 0; i < words; ++i) {
        dst[i] &= src[i];
    }
}

}  // namespace

SpecLogitsVerifyRunner::SpecLogitsVerifyRunner(): copy_stream_(cuda_graph::graphGetStreamFromPool(true)) {}

void SpecLogitsVerifyRunner::ensureBuffersFit(size_t total_streams,
                                              int    propose_step,
                                              size_t vocab_size,
                                              size_t bitmask_words) {
    const int64_t B    = static_cast<int64_t>(total_streams);
    const int64_t P    = static_cast<int64_t>(propose_step);
    const int64_t rows = B * (P + 1);
    const int64_t W    = static_cast<int64_t>(bitmask_words);
    const int64_t V    = static_cast<int64_t>(vocab_size);

    auto cpu_i32     = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto pinned_i32  = cpu_i32.pinned_memory(true);
    auto pinned_bool = torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU).pinned_memory(true);
    auto cuda_i32    = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto cuda_bool   = torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA);

    if (!draft_tokens_cpu_.defined() || draft_tokens_cpu_.numel() < B * P) {
        draft_tokens_cpu_ = torch::empty({B, P}, pinned_i32);
    }
    if (!processor_bitmask_cpu_.defined() || processor_bitmask_cpu_.numel() < (P + 1) * W) {
        processor_bitmask_cpu_ = torch::empty({P + 1, W}, cpu_i32);
    }
    if (!merged_bitmask_cpu_.defined() || merged_bitmask_cpu_.numel() < rows * W) {
        merged_bitmask_cpu_ = torch::empty({rows, W}, cpu_i32);
    }
    if (!spec_vocab_mask_cpu_.defined() || spec_vocab_mask_cpu_.numel() < rows * V) {
        spec_vocab_mask_cpu_ = torch::empty({rows, V}, pinned_bool);
    }
    if (!spec_vocab_mask_gpu_.defined() || spec_vocab_mask_gpu_.numel() < rows * V) {
        spec_vocab_mask_gpu_ = torch::empty({rows, V}, cuda_bool);
    }
    if (!spec_cap_cpu_.defined() || spec_cap_cpu_.numel() < B) {
        spec_cap_cpu_ = torch::empty({B}, pinned_i32);
    }
    if (!spec_cap_gpu_.defined() || spec_cap_gpu_.numel() < B) {
        spec_cap_gpu_ = torch::empty({B}, cuda_i32);
    }
}

void SpecLogitsVerifyRunner::materializeDraftTokensToCpu(const LaunchTask& task) {
    const int64_t B = static_cast<int64_t>(task.total_streams);
    const int64_t P = static_cast<int64_t>(task.propose_step);
    if (B == 0 || P == 0) {
        return;
    }

    RTP_LLM_CHECK_WITH_INFO(task.draft_tokens.defined(), "spec logits runner requires draft tokens");
    auto draft = task.draft_tokens.reshape({B, P});
    auto dst   = draft_tokens_cpu_.narrow(0, 0, B).narrow(1, 0, P);
    if (!draft.is_cuda()) {
        auto draft_i32 =
            draft.scalar_type() == torch::kInt32 ? draft.contiguous() : draft.to(torch::kInt32).contiguous();
        dst.copy_(draft_i32);
        return;
    }

    cuda_graph::GraphStreamGuard stream_guard(cuda_graph::toGraphStream(copy_stream_));
    if (task.draft_tokens_ready_event) {
        task.draft_tokens_ready_event->block(copy_stream_);
    }
    auto draft_i32 = draft.scalar_type() == torch::kInt32 ? draft.contiguous() : draft.to(torch::kInt32).contiguous();
    dst.copy_(draft_i32, /*non_blocking=*/true);
    copy_stream_.synchronize();
}

void SpecLogitsVerifyRunner::unpackMergedBitmaskToVocabMask(size_t rows, size_t vocab_size, size_t bitmask_words) {
    const auto* merged = merged_bitmask_cpu_.data_ptr<int32_t>();
    auto*       mask   = spec_vocab_mask_cpu_.data_ptr<bool>();
    for (size_t row = 0; row < rows; ++row) {
        const auto* row_bits = merged + row * bitmask_words;
        auto*       row_mask = mask + row * vocab_size;
        for (size_t token = 0; token < vocab_size; ++token) {
            const uint32_t word    = static_cast<uint32_t>(row_bits[token / 32]);
            const bool     allowed = (word & (1u << (token % 32))) != 0u;
            row_mask[token]        = !allowed;
        }
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
    RTP_LLM_CHECK_WITH_INFO(B > 0 && P > 0 && V > 0, "invalid spec logits runner task");

    ensureBuffersFit(B, P, V, W);
    materializeDraftTokensToCpu(task);

    auto merged = merged_bitmask_cpu_.narrow(0, 0, static_cast<int64_t>(rows)).narrow(1, 0, static_cast<int64_t>(W));
    fillAllAllow(merged);
    std::fill_n(spec_cap_cpu_.data_ptr<int32_t>(), B, P);

    auto proc_mask = processor_bitmask_cpu_.narrow(0, 0, P + 1).narrow(1, 0, static_cast<int64_t>(W));
    for (const auto& item : task.active) {
        if (!item.processor || !item.processor->isSpecVerifyEligible()) {
            continue;
        }

        fillAllAllow(proc_mask);
        SpecLogitsProcessorRequest request;
        request.draft_tokens       = draft_tokens_cpu_.data_ptr<int32_t>() + item.stream_idx * P;
        request.propose_step       = P;
        request.bitmask_cpu_out    = proc_mask.data_ptr<int32_t>();
        request.bitmask_size_int32 = W;
        request.vocab_size         = V;
        request.stream_id          = item.stream_id;
        request.base_seq_len       = item.base_seq_len;
        request.base_output_len    = item.base_output_len;

        int cap = item.processor->tryAcceptAndFillBitmask(request);
        cap     = std::max(0, std::min(cap, P));

        auto* merged_row = merged_bitmask_cpu_.data_ptr<int32_t>() + item.stream_idx * (P + 1) * W;
        bitwiseAndInplace(merged_row, proc_mask.data_ptr<int32_t>(), static_cast<size_t>(P + 1) * W);
        auto* cap_ptr            = spec_cap_cpu_.data_ptr<int32_t>();
        cap_ptr[item.stream_idx] = std::min<int32_t>(cap_ptr[item.stream_idx], cap);
    }

    unpackMergedBitmaskToVocabMask(rows, V, W);

    cuda_graph::GraphStreamGuard stream_guard(cuda_graph::toGraphStream(copy_stream_));
    auto mask_cpu = spec_vocab_mask_cpu_.narrow(0, 0, static_cast<int64_t>(rows)).narrow(1, 0, static_cast<int64_t>(V));
    auto mask_gpu = spec_vocab_mask_gpu_.narrow(0, 0, static_cast<int64_t>(rows)).narrow(1, 0, static_cast<int64_t>(V));
    auto cap_cpu  = spec_cap_cpu_.narrow(0, 0, static_cast<int64_t>(B));
    auto cap_gpu  = spec_cap_gpu_.narrow(0, 0, static_cast<int64_t>(B));
    mask_gpu.copy_(mask_cpu, /*non_blocking=*/true);
    cap_gpu.copy_(cap_cpu, /*non_blocking=*/true);
    auto ready = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
    ready->record(copy_stream_);

    result.spec_vocab_mask_gpu  = mask_gpu;
    result.spec_cap_gpu         = cap_gpu;
    result.ready_event          = ready;
    result.has_active_processor = true;
    return result;
}

}  // namespace rtp_llm
