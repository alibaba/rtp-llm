#include "rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.h"

#include <algorithm>
#include <limits>

#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include "rtp_llm/models_py/bindings/cuda/ops/StandaloneOps.h"
#endif

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

bool has1DCapacity(const torch::Tensor& tensor, int64_t size) {
    return tensor.defined() && tensor.dim() == 1 && tensor.size(0) >= size;
}

bool has2DCapacity(const torch::Tensor& tensor, int64_t rows, int64_t cols) {
    return tensor.defined() && tensor.dim() == 2 && tensor.size(0) >= rows && tensor.size(1) == cols;
}

}  // namespace

SpecLogitsVerifyRunner::SpecLogitsVerifyRunner(): copy_stream_(cuda_graph::graphGetStreamFromPool(true)) {}

void SpecLogitsVerifyRunner::ensureBuffersFit(size_t total_streams,
                                              int    propose_step,
                                              size_t bitmask_words,
                                              size_t compact_rows) {
    const int64_t B    = static_cast<int64_t>(total_streams);
    const int64_t P    = static_cast<int64_t>(propose_step);
    const int64_t rows = static_cast<int64_t>(compact_rows);
    const int64_t W    = static_cast<int64_t>(bitmask_words);
    auto cpu_i32       = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto pinned_i32    = cpu_i32.pinned_memory(true);

    if (!has2DCapacity(draft_tokens_cpu_, B, P)) {
        draft_tokens_cpu_ = torch::empty({B, P}, pinned_i32);
    }
    if (!has2DCapacity(processor_bitmask_cpu_, P + 1, W)) {
        processor_bitmask_cpu_ = torch::empty({P + 1, W}, cpu_i32);
    }
    if (!has2DCapacity(merged_bitmask_cpu_, rows, W)) {
        merged_bitmask_cpu_ = torch::empty({rows, W}, pinned_i32);
    }
    if (!has1DCapacity(logits_row_indices_cpu_, rows)) {
        logits_row_indices_cpu_ = torch::empty({rows}, pinned_i32);
    }
    if (!has1DCapacity(spec_cap_cpu_, B)) {
        spec_cap_cpu_ = torch::empty({B}, pinned_i32);
    }
#if USING_CUDA
    auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    if (!has2DCapacity(merged_bitmask_gpu_, rows, W)) {
        merged_bitmask_gpu_ = torch::empty({rows, W}, cuda_i32);
    }
    if (!has1DCapacity(logits_row_indices_gpu_, rows)) {
        logits_row_indices_gpu_ = torch::empty({rows}, cuda_i32);
    }
    if (!has1DCapacity(spec_cap_gpu_, B)) {
        spec_cap_gpu_ = torch::empty({B}, cuda_i32);
    }
#endif
}

void SpecLogitsVerifyRunner::materializeDraftTokensToCpu(const LaunchTask& task) {
    const int64_t B = static_cast<int64_t>(task.total_streams);
    const int64_t P = static_cast<int64_t>(task.propose_step);
    if (B == 0 || P == 0) {
        return;
    }

    RTP_LLM_CHECK_WITH_INFO(task.draft_tokens.defined(), "spec logits runner requires draft tokens");
    RTP_LLM_CHECK_WITH_INFO(task.draft_tokens.numel() % B == 0, "spec logits runner draft token shape mismatch");
    const int64_t draft_cols   = task.draft_tokens.numel() / B;
    const int64_t draft_offset = draft_cols == P + 1 ? 1 : 0;
    RTP_LLM_CHECK_WITH_INFO(draft_cols == P || draft_cols == P + 1,
                            "spec logits runner draft token columns must be P or P+1");
    auto draft = task.draft_tokens.reshape({B, draft_cols}).narrow(1, draft_offset, P);
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

void SpecLogitsVerifyRunner::applyMaskToLogits(const torch::Tensor& logits,
                                               const LaunchResult& result,
                                               size_t              vocab_size) {
#if USING_CUDA
    if (result.packed_allow_mask_gpu.defined() && result.packed_allow_mask_gpu.is_cuda()) {
        cudaApplyPackedMaskLogits(logits,
                                  result.packed_allow_mask_gpu,
                                  result.logits_row_indices_gpu,
                                  vocab_size,
                                  at::cuda::getCurrentCUDAStream(logits.device().index()).stream());
        return;
    }
#endif
    const auto& packed_cpu = result.packed_allow_mask_cpu_lifetime.defined() ?
                                 result.packed_allow_mask_cpu_lifetime :
                                 result.packed_allow_mask_gpu;
    const auto& rows_cpu = result.logits_row_indices_cpu_lifetime.defined() ?
                               result.logits_row_indices_cpu_lifetime :
                               result.logits_row_indices_gpu;
    if (!packed_cpu.defined()) {
        return;
    }
    auto mask = packed_cpu;
    auto rows = rows_cpu;
    RTP_LLM_CHECK_WITH_INFO(!logits.is_cuda() && !mask.is_cuda() && !rows.is_cuda(),
                            "packed mask CPU fallback requires CPU tensors");
    for (int64_t compact_row = 0; compact_row < mask.size(0); ++compact_row) {
        const int32_t logits_row = rows[compact_row].item<int32_t>();
        auto          dense_mask = torch::empty({static_cast<int64_t>(vocab_size)}, torch::kBool);
        auto*         dense_ptr  = dense_mask.data_ptr<bool>();
        const auto*   bits       = mask[compact_row].data_ptr<int32_t>();
        for (size_t token = 0; token < vocab_size; ++token) {
            dense_ptr[token] = (static_cast<uint32_t>(bits[token / 32]) & (1u << (token % 32))) == 0u;
        }
        logits[logits_row].narrow(0, 0, static_cast<int64_t>(vocab_size)).masked_fill_(
            dense_mask, -std::numeric_limits<float>::max());
    }
}

SpecLogitsVerifyRunner::LaunchResult SpecLogitsVerifyRunner::buildInline(const LaunchTask& task) {
    RTP_LLM_PROFILE_SCOPE("spec_logits_verify_runner.build_inline");
    LaunchResult result;
    if (task.active.empty()) {
        return result;
    }
    if (last_consumed_event_) {
        last_consumed_event_->synchronize();
        last_consumed_event_.reset();
    }

    const size_t B = task.total_streams;
    const int    P = task.propose_step;
    const size_t V = task.vocab_size;
    RTP_LLM_CHECK_WITH_INFO(B > 0 && P > 0 && V > 0, "invalid spec logits runner task");
    RTP_LLM_CHECK_WITH_INFO(P < std::numeric_limits<int32_t>::max(), "spec logits propose step exceeds int32");
    RTP_LLM_CHECK_WITH_INFO(V <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                            "spec logits vocab exceeds int32");

    std::vector<int32_t> compact_slot_by_stream(B, -1);
    std::vector<size_t>  active_stream_indices;
    active_stream_indices.reserve(task.active.size());
    for (const auto& item : task.active) {
        RTP_LLM_CHECK_WITH_INFO(item.processor != nullptr, "spec logits active processor is null");
        RTP_LLM_CHECK_WITH_INFO(item.stream_idx < B, "spec logits stream index out of range");
        if (compact_slot_by_stream[item.stream_idx] < 0) {
            compact_slot_by_stream[item.stream_idx] = static_cast<int32_t>(active_stream_indices.size());
            active_stream_indices.push_back(item.stream_idx);
        }
    }

    const size_t W               = SpecLogitsProcessor::bitmaskWordCount(V);
    const size_t rows_per_stream = static_cast<size_t>(P + 1);
    const size_t compact_rows    = active_stream_indices.size() * rows_per_stream;
    ensureBuffersFit(B, P, W, compact_rows);
    materializeDraftTokensToCpu(task);

    auto merged = merged_bitmask_cpu_.narrow(0, 0, static_cast<int64_t>(compact_rows))
                      .narrow(1, 0, static_cast<int64_t>(W));
    fillAllAllow(merged);
    std::fill_n(spec_cap_cpu_.data_ptr<int32_t>(), B, P);

    auto* row_indices = logits_row_indices_cpu_.data_ptr<int32_t>();
    for (size_t compact_stream = 0; compact_stream < active_stream_indices.size(); ++compact_stream) {
        const size_t stream_idx = active_stream_indices[compact_stream];
        for (int offset = 0; offset <= P; ++offset) {
            const size_t compact_row = compact_stream * rows_per_stream + static_cast<size_t>(offset);
            row_indices[compact_row] = static_cast<int32_t>(stream_idx * rows_per_stream + offset);
        }
    }

    auto proc_mask = processor_bitmask_cpu_.narrow(0, 0, P + 1).narrow(1, 0, static_cast<int64_t>(W));
    for (const auto& item : task.active) {
        if (!item.processor->isSpecVerifyEligible()) {
            return {};
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
        const size_t compact_slot = static_cast<size_t>(compact_slot_by_stream[item.stream_idx]);
        auto* merged_row = merged_bitmask_cpu_.data_ptr<int32_t>() + compact_slot * rows_per_stream * W;
        bitwiseAndInplace(merged_row, proc_mask.data_ptr<int32_t>(), rows_per_stream * W);
        spec_cap_cpu_.data_ptr<int32_t>()[item.stream_idx] =
            std::min<int32_t>(spec_cap_cpu_.data_ptr<int32_t>()[item.stream_idx], cap);
        result.applied_processors.push_back({item.stream_id, item.processor_idx});
    }

    auto packed_cpu = merged_bitmask_cpu_.narrow(0, 0, static_cast<int64_t>(compact_rows))
                          .narrow(1, 0, static_cast<int64_t>(W));
    auto rows_cpu = logits_row_indices_cpu_.narrow(0, 0, static_cast<int64_t>(compact_rows));
    auto cap_cpu  = spec_cap_cpu_.narrow(0, 0, static_cast<int64_t>(B));
#if USING_CUDA
    auto packed_gpu = merged_bitmask_gpu_.narrow(0, 0, static_cast<int64_t>(compact_rows))
                          .narrow(1, 0, static_cast<int64_t>(W));
    auto rows_gpu = logits_row_indices_gpu_.narrow(0, 0, static_cast<int64_t>(compact_rows));
    auto cap_gpu  = spec_cap_gpu_.narrow(0, 0, static_cast<int64_t>(B));
    cuda_graph::GraphStreamGuard stream_guard(cuda_graph::toGraphStream(copy_stream_));
    packed_gpu.copy_(packed_cpu, /*non_blocking=*/true);
    rows_gpu.copy_(rows_cpu, /*non_blocking=*/true);
    cap_gpu.copy_(cap_cpu, /*non_blocking=*/true);
    result.packed_allow_mask_gpu  = packed_gpu;
    result.logits_row_indices_gpu = rows_gpu;
    result.spec_cap_gpu           = cap_gpu;
    result.ready_event            = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
    result.ready_event->record(copy_stream_);
    result.consumed_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
    last_consumed_event_  = result.consumed_event;
#endif
    result.packed_allow_mask_cpu_lifetime  = packed_cpu;
    result.logits_row_indices_cpu_lifetime = rows_cpu;
    result.spec_cap_cpu_lifetime           = cap_cpu;
    result.has_active_processor            = true;
    return result;
}

}  // namespace rtp_llm
