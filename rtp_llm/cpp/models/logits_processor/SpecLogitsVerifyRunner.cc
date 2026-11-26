#include "rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.h"

#include <algorithm>
#include <limits>
#include <new>
#include <utility>

#include <c10/util/Exception.h>
#if USING_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/models/logits_processor/BitmaskUtils.h"
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

SpecLogitsVerifyRunner::LaunchResult makeFailureResultForActiveStreams(const SpecLogitsVerifyRunner::LaunchTask& task,
                                                                       const ErrorInfo& error) {
    SpecLogitsVerifyRunner::LaunchResult result;
    result.has_active_processor = !task.active.empty();
    result.processor_errors.resize(task.total_streams);
    for (const auto& item : task.active) {
        if (item.stream_idx < task.total_streams && !result.processor_errors[item.stream_idx].has_value()) {
            result.processor_errors[item.stream_idx] = error;
        }
    }
    return result;
}

}  // namespace

SpecLogitsVerifyRunner::~SpecLogitsVerifyRunner() {
#if USING_CUDA
    // Never drop an in-flight draft-token copy: its source must outlive the
    // transfer, so sync a pending event before the members (and the keep-alive
    // source tensor) are destroyed. Best effort only: destruction must not throw.
    try {
        waitDraftTokensOnCpu();
    } catch (...) {
        // A failed sync here means the CUDA context is already broken; nothing
        // further can be done in a destructor.
    }
#endif
}

void SpecLogitsVerifyRunner::waitDraftTokensOnCpu() {
#if USING_CUDA
    if (draft_transfer_event_.has_value()) {
        // Wait for just the draft-token transfer (and, transitively, the producer
        // event it was ordered after) instead of synchronizing the whole current
        // stream. An event whose record() never succeeded is an inert no-op.
        draft_transfer_event_->synchronize();
        draft_transfer_event_.reset();
        // The in-flight copy has completed; the materialized source can go.
        draft_transfer_src_ = torch::Tensor();
    }
#endif
}

void SpecLogitsVerifyRunner::applyMaskToLogits(torch::Tensor& logits, const LaunchResult& result, size_t vocab_size) {
#if USING_CUDA
    const auto& packed_allow_mask  = result.packed_allow_mask_gpu;
    const auto& logits_row_indices = result.logits_row_indices_gpu;
#else
    const auto& packed_allow_mask  = result.packed_allow_mask_cpu_lifetime;
    const auto& logits_row_indices = result.logits_row_indices_cpu_lifetime;
#endif
    if (!packed_allow_mask.defined()) {
        return;
    }
    runtimeApplyPackedMaskLogits(logits, packed_allow_mask, logits_row_indices, vocab_size);
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
#if USING_CUDA
    auto device_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    if (!has2DCapacity(merged_bitmask_gpu_, rows, W)) {
        merged_bitmask_gpu_ = torch::empty({rows, W}, device_i32);
    }
    if (!has1DCapacity(logits_row_indices_gpu_, rows)) {
        logits_row_indices_gpu_ = torch::empty({rows}, device_i32);
    }
#endif
    if (!has1DCapacity(spec_cap_cpu_, B)) {
        spec_cap_cpu_ = torch::empty({B}, pinned_i32);
    }
#if USING_CUDA
    if (!has1DCapacity(spec_cap_gpu_, B)) {
        spec_cap_gpu_ = torch::empty({B}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    }
#endif
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
#if USING_CUDA
    if (draft_tokens.is_cuda()) {
        // Guard the copy device so ensure/pool streams match draft_tokens.
        c10::cuda::CUDAGuard device_guard(draft_tokens.device());
        if (draft_copy_stream_.has_value() && draft_copy_stream_->device() != draft_tokens.device()) {
            draft_copy_stream_.reset();
        }
        if (!draft_copy_stream_.has_value()) {
            // Pool stream on the runner's device; matches how MtpExecutor owns
            // its AsyncRunner streams.
            draft_copy_stream_ = cuda_graph::graphGetStreamFromPool(/*is_high_priority=*/true);
        }
        std::optional<torch::Event> current_stream_done;
        if (!task.draft_tokens_ready_event) {
            // No producer event: preserve the old same-stream ordering by
            // ordering the copy after work already enqueued on the current
            // stream. Recorded before the guard below switches streams.
            current_stream_done = cuda_graph::makeGraphEvent();
            current_stream_done->record(cuda_graph::graphGetCurrentStream());
        }

        // Guarantee an unwind before commit never leaves an unrecorded
        // draft_transfer_event_ (whose synchronize() would be a no-op) or a
        // dangling source reference.
        struct CopyRollbackGuard {
            SpecLogitsVerifyRunner* runner;
            bool                    committed = false;
            ~CopyRollbackGuard() {
                if (!committed) {
                    if (runner->draft_copy_stream_.has_value()) {
                        try {
                            runner->draft_copy_stream_->synchronize();
                        } catch (...) {}
                    }
                    runner->draft_transfer_event_.reset();
                    runner->draft_transfer_src_ = torch::Tensor();
                }
            }
        } rollback_guard{this};

        cuda_graph::GraphStreamGuard copy_guard(cuda_graph::toGraphStream(*draft_copy_stream_));
        if (task.draft_tokens_ready_event) {
            // Async MTP records this on the producer stream after draft_tokens
            // are final; order our D2H after it.
            task.draft_tokens_ready_event->block(*draft_copy_stream_);
        } else {
            current_stream_done->block(*draft_copy_stream_);
        }
        auto draft = draft_tokens.reshape({B, draft_cols}).narrow(1, draft_offset, P);
        auto dst   = draft_tokens_cpu_.narrow(0, 0, B).narrow(1, 0, P);
        auto cuda_copy_stream = cuda_graph::toGraphStream(*draft_copy_stream_);
        // Protect caller's producer buffer while reshaped/read on the copy stream.
        c10::cuda::CUDACachingAllocator::recordStream(draft_tokens.storage().data_ptr(), cuda_copy_stream);
        // Pinned destination + the explicit event ordering above make the async
        // copy safe; the source is captured and recorded on the copy stream so
        // the caching allocator cannot recycle it before D2H finishes.
        draft_transfer_src_ =
            draft.scalar_type() == torch::kInt32 ? draft.contiguous() : draft.to(torch::kInt32).contiguous();
        c10::cuda::CUDACachingAllocator::recordStream(draft_transfer_src_.storage().data_ptr(), cuda_copy_stream);
        auto transfer_event = cuda_graph::makeGraphEvent();
        dst.copy_(draft_transfer_src_, /*non_blocking=*/true);
        transfer_event.record(*draft_copy_stream_);
        draft_transfer_event_.emplace(std::move(transfer_event));
        rollback_guard.committed = true;
        return;
    }
#endif
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
    auto row_indices_cpu = logits_row_indices_cpu_.narrow(0, 0, static_cast<int64_t>(shape.compact_rows));

    LaunchResult result;
#if USING_CUDA
    auto packed_mask_gpu = merged_bitmask_gpu_.narrow(0, 0, static_cast<int64_t>(shape.compact_rows))
                               .narrow(1, 0, static_cast<int64_t>(shape.bitmask_words));
    auto row_indices_gpu = logits_row_indices_gpu_.narrow(0, 0, static_cast<int64_t>(shape.compact_rows));
    // Pinned sources + ready_event ordering make non-blocking uploads safe for
    // both the sync path (same-stream order) and the async worker path.
    packed_mask_gpu.copy_(packed_mask_cpu, /*non_blocking=*/true);
    row_indices_gpu.copy_(row_indices_cpu, /*non_blocking=*/true);
    auto spec_cap_gpu = spec_cap_gpu_.narrow(0, 0, static_cast<int64_t>(shape.batch_size));
    spec_cap_gpu.copy_(spec_cap_cpu_.narrow(0, 0, static_cast<int64_t>(shape.batch_size)), /*non_blocking=*/true);
    result.packed_allow_mask_gpu  = std::move(packed_mask_gpu);
    result.logits_row_indices_gpu = std::move(row_indices_gpu);
    result.spec_cap_gpu           = std::move(spec_cap_gpu);
    result.ready_event            = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
    result.ready_event->record(cuda_graph::graphGetCurrentStream());
    result.consumed_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
    last_consumed_event_  = result.consumed_event;
#endif
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
#if USING_CUDA
    std::optional<c10::cuda::CUDAGuard> device_guard;
    if (task.draft_tokens.defined() && task.draft_tokens.is_cuda()) {
        device_guard.emplace(task.draft_tokens.device());
    }
    if (last_consumed_event_) {
        // Pinned scratch is reused across rounds; wait for the previous
        // consumer's recorded GPU work before overwriting it. An event the
        // consumer never recorded completes immediately.
        last_consumed_event_->synchronize();
        last_consumed_event_.reset();
    }
    // A previously aborted round can leave a draft-token transfer in flight;
    // complete it before this round reuses the pinned scratch or the source
    // keep-alive.
    waitDraftTokensOnCpu();
#endif
    try {
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
        // CPU-side mask processing waits here, at the explicit transfer-event
        // sync point; initializeCompactRows above ran while the GPU was still
        // executing the propose graph and the draft-token D2H.
        waitDraftTokensOnCpu();
        auto merge_result       = mergeProcessorMasks(task, layout, shape);
        auto result             = makeResult(shape);
        result.processor_errors = std::move(merge_result.processor_errors);
        return result;
    } catch (const std::bad_alloc& e) {
        return makeFailureResultForActiveStreams(task, detail::grammarMaskBuildError("MTP verify", e));
    } catch (const c10::Error& e) {
        return makeFailureResultForActiveStreams(task, detail::grammarMaskBuildError("MTP verify", e));
    }
}

}  // namespace rtp_llm
