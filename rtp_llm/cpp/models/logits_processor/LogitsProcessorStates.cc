#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"

#include <algorithm>
#include <cstring>

#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/models/logits_processor/GrammarLogitsProcessor.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

#if USING_CUDA
#include <c10/cuda/CUDAGuard.h>
#endif

using namespace std;

namespace rtp_llm {

namespace {

void ensureSlotCapacity(BatchGrammarScratchPool::Slot& slot, int device_index, int64_t required_ints) {
    if (slot.capacity_ints >= required_ints && slot.device_index == device_index && slot.pinned_cpu.defined()
        && (device_index < 0 || slot.device_gpu.defined())) {
        return;
    }
    const int64_t alloc_ints = std::max(required_ints, slot.capacity_ints);
    if (!slot.pinned_cpu.defined() || slot.pinned_cpu.numel() < alloc_ints) {
        auto cpu_opts = torch::TensorOptions().dtype(torch::kInt32);
#if USING_CUDA
        if (device_index >= 0) {
            cpu_opts = cpu_opts.pinned_memory(true);
        }
#endif
        slot.pinned_cpu = torch::empty({alloc_ints}, cpu_opts);
    }
    if (device_index >= 0 && (!slot.device_gpu.defined() || slot.device_index != device_index
                              || slot.device_gpu.numel() < alloc_ints)) {
        auto cuda_opts =
            torch::TensorOptions().dtype(torch::kInt32).device(torch::Device(torch::kCUDA, device_index));
        slot.device_gpu = torch::empty({alloc_ints}, cuda_opts);
    } else if (device_index < 0) {
        slot.device_gpu = torch::Tensor();
    }
    slot.device_index  = device_index;
    slot.capacity_ints = alloc_ints;
}

}  // namespace

BatchGrammarScratchPool::~BatchGrammarScratchPool() {
#if USING_CUDA
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& slot : pool_) {
        if (slot && slot->recorded && slot->done_event.has_value() && !slot->done_event->query()) {
            slot->done_event->synchronize();
        }
    }
#endif
}

std::shared_ptr<BatchGrammarScratchPool::Slot> BatchGrammarScratchPool::acquire(int     device_index,
                                                                                int64_t required_ints) {
    std::shared_ptr<Slot> slot;
    bool                  need_sync = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!pool_.empty()) {
            const auto& front = pool_.front();
#if USING_CUDA
            const bool ready = front && front->device_index == device_index
                               && (!front->recorded || !front->done_event.has_value() || front->done_event->query());
            if (ready) {
                slot = std::move(pool_.front());
                pool_.pop_front();
            } else if (pool_.size() >= kMaxSlots) {
                slot      = std::move(pool_.front());
                need_sync = true;
                pool_.pop_front();
            }
#else
            if (front) {
                slot = std::move(pool_.front());
                pool_.pop_front();
            }
#endif
        }
    }
    if (!slot) {
        slot = std::make_shared<Slot>();
    } else {
#if USING_CUDA
        if (need_sync && slot->recorded && slot->done_event.has_value()) {
            slot->done_event->synchronize();
        }
        slot->recorded = false;
#endif
    }
    (void)need_sync;
    ensureSlotCapacity(*slot, device_index, required_ints);
    return slot;
}

void BatchGrammarScratchPool::retire(std::shared_ptr<Slot> slot) {
    if (!slot) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (pool_.size() >= kMaxSlots) {
        pool_.pop_front();
    }
    pool_.push_back(std::move(slot));
}

LogitsProcessorStates::LogitsProcessorStates() {};

void LogitsProcessorStates::setIntervalError(std::vector<std::optional<ErrorInfo>>& errors,
                                             const std::pair<size_t, size_t>&       interval,
                                             const ErrorInfo&                       error) {
    const size_t begin = std::min(interval.first, errors.size());
    const size_t end   = std::min(interval.second, errors.size());
    for (size_t row = begin; row < end; ++row) {
        if (!errors[row].has_value()) {
            errors[row] = error;
        }
    }
}

std::vector<std::optional<ErrorInfo>> LogitsProcessorStates::batchProcess(const SamplerInputs& inputs) {
    return batchProcess(inputs, nullptr);
}

std::vector<std::optional<ErrorInfo>> LogitsProcessorStates::batchProcess(const SamplerInputs&     inputs,
                                                                          BatchGrammarScratchPool* scratch_pool) {
    std::vector<std::optional<ErrorInfo>> processor_errors(inputs.batch_size);
    if (invocations_.empty()) {
        return processor_errors;
    }

    struct QueuedGrammar {
        GrammarLogitsProcessor*   processor = nullptr;
        size_t                    row_idx   = 0;
        std::pair<size_t, size_t> interval;
    };

    std::vector<QueuedGrammar> queued;
    queued.reserve(invocations_.size());
    std::vector<uint8_t> row_in_queued(inputs.batch_size, 0);

    std::vector<int32_t> row_indices_tmp;
    std::vector<int32_t> row_modes_tmp;
    std::vector<int32_t> vocab_limits_tmp;

    auto flush_queued = [&]() {
        if (queued.empty()) {
            return;
        }
        RTP_LLM_PROFILE_SCOPE("logits_processor.grammar_batch");
        // Fallback sequentially if logits is not a contiguous-row 2D tensor; individual
        // process() calls return the exact legacy validation error.
        if (!inputs.logits.defined() || inputs.logits.dim() != 2 || inputs.logits.size(1) <= 0) {
            for (const auto& q : queued) {
                auto err = q.processor->process(inputs, q.interval.first, q.interval.second);
                if (err.has_value()) {
                    setIntervalError(processor_errors, q.interval, err.value());
                }
                if (q.row_idx < row_in_queued.size()) {
                    row_in_queued[q.row_idx] = 0;
                }
            }
            queued.clear();
            return;
        }

        const int64_t logits_vocab_size  = inputs.logits.size(1);
        const int32_t bitmask_row_stride = static_cast<int32_t>((logits_vocab_size + 31) / 32);
        const int32_t max_rows           = static_cast<int32_t>(queued.size());
        const int32_t header_words       = ((3 * max_rows + 31) / 32) * 32;
        const int64_t max_required_ints =
            static_cast<int64_t>(header_words) + static_cast<int64_t>(max_rows) * bitmask_row_stride;

        const bool use_cuda     = inputs.logits.is_cuda();
        const int  device_index = use_cuda ? static_cast<int>(inputs.logits.device().index()) : -1;

        std::shared_ptr<BatchGrammarScratchPool::Slot> slot;
        torch::Tensor                                  standalone_cpu;
        int32_t*                                       host_ptr = nullptr;
        if (scratch_pool != nullptr) {
            slot     = scratch_pool->acquire(device_index, max_required_ints);
            host_ptr = slot->pinned_cpu.data_ptr<int32_t>();
        } else {
            auto cpu_opts = torch::TensorOptions().dtype(torch::kInt32);
#if USING_CUDA
            if (use_cuda) {
                cpu_opts = cpu_opts.pinned_memory(true);
            }
#endif
            standalone_cpu = torch::empty({max_required_ints}, cpu_opts);
            host_ptr       = standalone_cpu.data_ptr<int32_t>();
        }

        row_indices_tmp.clear();
        row_modes_tmp.clear();
        vocab_limits_tmp.clear();
        row_indices_tmp.reserve(queued.size());
        row_modes_tmp.reserve(queued.size());
        vocab_limits_tmp.reserve(queued.size());

        int32_t bitmask_row_count = 0;
        for (const auto& q : queued) {
            if (q.row_idx < row_in_queued.size()) {
                row_in_queued[q.row_idx] = 0;
            }
            int32_t* dst_row =
                host_ptr + header_words + static_cast<int64_t>(bitmask_row_count) * bitmask_row_stride;
            GrammarLogitsProcessor::BatchMaskEntry entry;
            auto err = q.processor->prepareBatchMaskEntry(
                inputs, q.row_idx, dst_row, static_cast<size_t>(bitmask_row_stride), entry);
            if (err.has_value()) {
                setIntervalError(processor_errors, q.interval, err.value());
                continue;
            }
            switch (entry.action) {
                case GrammarLogitsProcessor::BatchMaskAction::SKIP:
                    break;
                case GrammarLogitsProcessor::BatchMaskAction::TAIL_ONLY:
                    row_indices_tmp.push_back(static_cast<int32_t>(q.row_idx));
                    row_modes_tmp.push_back(-1);
                    vocab_limits_tmp.push_back(entry.grammar_vocab_size);
                    break;
                case GrammarLogitsProcessor::BatchMaskAction::FORCE_TOKEN:
                    row_indices_tmp.push_back(static_cast<int32_t>(q.row_idx));
                    row_modes_tmp.push_back(static_cast<int32_t>(entry.force_token_id));
                    vocab_limits_tmp.push_back(0);
                    break;
                case GrammarLogitsProcessor::BatchMaskAction::BITMASK_AND_TAIL:
                    row_indices_tmp.push_back(static_cast<int32_t>(q.row_idx));
                    row_modes_tmp.push_back(-2 - bitmask_row_count);
                    vocab_limits_tmp.push_back(entry.grammar_vocab_size);
                    ++bitmask_row_count;
                    break;
            }
        }
        queued.clear();

        const int32_t active_rows = static_cast<int32_t>(row_indices_tmp.size());
        if (active_rows > 0) {
            std::memcpy(host_ptr, row_indices_tmp.data(), static_cast<size_t>(active_rows) * sizeof(int32_t));
            std::memcpy(
                host_ptr + active_rows, row_modes_tmp.data(), static_cast<size_t>(active_rows) * sizeof(int32_t));
            std::memcpy(host_ptr + 2 * active_rows,
                        vocab_limits_tmp.data(),
                        static_cast<size_t>(active_rows) * sizeof(int32_t));

            const int32_t total_ints =
                bitmask_row_count > 0 ? (header_words + bitmask_row_count * bitmask_row_stride) : (3 * active_rows);

            if (slot) {
                runtimeApplyBatchGrammarMaskLogits(inputs.logits,
                                                   slot->pinned_cpu,
                                                   slot->device_gpu,
                                                   total_ints,
                                                   header_words,
                                                   active_rows,
                                                   bitmask_row_stride);
#if USING_CUDA
                if (use_cuda) {
                    c10::cuda::CUDAGuard device_guard(inputs.logits.device());
                    if (!slot->done_event.has_value()) {
                        slot->done_event.emplace(cuda_graph::makeGraphEvent());
                    }
                    slot->done_event->record(cuda_graph::graphGetCurrentStream());
                    slot->recorded = true;
                }
#endif
                scratch_pool->retire(std::move(slot));
            } else {
                runtimeApplyBatchGrammarMaskLogits(inputs.logits,
                                                   standalone_cpu,
                                                   torch::Tensor(),
                                                   total_ints,
                                                   header_words,
                                                   active_rows,
                                                   bitmask_row_stride);
            }
        } else if (slot) {
            scratch_pool->retire(std::move(slot));
        }
    };

    for (const auto& invocation : invocations_) {
        const auto& interval = invocation.interval;
        auto*       grammar_proc =
            (interval.second > interval.first && interval.second - interval.first == 1) ?
                dynamic_cast<GrammarLogitsProcessor*>(invocation.processor.get()) :
                nullptr;
        if (grammar_proc != nullptr && interval.first < inputs.batch_size) {
            if (row_in_queued[interval.first]) {
                flush_queued();
            }
            row_in_queued[interval.first] = 1;
            queued.push_back({grammar_proc, interval.first, interval});
        } else {
            const size_t begin = std::min(interval.first, row_in_queued.size());
            const size_t end   = std::min(interval.second, row_in_queued.size());
            for (size_t r = begin; r < end; ++r) {
                if (row_in_queued[r]) {
                    flush_queued();
                    break;
                }
            }
            auto error = invocation.processor->process(inputs, interval.first, interval.second);
            if (error.has_value()) {
                setIntervalError(processor_errors, interval, error.value());
            }
        }
    }
    flush_queued();

    return processor_errors;
}

void LogitsProcessorStates::insert(const BaseLogitsProcessorPtr& ptr, size_t start, size_t finish) {
    invocations_.push_back({ptr, std::make_pair(start, finish)});
}

}  // namespace rtp_llm
