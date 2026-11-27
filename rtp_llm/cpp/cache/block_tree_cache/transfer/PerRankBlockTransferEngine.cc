#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferStageState.h"

namespace rtp_llm {
namespace {

std::shared_ptr<TransferAsyncContext> invalidTransferContext(const TransferTask& task) {
    TransferResult result(ErrorInfo(ErrorCode::INVALID_PARAMS, "invalid block transfer request"));
    result.buffer_safe = true;
    return std::make_shared<CompletedTransferAsyncContext>(
        std::move(result), task.descriptors().size(), task.completionProof());
}

}  // namespace

PerRankBlockTransferEngine::PerRankBlockTransferEngine(std::vector<GroupSetPtr> group_sets,
                                                       PerRankWorkerOptions options,
                                                       PerRankWorkerIO io):
    group_sets_(std::move(group_sets)),
    max_descriptors_per_batch_(options.max_descriptors_per_batch),
    transfer_worker_count_(options.transfer_worker_count) {
    validateLimits(options.transfer_queue_max_size);
    if (!io.validate || !io.submit)
        throw std::invalid_argument("worker I/O requires validation and submission");
    validate_batch_ = [validate = std::move(io.validate)](const TransferTask& task) {
        const auto descriptors = task.descriptors();
        return std::all_of(descriptors.begin(), descriptors.end(),
                           [](const auto& descriptor) { return descriptor.isExecutable(); }) && validate(task);
    };
    initializeLoops(options.transfer_queue_max_size, [&](TransferOperation operation, size_t worker, size_t capacity) {
        return std::make_unique<TransferWorkLoop>(
            operation, batchLimit(), capacity, nullptr, options.world_rank, worker, io.submit);
    });
    startLoops();
}

void PerRankBlockTransferEngine::validateLimits(size_t queue_capacity) const {
    if (max_descriptors_per_batch_ == 0 || transfer_worker_count_ == 0
        || transfer_worker_count_ > std::numeric_limits<size_t>::max() / 2
        || queue_capacity < transfer_worker_count_
        || queue_capacity > std::numeric_limits<size_t>::max() / max_descriptors_per_batch_)
        throw std::invalid_argument("WorkLoop transfer requires Q >= worker count and nonoverflowing Q*M");
}

void PerRankBlockTransferEngine::initializeLoops(size_t queue_capacity, const MakeWorkLoop& make_loop) {
    load_loops_.reserve(transfer_worker_count_);
    evict_loops_.reserve(transfer_worker_count_);
    for (size_t worker = 0; worker < transfer_worker_count_; ++worker) {
        const auto capacity = queue_capacity / transfer_worker_count_
                              + (worker < queue_capacity % transfer_worker_count_ ? 1 : 0);
        load_loops_.push_back(make_loop(TransferOperation::LOAD, worker, capacity));
        evict_loops_.push_back(make_loop(TransferOperation::EVICT, worker, capacity));
    }
}

void PerRankBlockTransferEngine::startLoops() {
    for (auto& loop : load_loops_)
        loop->start();
    for (auto& loop : evict_loops_)
        loop->start();
}

PerRankBlockTransferEngine::~PerRankBlockTransferEngine() {
    shutdown();
}

void PerRankBlockTransferEngine::stopAdmission() {
    for (auto& loop : load_loops_)
        loop->stopAdmission();
    for (auto& loop : evict_loops_)
        loop->stopAdmission();
}

void PerRankBlockTransferEngine::shutdown() {
    stopAdmission();
    for (auto& loop : load_loops_)
        loop->shutdown();
    for (auto& loop : evict_loops_)
        loop->shutdown();
    // Accepted CRC stages may still submit their disk successors while draining.
    if (stop_io_)
        stop_io_();
}

void PerRankBlockTransferEngine::cancelPendingStagingTransfers() {
    for (auto& loop : load_loops_)
        loop->cancelPendingStagingTransfers();
}

BlockTreeQueueSizes PerRankBlockTransferEngine::queueSizes() const {
    BlockTreeQueueSizes sizes;
    for (const auto& loop : load_loops_)
        sizes.load += loop->outstandingBatches();
    for (const auto& loop : evict_loops_)
        sizes.background += loop->outstandingBatches();
    return sizes;
}

std::shared_ptr<TransferAsyncContext> PerRankBlockTransferEngine::execute(TransferTask task) {
    const auto                             proof = task.completionProof();
    block_tree_cache_detail::ScopeRollback unsubmitted([proof] {
        if (proof)
            proof->report(true);
    });
    const auto                             descriptors = task.descriptors();
    const size_t                           count       = descriptors.size();
    if (count == 0 || !task.sanityCheck()) {
        return invalidTransferContext(task);
    }
    if (task.expired()) {
        TransferResult result(ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "transfer deadline exceeded before dispatch"),
                              CacheCopyError::NOT_EXECUTED);
        result.buffer_safe = true;
        return std::make_shared<CompletedTransferAsyncContext>(std::move(result), count, proof);
    }

    auto context = std::make_shared<TransferBatchAsyncContext>(count);
    auto state   = std::make_shared<TransferStageState>(
        [context](TransferResult result) { context->complete(std::move(result)); }, count);
    auto                                   safety = std::make_shared<TransferBufferSafetyTracker>([proof] {
        if (proof)
            proof->report(true);
    });
    block_tree_cache_detail::ScopeRollback close_submission([safety] { safety->finishSubmitting(); });
    unsubmitted.dismiss();
    bool any_registered = false;
    for (size_t begin = 0; begin < count;) {
        std::vector<size_t>                   indices;
        bool                                  registered = false;
        TransferResult                        error;
        std::shared_ptr<TransferBufferSafety> batch_proof;
        std::shared_ptr<TransferAsyncContext> child;
        TransferAsyncContext::DoneCallback    complete;
        bool                                  attempted = false;
        try {
            const size_t batch_begin = begin;
            const auto&  first       = descriptors[begin];
            const size_t limit       = batchLimit();
            const size_t max_end     = begin + std::min(limit, count - begin);
            size_t       end         = begin + 1;
            while (end < max_end && descriptors[end].group_set_id == first.group_set_id
                   && descriptors[end].source_tier == first.source_tier
                   && descriptors[end].target_tier == first.target_tier) {
                ++end;
            }
            begin = end;
            indices.resize(end - batch_begin);
            std::iota(indices.begin(), indices.end(), batch_begin);
            batch_proof    = safety->addBatch();
            auto completed = std::make_shared<std::atomic<bool>>(false);
            complete       = [state, indices, completed](TransferResult result) {
                if (!completed->exchange(true, std::memory_order_acq_rel))
                    state->completeBatch(std::move(result), indices);
            };
            auto batch_task = task.subtask(batch_begin, end, batch_proof);
            state->addBatch();
            registered = any_registered = true;
            attempted                   = true;
            child                       = executeBatch(std::move(batch_task));
            if (!child) {
                error = ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "transfer engine returned a null context");
            } else {
                child->onDone(complete);
            }
        } catch (const std::exception& exception) {
            if (batch_proof && !attempted)
                batch_proof->report(true);
            if (!any_registered) {
                throw;
            }
            error = TransferResult::executionException(false, exception.what());
            if (child)
                error.status.setErrorCode(ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
        } catch (...) {
            if (batch_proof && !attempted)
                batch_proof->report(true);
            if (!any_registered) {
                throw;
            }
            error = TransferResult::executionException(false, "unknown transfer submission exception");
            if (child)
                error.status.setErrorCode(ErrorCode::TRANSFER_COMPLETION_UNKNOWN);
        }
        if (!error.ok()) {
            if (!registered) {
                close_submission.run();
                state->finishSubmitting(std::move(error));
                return context;
            }
            complete(std::move(error));
        }
    }
    // Already submitted batches retain their resources until every completion
    // arrives, including when a later batch is rejected or throws.
    close_submission.run();
    state->finishSubmitting();
    return context;
}

std::shared_ptr<TransferAsyncContext> PerRankBlockTransferEngine::executeBatch(TransferTask task) {
    block_tree_cache_detail::ScopeRollback unsubmitted([proof = task.completionProof()] {
        if (proof)
            proof->report(true);
    });
    const auto&                            descriptors  = task.descriptors();
    const Tier                             source       = descriptors.front().source_tier;
    const Tier                             target       = descriptors.front().target_tier;
    if (!validate_batch_(task))
        return invalidTransferContext(task);
    const bool load = isLoadTransfer(source, target) || (source == Tier::DISK && target == Tier::HOST);
    auto& loops = load ? load_loops_ : evict_loops_;
    auto& next = load ? next_load_ : next_evict_;
    const auto worker = loops.size() == 1 ? 0 : next.fetch_add(1, std::memory_order_relaxed) % loops.size();
    std::shared_ptr<TransferAsyncContext> context;
    unsubmitted.dismiss();
    for (size_t offset = 0; offset < loops.size(); ++offset) {
        bool retryable = false;
        // The last owner returns the final rejection and records it once.
        context = loops[(worker + offset) % loops.size()]->execute(
            std::move(task), {}, offset + 1 < loops.size() ? &retryable : nullptr);
        if (!retryable) {
            break;
        }
    }
    return context;
}

void PerRankBlockTransferEngine::waitForIdle() {
    BlockTreeQueueSizes pending;
    do {
        for (auto& loop : load_loops_)
            loop->waitForIdle();
        for (auto& loop : evict_loops_)
            loop->waitForIdle();
        // A later owner's callback may have submitted to an earlier owner.
        pending = queueSizes();
    } while (pending.load != 0 || pending.background != 0);
}

}  // namespace rtp_llm
