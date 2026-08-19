#include "rtp_llm/cpp/cache/block_tree_cache/benchmark/OnlineTreeScheduler.h"

#include <algorithm>
#include <thread>

namespace rtp_llm::benchmark {

namespace {

int64_t elapsedNs(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

}  // anonymous namespace

bool boundedPendingTaskDrain(const std::function<size_t()>& pending_count, std::chrono::milliseconds budget) {
    const auto deadline = std::chrono::steady_clock::now() + budget;
    while (pending_count() != 0) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            return false;
        }
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
        std::this_thread::sleep_for(std::min(std::chrono::milliseconds(5), remaining));
    }
    return true;
}

OnlineTreeScheduler::OnlineTreeScheduler(OnlineCacheApi& cache, const OnlineTreeWorkloadConfig& config):
    cache_(cache), config_(config) {}

OnlineRequestContext OnlineTreeScheduler::makeContext(const OnlineRequestDescriptor& descriptor) {
    OnlineRequestContext context;
    context.request_id           = descriptor.request_id;
    context.path                 = descriptor.path;
    context.planned_reuse_blocks = descriptor.planned_reuse_blocks;
    context.target_tokens        = descriptor.target_tokens;
    context.family_id            = descriptor.family_id;
    context.epoch_id             = descriptor.epoch_id;
    context.generation           = descriptor.generation;
    context.predecessor_id       = descriptor.predecessor_id;
    context.is_continuation      = descriptor.is_continuation;
    return context;
}

OnlineTreeScheduler::RequestOutcome
OnlineTreeScheduler::predecessorOutcome(const OnlineRequestContext& ctx) const {
    if (ctx.predecessor_id < 0) {
        return RequestOutcome::PENDING;
    }
    const size_t predecessor_id = static_cast<size_t>(ctx.predecessor_id);
    return predecessor_id < request_outcomes_.size() ? request_outcomes_[predecessor_id] : RequestOutcome::PENDING;
}

void OnlineTreeScheduler::recordOutcome(const OnlineRequestContext& ctx, RequestOutcome outcome) {
    request_outcomes_.resize(std::max(request_outcomes_.size(), ctx.request_id + 1), RequestOutcome::PENDING);
    request_outcomes_[ctx.request_id] = outcome;
}

OnlineTreeScheduler::AdmitResult OnlineTreeScheduler::admit(OnlineRequestContext& ctx) {
    // Check dependency: continuation requests need the parent published first.
    if (ctx.is_continuation) {
        const auto predecessor = predecessorOutcome(ctx);
        if (predecessor == RequestOutcome::FAILED) {
            ++metrics_.dependency_failed_descendants;
            recordOutcome(ctx, RequestOutcome::FAILED);
            ctx.state = OnlineRequestState::FINISHED;
            return AdmitResult::OK;
        }
        if (predecessor == RequestOutcome::ABANDONED) {
            recordOutcome(ctx, RequestOutcome::ABANDONED);
            ++metrics_.dropped_waiting_at_deadline;
            ctx.state = OnlineRequestState::FINISHED;
            return AdmitResult::OK;
        }
        if (predecessor != RequestOutcome::PUBLISHED) {
            return AdmitResult::SKIP;  // caller skips this context and continues
        }
    }

    if (active_tokens_ + ctx.target_tokens > config_.active_token_budget) {
        return AdmitResult::BLOCKED;  // FIFO: the caller stops admitting
    }

    const auto admit_start = std::chrono::steady_clock::now();
    ctx.admitted_at        = admit_start;
    ctx.match_at           = admit_start;

    auto outcome = cache_.match(ctx.path);
    metrics_.match_ns.push_back(elapsedNs(admit_start, std::chrono::steady_clock::now()));
    cache_.materializeRequestBlocks(outcome);
    if (outcome.actual_matched_depth > ctx.planned_reuse_blocks) {
        ++metrics_.unexpected_extra_match_count;
    }

    // Move match-held resources into the context before any allocation so the
    // failure paths release them through cleanupRequest.
    ctx.matched_depth            = outcome.actual_matched_depth;
    ctx.matched_device_blocks    = outcome.matched_device_blocks;
    ctx.host_matched_blocks      = outcome.host_matched_blocks;
    ctx.request_blocks            = std::move(outcome.request_blocks);
    ctx.joined_target_block_count = outcome.joined_target_block_count;

    const size_t suffix_block_count = ctx.path.size() - ctx.matched_depth;
    enum class AdmissionFailure {
        NONE,
        LOAD_TARGETS,
        SUFFIX,
        COMMIT
    };
    AdmissionFailure failure = AdmissionFailure::NONE;

    for (size_t attempt = 0; attempt <= config_.admission_allocation_retry_limit; ++attempt) {
        if (attempt > 0) {
            ++metrics_.admission_allocation_retries;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        if (outcome.load_ticket) {
            const auto commit_start = std::chrono::steady_clock::now();
            if (!cache_.allocateLoadTargets(outcome, ctx.prepared)) {
                failure = AdmissionFailure::LOAD_TARGETS;
            } else if (!cache_.allocateSuffixBlocks(suffix_block_count, ctx.prepared)) {
                failure = AdmissionFailure::SUFFIX;
            } else if (!cache_.commitLoad(outcome.load_ticket, ctx.prepared)) {
                failure = AdmissionFailure::COMMIT;
            } else {
                failure = AdmissionFailure::NONE;
                ++metrics_.loads_committed;
                metrics_.load_commit_ns.push_back(elapsedNs(commit_start, std::chrono::steady_clock::now()));
            }
        } else if (!cache_.allocateSuffixBlocks(suffix_block_count, ctx.prepared)) {
            failure = AdmissionFailure::SUFFIX;
        } else {
            failure = AdmissionFailure::NONE;
        }
        if (failure == AdmissionFailure::NONE || failure == AdmissionFailure::COMMIT
            || attempt == config_.admission_allocation_retry_limit) {
            break;
        }
        // Allocation hit a temporarily exhausted pool: roll back the partial
        // allocations and wait for the cache's event-driven watermark
        // eviction to free capacity. The scheduler never evicts directly.
        std::vector<BlockIndicesType> no_request_blocks;
        cache_.rollback(ctx.prepared, no_request_blocks);
    }

    if (failure != AdmissionFailure::NONE) {
        switch (failure) {
            case AdmissionFailure::LOAD_TARGETS:
                ++metrics_.load_target_allocation_failed;
                break;
            case AdmissionFailure::SUFFIX:
                ++metrics_.suffix_allocation_failed;
                break;
            case AdmissionFailure::COMMIT:
                ++metrics_.load_commit_failed;
                break;
            case AdmissionFailure::NONE:
                break;
        }
        recordOutcome(ctx, RequestOutcome::FAILED);
        cleanupRequest(ctx);
        return AdmitResult::OK;  // handled (failed admission); keep scanning
    }

    ctx.load_ticket = std::move(outcome.load_ticket);
    metrics_.joined_target_blocks_total += ctx.joined_target_block_count;
    active_tokens_ += ctx.target_tokens;
    ctx.tokens_counted = true;

    if (ctx.load_ticket && !ctx.load_ticket->done()) {
        ctx.state = OnlineRequestState::LOADING_CACHE;
    } else if (ctx.load_ticket && !ctx.load_ticket->success()) {
        ++metrics_.loads_failed;
        recordOutcome(ctx, RequestOutcome::FAILED);
        cleanupRequest(ctx);
        return AdmitResult::OK;
    } else {
        if (ctx.load_ticket) {
            ++metrics_.loads_succeeded;
        }
        ctx.state         = OnlineRequestState::READY;
        const auto now    = std::chrono::steady_clock::now();
        metrics_.match_to_ready_ns.push_back(elapsedNs(ctx.match_at, now));
    }
    return AdmitResult::OK;
}

void OnlineTreeScheduler::cleanupRequest(OnlineRequestContext& ctx) {
    if (ctx.state == OnlineRequestState::FINISHED) {
        return;
    }
    if (ctx.prepared.holdsBlocks()) {
        cache_.rollback(ctx.prepared, ctx.request_blocks);
    } else if (!ctx.request_blocks.empty()) {
        cache_.releaseRequestBlocks(ctx.request_blocks);
    }
    if (ctx.tokens_counted) {
        active_tokens_ -= ctx.target_tokens;
        ctx.tokens_counted = false;
    }
    ctx.load_ticket.reset();
    ctx.state = OnlineRequestState::FINISHED;
}

void OnlineTreeScheduler::removeFinished() {
    contexts_.erase(
        std::remove_if(contexts_.begin(),
                       contexts_.end(),
                       [](const OnlineRequestContext& ctx) { return ctx.state == OnlineRequestState::FINISHED; }),
        contexts_.end());
}

size_t OnlineTreeScheduler::pendingLoadTickets() const {
    return static_cast<size_t>(std::count_if(contexts_.begin(), contexts_.end(), [](const auto& ctx) {
        return ctx.load_ticket != nullptr && !ctx.load_ticket->done();
    }));
}

bool OnlineTreeScheduler::runPhase(const std::vector<OnlineRequestDescriptor>& trace,
                                   size_t&                                     next_trace_index,
                                   std::chrono::milliseconds                   duration,
                                   int64_t&                                    measured_ns) {
    const auto phase_start     = std::chrono::steady_clock::now();
    const auto deadline        = phase_start + duration;
    bool       success         = true;
    bool       trace_exhausted = false;

    auto updatePeaks = [&]() {
        size_t active = 0, waiting = 0, loading = 0, tickets_pending = 0, held = 0, dep_wait = 0;
        for (const auto& ctx : contexts_) {
            switch (ctx.state) {
                case OnlineRequestState::WAITING:
                    ++waiting;
                    break;
                case OnlineRequestState::LOADING_CACHE:
                    ++loading;
                    ++active;
                    if (ctx.load_ticket && !ctx.load_ticket->done()) {
                        ++tickets_pending;
                    }
                    break;
                case OnlineRequestState::READY:
                    ++active;
                    break;
                case OnlineRequestState::FINISHED:
                    break;
            }
            if (ctx.state == OnlineRequestState::LOADING_CACHE || ctx.state == OnlineRequestState::READY) {
                for (const auto& blocks : ctx.request_blocks)
                    held += blocks.size();
                for (const auto& blocks : ctx.prepared.load_target_blocks)
                    held += blocks.size();
                for (const auto& blocks : ctx.prepared.suffix_blocks)
                    held += blocks.size();
            }
            if (ctx.state == OnlineRequestState::WAITING && ctx.is_continuation
                && predecessorOutcome(ctx) == RequestOutcome::PENDING) {
                ++dep_wait;
            }
        }
        metrics_.active_requests_peak      = std::max(metrics_.active_requests_peak, active);
        metrics_.waiting_requests_peak     = std::max(metrics_.waiting_requests_peak, waiting);
        metrics_.loading_requests_peak     = std::max(metrics_.loading_requests_peak, loading);
        metrics_.load_tickets_pending_peak = std::max(metrics_.load_tickets_pending_peak, tickets_pending);
        metrics_.held_request_blocks_peak  = std::max(metrics_.held_request_blocks_peak, held);
        metrics_.dependency_waiting_peak   = std::max(metrics_.dependency_waiting_peak, dep_wait);
        return loading != 0;
    };

    std::vector<OnlineRequestContext*> batch;
    batch.reserve(config_.logical_concurrency);
    while (true) {
        // 1. Resolve load completion, dependency failure and timeout in state
        // order. Predecessors always appear before descendants in contexts_.
        const auto now = std::chrono::steady_clock::now();
        for (auto& ctx : contexts_) {
            if (ctx.state == OnlineRequestState::LOADING_CACHE && ctx.load_ticket && ctx.load_ticket->done()) {
                if (ctx.load_ticket->success()) {
                    ++metrics_.loads_succeeded;
                    ctx.state = OnlineRequestState::READY;
                    metrics_.match_to_ready_ns.push_back(elapsedNs(ctx.match_at, now));
                } else {
                    ++metrics_.loads_failed;
                    recordOutcome(ctx, RequestOutcome::FAILED);
                    cleanupRequest(ctx);
                }
            }
            if (ctx.state == OnlineRequestState::WAITING && ctx.is_continuation
                && predecessorOutcome(ctx) == RequestOutcome::FAILED) {
                ++metrics_.dependency_failed_descendants;
                recordOutcome(ctx, RequestOutcome::FAILED);
                ctx.state = OnlineRequestState::FINISHED;
            }
            if (ctx.state == OnlineRequestState::LOADING_CACHE && elapsedNs(ctx.admitted_at, now)
                > static_cast<int64_t>(config_.request_lifecycle_timeout_ms) * 1'000'000) {
                ++metrics_.lifecycle_timeouts;
                success = false;
                recordOutcome(ctx, RequestOutcome::FAILED);
                cleanupRequest(ctx);
            }
        }
        removeFinished();

        // 2. Refill and admission only while before the phase deadline.
        if (std::chrono::steady_clock::now() < deadline) {
            while (contexts_.size() < config_.logical_concurrency) {
                if (next_trace_index >= trace.size()) {
                    trace_exhausted = true;
                    break;
                }
                contexts_.push_back(makeContext(trace[next_trace_index]));
                ++next_trace_index;
            }
            // SKIP allows scanning past an unresolved dependency; BLOCKED
            // stops the pass because later requests cannot bypass FIFO.
            for (auto& ctx : contexts_) {
                if (ctx.state != OnlineRequestState::WAITING) {
                    continue;
                }
                const auto result = admit(ctx);
                if (result == AdmitResult::BLOCKED) {
                    break;
                } else if (result == AdmitResult::SKIP) {
                    ++metrics_.dependency_skip_count;
                    // Continue scanning; do NOT mark this context finished
                    // so it remains WAITING for the parent's completion.
                }
            }
            // Admission failures and dependency-terminal descendants hold no
            // live work. Remove them now so final-zero reflects real active
            // contexts even when an entire admission pass fails.
            removeFinished();
        }

        // 3. Forward: exactly one fixed sleep per non-empty READY batch.
        const bool has_loading = updatePeaks();
        batch.clear();
        for (auto& ctx : contexts_) {
            if (ctx.state == OnlineRequestState::READY) {
                batch.push_back(&ctx);
            }
        }
        if (!batch.empty()) {
            ++metrics_.forward_batches;
            metrics_.forward_requests += batch.size();
            metrics_.ready_batch_sizes.push_back(batch.size());
            metrics_.ready_batch_max = std::max(metrics_.ready_batch_max, batch.size());
            std::this_thread::sleep_for(std::chrono::milliseconds(config_.forward_sleep_ms));
            for (auto* ctx : batch) {
                const auto insert_start = std::chrono::steady_clock::now();
                cache_.publishInsert(ctx->path, ctx->matched_depth, ctx->prepared, ctx->request_blocks);
                metrics_.insert_ns.push_back(elapsedNs(insert_start, std::chrono::steady_clock::now()));
                if (ctx->tokens_counted) {
                    active_tokens_ -= ctx->target_tokens;
                    ctx->tokens_counted = false;
                }

                // Publish family state: after a successful insert, the family
                // has a published ancestor that can unlock continuations.
                recordOutcome(*ctx, RequestOutcome::PUBLISHED);
                ctx->state = OnlineRequestState::FINISHED;
                ++metrics_.completed_transactions;
                metrics_.planned_reuse_blocks_total += ctx->planned_reuse_blocks;
                metrics_.actual_matched_depth_total += ctx->matched_depth;
                metrics_.device_matched_blocks_total += ctx->matched_device_blocks;
                metrics_.host_matched_blocks_total += ctx->host_matched_blocks;
                metrics_.insert_path_keys_total += ctx->path.size();
                metrics_.insert_new_nodes_total += ctx->path.size() - ctx->matched_depth;
                metrics_.planned_reuse_blocks_samples.push_back(static_cast<int64_t>(ctx->planned_reuse_blocks));
                metrics_.actual_matched_depth_samples.push_back(static_cast<int64_t>(ctx->matched_depth));
                metrics_.reuse_delta_blocks_samples.push_back(static_cast<int64_t>(ctx->matched_depth)
                                                              - static_cast<int64_t>(ctx->planned_reuse_blocks));
                ++metrics_.completed_requests_by_family[ctx->family_id];
                metrics_.completed_generation_samples.push_back(static_cast<int64_t>(ctx->generation));
                metrics_.completed_family_epochs.emplace(ctx->family_id, ctx->epoch_id);
                metrics_.max_completed_generation = std::max(metrics_.max_completed_generation, ctx->generation);
                if (ctx->is_continuation) {
                    ++metrics_.completed_continuation_transactions;
                    metrics_.completed_continuation_families.insert(ctx->family_id);
                } else {
                    ++metrics_.completed_base_transactions;
                }
            }
            removeFinished();
            continue;
        }

        // 4. No READY: poll pending loads (bounded 1ms, never an unbounded
        // waitDone), or finish when the backlog is empty.
        if (contexts_.empty()) {
            break;
        }
        if (has_loading) {
            const auto poll_start = std::chrono::steady_clock::now();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            metrics_.scheduler_no_ready_wait_ns += elapsedNs(poll_start, std::chrono::steady_clock::now());
            continue;
        }

        // No READY and no LOADING: the only remaining contexts are WAITING
        // left over after the deadline (a dependency block or token-budget
        // block cannot outlive its LOADING/READY holders). Drop them without
        // resources so the drain terminates.
        for (auto& ctx : contexts_) {
            if (ctx.state == OnlineRequestState::WAITING) {
                ++metrics_.dropped_waiting_at_deadline;
                recordOutcome(ctx, RequestOutcome::ABANDONED);
                ctx.state = OnlineRequestState::FINISHED;
            }
        }
        removeFinished();
    }

    measured_ns = elapsedNs(phase_start, std::chrono::steady_clock::now());

    if (trace_exhausted) {
        ++metrics_.trace_exhaustions;
        success = false;
    }
    return success;
}

}  // namespace rtp_llm::benchmark
