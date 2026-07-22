#include "rtp_llm/cpp/model_rpc/PrefillBatchRpcServer.h"

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/AtomicUtil.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <unistd.h>

using namespace std;
namespace rtp_llm {

namespace {

// RAII cleanup used by the async batch tasks to keep metric/worker bookkeeping paired on every exit.
class ScopeExit {
public:
    explicit ScopeExit(std::function<void()> fn): fn_(std::move(fn)) {}
    ~ScopeExit() {
        if (fn_) {
            fn_();
        }
    }
    ScopeExit(const ScopeExit&)            = delete;
    ScopeExit& operator=(const ScopeExit&) = delete;

private:
    std::function<void()> fn_;
};

grpc::Status statusFromErrorInfo(const ErrorInfo& error_info) {
    if (!error_info.hasError()) {
        return grpc::Status::OK;
    }
    return grpc::Status(transErrorCodeToGrpc(error_info.code()), error_info.ToString());
}

void addBatchSuccess(EnqueueBatchResponsePB* response, int64_t request_id) {
    auto* success = response->add_successes();
    success->set_request_id(request_id);
}

void addBatchError(EnqueueBatchResponsePB* response, int64_t request_id, int64_t code, const std::string& msg) {
    auto* error = response->add_errors();
    error->set_request_id(request_id);
    auto* error_info = error->mutable_error_info();
    error_info->set_error_code(code);
    error_info->set_error_message(msg);
}

}  // namespace

PrefillBatchRpcServer::~PrefillBatchRpcServer() {
    stopAsyncResponseWorkers();
    stopResponseRegistryGc();
    if (prepare_resource_worker_pool_) {
        prepare_resource_worker_pool_->stop();
        prepare_resource_worker_pool_.reset();
    }
    if (worker_run_pool_) {
        worker_run_pool_->stop();
        worker_run_pool_.reset();
    }
}

grpc::Status PrefillBatchRpcServer::init(const EngineInitParams&                                maga_init_params,
                                         py::object                                             mm_process_engine,
                                         std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) {
    auto ret = PrefillRpcServer::init(maga_init_params, mm_process_engine, std::move(propose_params));
    if (!ret.ok()) {
        return ret;
    }
    initThreadPools();
    startResponseRegistryGc();
    return grpc::Status::OK;
}

// ---------------------------------------------------------------------------
// Batch infrastructure: response registry GC, async-worker counting, thread pools
// ---------------------------------------------------------------------------

void PrefillBatchRpcServer::startResponseRegistryGc() {
    if (response_gc_thread_.joinable()) {
        return;
    }
    response_gc_stop_.store(false);
    response_gc_thread_ = std::thread([this] {
        std::unique_lock<std::mutex> lock(response_gc_mu_);
        int                          gc_counter = 0;
        while (!response_gc_stop_.load()) {
            response_gc_cv_.wait_for(lock, std::chrono::seconds(10), [this] { return response_gc_stop_.load(); });
            if (response_gc_stop_.load()) {
                break;
            }
            lock.unlock();
            reportPoolMetrics();
            gc_counter++;
            if (gc_counter >= 6) {  // GC every 60 seconds
                response_registry_.gc(std::chrono::minutes(10));
                gc_counter = 0;
            }
            lock.lock();
        }
    });
}

void PrefillBatchRpcServer::stopResponseRegistryGc() {
    response_gc_stop_.store(true);
    response_gc_cv_.notify_all();
    if (response_gc_thread_.joinable()) {
        response_gc_thread_.join();
    }
}

bool PrefillBatchRpcServer::tryStartAsyncResponseWorker() {
    std::lock_guard<std::mutex> lock(response_worker_mu_);
    if (response_worker_stop_) {
        return false;
    }
    ++response_worker_count_;
    return true;
}

void PrefillBatchRpcServer::finishAsyncResponseWorker() {
    bool notify = false;
    {
        std::lock_guard<std::mutex> lock(response_worker_mu_);
        RTP_LLM_CHECK_WITH_INFO(response_worker_count_ > 0, "unbalanced async response worker finish");
        --response_worker_count_;
        notify = response_worker_stop_ && response_worker_count_ == 0;
    }
    if (notify) {
        response_worker_cv_.notify_all();
    }
}

void PrefillBatchRpcServer::stopAsyncResponseWorkers() {
    std::unique_lock<std::mutex> lock(response_worker_mu_);
    response_worker_stop_ = true;
    const bool stopped =
        response_worker_cv_.wait_for(lock, std::chrono::minutes(10), [this] { return response_worker_count_ == 0; });
    if (!stopped) {
        RTP_LLM_LOG_WARNING("timed out waiting for %zu async response workers; cancelling remaining responses",
                            response_worker_count_);
    }
    lock.unlock();

    if (!stopped) {
        response_registry_.cancelAll();
        lock.lock();
        const bool cancelled_stopped =
            response_worker_cv_.wait_for(lock, std::chrono::minutes(5), [this] { return response_worker_count_ == 0; });
        RTP_LLM_CHECK_WITH_INFO(cancelled_stopped,
                                "timed out waiting for %zu async response workers five minutes after cancellation",
                                response_worker_count_);
    }
}

void PrefillBatchRpcServer::initThreadPools() {
    const auto&   pd_sep_config     = maga_init_params_.pd_sep_config;
    const int64_t concurrency_limit = std::max<int64_t>(1, maga_init_params_.concurrency_config.concurrency_limit);
    const int64_t configured_prepare_size       = pd_sep_config.prefill_prepare_resource_pool_size > 0 ?
                                                      pd_sep_config.prefill_prepare_resource_pool_size :
                                                      concurrency_limit * 2;
    const int64_t prepare_resource_threads      = std::max<int64_t>(128, configured_prepare_size);
    const int64_t prepare_resource_queue        = prepare_resource_threads;
    const size_t  prepare_resource_thread_count = static_cast<size_t>(prepare_resource_threads);
    const size_t  prepare_resource_queue_size   = static_cast<size_t>(prepare_resource_queue);

    prepare_resource_worker_pool_ = std::make_shared<autil::LockFreeThreadPool>(
        prepare_resource_thread_count, prepare_resource_queue_size, nullptr, "PrefillPrepareResource");
    RTP_LLM_CHECK_WITH_INFO(prepare_resource_worker_pool_->start(),
                            "PrefillRpcServer prepare-resource thread pool start failed");
    RTP_LLM_LOG_INFO("PrefillRpcServer prepare-resource pool started: threads=%ld queue=%ld "
                     "(configured=%ld concurrency_limit=%ld)",
                     prepare_resource_threads,
                     prepare_resource_queue,
                     pd_sep_config.prefill_prepare_resource_pool_size,
                     concurrency_limit);
    prepare_resource_pool_metrics_.thread_max = prepare_resource_thread_count;
    prepare_resource_pool_metrics_.queue_max  = prepare_resource_queue_size;

    constexpr int64_t default_worker_run_threads    = 1024;
    const int64_t     configured_worker_run_threads = pd_sep_config.prefill_worker_run_pool_size > 0 ?
                                                          pd_sep_config.prefill_worker_run_pool_size :
                                                          default_worker_run_threads;
    const int64_t     worker_run_threads            = std::max<int64_t>(1, configured_worker_run_threads);
    const size_t      worker_run_thread_count       = static_cast<size_t>(worker_run_threads);
    constexpr size_t  worker_run_queue_size         = autil::ThreadPoolBase::DEFAULT_QUEUESIZE;
    worker_run_pool_                                = std::make_shared<autil::LockFreeThreadPool>(
        worker_run_thread_count, worker_run_queue_size, nullptr, "PrefillWorkerRun");
    RTP_LLM_CHECK_WITH_INFO(worker_run_pool_->start(), "PrefillRpcServer worker-run thread pool start failed");
    RTP_LLM_LOG_INFO("PrefillRpcServer worker-run pool started: threads=%ld queue=%zu (configured=%ld)",
                     worker_run_threads,
                     worker_run_queue_size,
                     pd_sep_config.prefill_worker_run_pool_size);
    worker_run_pool_metrics_.thread_max = worker_run_thread_count;
    worker_run_pool_metrics_.queue_max  = worker_run_queue_size;
}

void PrefillBatchRpcServer::reportPoolMetrics() {
    size_t response_worker_count = 0;
    {
        std::lock_guard<std::mutex> lock(response_worker_mu_);
        response_worker_count = response_worker_count_;
    }
    // Report to kmonitor (called every 10s from GC thread)
    reportPoolMetricsToKmonitor(metrics_reporter_, "prepare_resource", prepare_resource_pool_metrics_);
    reportPoolMetricsToKmonitor(metrics_reporter_, "worker_run", worker_run_pool_metrics_);
    RTP_LLM_LOG_DEBUG("PoolMetrics prepare_resource: active=%zu queued=%zu completed=%zu rejected=%zu fallback=%zu "
                      "thread_max=%zu queue_max=%zu",
                      prepare_resource_pool_metrics_.active.load(),
                      prepare_resource_pool_metrics_.queued.load(),
                      prepare_resource_pool_metrics_.completed.load(),
                      prepare_resource_pool_metrics_.rejected.load(),
                      prepare_resource_pool_metrics_.fallback.load(),
                      prepare_resource_pool_metrics_.thread_max,
                      prepare_resource_pool_metrics_.queue_max);
    RTP_LLM_LOG_DEBUG(
        "PoolMetrics worker_run: active=%zu queued=%zu completed=%zu rejected=%zu fallback=%zu response_workers=%zu "
        "thread_max=%zu queue_max=%zu",
        worker_run_pool_metrics_.active.load(),
        worker_run_pool_metrics_.queued.load(),
        worker_run_pool_metrics_.completed.load(),
        worker_run_pool_metrics_.rejected.load(),
        worker_run_pool_metrics_.fallback.load(),
        response_worker_count,
        worker_run_pool_metrics_.thread_max,
        worker_run_pool_metrics_.queue_max);
}

// ---------------------------------------------------------------------------
// EnqueueBatch — single-DP adapter for EnqueueGroup
// ---------------------------------------------------------------------------

grpc::Status PrefillBatchRpcServer::EnqueueBatch(grpc::ServerContext*         context,
                                                 const EnqueueBatchRequestPB* request,
                                                 EnqueueBatchResponsePB*      response) {
    RTP_LLM_PROFILE_FUNCTION();
    const auto& parallelism_config = maga_init_params_.parallelism_config;
    RTP_LLM_CHECK_WITH_INFO(parallelism_config.dp_size == 1,
                            "EnqueueBatch only supports single-DP mode, dp_size=%ld",
                            parallelism_config.dp_size);

    const int             local_dp_rank = static_cast<int>(parallelism_config.dp_rank);
    EnqueueGroupRequestPB group_request;
    group_request.set_batch_id(request->batch_id());
    group_request.set_dp_rank(local_dp_rank);
    response->set_batch_id(request->batch_id());

    int                         input_count          = 0;
    bool                        duplicate_request_id = false;
    std::unordered_set<int64_t> seen_request_ids;
    for (const auto& dp_slot : request->dp_slots()) {
        for (const auto& external_input : dp_slot.requests()) {
            ++input_count;
            if (external_input.has_input() && !seen_request_ids.insert(external_input.input().request_id()).second) {
                duplicate_request_id = true;
            }
        }
    }
    if (duplicate_request_id) {
        for (const auto& dp_slot : request->dp_slots()) {
            for (const auto& external_input : dp_slot.requests()) {
                if (external_input.has_input()) {
                    addBatchError(response,
                                  external_input.input().request_id(),
                                  grpc::StatusCode::ALREADY_EXISTS,
                                  "duplicate request_id in EnqueueBatch");
                } else {
                    addBatchError(response,
                                  /*request_id=*/0,
                                  grpc::StatusCode::INVALID_ARGUMENT,
                                  "EnqueueBatch external request missing input");
                }
            }
        }
        return grpc::Status::OK;
    }

    for (const auto& dp_slot : request->dp_slots()) {
        for (const auto& external_input : dp_slot.requests()) {
            if (dp_slot.dp_rank() != local_dp_rank) {
                addBatchError(response,
                              external_input.has_input() ? external_input.input().request_id() : 0,
                              grpc::StatusCode::INVALID_ARGUMENT,
                              "EnqueueBatch dp_rank mismatch, request dp_rank " + std::to_string(dp_slot.dp_rank())
                                  + ", local dp_rank " + std::to_string(local_dp_rank));
                continue;
            }
            auto* group_input = group_request.add_requests();
            if (external_input.has_input()) {
                group_input->mutable_input()->CopyFrom(external_input.input());
            }
        }
    }

    EnqueueBatchResponsePB group_response;
    auto                   status = EnqueueGroup(context, &group_request, &group_response);
    response->mutable_successes()->MergeFrom(group_response.successes());
    response->mutable_errors()->MergeFrom(group_response.errors());
    RTP_LLM_CHECK_WITH_INFO(response->successes_size() + response->errors_size() == input_count,
                            "EnqueueBatch result size mismatch: request=%d response=%d",
                            input_count,
                            response->successes_size() + response->errors_size());
    return status;
}

// ---------------------------------------------------------------------------
// EnqueueGroup — mirrors GenerateStreamCall: linear top level over named phases
// ---------------------------------------------------------------------------

grpc::Status PrefillBatchRpcServer::EnqueueGroup(grpc::ServerContext* /*context*/,
                                                 const EnqueueGroupRequestPB* request,
                                                 EnqueueBatchResponsePB*      response) {
    RTP_LLM_PROFILE_FUNCTION();
    response->set_batch_id(request->batch_id());

    std::vector<BatchSlot> slots;
    auto                   status = admitGroup(request, response, slots);
    if (status.ok() && !slots.empty()) {
        status = acceptGroup(std::move(slots), response);
    }
    const int response_size = response->successes_size() + response->errors_size();
    RTP_LLM_CHECK_WITH_INFO(response_size == request->requests_size(),
                            "EnqueueGroup result size mismatch: request=%d response=%d",
                            request->requests_size(),
                            response_size);
    return status;
}

grpc::Status PrefillBatchRpcServer::admitGroup(const EnqueueGroupRequestPB* request,
                                               EnqueueBatchResponsePB*      response,
                                               std::vector<BatchSlot>&      slots) {
    std::vector<const GenerateInputPB*> all_inputs;
    all_inputs.reserve(request->requests_size());
    std::unordered_set<int64_t> seen_request_ids;
    bool                        duplicate_request_id = false;
    for (const auto& dp_input : request->requests()) {
        if (!dp_input.has_input()) {
            addBatchError(response,
                          /*request_id=*/0,
                          grpc::StatusCode::INVALID_ARGUMENT,
                          "EnqueueGroup request missing input");
            continue;
        }
        all_inputs.push_back(&dp_input.input());
        if (!seen_request_ids.insert(dp_input.input().request_id()).second) {
            duplicate_request_id = true;
        }
    }

    response->mutable_successes()->Reserve(static_cast<int>(all_inputs.size()));
    response->mutable_errors()->Reserve(static_cast<int>(all_inputs.size()));

    auto add_error_for_all = [&](int64_t code, const std::string& message) {
        for (const auto* input : all_inputs) {
            addBatchError(response, input->request_id(), code, message);
        }
    };

    const int local_dp_rank = static_cast<int>(maga_init_params_.parallelism_config.dp_rank);
    if (request->dp_rank() != local_dp_rank) {
        add_error_for_all(grpc::StatusCode::INVALID_ARGUMENT,
                          "EnqueueGroup dp_rank mismatch, request dp_rank " + std::to_string(request->dp_rank())
                              + ", local dp_rank " + std::to_string(local_dp_rank));
        return grpc::Status::OK;
    }
    if (duplicate_request_id) {
        add_error_for_all(grpc::StatusCode::ALREADY_EXISTS, "duplicate request_id in EnqueueGroup");
        return grpc::Status::OK;
    }

    slots.reserve(all_inputs.size());
    for (const auto* input : all_inputs) {
        auto input_copy = std::make_shared<GenerateInputPB>(*input);
        // EnqueueGroup owns group membership; legacy stream metadata must not alter fallback scheduling.
        input_copy->clear_group_id();
        input_copy->clear_group_size();

        BatchSlot slot;
        slot.input = std::move(input_copy);
        slots.push_back(std::move(slot));
    }

    return grpc::Status::OK;
}

grpc::Status PrefillBatchRpcServer::acceptGroup(std::vector<BatchSlot> slots, EnqueueBatchResponsePB* response) {
    buildSlotContexts(slots);
    auto prepare_results = prepareGroup(slots);

    std::vector<ReadySlot> ready_slots;
    ready_slots.reserve(slots.size());
    for (size_t i = 0; i < slots.size(); ++i) {
        auto& slot       = slots[i];
        auto& result     = prepare_results[i];
        auto  request_id = slot.input->request_id();
        if (!result.prepared) {
            if (result.stage_status.ok()) {
                result.stage_status = grpc::Status(grpc::StatusCode::INTERNAL, "prepareAllocateResource failed");
            }
            addBatchError(response, request_id, result.stage_status.error_code(), result.stage_status.error_message());
            continue;
        }

        auto entry = response_registry_.reserve(request_id);
        if (!entry) {
            addBatchError(response, request_id, grpc::StatusCode::ALREADY_EXISTS, "request already enqueued");
            continue;
        }

        ready_slots.push_back(ReadySlot{&slot, std::move(entry)});
    }

    auto enqueue_status = enqueueGroupStreams(ready_slots, response);
    if (!enqueue_status.ok()) {
        for (auto& ready_slot : ready_slots) {
            rejectSlot(ready_slot, enqueue_status, response);
        }
        return grpc::Status::OK;
    }

    for (auto& ready_slot : ready_slots) {
        auto launch_status = launchSlotRunner(ready_slot);
        if (!launch_status.ok()) {
            rejectSlot(ready_slot, launch_status, response);
        } else {
            publishSlot(ready_slot, response);
        }
    }
    return grpc::Status::OK;
}

void PrefillBatchRpcServer::buildSlotContexts(std::vector<BatchSlot>& slots) {
    for (auto& slot : slots) {
        RPCContext rpc_ctx{slot.input.get(), nullptr};
        auto       pfx_ctx = std::make_unique<PrefillGenerateContext>(
            &this->resource(),
            rpc_ctx,
            slot.input->generate_config().timeout_ms(),
            /*server_context=*/nullptr,
            metrics_reporter_,
            meta_,
            maga_init_params_.pd_sep_config.prefill_stop_stream_wait_timeout_ms);
        pfx_ctx->onflight_requests      = onflight_requests_;
        pfx_ctx->loading_cache_requests = loading_cache_requests_;
        auto guard                      = std::make_shared<AtomicGuard>(onflight_requests_);
        slot.prefill_context            = std::move(pfx_ctx);
        slot.request_guard              = guard;
    }
}

std::vector<PrefillBatchRpcServer::PrepareResult> PrefillBatchRpcServer::prepareGroup(std::vector<BatchSlot>& slots) {
    const auto max_retry_times      = maga_init_params_.pd_sep_config.prefill_retry_times;
    const auto max_retry_timeout_ms = maga_init_params_.pd_sep_config.prefill_retry_timeout_ms;

    std::vector<PrepareResult>                       results(slots.size());
    std::vector<autil::ThreadPoolBase::Future<void>> prepare_futures;
    prepare_futures.reserve(slots.size());
    for (size_t i = 0; i < slots.size(); ++i) {
        auto* slot   = &slots[i];
        auto* result = &results[i];
        prepare_resource_pool_metrics_.queued++;
        try {
            auto future =
                prepare_resource_worker_pool_->async([this, slot, result, max_retry_times, max_retry_timeout_ms] {
                    try {
                        prepare_resource_pool_metrics_.queued--;
                        prepare_resource_pool_metrics_.active++;
                        ScopeExit slot_prepare_guard([this] {
                            prepare_resource_pool_metrics_.active--;
                            prepare_resource_pool_metrics_.completed++;
                        });
                        int64_t   begin_time_us = currentTimeUs();
                        auto      stage         = slot->prefill_context->stat_info.saveStage();
                        for (int attempt = 0; attempt <= max_retry_times; ++attempt) {
                            slot->prefill_context->reset();
                            slot->prefill_context->stat_info.restoreStage(stage);
                            slot->prefill_context->retry_times++;
                            prepareAllocateResource(*slot->prefill_context);
                            if (slot->prefill_context->ok()) {
                                result->prepared = true;
                                return;
                            }
                            auto cost_time_us                         = currentTimeUs() - begin_time_us;
                            slot->prefill_context->retry_cost_time_ms = cost_time_us / 1000;
                            if (max_retry_timeout_ms > 0 && cost_time_us >= max_retry_timeout_ms * 1000) {
                                break;
                            }
                            usleep(1000);
                        }
                        result->stage_status = slot->prefill_context->error_status.ok() ?
                                                   statusFromErrorInfo(slot->prefill_context->error_info) :
                                                   slot->prefill_context->error_status;
                        if (result->stage_status.ok()) {
                            result->stage_status =
                                grpc::Status(grpc::StatusCode::INTERNAL, "prepareAllocateResource failed");
                        }
                    } catch (const std::exception& e) {
                        result->stage_status = grpc::Status(
                            grpc::StatusCode::INTERNAL, "prepareAllocateResource exception: " + std::string(e.what()));
                    } catch (...) {
                        result->stage_status =
                            grpc::Status(grpc::StatusCode::INTERNAL, "prepareAllocateResource unknown exception");
                    }
                });
            prepare_futures.emplace_back(std::move(future));
        } catch (const std::exception& e) {
            prepare_resource_pool_metrics_.queued--;
            prepare_resource_pool_metrics_.rejected++;
            result->stage_status =
                grpc::Status(grpc::StatusCode::INTERNAL, "submit prepare task exception: " + std::string(e.what()));
        } catch (...) {
            prepare_resource_pool_metrics_.queued--;
            prepare_resource_pool_metrics_.rejected++;
            result->stage_status = grpc::Status(grpc::StatusCode::INTERNAL, "submit prepare task unknown exception");
        }
    }
    for (auto& future : prepare_futures) {
        future.get();
    }
    return results;
}

grpc::Status PrefillBatchRpcServer::enqueueGroupStreams(std::vector<ReadySlot>& ready_slots,
                                                        EnqueueBatchResponsePB* response) {
    if (ready_slots.empty()) {
        return grpc::Status::OK;
    }
    std::vector<std::shared_ptr<GenerateInput>> generate_inputs;
    generate_inputs.reserve(ready_slots.size());
    for (auto& ready_slot : ready_slots) {
        auto* slot = ready_slot.slot;
        slot->prefill_context->stat_info.nextStage();
        generate_inputs.push_back(slot->prefill_context->generate_input);
    }

    std::vector<bool>              enqueue_successes;
    std::vector<GenerateStreamPtr> streams;
    try {
        std::tie(enqueue_successes, streams) = engine_->enqueueMultiple(generate_inputs);
    } catch (const std::exception& e) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "enqueueMultiple exception: " + std::string(e.what()));
    } catch (...) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "enqueueMultiple unknown exception");
    }

    RTP_LLM_CHECK_WITH_INFO(enqueue_successes.size() == generate_inputs.size()
                                && streams.size() == generate_inputs.size(),
                            "enqueueMultiple result size mismatch: input=%zu status=%zu stream=%zu",
                            generate_inputs.size(),
                            enqueue_successes.size(),
                            streams.size());

    std::vector<ReadySlot> admitted_slots;
    admitted_slots.reserve(ready_slots.size());
    for (size_t i = 0; i < ready_slots.size(); ++i) {
        auto& stream = streams[i];
        RTP_LLM_CHECK_WITH_INFO(stream != nullptr, "enqueueMultiple returned null stream");
        auto& ready_slot = ready_slots[i];
        ready_slot.slot->prefill_context->setStream(stream);
        if (!enqueue_successes[i]) {
            auto status = statusFromErrorInfo(stream->statusInfo());
            if (status.ok()) {
                status = grpc::Status(grpc::StatusCode::INTERNAL, "scheduler rejected request");
            }
            rejectSlot(ready_slot, status, response);
            continue;
        }
        admitted_slots.push_back(std::move(ready_slot));
    }
    ready_slots = std::move(admitted_slots);
    return grpc::Status::OK;
}

grpc::Status PrefillBatchRpcServer::launchSlotRunner(ReadySlot& ready_slot) {
    auto& slot                               = *ready_slot.slot;
    auto  entry                              = ready_slot.entry;
    auto  writer                             = std::make_shared<ResponseBufferWriter>(entry);
    slot.prefill_context->rpc_context.writer = writer.get();

    if (!tryStartAsyncResponseWorker()) {
        slot.prefill_context->rpc_context.writer = nullptr;
        return grpc::Status(grpc::StatusCode::UNAVAILABLE, "EnqueueGroup server is stopping");
    }

    auto cancel_state = slot.prefill_context->cancel_state;
    entry->installCancelProducer([cancel_state] { cancel_state->store(true); });

    auto state             = std::make_shared<SlotRunnerState>();
    state->prefill_context = std::move(slot.prefill_context);
    state->input           = slot.input;
    state->writer          = std::move(writer);
    state->entry           = entry;
    state->request_guard   = std::move(slot.request_guard);
    auto request_id        = slot.input->request_id();
    worker_run_pool_metrics_.queued++;
    autil::ThreadPoolBase::Task stream_runner = [this, state, request_id]() mutable {
        worker_run_pool_metrics_.queued--;
        runSlotStream(std::move(*state), request_id);
    };
    // Never block the EnqueueGroup RPC handler. If the worker pool is saturated, reject the slot;
    // rejectSlot() tears down both the local stream and the prepared remote decode stream.
    auto error = worker_run_pool_->pushTask(std::move(stream_runner), /*isBlocked=*/false);
    if (error != autil::ThreadPoolBase::ERROR_NONE) {
        worker_run_pool_metrics_.queued--;
        worker_run_pool_metrics_.rejected++;
        finishAsyncResponseWorker();
        slot.prefill_context                     = std::move(state->prefill_context);
        slot.request_guard                       = std::move(state->request_guard);
        slot.prefill_context->rpc_context.writer = nullptr;
        return grpc::Status(grpc::StatusCode::UNAVAILABLE,
                            "EnqueueGroup worker-run pool rejected task with error=" + std::to_string(error));
    }
    return grpc::Status::OK;
}

void PrefillBatchRpcServer::runSlotStream(SlotRunnerState state, int64_t request_id) {
    worker_run_pool_metrics_.active++;
    ScopeExit    slot_finish_task_guard([this] {
        worker_run_pool_metrics_.active--;
        worker_run_pool_metrics_.completed++;
    });
    ScopeExit    worker_finish_guard([this] { finishAsyncResponseWorker(); });
    ScopeExit    release_state_guard([&] {
        state.prefill_context.reset();
        state.input.reset();
        state.writer.reset();
        state.entry.reset();
        state.request_guard.reset();
    });
    grpc::Status finish_status;
    try {
        finish_status = finishStream(*state.prefill_context);
        RTP_LLM_LOG_DEBUG("request [%ld] worker run returned, ok=%d, has_stream=%d",
                          request_id,
                          finish_status.ok(),
                          state.prefill_context->getStream() ? 1 : 0);
    } catch (const std::exception& e) {
        auto error_msg =
            "request [" + state.prefill_context->request_key + "] finishStream exception [" + e.what() + "]";
        finish_status = grpc::Status(grpc::StatusCode::INTERNAL, error_msg);
    } catch (...) {
        finish_status = grpc::Status(grpc::StatusCode::INTERNAL, "finishStream unknown exception");
    }
    if (finish_status.ok() && state.prefill_context->isRequestCancelled()) {
        finish_status = grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled");
    }
    if (!finish_status.ok()) {
        state.prefill_context->error_status = finish_status;
    }
    state.prefill_context.reset();
    response_registry_.finish(request_id, state.entry, finish_status);
    RTP_LLM_LOG_DEBUG("EnqueueGroup request [%ld] finishStream done, ok=%d", request_id, finish_status.ok());
}

void PrefillBatchRpcServer::publishSlot(ReadySlot& ready_slot, EnqueueBatchResponsePB* response) {
    auto request_id = ready_slot.slot->input->request_id();
    response_registry_.publish(request_id, ready_slot.entry);
    addBatchSuccess(response, request_id);
}

void PrefillBatchRpcServer::rejectSlot(ReadySlot&              ready_slot,
                                       const grpc::Status&     status,
                                       EnqueueBatchResponsePB* response) {
    auto&   slot       = *ready_slot.slot;
    int64_t request_id = slot.input->request_id();
    if (slot.prefill_context) {
        // The context destructor owns gRPC teardown. Mark this rejected request as cancelled so
        // closeGrpcStream() uses TryCancel() before waiting for the remote stream to finish.
        slot.prefill_context->cancel_state->store(true);
        if (slot.prefill_context->getStream()) {
            auto stream = slot.prefill_context->getStream();
            if (!stream->hasError()) {
                stream->reportError(status.error_code() == grpc::StatusCode::CANCELLED ? ErrorCode::CANCELLED :
                                                                                         ErrorCode::UNKNOWN_ERROR,
                                    std::string(status.error_message()));
            }
        }
    }
    slot.prefill_context.reset();
    response_registry_.abort(request_id, ready_slot.entry);
    addBatchError(response, request_id, status.error_code(), status.error_message());
}

// ---------------------------------------------------------------------------
// FetchResponse — claim and drain a per-request ResponseBuffer
// ---------------------------------------------------------------------------

grpc::Status PrefillBatchRpcServer::FetchResponse(grpc::ServerContext*                   context,
                                                  const FetchRequestPB*                  request,
                                                  grpc::ServerWriter<GenerateOutputsPB>* writer) {
    RTP_LLM_PROFILE_FUNCTION();
    const auto request_id   = request->request_id();
    auto       claim_result = response_registry_.claim(request_id);
    if (claim_result.status == ResponseBufferRegistry::ClaimStatus::NOT_FOUND) {
        return grpc::Status(grpc::StatusCode::NOT_FOUND,
                            "request [" + std::to_string(request_id) + "] not found in response registry");
    }
    if (claim_result.status == ResponseBufferRegistry::ClaimStatus::ALREADY_CLAIMED) {
        return grpc::Status(grpc::StatusCode::ALREADY_EXISTS,
                            "request [" + std::to_string(request_id) + "] response is already being fetched");
    }
    auto      entry = std::move(claim_result.entry);
    ScopeExit release_claim([this, request_id, entry] { response_registry_.releaseClaim(request_id, entry); });

    while (true) {
        if (context && context->IsCancelled()) {
            entry->cancel();
            return grpc::Status(grpc::StatusCode::CANCELLED, "fetch response cancelled by client");
        }

        auto drained = entry->waitAndDrain(std::chrono::milliseconds(100));

        for (auto& output : drained.outputs) {
            if (!writer->Write(output)) {
                entry->cancel();
                return grpc::Status(grpc::StatusCode::CANCELLED, "client writer closed");
            }
        }

        if (drained.terminal) {
            return drained.terminal_status;
        }
    }
}

}  // namespace rtp_llm
