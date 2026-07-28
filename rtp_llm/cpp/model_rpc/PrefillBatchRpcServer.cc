#include "rtp_llm/cpp/model_rpc/PrefillBatchRpcServer.h"

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/AtomicUtil.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <unistd.h>

using namespace std;
namespace rtp_llm {

namespace {

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

void DeferredPrefillContext::cancel(const grpc::Status& status) {
    if (!context) {
        return;
    }
    context->error_status = status;
    context->cancel_state->store(true);
    if (context->client_context) {
        context->client_context->TryCancel();
    }
    auto stream = context->getStream();
    if (stream && !stream->hasError() && stream->getStatus() != StreamState::FINISHED) {
        stream->reportError(status.error_code() == grpc::StatusCode::CANCELLED ? ErrorCode::CANCELLED :
                                                                                 ErrorCode::UNKNOWN_ERROR,
                            status.error_message());
    }
}

grpc::Status DeferredPrefillContextMap::store(int64_t                                        request_id,
                                              const std::shared_ptr<DeferredPrefillContext>& deferred) {
    std::lock_guard<std::mutex> lock(mu_);
    if (stopping_) {
        return grpc::Status(grpc::StatusCode::UNAVAILABLE, "Prefill batch server is shutting down");
    }
    if (!contexts_.emplace(request_id, deferred).second) {
        return grpc::Status(grpc::StatusCode::ALREADY_EXISTS, "request already exists in deferred context map");
    }
    return grpc::Status::OK;
}

grpc::Status DeferredPrefillContextMap::armTtl(int64_t                                        request_id,
                                               const std::shared_ptr<DeferredPrefillContext>& deferred,
                                               std::chrono::milliseconds                      ttl) {
    auto alarm = std::make_shared<grpc::Alarm>();
    auto weak  = weak_from_this();
    ttl        = std::max(ttl, std::chrono::milliseconds(1));
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (stopping_) {
            return grpc::Status(grpc::StatusCode::UNAVAILABLE, "Prefill batch server is shutting down");
        }
        auto it = contexts_.find(request_id);
        if (it == contexts_.end() || it->second.get() != deferred.get()) {
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                                "request context is missing from deferred context map");
        }
        deferred->ttl_alarm = alarm;
        // Arm before returning success. cancelAll() cannot remove this context
        // until Set() has completed, so Cancel() never races an unset Alarm.
        alarm->experimental().Set(std::chrono::system_clock::now() + ttl,
                                  [weak, request_id, expected = deferred.get()](bool ok) {
                                      if (ok) {
                                          if (auto contexts = weak.lock()) {
                                              contexts->expire(request_id, expected);
                                          }
                                      }
                                  });
    }
    return grpc::Status::OK;
}

grpc::Status DeferredPrefillContextMap::take(int64_t request_id, std::shared_ptr<DeferredPrefillContext>& deferred) {
    deferred.reset();
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto                        it = contexts_.find(request_id);
        if (it == contexts_.end()) {
            return grpc::Status(grpc::StatusCode::NOT_FOUND,
                                "request [" + std::to_string(request_id) + "] not found in deferred context map");
        }
        deferred = std::move(it->second);
        contexts_.erase(it);
    }
    if (deferred->ttl_alarm) {
        deferred->ttl_alarm->Cancel();
    }
    return grpc::Status::OK;
}

std::shared_ptr<DeferredPrefillContext> DeferredPrefillContextMap::remove(int64_t                       request_id,
                                                                          const DeferredPrefillContext* expected) {
    std::shared_ptr<DeferredPrefillContext> deferred;
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto                        it = contexts_.find(request_id);
        if (it == contexts_.end() || it->second.get() != expected) {
            return nullptr;
        }
        deferred = std::move(it->second);
        contexts_.erase(it);
    }
    if (deferred->ttl_alarm) {
        deferred->ttl_alarm->Cancel();
    }
    return deferred;
}

void DeferredPrefillContextMap::expire(int64_t request_id, const DeferredPrefillContext* expected) {
    std::shared_ptr<DeferredPrefillContext> deferred;
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto                        it = contexts_.find(request_id);
        if (it == contexts_.end() || it->second.get() != expected) {
            return;
        }
        deferred = std::move(it->second);
        contexts_.erase(it);
    }
    deferred->cancel(grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "FetchResponse context TTL expired"));
}

void DeferredPrefillContextMap::stopAccepting() {
    std::lock_guard<std::mutex> lock(mu_);
    stopping_ = true;
}

void DeferredPrefillContextMap::cancelAll(const grpc::Status& status) {
    std::vector<std::shared_ptr<DeferredPrefillContext>> deferred_contexts;
    {
        std::lock_guard<std::mutex> lock(mu_);
        stopping_ = true;
        deferred_contexts.reserve(contexts_.size());
        for (auto& entry : contexts_) {
            deferred_contexts.push_back(std::move(entry.second));
        }
        contexts_.clear();
    }
    for (auto& deferred : deferred_contexts) {
        if (deferred->ttl_alarm) {
            deferred->ttl_alarm->Cancel();
        }
        deferred->cancel(status);
    }
}

size_t DeferredPrefillContextMap::size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return contexts_.size();
}

PrefillBatchRpcServer::~PrefillBatchRpcServer() {
    beginShutdown();
    deferred_contexts_->cancelAll(grpc::Status(grpc::StatusCode::UNAVAILABLE, "Prefill batch server is shutting down"));
    if (prepare_resource_worker_pool_) {
        prepare_resource_worker_pool_->stop();
        prepare_resource_worker_pool_.reset();
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
    return grpc::Status::OK;
}

// ---------------------------------------------------------------------------
// Batch infrastructure: short-lived resource preparation only
// ---------------------------------------------------------------------------

void PrefillBatchRpcServer::beginShutdown() {
    if (!stopping_.exchange(true)) {
        deferred_contexts_->stopAccepting();
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
    group_request.set_fetch_attach_timeout_ms(request->fetch_attach_timeout_ms());
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
    if (stopping_.load()) {
        for (const auto& item : request->requests()) {
            addBatchError(response,
                          item.has_input() ? item.input().request_id() : 0,
                          grpc::StatusCode::UNAVAILABLE,
                          "Prefill batch server is shutting down");
        }
        return grpc::Status::OK;
    }

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
    const int group_size = static_cast<int>(all_inputs.size());
    for (const auto* input : all_inputs) {
        auto input_copy = std::make_shared<GenerateInputPB>(*input);
        // Worker status derives batch_id from stream metadata; the batch RPC envelope is authoritative.
        input_copy->set_group_size(group_size);
        input_copy->mutable_group_id()->set_value(request->batch_id());

        BatchSlot slot;
        slot.input                   = std::move(input_copy);
        slot.fetch_attach_timeout_ms = request->fetch_attach_timeout_ms();
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

        // Frontend sends FetchResponse only after the master has received this
        // request's success ACK and set enqueued_by_master. The supported
        // frontend therefore cannot Fetch in this interval. Storing first
        // atomically prevents concurrent batches from enqueueing the same ID.
        auto deferred = storeSlot(slot, response);
        if (deferred) {
            ready_slots.push_back(ReadySlot{&slot, std::move(deferred)});
        }
    }

    auto enqueue_status = enqueueGroupStreams(ready_slots, response);
    if (!enqueue_status.ok()) {
        for (auto& ready_slot : ready_slots) {
            rejectSlot(ready_slot, enqueue_status, response);
        }
        return grpc::Status::OK;
    }

    for (auto& ready_slot : ready_slots) {
        publishSlot(ready_slot, response);
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
        if (slot.input->has_group_id()) {
            pfx_ctx->setDispatchGeneration(slot.input->group_id().value());
        }
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
        try {
            auto future =
                prepare_resource_worker_pool_->async([this, slot, result, max_retry_times, max_retry_timeout_ms] {
                    try {
                        int64_t begin_time_us = currentTimeUs();
                        auto    stage         = slot->prefill_context->stat_info.saveStage();
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
            result->stage_status =
                grpc::Status(grpc::StatusCode::INTERNAL, "submit prepare task exception: " + std::string(e.what()));
        } catch (...) {
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
        auto& prefill_context = *ready_slot.deferred->context;
        prefill_context.stat_info.nextStage();
        generate_inputs.push_back(prefill_context.generate_input);
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

    if (enqueue_successes.size() != generate_inputs.size() || streams.size() != generate_inputs.size()) {
        return grpc::Status(grpc::StatusCode::INTERNAL,
                            "enqueueMultiple result size mismatch: input=" + std::to_string(generate_inputs.size())
                                + " status=" + std::to_string(enqueue_successes.size())
                                + " stream=" + std::to_string(streams.size()));
    }

    std::vector<ReadySlot> admitted_slots;
    admitted_slots.reserve(ready_slots.size());
    for (size_t i = 0; i < ready_slots.size(); ++i) {
        auto& stream     = streams[i];
        auto& ready_slot = ready_slots[i];
        if (!stream) {
            return grpc::Status(grpc::StatusCode::INTERNAL, "enqueueMultiple returned null stream");
        }
        if (stream->streamId() != ready_slot.slot->input->request_id()) {
            return grpc::Status(grpc::StatusCode::INTERNAL,
                                "enqueueMultiple result order mismatch: expected request_id="
                                    + std::to_string(ready_slot.slot->input->request_id())
                                    + " actual=" + std::to_string(stream->streamId()));
        }
        ready_slot.deferred->context->setStream(stream);
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

std::shared_ptr<DeferredPrefillContext> PrefillBatchRpcServer::storeSlot(BatchSlot&              slot,
                                                                         EnqueueBatchResponsePB* response) {
    const auto request_id   = slot.input->request_id();
    auto       deferred     = std::make_shared<DeferredPrefillContext>();
    deferred->context       = std::move(slot.prefill_context);
    deferred->input         = slot.input;
    deferred->request_guard = std::move(slot.request_guard);

    const auto store_status = deferred_contexts_->store(request_id, deferred);
    if (!store_status.ok()) {
        deferred->cancel(store_status);
        addBatchError(response, request_id, store_status.error_code(), store_status.error_message());
        return nullptr;
    }
    return deferred;
}

void PrefillBatchRpcServer::publishSlot(ReadySlot& ready_slot, EnqueueBatchResponsePB* response) {
    auto&             slot                 = *ready_slot.slot;
    const auto        request_id           = slot.input->request_id();
    const auto&       deferred             = ready_slot.deferred;
    constexpr int64_t kDefaultContextTtlMs = 10 * 60 * 1000;
    int64_t           ttl_ms = slot.fetch_attach_timeout_ms > 0 ? slot.fetch_attach_timeout_ms : kDefaultContextTtlMs;
    if (deferred->context->request_timeout_ms > 0) {
        const int64_t elapsed_ms   = (currentTimeUs() - deferred->context->request_begin_time_us) / 1000;
        const int64_t remaining_ms = deferred->context->request_timeout_ms - elapsed_ms;
        if (remaining_ms <= 0) {
            rejectSlot(ready_slot,
                       grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED,
                                    "request deadline expired before FetchResponse became available"),
                       response);
            return;
        }
        ttl_ms = std::min(ttl_ms, remaining_ms);
    }
    auto publish_status = deferred_contexts_->armTtl(request_id, deferred, std::chrono::milliseconds(ttl_ms));
    if (!publish_status.ok()) {
        rejectSlot(ready_slot, publish_status, response);
        return;
    }
    addBatchSuccess(response, request_id);
}

void PrefillBatchRpcServer::rejectSlot(ReadySlot&              ready_slot,
                                       const grpc::Status&     status,
                                       EnqueueBatchResponsePB* response) {
    auto&   slot       = *ready_slot.slot;
    int64_t request_id = slot.input->request_id();
    auto    deferred   = deferred_contexts_->remove(request_id, ready_slot.deferred.get());
    if (!deferred) {
        deferred = ready_slot.deferred;
    }
    if (deferred && deferred->context && !deferred->context->cancel_state->load()) {
        deferred->cancel(status);
    }
    ready_slot.deferred.reset();
    addBatchError(response, request_id, status.error_code(), status.error_message());
}

// ---------------------------------------------------------------------------
// FetchResponse — atomically take the stored context and continue its Decode stream
// ---------------------------------------------------------------------------

grpc::Status PrefillBatchRpcServer::FetchResponse(grpc::ServerContext*                   context,
                                                  const FetchRequestPB*                  request,
                                                  grpc::ServerWriter<GenerateOutputsPB>* writer) {
    RTP_LLM_PROFILE_FUNCTION();
    const int64_t                           request_id = request->request_id();
    std::shared_ptr<DeferredPrefillContext> deferred;
    const auto                              take_status = deferred_contexts_->take(request_id, deferred);
    if (!take_status.ok()) {
        return take_status;
    }

    auto& prefill_context              = *deferred->context;
    prefill_context.server_context     = context;
    prefill_context.rpc_context.writer = writer;
    grpc::Status status                = grpc::Status::OK;
    try {
        status = finishStream(prefill_context);
    } catch (const std::exception& e) {
        status =
            grpc::Status(grpc::StatusCode::INTERNAL,
                         "request [" + prefill_context.request_key + "] finishStream exception [" + e.what() + "]");
    } catch (...) {
        status = grpc::Status(grpc::StatusCode::INTERNAL, "finishStream unknown exception");
    }
    if (!status.ok()) {
        prefill_context.error_status = status;
    }
    return status;
}

}  // namespace rtp_llm
