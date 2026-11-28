#include <limits>
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/GrpcAddressUtil.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include <limits.h>
#include <condition_variable>
#include <unordered_map>
#include <unordered_set>
#include <c10/core/DeviceGuard.h>
#include <c10/core/InferenceMode.h>
#include "autil/Scope.h"

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/utils/KVCacheUtils.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"
#include "rtp_llm/cpp/model_rpc/RpcTimeoutUtils.h"
#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/utils/RpcCompletionQueue.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include "rtp_llm/cpp/model_rpc/PDRequestUtils.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/engine_base/Host.h"
#include "autil/NetUtil.h"
#include <cerrno>
#include <cstring>
#include <deque>
#include <unistd.h>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <c10/core/InferenceMode.h>

namespace rtp_llm {

namespace {

constexpr int64_t kMinGrpcPort = 1;
constexpr int64_t kMaxGrpcPort = 65535;

struct ReuseLens {
    int64_t total  = 0;
    int64_t local  = 0;
    int64_t remote = 0;
    int64_t memory = 0;
    int64_t disk   = 0;
};

ReuseLens getPrefillReuseLens(std::shared_ptr<GenerateStream>& stream) {
    ReuseLens reuse_lens;
    reuse_lens.total  = stream->prefillTotalReuseLen();
    reuse_lens.local  = stream->prefillLocalReuseLen();
    reuse_lens.remote = stream->prefillRemoteReuseLen();
    reuse_lens.memory = stream->prefillMemoryReuseLen();
    reuse_lens.disk   = stream->prefillDiskReuseLen();
    return reuse_lens;
}

bool parseGrpcPort(const std::string& port_text, uint32_t* port) {
    if (!port || port_text.empty()) {
        return false;
    }
    char* end             = nullptr;
    errno                 = 0;
    auto       value      = std::strtoll(port_text.c_str(), &end, 10);
    const bool parsed_all = end == port_text.c_str() + port_text.size();
    if (errno != 0 || !parsed_all || value < kMinGrpcPort || value > kMaxGrpcPort) {
        return false;
    }
    *port = static_cast<uint32_t>(value);
    return true;
}

std::string prefillAddress(const GenerateInputPB& request) {
    for (const auto& addr : request.generate_config().role_addrs()) {
        if (addr.role() == RoleAddrPB::PREFILL) {
            return formatGrpcHostPort(addr.ip(), addr.grpc_port());
        }
    }
    return {};
}

}  // namespace

std::string makeDecodeEntranceUniqueKey(const std::string& bind_ip, int64_t unique_key_id, int64_t current_time_us) {
    return bind_ip + "_" + std::to_string(getpid()) + "_" + std::to_string(unique_key_id) + "_"
           + std::to_string(current_time_us);
}

GenerateInputPB makeDecodeEntranceHandoffRequest(const GenerateInputPB& request,
                                                 const std::string&     handoff_unique_key) {
    GenerateInputPB handoff_request;
    handoff_request.CopyFrom(request);
    handoff_request.mutable_generate_config()->set_unique_key(handoff_unique_key);
    return handoff_request;
}

grpc::Status DecodeRpcServer::parsePrefillDpAddr(const std::string& addr, std::string* ip, uint32_t* port) {
    if (!ip || !port || addr.empty()) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "malformed prefill dp_addr: " + addr);
    }

    std::string host;
    std::string port_text;
    if (addr.front() == '[') {
        const auto close_pos = addr.find(']');
        if (close_pos == std::string::npos || close_pos + 1 >= addr.size() || addr[close_pos + 1] != ':') {
            RTP_LLM_LOG_ERROR("malformed prefill dp_addr: %s", addr.c_str());
            return grpc::Status(grpc::StatusCode::INTERNAL, "malformed prefill dp_addr: " + addr);
        }
        host      = addr.substr(0, close_pos + 1);
        port_text = addr.substr(close_pos + 2);
    } else {
        const auto colon_pos = addr.rfind(':');
        if (colon_pos == std::string::npos || colon_pos == 0 || colon_pos + 1 >= addr.size()) {
            RTP_LLM_LOG_ERROR("malformed prefill dp_addr: %s", addr.c_str());
            return grpc::Status(grpc::StatusCode::INTERNAL, "malformed prefill dp_addr: " + addr);
        }
        host      = addr.substr(0, colon_pos);
        port_text = addr.substr(colon_pos + 1);
        if (host.find(':') != std::string::npos) {
            if (host.back() == ':' || host.find('.') != std::string::npos) {
                RTP_LLM_LOG_ERROR("malformed prefill dp_addr: %s", addr.c_str());
                return grpc::Status(grpc::StatusCode::INTERNAL, "malformed prefill dp_addr: " + addr);
            }
            host = "[" + host + "]";
        }
    }

    if (host.empty() || !parseGrpcPort(port_text, port)) {
        RTP_LLM_LOG_ERROR("invalid port in prefill dp_addr: %s", addr.c_str());
        return grpc::Status(grpc::StatusCode::INTERNAL, "invalid port in prefill dp_addr: " + addr);
    }
    *ip = std::move(host);
    return grpc::Status::OK;
}

class DecodeRpcServer::CancelClient: public std::enable_shared_from_this<CancelClient> {
public:
    using Handle = PDCancelRegistry::Handle;

    explicit CancelClient(std::weak_ptr<PDCancelRegistry> registry): registry_(std::move(registry)) {}

    void schedule(const Handle& handle, int64_t delay_ms = 0) {
        RpcCompletionQueue::instance().scheduleAfter(
            delay_ms, [weak = weak_from_this(), request = std::weak_ptr<PDCancelRegistry::Entry>(handle)](bool ok) {
                auto client = weak.lock();
                auto entry  = request.lock();
                if (ok && client && entry) {
                    client->ready_.push_back(request);
                    client->pump();
                }
            });
    }

private:
    template<typename Response, typename Request, typename Start, typename Completion>
    bool submit(const std::shared_ptr<RpcService::Stub>& stub, Request request, Start start, Completion complete) {
        struct Call {
            std::shared_ptr<grpc::ClientContext> context = std::make_shared<grpc::ClientContext>();
            std::unique_ptr<grpc::ClientAsyncResponseReader<Response>> reader;
            std::shared_ptr<RpcService::Stub>                          stub;
            Request                                                    request;
            Response                                                   response;
            grpc::Status                                               status;
        };
        auto call     = std::make_shared<Call>();
        call->stub    = stub;
        call->request = std::move(request);
        call->context->set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(200));
        return RpcCompletionQueue::instance().submit(
            call->context,
            [call, start](grpc::CompletionQueue* queue, void* tag) {
                call->reader = start(*call->stub, call->context.get(), call->request, queue);
                if (!call->reader)
                    return false;
                call->reader->Finish(&call->response, &call->status, tag);
                return true;
            },
            [call, complete](bool ok) { complete(ok && call->status.ok(), call->response); });
    }

    void pump() {
        // Preserve one in-flight Prefill cancellation round per Decode server.
        // CQ callbacks own this queue; callers only post events.
        auto registry = registry_.lock();
        while (registry && !active_ && !ready_.empty()) {
            auto handle = ready_.front().lock();
            ready_.pop_front();
            if (!handle || !registry->needsDownstreamCancel(handle))
                continue;
            auto connection = pool_.getConnection(handle->downstream_address);
            if (!connection.ok()) {
                retry(handle);
                continue;
            }
            CancelRequestPB request;
            request.set_request_id(handle->identity.request_id);
            request.set_cancel_error_code(static_cast<int64_t>(handle->cancel_reason.code()));
            request.set_cancel_error_message(handle->cancel_reason.ToString());
            const auto& stub = connection->stub;
            active_          = submit<CancelResponsePB>(
                stub,
                std::move(request),
                [](auto& rpc, auto* context, const auto& request, auto* queue) {
                    return rpc.AsyncCancel(context, request, queue);
                },
                [weak = weak_from_this(), handle, stub](bool ok, const CancelResponsePB& response) {
                    auto client = weak.lock();
                    if (!client)
                        return;
                    const bool confirmed = ok
                                           && (response.status() == CANCEL_STATUS_TOMBSTONED
                                               || response.status() == CANCEL_STATUS_NOT_FOUND);
                    if (ok && response.status() == CANCEL_STATUS_ACCEPTED && client->queryStatus(handle, stub))
                        return;
                    client->finishRound(handle, confirmed);
                });
            if (!active_)
                retry(handle);
        }
    }

    bool queryStatus(const Handle& handle, const std::shared_ptr<RpcService::Stub>& stub) {
        auto registry = registry_.lock();
        if (!registry || !registry->needsDownstreamCancel(handle))
            return false;
        StatusVersionPB request;
        request.set_latest_finished_version(0);
        return submit<WorkerStatusPB>(
            stub,
            std::move(request),
            [](auto& rpc, auto* context, const auto& request, auto* queue) {
                return rpc.AsyncGetWorkerStatus(context, request, queue);
            },
            [weak = weak_from_this(), handle](bool ok, const WorkerStatusPB& status) {
                auto client = weak.lock();
                if (!client)
                    return;
                const bool confirmed =
                    ok
                    && std::any_of(status.finished_task_list().begin(),
                                   status.finished_task_list().end(),
                                   [&](const auto& task) { return task.request_id() == handle->identity.request_id; });
                client->finishRound(handle, confirmed);
            });
    }

    void retry(const Handle& handle) {
        auto registry = registry_.lock();
        if (registry && registry->needsDownstreamCancel(handle)) {
            const auto delay = handle->cancel_retry_delay_ms.load();
            handle->cancel_retry_delay_ms.store(std::min<int64_t>(delay * 2, 1000));
            schedule(handle, delay);
        }
    }

    void finishRound(const Handle& handle, bool confirmed) {
        if (confirmed) {
            if (auto registry = registry_.lock())
                registry->finishDownstream(handle);
        } else {
            retry(handle);
        }
        active_ = false;
        pump();
    }

    std::weak_ptr<PDCancelRegistry>                    registry_;
    Pool<RpcService>                                   pool_;
    std::deque<std::weak_ptr<PDCancelRegistry::Entry>> ready_;
    bool                                               active_{false};
};

DecodeRpcServer::~DecodeRpcServer() = default;

void DecodeRpcServer::initCancelRegistry() {
    cancel_registry_ = std::make_shared<PDCancelRegistry>(meta_);
    cancel_client_   = std::make_shared<CancelClient>(cancel_registry_);
    cancel_registry_->setCancelStarted([weak = std::weak_ptr<CancelClient>(cancel_client_)](const auto& handle) {
        if (auto client = weak.lock())
            client->schedule(handle);
    });
}

grpc::Status DecodeRpcServer::Cancel(grpc::ServerContext*, const CancelRequestPB* request, CancelResponsePB* response) {
    if (!request || request->request_id() <= 0 || !response)
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "cancel request missing request_id");
    if (!cancel_registry_)
        return grpc::Status(grpc::StatusCode::UNAVAILABLE, "Decode is not initialized");
    auto handle = cancel_registry_->find(request->request_id());
    if (!handle && !request->prefill_address().empty()) {
        // Create a control-only record so retries can finish Prefill cancellation
        // even when the frontend has not connected to Decode yet.
        GenerateInputPB pending_request;
        pending_request.set_request_id(request->request_id());
        const auto admission = cancel_registry_->admit(pending_request,
                                                       masterEnqueuedHandoffUniqueKey(request->request_id()),
                                                       request->prefill_address(),
                                                       currentTimeMs() + 10 * 60 * 1000,
                                                       handle,
                                                       /*control_only=*/true);
        if (!admission.ok())
            handle = cancel_registry_->find(request->request_id());
        if (admission.ok())
            cancel_registry_->finishLocal(handle);
    }
    if (handle && handle->terminal.load()) {
        // Internal forwarding may retry after WorkerStatus's short delta window.
        // This retained terminal proves Decode physical cleanup; Prefill cleanup
        // remains in the independent retry record if it is still unreachable.
        response->set_status(CANCEL_STATUS_NOT_FOUND);
    } else {
        const auto cancellation = cancel_registry_->cancel(
            request->request_id(),
            {request->cancel_error_code() == 0 ? ErrorCode::PRIORITY_PREEMPTED :
                                                 static_cast<ErrorCode>(request->cancel_error_code()),
             request->cancel_error_message().empty() ? "PD request canceled" : request->cancel_error_message()});
        if (!cancellation.ok()) {
            return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, std::string(cancellation.status().message()));
        }
        response->set_status(cancellation.value());
    }
    return grpc::Status::OK;
}

grpc::Status DecodeRpcServer::init(const EngineInitParams&                                maga_init_params,
                                   std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params,
                                   py::object                                             mm_process_engine) {
    auto ret = LocalRpcServer::init(maga_init_params, std::move(propose_params), mm_process_engine);
    if (!ret.ok()) {
        RTP_LLM_LOG_ERROR("decode rpc server new2 init failed, err: %s", ret.error_message().c_str());
        return ret;
    }

    auto kvcache_manager = engine_->getCacheManager();
    if (!kvcache_manager) {
        RTP_LLM_LOG_WARNING("decode rpc server new2 init failed, kvcache manager is null");
        return grpc::Status(grpc::StatusCode::INTERNAL, "kvcache manager is null");
    }
    if (!kvcache_manager->hasP2PConnector()) {
        RTP_LLM_LOG_WARNING("decode rpc server new2 init failed, PD requires P2P connector");
        return grpc::Status(grpc::StatusCode::INTERNAL, "PD requires P2P connector");
    }

    prefill_server_caller_ = std::make_shared<PrefillServerCaller>();
    initCancelRegistry();

    RTP_LLM_LOG_INFO("decode rpc server new2 init");
    return grpc::Status::OK;
}

grpc::Status DecodeRpcServer::BatchGenerateCall(grpc::ServerContext*        context,
                                                const BatchGenerateInputPB* request,
                                                BatchGenerateOutputsPB*     response) {
    c10::InferenceMode inference_guard(true);
    AtomicGuard request_guard(onflight_requests_);
    if (!request || !response)
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "batch request and response are required");
    response->Clear();
    if (request->inputs().empty())
        return grpc::Status::OK;
    if (!engine_ || !prefill_server_caller_ || !cancel_registry_)
        return grpc::Status(grpc::StatusCode::UNAVAILABLE, "Decode is not initialized");

    const auto entry_ms = currentTimeMs();
    const auto address  = prefillAddress(request->inputs(0));
    if (address.empty())
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "batch requires a Prefill address");
    std::unordered_set<int64_t> request_ids;
    bool                        all_pd            = true;
    bool                        master_admitted   = false;
    bool                        batch_has_unbounded_request = false;
    int64_t                     batch_deadline_ms = entry_ms;
    BatchGenerateInputPB        effective;
    for (const auto& item : request->inputs()) {
        const bool multimodal =
            item.multimodal_inputs_size() != 0 || item.has_v41_inputs() || item.has_multimodal_token_layout();
        if (item.request_id() <= 0 || !request_ids.insert(item.request_id()).second)
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "batch requires distinct positive request_ids");
        if (prefillAddress(item) != address)
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "batch items require the same Prefill address");
        // Prefill's local batch path owns multimodal preprocessing and execution.
        all_pd          = all_pd && !multimodal && (item.enqueued_by_master() || checkPDSupport(item).supported);
        master_admitted = master_admitted || item.enqueued_by_master();
        auto* copy      = effective.add_inputs();
        copy->CopyFrom(item);
        const auto timeout = clampRpcTimeoutMsToInt32(configuredRpcTimeoutMs(
            item.generate_config().timeout_ms(), maga_init_params_.pd_sep_config.max_rpc_timeout_ms));
        copy->mutable_generate_config()->set_timeout_ms(timeout);
        batch_has_unbounded_request = batch_has_unbounded_request || timeout == 0;
        batch_deadline_ms = std::max(batch_deadline_ms, entry_ms + (timeout > 0 ? timeout : kDefaultRpcTimeoutMs));
    }
    const auto client_cancelled = [context]() { return context && context->IsCancelled(); };
    const bool single_dp        = maga_init_params_.parallelism_config.dp_size == 1;
    if ((!all_pd || !single_dp) && !master_admitted) {
        // Match single-request fallback, while retaining the existing batch result contract.
        for (auto& item : *effective.mutable_inputs())
            item.mutable_generate_config()->set_can_use_pd_separation(false);
        return prefill_server_caller_->forwardBatchGenerate(
            effective, response, address, batch_has_unbounded_request ? 0 : batch_deadline_ms, client_cancelled);
    }
    // Preserve client semantics before adding P2P-only keys, groups and handoff timeouts.
    const auto fallback_request = effective;
    struct BatchContextGuard {
        PDCancelRegistry&                             registry;
        EngineBase&                                   engine;
        std::vector<PDCancelRegistry::Handle>         handles;
        std::vector<std::unique_ptr<GenerateContext>> contexts;
        std::vector<GenerateStreamPtr>                streams;
        std::unordered_set<int64_t>                   submitted_prefill;
        grpc::Status                                  status;
        size_t                                        enqueued_stream_count = 0;
        bool                                          completed             = false;
        bool                                          cleaned               = false;
        bool                                          forwarded             = false;
        ~BatchContextGuard() {
            cleanup();
        }
        void cleanup() {
            if (cleaned)
                return;
            cleaned            = true;
            const auto cleanup = [](auto&& operation) noexcept {
                try {
                    operation();
                } catch (...) {
                    // Continue cleanup for the other items and stages.
                }
            };
            if (!completed) {
                ErrorInfo reason(ErrorCode::CANCELLED, "");
                cleanup([&] {
                    reason = status.ok() ? ErrorInfo(ErrorCode::CANCELLED, "batch generation abandoned") :
                                           errorInfoFromGrpcStatus(status);
                });
                // Also cancel attempted admissions when the EnqueueBatch ACK was lost.
                for (const auto& handle : handles) {
                    cleanup([&] {
                        const auto request_id = handle->identity.request_id;
                        if (submitted_prefill.count(request_id)) {
                            const auto cancellation = registry.cancel(request_id, reason);
                            if (!cancellation.ok()) {
                                RTP_LLM_LOG_WARNING("PD batch cleanup cancel failed, request_id=%ld status=%s",
                                                    request_id,
                                                    cancellation.status().ToString().c_str());
                            }
                        } else {
                            registry.finishDownstream(handle);
                        }
                    });
                }
                for (const auto& stream : streams) {
                    cleanup([&] {
                        if (!stream->hasError() && stream->getStatus() != StreamState::FINISHED)
                            stream->reportError(reason.code(), reason.ToString());
                    });
                }
                for (auto& ctx : contexts) {
                    cleanup([&] {
                        ctx->error_info   = reason;
                        ctx->error_status = grpcStatusFromErrorInfo(reason);
                    });
                }
                // Canceled streams still need the engine's terminal/resource cleanup.
                // Resume after any partial submission without enqueueing a stream twice.
                for (size_t i = enqueued_stream_count; i < streams.size(); ++i) {
                    cleanup([&] {
                        if (forwarded)
                            streams[i]->finishWithoutGenerate();
                        else
                            engine.enqueue(streams[i]);
                    });
                }
            }
            for (auto& ctx : contexts) {
                cleanup([&] { ctx->markRpcHandlingCompleted(); });
            }
            contexts.clear();
            for (const auto& handle : handles) {
                if (completed && !handle->canceled.load())
                    cleanup([&] { registry.finishDownstream(handle); });
                cleanup([&] { registry.finishLocal(handle); });
            }
        }
    } guard{*cancel_registry_, *engine_, {}, {}, {}, {}, grpc::Status::OK};

    try {
        EnqueueBatchRequestPB prefill_request;
        prefill_request.set_batch_id(request->inputs(0).request_id());
        auto* slot = prefill_request.add_dp_slots();
        slot->set_dp_rank(0);
        // The batch validated that all items target this exact Prefill endpoint.
        std::optional<PrefillPeerInfo> peer;
        std::unordered_map<int64_t, int> response_indices;
        std::unordered_map<int64_t, PDCancelRegistry::Handle> handles_by_id;
        std::vector<GenerateStreamPtr> output_streams(request->inputs_size());
        std::vector<GenerateContext*> output_contexts(request->inputs_size(), nullptr);
        const auto fail_item = [&](int64_t request_id, const ErrorInfo& error) {
            const auto index = response_indices.at(request_id);
            auto* item_error = response->mutable_results(index)->mutable_error_info();
            item_error->set_error_code(transErrorCodeToRPC(error.code()));
            item_error->set_error_message(error.ToString());
            if (auto* context = output_contexts[index])
                context->recordRpcFailure(error);
            if (const auto& stream = output_streams[index]; stream && !stream->hasError())
                stream->reportError(error.code(), error.ToString());
            const auto handle = handles_by_id.find(request_id);
            if (handle != handles_by_id.end()) {
                if (guard.submitted_prefill.count(request_id)) {
                    (void)cancel_registry_->cancel(request_id, error);
                } else {
                    cancel_registry_->finishDownstream(handle->second);
                }
            }
        };
        for (auto& item : *effective.mutable_inputs()) {
            response_indices.emplace(item.request_id(), response->results_size());
            response->add_results();
            auto* config = item.mutable_generate_config();
            config->set_unique_key(masterEnqueuedHandoffUniqueKey(item.request_id()));
            const auto configured_timeout = config->timeout_ms();
            if (configured_timeout == 0) {
                config->set_timeout_ms(kDefaultRpcTimeoutMs);
            }
            if (!item.has_group_id())
                item.mutable_group_id()->set_value(prefill_request.batch_id());
            item.set_group_size(request->inputs_size());
            const auto               deadline_ms = entry_ms + item.generate_config().timeout_ms();
            PDCancelRegistry::Handle handle;
            const auto               admission =
                cancel_registry_->admit(item, item.generate_config().unique_key(), address, deadline_ms, handle);
            if (!admission.ok()) {
                fail_item(item.request_id(), admission);
                continue;
            }
            guard.handles.push_back(handle);
            handles_by_id.emplace(item.request_id(), handle);
            if (item.enqueued_by_master())
                guard.submitted_prefill.insert(item.request_id());

            if (config->return_logprobs()) {
                fail_item(item.request_id(), {ErrorCode::INVALID_PARAMS, "PD batch does not support compact logprobs"});
                continue;
            }

            const bool multimodal =
                item.multimodal_inputs_size() != 0 || item.has_v41_inputs() || item.has_multimodal_token_layout();
            if (multimodal) {
                fail_item(item.request_id(),
                          {ErrorCode::INVALID_PARAMS, "P2P batch inference cannot attach this multimodal item"});
                continue;
            }

            std::shared_ptr<GenerateInput> input;
            GenerateInputPB                prefill_input;
            try {
                const auto status = preparePDRequest(item, deadline_ms, input, prefill_input, peer, context);
                if (!status.ok()) {
                    fail_item(item.request_id(), errorInfoFromGrpcStatus(status));
                    continue;
                }
            } catch (const std::exception& error) {
                fail_item(item.request_id(), {ErrorCode::EXECUTION_EXCEPTION, error.what()});
                continue;
            }
            input->request_deadline_is_handoff_only = configured_timeout == 0;
            input->generate_config->timeout_ms      = configured_timeout;
            auto stream = engine_->makeStream(input);
            stream->streamCacheResource().setPDDecodeLinearTailOnly(
                maga_init_params_.pd_sep_config.decode_linear_tail_only);
            stream->setPrefillTpSize(peer->tp_size);
            stream->setPrefillCpSize(peer->cp_size);
            stream->setPrefillPipelineLayout(peer->pipeline_layout);
            guard.streams.push_back(stream);
            output_streams[response_indices.at(item.request_id())] = stream;
            cancel_registry_->attach(handle, stream);
            auto ctx = std::make_unique<GenerateContext>(
                item.request_id(), configured_timeout, context, metrics_reporter_, meta_);
            ctx->setStream(stream);
            output_contexts[response_indices.at(item.request_id())] = ctx.get();
            guard.contexts.push_back(std::move(ctx));
            if (!item.enqueued_by_master()) {
                auto* input_pb = slot->add_requests()->mutable_input();
                input_pb->Swap(&prefill_input);
            }
        }
        if (client_cancelled())
            return guard.status = grpcStatusFromErrorInfo({ErrorCode::CANCELLED, "batch cancelled"});
        auto* prefill_items = slot->mutable_requests();
        for (int i = prefill_items->size() - 1; i >= 0; --i) {
            auto* item = prefill_items->Mutable(i)->mutable_input();
            const auto request_id = item->request_id();
            const auto remaining = entry_ms + item->generate_config().timeout_ms() - currentTimeMs();
            const auto& handle = handles_by_id.at(request_id);
            if (handle->canceled.load() || remaining <= 0) {
                fail_item(request_id, handle->canceled.load() ? handle->cancel_reason :
                    ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "batch item expired before Prefill admission"));
                prefill_items->DeleteSubrange(i, 1);
            } else {
                item->mutable_generate_config()->set_timeout_ms(clampRpcTimeoutMsToInt32(remaining));
            }
        }
        if (!prefill_items->empty()) {
            std::unordered_set<int64_t> attempted;
            for (const auto& external : *prefill_items) {
                attempted.insert(external.input().request_id());
                guard.submitted_prefill.insert(external.input().request_id());
            }
            EnqueueBatchResponsePB ack;
            const auto             status = prefill_server_caller_->enqueueBatch(
                prefill_request, &ack, address, batch_deadline_ms, client_cancelled);
            if (client_cancelled())
                return guard.status = grpcStatusFromErrorInfo({ErrorCode::CANCELLED, "batch cancelled"});
            if (status.error_code() == grpc::StatusCode::UNIMPLEMENTED && !master_admitted) {
                // UNIMPLEMENTED admits no P2P items, but the request is still live.
                // Keep the control records and contexts until the forwarded RPC returns:
                // cleanup here would publish CANCELLED and disable downstream Cancel.
                BatchGenerateInputPB forwarded;
                for (const auto& original : fallback_request.inputs()) {
                    if (!attempted.count(original.request_id()))
                        continue;
                    auto* item = forwarded.add_inputs();
                    item->CopyFrom(original);
                    auto* config = item->mutable_generate_config();
                    config->set_can_use_pd_separation(false);
                    if (config->timeout_ms() > 0) {
                        config->set_timeout_ms(clampRpcTimeoutMsToInt32(
                            std::max<int64_t>(1, entry_ms + config->timeout_ms() - currentTimeMs())));
                    }
                }
                BatchGenerateOutputsPB forwarded_response;
                guard.forwarded = true;
                guard.status =
                    prefill_server_caller_->forwardBatchGenerate(forwarded,
                                                                 &forwarded_response,
                                                                 address,
                                                                 batch_has_unbounded_request ? 0 : batch_deadline_ms,
                                                                 client_cancelled);
                if (guard.status.ok() && forwarded_response.results_size() != forwarded.inputs_size())
                    guard.status = grpc::Status(grpc::StatusCode::INTERNAL, "Prefill fallback result size mismatch");
                if (guard.status.ok()) {
                    for (int i = 0; i < forwarded.inputs_size(); ++i)
                        response->mutable_results(response_indices.at(forwarded.inputs(i).request_id()))
                            ->Swap(forwarded_response.mutable_results(i));
                }
                // These Decode streams were only prepared, never submitted to the engine.
                // Mirror the actual result before releasing them and publishing completion.
                for (const auto& stream : guard.streams) {
                    const auto index = response_indices.at(stream->generateInput()->request_id);
                    if (!guard.status.ok()) {
                        const auto error = errorInfoFromGrpcStatus(guard.status);
                        stream->reportError(error.code(), error.ToString());
                    } else if (index < response->results_size() && response->results(index).has_error_info()) {
                        const auto& error = response->results(index).error_info();
                        stream->reportError(transRPCErrorCode(error.error_code()), error.error_message());
                    }
                    stream->finishWithoutGenerate();
                }
                guard.enqueued_stream_count = guard.streams.size();
                guard.completed             = guard.status.ok() && !client_cancelled();
                return guard.status;
            }
            std::unordered_set<int64_t> accepted;
            if (status.ok() && ack.batch_id() == prefill_request.batch_id()) {
                for (const auto& success : ack.successes())
                    accepted.insert(success.request_id());
                for (const auto& error : ack.errors()) {
                    const auto request_id = error.request_id();
                    if (!attempted.count(request_id)
                        || response->results(response_indices.at(request_id)).has_error_info())
                        continue;
                    // A contradictory ACK leaves admission uncertain, so cancel it.
                    if (!accepted.erase(request_id))
                        guard.submitted_prefill.erase(request_id);
                    const auto code = static_cast<ErrorCode>(error.error_info().error_code());
                    fail_item(request_id,
                              {code == ErrorCode::NONE_ERROR ? ErrorCode::EXECUTION_EXCEPTION : code,
                               error.error_info().error_message()});
                }
            }
            for (const auto& external : *prefill_items) {
                const auto request_id = external.input().request_id();
                if (!accepted.count(request_id)
                    && !response->results(response_indices.at(request_id)).has_error_info()) {
                    fail_item(request_id,
                              status.ok() ? ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "Prefill batch ACK mismatch") :
                                            errorInfoFromGrpcStatus(status));
                }
            }
        }
        for (auto& stream : guard.streams) {
            const auto request_id = stream->generateInput()->request_id;
            const auto index = response_indices.at(request_id);
            const auto& handle = handles_by_id.at(request_id);
            if (handle->canceled.load() && !response->results(index).has_error_info())
                fail_item(request_id, handle->cancel_reason);
            if (response->results(index).has_error_info()) {
                stream->finishWithoutGenerate();
                output_streams[index].reset();
            } else {
                engine_->enqueue(stream);
            }
            ++guard.enqueued_stream_count;
        }
        guard.status = pollBatchStreamOutput(context, output_streams, response, fail_item);
        guard.completed = guard.status.ok() && !client_cancelled();
        return guard.status;
    } catch (const std::exception& e) {
        return guard.status = serializeErrorMsg("batch", {ErrorCode::EXECUTION_EXCEPTION, e.what()});
    } catch (...) {
        return guard.status = serializeErrorMsg("batch", {ErrorCode::EXECUTION_EXCEPTION, "unknown batch exception"});
    }
}

grpc::Status DecodeRpcServer::preparePDRequest(const GenerateInputPB&          request,
                                               int64_t                         deadline_ms,
                                               std::shared_ptr<GenerateInput>& input,
                                               GenerateInputPB&                prefill_request,
                                               std::optional<PrefillPeerInfo>& peer_info,
                                               grpc::ServerContext*            server_context) {
    const auto& config = request.generate_config();
    if (config.return_logprobs() && !config.is_streaming()) {
        return serializeErrorMsg(std::to_string(request.request_id()),
                                 {ErrorCode::INVALID_PARAMS, "PD compact logprobs require streaming output"});
    }
    input                      = QueryConverter::transQuery(&request);
    input->request_deadline_ms = deadline_ms;
    const auto support         = checkPDSupport(request);
    if (!support.supported) {
        return serializeErrorMsg(std::to_string(request.request_id()), {ErrorCode::INVALID_PARAMS, support.reason});
    }
    // Preserve the configured duration on the wire; D keeps its own deadline.
    input->generate_config->timeout_ms = request.generate_config().timeout_ms();
    auto status = preprocessForPD(
        input, mm_processor_.get(), engine_->isMTPEagle(), RoleType::DECODE, server_context, {});
    if (!status.ok())
        return serializeErrorMsg(std::to_string(request.request_id()), status);
    prefill_request.CopyFrom(request);
    prefill_request.mutable_generate_config()->set_timeout_ms(input->generate_config->timeout_ms);
    if (!request.enqueued_by_master() && !request.has_multimodal_token_layout() && !request.has_v41_inputs()
        && input->multimodal_features && input->mm_locs) {
        // Local ViT has already expanded the tokens used to register Decode's KV.
        // Prefill validates the same spans against its fetched embeddings.
        const auto* token_ids = input->input_ids.data_ptr<int32_t>();
        auto* wire_tokens = prefill_request.mutable_token_ids();
        wire_tokens->Clear();
        wire_tokens->Reserve(input->input_ids.numel());
        for (int64_t i = 0; i < input->input_ids.numel(); ++i) {
            wire_tokens->Add(token_ids[i]);
        }
        auto* layout = prefill_request.mutable_multimodal_token_layout();
        for (size_t i = 0; i < input->multimodal_features->size(); ++i) {
            auto* span = layout->add_spans();
            span->set_offset(input->mm_locs->data_ptr<int32_t>()[i]);
            span->set_length(input->multimodal_features->at(i).size(0));
        }
    }
    const auto remaining = deadline_ms - currentTimeMs();
    if (remaining <= 0)
        return grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "PD preprocessing deadline exceeded");
    if (peer_info) {
        return grpc::Status::OK;
    }
    const auto  address = prefillAddress(request);
    std::string ip;
    uint32_t    port         = 0;
    auto        parse_status = parsePrefillDpAddr(address, &ip, &port);
    if (!parse_status.ok())
        return parse_status;
    auto peer_result = prefill_server_caller_->getPrefillPeerInfo(
        ip, port, clampRpcTimeoutMsToInt32(remaining), maga_init_params_.parallelism_config.pp_size);
    if (!peer_result.ok())
        return grpcStatusFromErrorInfo(peer_result.status());
    if (peer_result.value().tp_size <= 0 || peer_result.value().cp_size <= 0) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "prefill peer info is not available");
    }
    peer_info = std::move(peer_result.value());
    return grpc::Status::OK;
}

grpc::Status DecodeRpcServer::GenerateStreamCall(grpc::ServerContext*                   server_context,
                                                 const GenerateInputPB*                 request,
                                                 grpc::ServerWriter<GenerateOutputsPB>* response_writer) try {
    if (!request) {
        return serializeErrorMsg("unknown", ErrorInfo(ErrorCode::INVALID_PARAMS, "null generate request"));
    }
    AtomicGuard request_guard(onflight_requests_);
    struct PrefillContextGuard {
        std::shared_ptr<PrefillServerCallerContext> context;

        ~PrefillContextGuard() {
            if (!context) {
                return;
            }
            try {
                if (!context->done()) {
                    context->cancel();
                }
            } catch (...) {
                // Waiting is still necessary once the call has been started.
            }
            try {
                context->wait();
            } catch (...) {
                // PrefillServerCallerContext owns completion queue shutdown.
            }
        }
    };

    const int64_t request_entry_ms          = currentTimeMs();
    const auto    configured_timeout_ms     = clampRpcTimeoutMsToInt32(configuredRpcTimeoutMs(
        request->generate_config().timeout_ms(), maga_init_params_.pd_sep_config.max_rpc_timeout_ms));
    const auto    handoff_timeout_ms        = configured_timeout_ms > 0 ? configured_timeout_ms : kDefaultRpcTimeoutMs;

    // Always isolate decode-entrance P2P handoff from any caller-provided business unique_key.
    // The external unique_key remains on the original request object, while the internal handoff
    // request gets a per-request key used only by the prefill/decode P2P pipeline.
    GenerateInputPB request_with_handoff_key;
    const auto*     effective_request = request;

    // Check if pd separation should be used
    // A successful Master batch admission is authoritative: Prefill has
    // already entered the P2P path, so Decode must attach to that handoff.
    const bool needs_prefill_multimodal_processing = request->multimodal_inputs_size() > 0 && !request->has_v41_inputs()
                                                     && !request->has_multimodal_token_layout() && !mm_processor_;
    const bool pd_separation =
        request->enqueued_by_master() || (checkPDSupport(*request).supported && !needs_prefill_multimodal_processing);
    if (pd_separation) {
        std::string handoff_unique_key;
        if (request->enqueued_by_master()) {
            handoff_unique_key = masterEnqueuedHandoffUniqueKey(request->request_id());
        } else {
            const auto handoff_id = unique_key_id_.fetch_add(1);
            handoff_unique_key = makeDecodeEntranceUniqueKey(autil::NetUtil::getBindIp(), handoff_id, currentTimeUs());
        }
        request_with_handoff_key = makeDecodeEntranceHandoffRequest(*request, handoff_unique_key);
        request_with_handoff_key.mutable_generate_config()->set_timeout_ms(handoff_timeout_ms);
        effective_request = &request_with_handoff_key;
    }
    if (!pd_separation) {
        RTP_LLM_LOG_DEBUG("pd separation is disabled, call prefill server");
        if (!cancel_registry_ || !prefill_server_caller_) {
            return grpc::Status(grpc::StatusCode::UNAVAILABLE, "Decode is not initialized");
        }
        PDCancelRegistry::Handle handle;
        const auto admission = cancel_registry_->admit(*request,
                                                       "",
                                                       prefillAddress(*request),
                                                       request_entry_ms + handoff_timeout_ms,
                                                       handle);
        if (!admission.ok()) {
            return serializeErrorMsg(std::to_string(request->request_id()),
                                     QueryConverter::transRequestInfo(request->request_info()),
                                     admission);
        }
        PDCancelRegistry::CallGuard call_guard(*cancel_registry_, handle);
        GenerateInputPB prefill_forward_request;
        prefill_forward_request.CopyFrom(*effective_request);
        prefill_forward_request.mutable_generate_config()->set_timeout_ms(configured_timeout_ms);
        prefill_forward_request.mutable_generate_config()->set_can_use_pd_separation(false);
        const auto status = prefill_server_caller_->callPrefill(server_context, &prefill_forward_request, response_writer);
        if (!handle->canceled.load()) {
            cancel_registry_->finishDownstream(handle);
        }
        return status;
    }

    const auto log_pd_error = [&](const char* stage, const grpc::Status& status) {
        const auto error = errorInfoFromGrpcStatus(status);
        const auto& trace_id = request->request_info().trace_id().empty()
                                   ? request->generate_config().trace_id().value()
                                   : request->request_info().trace_id();
        if (status.error_code() == grpc::StatusCode::CANCELLED && error.code() == ErrorCode::CANCELLED) {
            RTP_LLM_LOG_INFO("P2P Decode request cancelled, stage=%s request_id=%ld unique_key=%s trace_id=%s error=%s",
                             stage,
                             request->request_id(),
                             effective_request->generate_config().unique_key().c_str(),
                             trace_id.c_str(),
                             error.ToString().c_str());
            return;
        }
        RTP_LLM_LOG_WARNING("P2P Decode request failed, stage=%s request_id=%ld business_unique_key=%s unique_key=%s "
                            "grpc_code=%d error_code=%d error_name=%s trace_id=%s error=%s",
                            stage,
                            request->request_id(),
                            request->generate_config().unique_key().c_str(),
                            effective_request->generate_config().unique_key().c_str(),
                            static_cast<int>(status.error_code()),
                            static_cast<int>(error.code()),
                            ErrorCodeToString(error.code()).c_str(),
                            trace_id.c_str(),
                            error.ToString().c_str());
    };

    auto        request_id = request->request_id();
    RTP_LLM_LOG_DEBUG("receive request %ld", request_id);
    PDCancelRegistry::Handle cancel_handle;
    std::optional<PDCancelRegistry::CallGuard> cancel_guard;
    auto                                       generate_context =
        GenerateContext(request_id, configured_timeout_ms, server_context, metrics_reporter_, meta_);
    generate_context.onflight_requests = &onflight_requests_;
    generate_context.request_info = QueryConverter::transRequestInfo(request->request_info());
    if (generate_context.request_info.trace_id.empty()) {
        generate_context.request_info.trace_id = request->generate_config().trace_id().value();
    }
    if (generate_context.request_info.request_id.empty()) {
        generate_context.request_info.request_id = std::to_string(request_id);
    }
    if (telemetry::TelemetryRuntime::isActive()) {
        auto span = telemetry::startRpcServerSpan(
            "rtp_llm.generate_stream_call", server_context, true, "RpcService/GenerateStreamCall");
        generate_context.trace_span_guard =
            std::make_unique<telemetry::GrpcStatusSpanGuard>(span, &generate_context.error_status);
        generate_context.trace_span_guard->setAttribute(telemetry::kAttrRequestId, std::to_string(request_id));
        generate_context.trace_span_guard->setAttribute(telemetry::kAttrGenAiPdRole, telemetry::kValPdRoleConsumer);
    }
    telemetry::PhaseSpanSynthesisScope phase_span_scope([&generate_context](bool exception_unwinding) {
        generate_context.synthesizeTracePhases(telemetry::PhaseRole::Decode, exception_unwinding);
    });
    const int uncaught_exceptions = std::uncaught_exceptions();
    autil::ScopeGuard rpc_completion_guard([&generate_context, uncaught_exceptions] {
        if (std::uncaught_exceptions() == uncaught_exceptions) {
            generate_context.markRpcHandlingCompleted();
        }
    });
    const auto                         admission = cancel_registry_->admit(*effective_request,
                                                   effective_request->generate_config().unique_key(),
                                                   prefillAddress(*effective_request),
                                                   request_entry_ms + handoff_timeout_ms,
                                                   cancel_handle);
    if (!admission.ok()) {
        generate_context.error_info = admission;
        generate_context.error_status = serializeErrorMsg(generate_context.request_key, generate_context.request_info, admission);
        return generate_context.error_status;
    }
    cancel_guard.emplace(*cancel_registry_, cancel_handle);
    struct PrefillCancelGuard {
        PDCancelRegistry&        registry;
        PDCancelRegistry::Handle handle;
        GenerateContext&         context;
        bool                     completed{false};
        void                     cancel() {
            auto reason = errorInfoFromGrpcStatus(context.error_status);
            if (!reason.hasError())
                reason = {ErrorCode::CANCELLED, "PD request abandoned before completion"};
            const auto request_id   = handle->identity.request_id;
            const auto cancellation = registry.cancel(request_id, reason);
            if (!cancellation.ok()) {
                RTP_LLM_LOG_WARNING("PD stream cleanup cancel failed, request_id=%ld status=%s",
                                    request_id,
                                    cancellation.status().ToString().c_str());
            }
            completed = true;
        }
        ~PrefillCancelGuard() {
            if (!completed)
                cancel();
        }
    } prefill_cancel_guard{*cancel_registry_, cancel_handle, generate_context};
    try {
        std::shared_ptr<GenerateInput> input;
        GenerateInputPB                prefill_request;
        std::optional<PrefillPeerInfo> peer_info;
        const auto                     prepare_start_us = currentTimeUs();
        auto                           prepare_status   = preparePDRequest(*effective_request,
                                               request_entry_ms + handoff_timeout_ms,
                                               input,
                                               prefill_request,
                                               peer_info,
                                               server_context);
        {
            RpcMetricsCollector collector;
            collector.min_response_done_time_us      = 0;
            collector.prepare_generate_context_rt_us = currentTimeUs() - prepare_start_us;
            generate_context.reportMetrics(collector);
        }
        if (!prepare_status.ok()) {
            generate_context.error_info = errorInfoFromGrpcStatus(prepare_status);
            generate_context.error_status = serializeErrorMsg(
                generate_context.request_key, generate_context.request_info, generate_context.error_info);
            log_pd_error("prepare", generate_context.error_status);
            return generate_context.error_status;
        }
        if (cancel_handle->canceled.load()) {
            generate_context.error_info   = cancel_handle->cancel_reason;
            generate_context.error_status = serializeErrorMsg(std::to_string(request_id), generate_context.request_info, generate_context.error_info);
            return generate_context.error_status;
        }
        input->request_deadline_is_handoff_only = configured_timeout_ms == 0;
        input->generate_config->timeout_ms      = configured_timeout_ms;
        auto stream = engine_->makeStream(input);
        stream->streamCacheResource().setPDDecodeLinearTailOnly(
            maga_init_params_.pd_sep_config.decode_linear_tail_only);
        // Until enqueue succeeds, no scheduler owns this stream's terminal transition.
        struct UnenqueuedStreamGuard {
            GenerateStreamPtr stream;
            bool submitted = false;
            ~UnenqueuedStreamGuard() {
                if (!submitted) {
                    stream->finishWithoutGenerate();
                }
            }
        } unenqueued_guard{stream};
        stream->setPrefillTpSize(peer_info->tp_size);
        stream->setPrefillCpSize(peer_info->cp_size);
        stream->setPrefillPipelineLayout(std::move(peer_info->pipeline_layout));
        generate_context.setStream(stream);
        cancel_registry_->attach(cancel_handle, stream);

        std::shared_ptr<PrefillServerCallerContext> prefill_caller_ctx;
        PrefillContextGuard                         prefill_context_guard;
        if (!request->enqueued_by_master()) {
            if (cancel_handle->canceled.load()) {
                // NormalEngine settles rejected canceled streams without scheduler ownership.
                engine_->enqueue(stream);
                unenqueued_guard.submitted = true;
                generate_context.error_info   = cancel_handle->cancel_reason;
                generate_context.error_status = serializeErrorMsg(std::to_string(request_id), generate_context.request_info, generate_context.error_info);
                return generate_context.error_status;
            }
            const auto& unique_key          = input->generate_config->unique_key;
            const auto  request_deadline_ms = input->request_deadline_ms;
            // The frontend owns Prefill selection. Use the same endpoint as the peer-info probe.
            std::string target_ip;
            uint32_t    target_port  = 0;
            auto        parse_status = parsePrefillDpAddr(prefillAddress(prefill_request), &target_ip, &target_port);
            if (!parse_status.ok()) {
                generate_context.error_info = errorInfoFromGrpcStatus(parse_status);
                generate_context.error_status = serializeErrorMsg(
                    generate_context.request_key, generate_context.request_info, generate_context.error_info);
                log_pd_error("prefill_address", generate_context.error_status);
                return generate_context.error_status;
            }
            const auto prefill_call_start_us = currentTimeUs();
            auto       started               = prefill_server_caller_->callPrefill(
                &prefill_request, target_ip, target_port, unique_key, request_deadline_ms,
                generate_context.trace_span_guard ? generate_context.trace_span_guard->sharedSpan() :
                    opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>{});
            {
                RpcMetricsCollector collector;
                collector.min_response_done_time_us = 0;
                collector.remote_generate_rt_us     = currentTimeUs() - prefill_call_start_us;
                generate_context.reportMetrics(collector);
            }
            if (!started.ok()) {
                generate_context.error_info = started.status();
                generate_context.error_status =
                    serializeErrorMsg(generate_context.request_key, generate_context.request_info, generate_context.error_info);
                log_pd_error("call_prefill", generate_context.error_status);
                return generate_context.error_status;
            }
            prefill_caller_ctx            = std::move(started.value());
            prefill_context_guard.context = prefill_caller_ctx;
        }

        engine_->enqueue(stream);
        unenqueued_guard.submitted = true;

        // Keep the async prefill RPC alive for the handoff lifecycle only. StartLoad applies the first token to the
        // decode stream, so client output must come exclusively from pollStreamOutput().
        const auto check_progress = [&] {
            if (!prefill_caller_ctx) {
                return FirstError::Snapshot{};
            }
            const auto remote = prefill_caller_ctx->firstError();
            const auto first = FirstError::earlier(stream->firstError(), remote);
            if (remote.error.hasError() && first.order == remote.order) {
                generate_context.dependency_failure = remote;
            }
            return remote;
        };
        generate_context.error_status = pollStreamOutput(
            server_context, generate_context.request_key, response_writer, generate_context.getStream(), check_progress);
        prefill_cancel_guard.completed = generate_context.error_status.ok() && !server_context->IsCancelled()
                                         && !stream->hasError()
                                         && (stream->getStatus() == StreamState::FINISHED
                                             || stream->hasEvent(StreamEvents::GenerateDone));
        if (prefill_cancel_guard.completed)
            cancel_registry_->finishDownstream(cancel_handle);
        else
            prefill_cancel_guard.cancel();
        meta_->dequeue(generate_context.request_id, generate_context.getStream());

        if (prefill_caller_ctx && (!generate_context.error_status.ok() || server_context->IsCancelled())) {
            prefill_caller_ctx->cancel();
        }
        if (!generate_context.error_status.ok()) {
            log_pd_error("stream_output", generate_context.error_status);
        }
        return generate_context.error_status;
    } catch (const std::invalid_argument& error) {
        generate_context.recordRpcFailure({ErrorCode::INVALID_PARAMS, error.what()});
        throw;
    } catch (const std::exception& error) {
        generate_context.recordRpcFailure({ErrorCode::EXECUTION_EXCEPTION, error.what()});
        throw;
    } catch (...) {
        generate_context.recordRpcFailure({ErrorCode::EXECUTION_EXCEPTION, "unknown PD handler exception"});
        throw;
    }
} catch (const std::invalid_argument& error) {
    const auto key = request ? std::to_string(request->request_id()) : "unknown";
    return serializeErrorMsg(key, request ? QueryConverter::transRequestInfo(request->request_info()) : RequestInfo(), ErrorInfo(ErrorCode::INVALID_PARAMS, error.what()));
} catch (const std::exception& error) {
    const auto key = request ? std::to_string(request->request_id()) : "unknown";
    return serializeErrorMsg(key, request ? QueryConverter::transRequestInfo(request->request_info()) : RequestInfo(), ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error.what()));
} catch (...) {
    const auto key = request ? std::to_string(request->request_id()) : "unknown";
    return serializeErrorMsg(key, request ? QueryConverter::transRequestInfo(request->request_info()) : RequestInfo(), ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown decode RPC exception"));
}

void DecodeRpcServer::updateAuxInfo(GenerateOutputsPB& outputs_pb, std::shared_ptr<GenerateStream>& stream) {
    const auto time_info         = stream->getTimeInfo();
    const auto first_token_rt_us = time_info.first_token_rt_us;
    const auto cost_time_us      = autil::TimeUtility::currentTimeInMicroSeconds() - time_info.begin_time_us;
    const auto reuse_lens        = getPrefillReuseLens(stream);

    for (size_t i = 0; i < outputs_pb.flatten_output().aux_info_size(); i++) {
        auto       aux_info                = outputs_pb.mutable_flatten_output()->mutable_aux_info(i);
        const auto decode_total_reuse_len  = aux_info->total_reuse_len();
        const auto decode_local_reuse_len  = aux_info->local_reuse_len();
        const auto decode_remote_reuse_len = aux_info->remote_reuse_len();
        const auto decode_memory_reuse_len = aux_info->memory_reuse_len();
        const auto decode_disk_reuse_len   = aux_info->disk_reuse_len();
        aux_info->set_first_token_cost_time_us(first_token_rt_us);
        aux_info->set_cost_time_us(cost_time_us);
        aux_info->set_pd_sep(true);

        // Generic prompt-cache accounting describes Prefill reuse. Decode reuse
        // is reported separately and must not inflate OpenAI cached_tokens.
        aux_info->set_total_reuse_len(reuse_lens.total);
        aux_info->set_local_reuse_len(reuse_lens.local);
        aux_info->set_remote_reuse_len(reuse_lens.remote);
        aux_info->set_memory_reuse_len(reuse_lens.memory);
        aux_info->set_disk_reuse_len(reuse_lens.disk);

        aux_info->set_prefill_total_reuse_len(reuse_lens.total);
        aux_info->set_prefill_local_reuse_len(reuse_lens.local);
        aux_info->set_prefill_remote_reuse_len(reuse_lens.remote);
        aux_info->set_prefill_memory_reuse_len(static_cast<int32_t>(reuse_lens.memory));
        aux_info->set_prefill_disk_reuse_len(static_cast<int32_t>(reuse_lens.disk));

        aux_info->set_decode_total_reuse_len(decode_total_reuse_len);
        aux_info->set_decode_local_reuse_len(decode_local_reuse_len);
        aux_info->set_decode_remote_reuse_len(decode_remote_reuse_len);
        aux_info->set_decode_memory_reuse_len(decode_memory_reuse_len);
        aux_info->set_decode_disk_reuse_len(decode_disk_reuse_len);
    }
}

}  // namespace rtp_llm
