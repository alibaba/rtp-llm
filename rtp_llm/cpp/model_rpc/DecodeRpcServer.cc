#include <limits>
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/GrpcAddressUtil.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <limits>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include <limits.h>
#include <condition_variable>
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
#include "rtp_llm/cpp/multimodal_processor/MultimodalTypes.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/model_rpc/RpcTimeoutUtils.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include "rtp_llm/cpp/model_rpc/PDRequestUtils.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/utils/DebugUtils.h"
#include "rtp_llm/cpp/engine_base/Host.h"
#include "autil/NetUtil.h"
#include <cerrno>
#include <cstring>
#include <utility>
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
    return bind_ip + "_" + std::to_string(unique_key_id) + "_" + std::to_string(current_time_us);
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

DecodeRpcServer::~DecodeRpcServer() {
    if (cancel_cleanup_thread_)
        cancel_cleanup_thread_->stop();
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
                                                       handle);
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

void DecodeRpcServer::cancelCleanupTick() {
    const auto pending = cancel_registry_->pending();
    // Publish every locally safe terminal in this bounded batch before any
    // potentially slow remote operation. Local-completion selection has its own
    // fair cursor, independent of retained remote retry history.
    for (const auto& handle : pending) {
        cancel_registry_->complete(handle);
    }
    // Retry the oldest due record only. This bounds remote work per tick and
    // prevents one batch of dead peers from multiplying the RPC deadline.
    PDCancelRegistry::Handle next;
    int64_t                  earliest_retry = std::numeric_limits<int64_t>::max();
    for (const auto& handle : pending) {
        if (handle->downstream_address.empty()) {
            continue;
        }
        const auto retry = handle->next_cancel_retry_ms.load(std::memory_order_acquire);
        if (retry < earliest_retry) {
            earliest_retry = retry;
            next           = handle;
        }
    }
    if (!next) {
        return;
    }
    const auto& handle       = next;
    const auto  monotonic_ms = [] {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    };
    auto next_retry = handle->next_cancel_retry_ms.load(std::memory_order_acquire);
    if (monotonic_ms() < next_retry
        || !handle->next_cancel_retry_ms.compare_exchange_strong(
            next_retry, std::numeric_limits<int64_t>::max(), std::memory_order_acq_rel)) {
        return;
    }
    autil::ScopeGuard retry_guard([&] {
        const auto delay = handle->cancel_retry_delay_ms.load();
        handle->cancel_retry_delay_ms.store(std::min<int64_t>(delay * 2, 1000));
        handle->next_cancel_retry_ms.store(monotonic_ms() + delay, std::memory_order_release);
    });
    auto              connection = cancel_pool_.getConnection(handle->downstream_address);
    if (!connection.ok())
        return;
    CancelRequestPB request;
    request.set_request_id(handle->identity.request_id);
    request.set_cancel_error_code(static_cast<int64_t>(handle->cancel_reason.code()));
    request.set_cancel_error_message(handle->cancel_reason.ToString());
    CancelResponsePB    response;
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(200));
    if (!connection->stub->Cancel(&context, request, &response).ok())
        return;
    if (response.status() == CANCEL_STATUS_TOMBSTONED || response.status() == CANCEL_STATUS_NOT_FOUND) {
        cancel_registry_->finishDownstream(handle);
    } else if (response.status() == CANCEL_STATUS_ACCEPTED) {
        StatusVersionPB version;
        version.set_latest_finished_version(0);
        WorkerStatusPB      status;
        grpc::ClientContext status_context;
        status_context.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(200));
        if (connection->stub->GetWorkerStatus(&status_context, version, &status).ok()) {
            for (const auto& task : status.finished_task_list()) {
                if (task.request_id() == handle->identity.request_id) {
                    cancel_registry_->finishDownstream(handle);
                    break;
                }
            }
        }
    }
    cancel_registry_->complete(handle);
}

grpc::Status DecodeRpcServer::init(const EngineInitParams&                                maga_init_params,
                                   std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params,
                                   py::object                                             mm_process_engine) {
    auto ret = LocalRpcServer::init(maga_init_params, std::move(propose_params), mm_process_engine);
    if (!ret.ok()) {
        RTP_LLM_LOG_ERROR("decode rpc server new2 init failed, err: %s", ret.error_message().c_str());
        return ret;
    }
    requires_group_drain_ = maga_init_params.parallelism_config.world_size > 1;
    const auto& peers     = maga_init_params.runtime_config.shutdown_worker_grpc_addrs;
    if (peers.size() == size_t(maga_init_params.parallelism_config.world_size)) {
        shutdown_coordinator_ = std::make_unique<ShutdownCoordinator>(
            peers,
            maga_init_params.parallelism_config.world_rank,
            [this] {
                std::lock_guard<std::mutex> lock(admission_mutex_);
                return ShutdownCoordinator::State{draining_, onflight_requests_.load() + external_onflight_()};
            },
            [this] { beginShutdown(); });
    } else if (requires_group_drain_) {
        RTP_LLM_LOG_WARNING("missing complete shutdown group; graceful shutdown will wait until its deadline");
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

    decode_rpc_caller_     = std::make_shared<DecodeRpcCaller>();
    cancel_registry_       = std::make_unique<PDCancelRegistry>(meta_);
    cancel_cleanup_thread_ =
        autil::LoopThread::createLoopThread([this]() { cancelCleanupTick(); }, 10 * 1000, "P2PDecodeCancel");
    if (!cancel_cleanup_thread_)
        return grpc::Status(grpc::StatusCode::INTERNAL, "failed to start Decode cancel cleanup");

    RTP_LLM_LOG_INFO("decode rpc server new2 init");
    return grpc::Status::OK;
}

void DecodeRpcServer::beginShutdown() {
    std::lock_guard<std::mutex> lock(admission_mutex_);
    external_begin_shutdown_();
    draining_ = true;
}

bool DecodeRpcServer::waitForDrain(ShutdownCoordinator::Deadline deadline) {
    if (shutdown_coordinator_) {
        return shutdown_coordinator_->wait(deadline);
    }
    // Old/standalone configs have no group address list. Never mistake an idle
    // local rank for a drained distributed engine in that case.
    while (requires_group_drain_ || onflight_requests_.load() || external_onflight_()) {
        if (std::chrono::system_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_until(
            std::min(deadline, std::chrono::system_clock::now() + std::chrono::milliseconds(20)));
    }
    return true;
}

grpc::Status DecodeRpcServer::shutdownStatus(const ShutdownRequestPB& request, ShutdownResponsePB* response) {
    if (!shutdown_coordinator_) {
        return {grpc::StatusCode::FAILED_PRECONDITION, "shutdown group is unavailable"};
    }
    return shutdown_coordinator_->handle(request, response);
}

grpc::Status DecodeRpcServer::BatchGenerateCall(grpc::ServerContext*        context,
                                                const BatchGenerateInputPB* request,
                                                BatchGenerateOutputsPB*     response) {
    c10::InferenceMode           inference_guard(true);
    std::unique_lock<std::mutex> admission_lock(admission_mutex_);
    if (draining_)
        return {grpc::StatusCode::UNAVAILABLE, "Decode is draining"};
    AtomicGuard request_guard(onflight_requests_);
    admission_lock.unlock();
    if (!request || !response)
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "batch request and response are required");
    response->Clear();
    if (request->inputs().empty())
        return grpc::Status::OK;
    if (!engine_ || !decode_rpc_caller_ || !cancel_registry_)
        return grpc::Status(grpc::StatusCode::UNAVAILABLE, "Decode is not initialized");

    const auto entry_ms = currentTimeMs();
    const auto address  = prefillAddress(request->inputs(0));
    if (address.empty())
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "batch requires a Prefill address");
    std::unordered_set<int64_t> request_ids;
    bool                        all_pd            = true;
    bool                        master_admitted   = false;
    int64_t                     batch_deadline_ms = entry_ms;
    BatchGenerateInputPB        effective;
    for (const auto& item : request->inputs()) {
        if (item.request_id() <= 0 || !request_ids.insert(item.request_id()).second)
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "batch requires distinct positive request_ids");
        if (prefillAddress(item) != address)
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "batch items require the same Prefill address");
        all_pd          = all_pd && (item.enqueued_by_master() || checkPDSupport(item).supported);
        master_admitted = master_admitted || item.enqueued_by_master();
        auto* copy      = effective.add_inputs();
        copy->CopyFrom(item);
        const auto timeout = clampRpcTimeoutMsToInt32(normalizeRpcTimeoutMs(
            item.generate_config().timeout_ms(), maga_init_params_.pd_sep_config.max_rpc_timeout_ms));
        copy->mutable_generate_config()->set_timeout_ms(timeout);
        batch_deadline_ms = std::max(batch_deadline_ms, entry_ms + timeout);
    }
    const auto client_cancelled = [context]() { return context && context->IsCancelled(); };
    if (!all_pd) {
        if (master_admitted)
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                                "cannot forward a mixed batch after Master Prefill admission");
        // Match single-request fallback, while retaining the existing batch result contract.
        for (auto& item : *effective.mutable_inputs())
            item.mutable_generate_config()->set_can_use_pd_separation(false);
        return decode_rpc_caller_->forwardBatchGenerate(
            effective, response, address, batch_deadline_ms, client_cancelled);
    }
    if (maga_init_params_.parallelism_config.dp_size != 1)
        return grpc::Status(grpc::StatusCode::UNIMPLEMENTED, "P2P batch inference requires single-DP mode");

    struct BatchContextGuard {
        PDCancelRegistry&                             registry;
        EngineBase&                                   engine;
        std::vector<PDCancelRegistry::Handle>         handles;
        std::vector<std::unique_ptr<GenerateContext>> contexts;
        std::vector<GenerateStreamPtr>                streams;
        std::unordered_set<int64_t>                   submitted_prefill;
        grpc::Status                                  status;
        bool                                          scheduled = false;
        bool                                          completed = false;
        ~BatchContextGuard() {
            if (!completed) {
                auto reason = errorInfoFromGrpcStatus(status);
                if (!reason.hasError())
                    reason = {ErrorCode::CANCELLED, "batch generation abandoned"};
                // Also cancel attempted admissions when the EnqueueBatch ACK was lost.
                for (const auto& handle : handles) {
                    if (submitted_prefill.count(handle->identity.request_id))
                        registry.cancel(handle->identity.request_id, reason);
                    else
                        registry.finishDownstream(handle);
                }
                for (const auto& stream : streams) {
                    if (!stream->hasError() && stream->getStatus() != StreamState::FINISHED)
                        stream->reportError(reason.code(), reason.ToString());
                }
                for (auto& ctx : contexts) {
                    ctx->error_info   = reason;
                    ctx->error_status = grpcStatusFromErrorInfo(reason);
                }
                if (!scheduled) {
                    // Canceled streams still need the engine's terminal/resource cleanup.
                    for (auto& stream : streams)
                        engine.enqueue(stream);
                }
            }
            contexts.clear();
            for (const auto& handle : handles) {
                if (completed)
                    registry.finishDownstream(handle);
                registry.finishLocal(handle);
            }
        }
    } guard{*cancel_registry_, *engine_, {}, {}, {}, {}, grpc::Status::OK};

    try {
        EnqueueBatchRequestPB prefill_request;
        prefill_request.set_batch_id(request->inputs(0).request_id());
        auto* slot = prefill_request.add_dp_slots();
        slot->set_dp_rank(0);
        for (auto& item : *effective.mutable_inputs()) {
            item.mutable_generate_config()->set_unique_key(masterEnqueuedHandoffUniqueKey(item.request_id()));
            if (!item.has_group_id())
                item.mutable_group_id()->set_value(prefill_request.batch_id());
            item.set_group_size(request->inputs_size());
            const auto               deadline_ms = entry_ms + item.generate_config().timeout_ms();
            PDCancelRegistry::Handle handle;
            const auto               admission =
                cancel_registry_->admit(item, item.generate_config().unique_key(), address, deadline_ms, handle);
            if (!admission.ok())
                return guard.status = serializeErrorMsg(std::to_string(item.request_id()), admission);
            guard.handles.push_back(handle);
            if (item.enqueued_by_master())
                guard.submitted_prefill.insert(item.request_id());

            std::shared_ptr<GenerateInput> input;
            std::string                    prefill_ip;
            uint32_t                       prefill_port = 0;
            PrefillPeerInfo                peer;
            auto status = preparePDRequest(item, deadline_ms, input, prefill_ip, prefill_port, peer, context);
            if (!status.ok())
                return guard.status = status;
            auto stream = engine_->makeStream(input);
            stream->setPrefillTpSize(peer.tp_size);
            stream->setPrefillCpSize(peer.cp_size);
            guard.streams.push_back(stream);
            cancel_registry_->attach(handle, stream);
            auto ctx = std::make_unique<GenerateContext>(
                item.request_id(), item.generate_config().timeout_ms(), context, metrics_reporter_, meta_);
            ctx->setStream(stream);
            guard.contexts.push_back(std::move(ctx));
            if (!item.enqueued_by_master()) {
                auto* input_pb = slot->add_requests()->mutable_input();
                input_pb->CopyFrom(item);
            }
        }
        const auto is_cancelled = [&]() {
            return client_cancelled()
                   || std::any_of(guard.handles.begin(), guard.handles.end(), [](const auto& handle) {
                          return handle->canceled.load();
                      });
        };
        const auto cancellation_status = [&]() {
            for (const auto& handle : guard.handles) {
                if (handle->canceled.load())
                    return grpcStatusFromErrorInfo(handle->cancel_reason);
            }
            return grpcStatusFromErrorInfo({ErrorCode::CANCELLED, "batch cancelled before Prefill admission"});
        };
        if (is_cancelled())
            return guard.status = cancellation_status();
        if (!slot->requests().empty()) {
            for (auto& external : *slot->mutable_requests()) {
                auto*      item      = external.mutable_input();
                const auto remaining = entry_ms + item->generate_config().timeout_ms() - currentTimeMs();
                if (remaining <= 0)
                    return guard.status = grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED,
                                                       "batch item expired before Prefill admission");
                item->mutable_generate_config()->set_timeout_ms(clampRpcTimeoutMsToInt32(remaining));
            }
            EnqueueBatchResponsePB ack;
            for (const auto& external : slot->requests())
                guard.submitted_prefill.insert(external.input().request_id());
            auto status =
                decode_rpc_caller_->enqueueBatch(prefill_request, &ack, address, batch_deadline_ms, is_cancelled);
            if (!status.ok())
                return guard.status = is_cancelled() ? cancellation_status() : status;
            // Explicit rejections were not admitted by this call. In particular,
            // a duplicate ID must not cancel a different request already on P.
            for (const auto& error : ack.errors())
                guard.submitted_prefill.erase(error.request_id());
            for (const auto& error : ack.errors()) {
                const auto code     = static_cast<ErrorCode>(error.error_info().error_code());
                return guard.status = grpcStatusFromErrorInfo(
                           {code == ErrorCode::NONE_ERROR ? ErrorCode::EXECUTION_EXCEPTION : code,
                            "Prefill batch request " + std::to_string(error.request_id()) + ": "
                                + error.error_info().error_message()});
            }
            std::unordered_set<int64_t> accepted;
            for (const auto& success : ack.successes())
                accepted.insert(success.request_id());
            if (ack.batch_id() != prefill_request.batch_id() || ack.successes_size() != slot->requests_size()
                || accepted.size() != static_cast<size_t>(slot->requests_size()))
                return guard.status = grpc::Status(grpc::StatusCode::INTERNAL, "Prefill batch ACK size/id mismatch");
            for (const auto& external : slot->requests()) {
                if (!accepted.count(external.input().request_id()))
                    return guard.status =
                               grpc::Status(grpc::StatusCode::INTERNAL, "Prefill batch ACK request mismatch");
            }
        }
        // makeStream installs engine invariants; enqueueGroup preserves peer TP/CP and cancel bindings.
        const auto enqueued = engine_->getScheduler().enqueueGroup(guard.streams);
        guard.scheduled     = true;
        if (enqueued.first.size() != guard.streams.size())
            return guard.status = grpc::Status(grpc::StatusCode::INTERNAL, "Decode batch enqueue size mismatch");
        for (size_t i = 0; i < enqueued.first.size(); ++i) {
            if (!enqueued.first[i]) {
                const auto error    = guard.streams[i]->statusInfo();
                return guard.status = grpcStatusFromErrorInfo(
                           error.hasError() ? error :
                                              ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "Decode batch item rejected"));
            }
        }
        guard.status    = pollBatchStreamOutput(context, guard.streams, response);
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
                                               std::string&                    prefill_ip,
                                               uint32_t&                       prefill_port,
                                               PrefillPeerInfo&                peer_info,
                                               grpc::ServerContext*            server_context) {
    input                      = QueryConverter::transQuery(&request);
    input->request_deadline_ms = deadline_ms;
    // Preserve the configured duration on the wire; D keeps its own deadline.
    input->generate_config->timeout_ms = request.generate_config().timeout_ms();
    auto status = preprocessForPD(input, mm_processor_.get(), engine_->isMTPEagle(), RoleType::DECODE, nullptr, {});
    if (!status.ok())
        return serializeErrorMsg(std::to_string(request.request_id()), status);
    const auto  address = prefillAddress(request);
    auto        parse_status = parsePrefillDpAddr(address, &prefill_ip, &prefill_port);
    if (!parse_status.ok())
        return parse_status;
    const auto remaining = deadline_ms - currentTimeMs();
    if (remaining <= 0)
        return grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "PD preprocessing deadline exceeded");
    auto peer_result = decode_rpc_caller_->getPrefillPeerInfo(
        prefill_ip, prefill_port, clampRpcTimeoutMsToInt32(remaining), request, [this, server_context, &request] {
            return (server_context && server_context->IsCancelled())
                   || cancel_registry_->isCanceled(request.generate_config().unique_key());
        });
    if (!peer_result.ok())
        return grpcStatusFromErrorInfo(peer_result.status());
    peer_info = std::move(peer_result.value());
    if (peer_info.prepared_input) {
        const auto prepared_status = applyPDPreparedInput(*peer_info.prepared_input, *input);
        if (!prepared_status.ok())
            return grpcStatusFromErrorInfo(prepared_status);
    }
    if (peer_info.tp_size <= 0 || peer_info.cp_size <= 0) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "prefill peer info is not available");
    }
    return grpc::Status::OK;
}

grpc::Status DecodeRpcServer::GenerateStreamCall(grpc::ServerContext*                   server_context,
                                                 const GenerateInputPB*                 request,
                                                 grpc::ServerWriter<GenerateOutputsPB>* response_writer) try {
    if (!request) {
        return serializeErrorMsg("unknown", ErrorInfo(ErrorCode::INVALID_PARAMS, "null generate request"));
    }
    struct PrefillContextGuard {
        std::shared_ptr<DecodeRpcCallerContext> context;

        ~PrefillContextGuard() {
            if (!context) {
                return;
            }
            if (!context->done()) {
                context->cancel();
            }
            context->wait();
        }
    };

    const int64_t request_entry_ms          = currentTimeMs();
    const auto    normalized_timeout_ms     = normalizeRpcTimeoutMs(request->generate_config().timeout_ms(),
                                                             maga_init_params_.pd_sep_config.max_rpc_timeout_ms);
    const auto    normalized_timeout_ms_i32 = clampRpcTimeoutMsToInt32(normalized_timeout_ms);

    // Always isolate decode-entrance P2P handoff from any caller-provided business unique_key.
    // The external unique_key remains on the original request object, while the internal handoff
    // request gets a per-request key used only by the prefill/decode P2P pipeline.
    GenerateInputPB request_with_handoff_key;
    const auto*     effective_request = request;

    // Check if pd separation should be used
    // A successful Master batch admission is authoritative: Prefill has
    // already entered the P2P path, so Decode must attach to that handoff.
    auto pd_separation = checkPDSupport(*request).supported || request->enqueued_by_master();
    if (pd_separation) {
        std::string handoff_unique_key;
        if (request->enqueued_by_master()) {
            handoff_unique_key = masterEnqueuedHandoffUniqueKey(request->request_id());
        } else {
            const auto handoff_id = unique_key_id_.fetch_add(1);
            handoff_unique_key = makeDecodeEntranceUniqueKey(autil::NetUtil::getBindIp(), handoff_id, currentTimeUs());
        }
        request_with_handoff_key = makeDecodeEntranceHandoffRequest(*request, handoff_unique_key);
        request_with_handoff_key.mutable_generate_config()->set_timeout_ms(normalized_timeout_ms_i32);
        effective_request = &request_with_handoff_key;
    }
    if (!pd_separation) {
        RTP_LLM_LOG_DEBUG("pd separation is disabled, call prefill server");
        GenerateInputPB prefill_forward_request;
        prefill_forward_request.CopyFrom(*effective_request);
        prefill_forward_request.mutable_generate_config()->set_timeout_ms(normalized_timeout_ms_i32);
        return decode_rpc_caller_->forwardGenerate(server_context, &prefill_forward_request, response_writer);
    }

    const auto log_pd_error = [&](const char* stage, const grpc::Status& status) {
        const auto  error    = errorInfoFromGrpcStatus(status);
        const auto& trace_id = request->request_info().trace_id().empty() ?
                                   request->generate_config().trace_id().value() :
                                   request->request_info().trace_id();
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

    std::unique_lock<std::mutex> admission_lock(admission_mutex_);
    if (draining_)
        return {grpc::StatusCode::UNAVAILABLE, "Decode is draining"};
    AtomicGuard request_guard(onflight_requests_);
    admission_lock.unlock();
    auto request_id = request->request_id();
    RTP_LLM_LOG_DEBUG("receive request %ld", request_id);
    PDCancelRegistry::Handle                   cancel_handle;
    std::optional<PDCancelRegistry::CallGuard> cancel_guard;
    auto                                       generate_context =
        GenerateContext(request_id, normalized_timeout_ms, server_context, metrics_reporter_, meta_);
    generate_context.onflight_requests = &onflight_requests_;
    generate_context.request_info      = QueryConverter::transRequestInfo(request->request_info());
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
    const int                          uncaught_exceptions = std::uncaught_exceptions();
    autil::ScopeGuard                  rpc_completion_guard([&generate_context, uncaught_exceptions] {
        if (std::uncaught_exceptions() == uncaught_exceptions) {
            generate_context.markRpcHandlingCompleted();
        }
    });
    const auto                         admission = cancel_registry_->admit(*effective_request,
                                                   effective_request->generate_config().unique_key(),
                                                   prefillAddress(*effective_request),
                                                   request_entry_ms + normalized_timeout_ms,
                                                   cancel_handle);
    if (!admission.ok()) {
        generate_context.error_info = admission;
        generate_context.error_status =
            serializeErrorMsg(generate_context.request_key, generate_context.request_info, admission);
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
            registry.cancel(handle->identity.request_id, reason);
            completed = true;
        }
        ~PrefillCancelGuard() {
            if (!completed)
                cancel();
        }
    } prefill_cancel_guard{*cancel_registry_, cancel_handle, generate_context};
    try {
        std::shared_ptr<GenerateInput> input;
        std::string                    prefill_ip;
        uint32_t                       prefill_port = 0;
        PrefillPeerInfo                peer_info;
        const auto                     prepare_start_us = currentTimeUs();
        auto                           prepare_status   = preparePDRequest(*effective_request,
                                               request_entry_ms + normalized_timeout_ms,
                                               input,
                                               prefill_ip,
                                               prefill_port,
                                               peer_info,
                                               server_context);
        {
            RpcMetricsCollector collector;
            collector.min_response_done_time_us      = 0;
            collector.prepare_generate_context_rt_us = currentTimeUs() - prepare_start_us;
            generate_context.reportMetrics(collector);
        }
        if (!prepare_status.ok()) {
            generate_context.error_info   = errorInfoFromGrpcStatus(prepare_status);
            generate_context.error_status = serializeErrorMsg(
                generate_context.request_key, generate_context.request_info, generate_context.error_info);
            log_pd_error("prepare", generate_context.error_status);
            return generate_context.error_status;
        }
        if (cancel_handle->canceled.load()) {
            generate_context.error_info   = cancel_handle->cancel_reason;
            generate_context.error_status = serializeErrorMsg(
                std::to_string(request_id), generate_context.request_info, generate_context.error_info);
            return generate_context.error_status;
        }
        auto stream = engine_->makeStream(input);
        // Until enqueue succeeds, no scheduler owns this stream's terminal transition.
        struct UnenqueuedStreamGuard {
            GenerateStreamPtr stream;
            bool              submitted = false;
            ~UnenqueuedStreamGuard() {
                if (!submitted) {
                    stream->finishWithoutGenerate();
                }
            }
        } unenqueued_guard{stream};
        stream->setPrefillTpSize(peer_info.tp_size);
        stream->setPrefillCpSize(peer_info.cp_size);
        generate_context.setStream(stream);
        cancel_registry_->attach(cancel_handle, stream);

        std::shared_ptr<DecodeRpcCallerContext>     prefill_caller_ctx;
        PrefillContextGuard                         prefill_context_guard;
        if (!request->enqueued_by_master()) {
            if (cancel_handle->canceled.load()) {
                // NormalEngine settles rejected canceled streams without scheduler ownership.
                engine_->enqueue(stream);
                unenqueued_guard.submitted    = true;
                generate_context.error_info   = cancel_handle->cancel_reason;
                generate_context.error_status = serializeErrorMsg(
                    std::to_string(request_id), generate_context.request_info, generate_context.error_info);
                return generate_context.error_status;
            }
            const auto& unique_key          = input->generate_config->unique_key;
            const auto  request_deadline_ms = input->request_deadline_ms;
            const auto prefill_call_start_us = currentTimeUs();
            auto        started               = decode_rpc_caller_->startPrefill(
                effective_request,
                prefill_ip,
                prefill_port,
                unique_key,
                request_deadline_ms,
                generate_context.trace_span_guard ? generate_context.trace_span_guard->sharedSpan() :
                                                                         opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>{});
            {
                RpcMetricsCollector collector;
                collector.min_response_done_time_us = 0;
                collector.remote_generate_rt_us     = currentTimeUs() - prefill_call_start_us;
                generate_context.reportMetrics(collector);
            }
            if (!started.ok()) {
                generate_context.error_info   = started.status();
                generate_context.error_status = serializeErrorMsg(
                    generate_context.request_key, generate_context.request_info, generate_context.error_info);
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
        std::optional<StreamState> observed_state;
        const auto                 check_progress = [&](bool& remote_done) {
            const auto state = stream->getStatus();
            if (!observed_state || state != *observed_state) {
                observed_state = state;
                RpcMetricsCollector collector;
                generate_context.reportMetrics(collector);
            }
            if (!prefill_caller_ctx) {
                remote_done = true;
                return ErrorInfo{};
            }
            remote_done       = prefill_caller_ctx->done();
            const auto remote = prefill_caller_ctx->errorInfo();
            if (remote.hasError() && stream->statusInfo().ok()) {
                generate_context.dependency_failure = remote;
            }
            return remote;
        };
        // Also sample the Master-enqueued path, which has no Prefill RPC to poll.
        generate_context.error_status = pollStreamOutput(server_context,
                                                         generate_context.request_key,
                                                         response_writer,
                                                         generate_context.getStream(),
                                                         check_progress);
        prefill_cancel_guard.completed =
            generate_context.error_status.ok() && !server_context->IsCancelled() && !stream->hasError()
            && (stream->getStatus() == StreamState::FINISHED || stream->hasEvent(StreamEvents::GenerateDone));
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
    return serializeErrorMsg(key,
                             request ? QueryConverter::transRequestInfo(request->request_info()) : RequestInfo(),
                             ErrorInfo(ErrorCode::INVALID_PARAMS, error.what()));
} catch (const std::exception& error) {
    const auto key = request ? std::to_string(request->request_id()) : "unknown";
    return serializeErrorMsg(key,
                             request ? QueryConverter::transRequestInfo(request->request_info()) : RequestInfo(),
                             ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, error.what()));
} catch (...) {
    const auto key = request ? std::to_string(request->request_id()) : "unknown";
    return serializeErrorMsg(key,
                             request ? QueryConverter::transRequestInfo(request->request_info()) : RequestInfo(),
                             ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "unknown decode RPC exception"));
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

grpc::Status DecodeRpcServer::WriteReady(grpc::ServerContext*                   context,
                                         const P2PConnectorWriteReadyRequestPB* request,
                                         P2PConnectorWriteReadyResponsePB*      response) {
    if (context->IsCancelled()) {
        return grpc::Status(grpc::StatusCode::CANCELLED, "request is cancelled");
    }
    if (!engine_ || !engine_->getCacheManager()) {
        return grpc::Status(grpc::StatusCode::INTERNAL, "cache manager is unavailable");
    }
    engine_->getCacheManager()->handleWriteReady(*request, *response);
    return grpc::Status::OK;
}

}  // namespace rtp_llm
