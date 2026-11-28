#include "rtp_llm/cpp/model_rpc/PrefillServerCaller.h"
#include "rtp_llm/cpp/utils/GrpcAddressUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <vector>
#include <thread>

namespace rtp_llm {

constexpr int32_t kPeerInfoProbeDefaultMs = 15000;
constexpr int32_t kPeerInfoProbeMaxMs     = 15000;

namespace {

constexpr auto kCancelProbeInterval = std::chrono::milliseconds(5);

struct SyncPrefillRpcCancelEntry {
    bool poll(std::chrono::system_clock::time_point now) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!client_context) {
            return true;
        }
        try {
            if (is_cancelled && is_cancelled()) {
                cancel_status = grpc::Status(grpc::StatusCode::CANCELLED, "request cancelled by user");
            } else if (now >= deadline) {
                cancel_status = grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "prefill server rpc timeout");
            } else {
                return false;
            }
        } catch (...) {
            cancel_status =
                grpcStatusFromErrorInfo({ErrorCode::EXECUTION_EXCEPTION, "Prefill RPC cancellation check failed"});
        }
        client_context->TryCancel();
        client_context = nullptr;
        is_cancelled   = {};
        return true;
    }

    std::mutex mutex;
    std::function<bool()> is_cancelled;
    grpc::ClientContext* client_context = nullptr;
    std::chrono::system_clock::time_point deadline;
    grpc::Status                          cancel_status;
};

class SyncPrefillRpcCancelMonitor {
public:
    static SyncPrefillRpcCancelMonitor& instance() {
        static SyncPrefillRpcCancelMonitor monitor;
        return monitor;
    }

    void add(const std::shared_ptr<SyncPrefillRpcCancelEntry>& entry) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            entries_.push_back(entry);
        }
        cv_.notify_one();
    }

    ~SyncPrefillRpcCancelMonitor() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        cv_.notify_one();
        watcher_.join();
    }

private:
    SyncPrefillRpcCancelMonitor(): watcher_([this] { watch(); }) {}

    void watch() {
        std::vector<std::weak_ptr<SyncPrefillRpcCancelEntry>> active;
        std::unique_lock<std::mutex> lock(mutex_);
        while (!stopping_) {
            if (active.empty() && entries_.empty()) {
                cv_.wait(lock, [this] { return stopping_ || !entries_.empty(); });
                continue;
            }
            // Only registration uses the shared lock. Probe and compact the
            // active set outside it so request admission does not wait on scans.
            active.insert(active.end(), entries_.begin(), entries_.end());
            entries_.clear();
            lock.unlock();
            const auto now        = std::chrono::system_clock::now();
            auto       next_probe = now + kCancelProbeInterval;
            // poll() and stop() share the lock protecting the stack-owned contexts.
            auto completed = std::remove_if(active.begin(), active.end(), [&](const auto& weak) {
                auto entry = weak.lock();
                if (!entry || entry->poll(now)) {
                    return true;
                }
                next_probe = std::min(next_probe, entry->deadline);
                return false;
            });
            active.erase(completed, active.end());
            lock.lock();
            if (!active.empty()) {
                cv_.wait_until(lock, next_probe, [this] { return stopping_; });
            }
        }
    }

    std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<std::weak_ptr<SyncPrefillRpcCancelEntry>> entries_;
    bool stopping_ = false;
    std::thread watcher_;
};

class SyncPrefillRpcCancelGuard {
public:
    SyncPrefillRpcCancelGuard(const std::function<bool()>& is_cancelled,
                              grpc::ClientContext* client_context,
                              std::chrono::system_clock::time_point deadline):
        entry_(std::make_shared<SyncPrefillRpcCancelEntry>()) {
        entry_->is_cancelled = is_cancelled;
        entry_->client_context = client_context;
        entry_->deadline = deadline;
        SyncPrefillRpcCancelMonitor::instance().add(entry_);
    }

    ~SyncPrefillRpcCancelGuard() {
        stop();
    }

    void stop() {
        std::lock_guard<std::mutex> lock(entry_->mutex);
        entry_->client_context = nullptr;
        entry_->is_cancelled = {};
    }

    bool poll() {
        return entry_->poll(std::chrono::system_clock::now());
    }

    grpc::Status finish(grpc::Status status) {
        stop();
        if (!status.ok() && !entry_->cancel_status.ok()) {
            return entry_->cancel_status;
        }
        return status;
    }

private:
    std::shared_ptr<SyncPrefillRpcCancelEntry> entry_;
};

}  // namespace

PrefillServerCaller::PrefillServerCaller():
    rpc_pool_(std::make_shared<RPCPool>()),
    async_reader_factory_([](const std::shared_ptr<RpcService::Stub>& stub,
                             grpc::ClientContext*                     client_context,
                             const GenerateInputPB&                   request,
                             grpc::CompletionQueue*                   completion_queue) {
        if (!stub) {
            return std::unique_ptr<grpc::ClientAsyncReader<GenerateOutputsPB>>();
        }
        return stub->AsyncGenerateStreamCall(client_context, request, completion_queue, reinterpret_cast<void*>(0));
    }) {}

ErrorResult<std::shared_ptr<PrefillServerCallerContext>>
PrefillServerCaller::callPrefill(const GenerateInputPB* request,
                                 const std::string&     ip,
                                 uint32_t               port,
                                 const std::string&     unique_key,
                                 int64_t                request_deadline_ms,
                                 const opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>& request_span) {
    if (request_deadline_ms <= currentTimeMs()) {
        return ErrorInfo(ErrorCode::GENERATE_TIMEOUT,
                         "Prefill GenerateStreamCall request deadline exceeded key=" + unique_key);
    }
    std::string prefill_addr = formatGrpcHostPort(ip, port);
    if (prefill_addr.empty()) {
        RTP_LLM_LOG_WARNING("request [%lld] invalid prefill grpc address, ip: %s, port: %u",
                            request->request_id(),
                            ip.c_str(),
                            port);
        return ErrorInfo(ErrorCode::INVALID_PARAMS,
                         "Prefill GenerateStreamCall invalid peer=" + ip + ":" + std::to_string(port)
                             + " key=" + unique_key);
    }
    auto context = std::make_shared<PrefillServerCallerContext>(prefill_addr, unique_key);
    auto client_span = telemetry::startChildClientSpan("rtp_llm.call_prefill", request_span,
        ip, port, request->request_id(), 0, "RpcService/GenerateStreamCall");
    if (client_span) {
        context->async_state_->trace_guard = std::make_unique<telemetry::GrpcStatusSpanGuard>(
            client_span, &context->async_state_->trace_status);
    }
    telemetry::injectSpanToClientContext(context->async_state_->client_context.get(), client_span);
    auto        connect_status = rpc_pool_->getConnection(prefill_addr);
    if (!connect_status.ok()) {
        RTP_LLM_LOG_WARNING("request [%lld] get grpc connection to prefill failed, addr: %s",
                            request->request_id(),
                            prefill_addr.c_str());
        const ErrorInfo error(ErrorCode::GET_CONNECTION_FAILED,
                              "Prefill GenerateStreamCall getConnection peer=" + prefill_addr + " key=" + unique_key
                                  + ": " + connect_status.status().ToString());
        context->async_state_->trace_status = grpcStatusFromErrorInfo(error);
        return error;
    }

    auto stub = connect_status.value().stub;

    context->async_state_->stub = stub;

    // Preserve the original request shape for PD handoff so the
    // prefill side still recognizes it as a PD-separation request.
    context->async_state_->request.CopyFrom(*request);
    context->async_state_->request.set_start_time(currentTimeUs());
    context->async_state_->request.mutable_generate_config()->set_can_use_pd_separation(true);
    context->async_state_->request.mutable_generate_config()->set_unique_key(unique_key);

    context->async_state_->client_context->set_deadline(std::chrono::system_clock::time_point(
        std::chrono::milliseconds(request_deadline_ms)));

    context->async_state_->reader = async_reader_factory_(context->async_state_->stub,
                                                          context->async_state_->client_context.get(),
                                                          context->async_state_->request,
                                                          context->async_state_->completion_queue.get());
    if (!context->async_state_->reader) {
        RTP_LLM_LOG_WARNING("request [%lld] create async prefill reader failed, addr: %s",
                            request->request_id(),
                            prefill_addr.c_str());
        context->async_state_->trace_status = grpc::Status(grpc::StatusCode::INTERNAL, "async reader creation failed");
        return ErrorInfo(ErrorCode::RPC_FINISH_FAILED,
                         "Prefill GenerateStreamCall async reader creation failed peer=" + prefill_addr
                             + " key=" + unique_key);
    }
    context->rpc_started_ = true;
    context->startPolling();

    return std::move(context);
}

ErrorResult<PrefillPeerInfo> PrefillServerCaller::getPrefillPeerInfo(const std::string& ip,
                                                                     uint32_t           port,
                                                                     int32_t            request_timeout_ms,
                                                                     int                decode_pp_size) {
    std::string addr = formatGrpcHostPort(ip, port);
    if (addr.empty()) {
        RTP_LLM_LOG_WARNING("getPrefillPeerInfo: invalid prefill grpc address, ip: %s, port: %u", ip.c_str(), port);
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "GetPeerInfo invalid peer=" + ip + ":" + std::to_string(port));
    }

    auto conn = rpc_pool_->getConnection(addr);
    if (!conn.ok()) {
        RTP_LLM_LOG_WARNING("getPrefillPeerInfo: getConnection failed for %s", addr.c_str());
        return ErrorInfo(ErrorCode::GET_CONNECTION_FAILED,
                         "GetPeerInfo getConnection peer=" + addr + ": " + conn.status().ToString());
    }

    // Never enlarge a positive remaining request budget for the peer-info probe.
    const auto probe_timeout_ms = std::min(request_timeout_ms > 0 ? request_timeout_ms : kPeerInfoProbeDefaultMs,
                                          kPeerInfoProbeMaxMs);
    grpc::ClientContext ctx;
    ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(probe_timeout_ms));

    ctx.set_wait_for_ready(true);
    GetPeerInfoRequestPB  request;
    request.set_requester_pp_size(decode_pp_size);
    GetPeerInfoResponsePB response;
    auto                  status = conn.value().stub->GetPeerInfo(&ctx, request, &response);

    if (!status.ok()) {
        RTP_LLM_LOG_ERROR(
            "getPrefillPeerInfo: GetPeerInfo RPC failed for %s: %s", addr.c_str(), status.error_message().c_str());
        const auto error = errorInfoFromGrpcStatus(status);
        const auto code =
            status.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED ? ErrorCode::CONNECT_TIMEOUT : error.code();
        return ErrorInfo(code, "GetPeerInfo peer=" + addr + ": " + error.ToString());
    }

    PrefillPeerInfo info;
    info.pipeline_layout = std::move(*response.mutable_pipeline_layout());
    info.tp_size = response.tp_size();
    if (info.tp_size <= 0) {
        RTP_LLM_LOG_ERROR("getPrefillPeerInfo: invalid tp_size=%d from %s", info.tp_size, addr.c_str());
        return ErrorInfo(ErrorCode::INVALID_PARAMS,
                         "GetPeerInfo peer=" + addr + " invalid tp_size=" + std::to_string(response.tp_size()));
    }
    if (response.cp_size() <= 0 || response.cp_size() > info.tp_size || info.tp_size % response.cp_size() != 0) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS,
                         "GetPeerInfo peer=" + addr + " invalid cp_size=" + std::to_string(response.cp_size())
                             + " tp_size=" + std::to_string(info.tp_size));
    }
    info.cp_size = response.cp_size();

    RTP_LLM_LOG_INFO("getPrefillPeerInfo: prefill %s tp_size=%d, cp_size=%d",
                     addr.c_str(), info.tp_size, info.cp_size);

    return std::move(info);
}

grpc::Status PrefillServerCaller::enqueueBatch(const EnqueueBatchRequestPB& request,
                                           EnqueueBatchResponsePB*      response,
                                           const std::string&           prefill_address,
                                           int64_t                      deadline_ms,
                                           const std::function<bool()>& is_cancelled) {
    auto connection = rpc_pool_->getConnection(prefill_address);
    if (!connection.ok())
        return grpcStatusFromErrorInfo({ErrorCode::GET_CONNECTION_FAILED, connection.status().ToString()});
    grpc::ClientContext context;
    const auto          deadline = std::chrono::system_clock::time_point(std::chrono::milliseconds(deadline_ms));
    context.set_deadline(deadline);
    SyncPrefillRpcCancelGuard cancel_guard(is_cancelled, &context, deadline);
    return cancel_guard.finish(connection->stub->EnqueueBatch(&context, request, response));
}

}  // namespace rtp_llm
