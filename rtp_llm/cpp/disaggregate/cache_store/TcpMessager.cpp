#include "rtp_llm/cpp/disaggregate/cache_store/TcpMessager.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <algorithm>

#include "autil/EnvUtil.h"
#include "autil/TimeUtility.h"

#include "rtp_llm/cpp/disaggregate/cache_store/TcpCacheStoreLoadServiceClosure.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TimerManager.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreMetricsCollector.h"

namespace rtp_llm {

namespace {

// retry only on transport-level failures; business errors (buffer mismatch,
// invalid params, server overload) gain nothing from a fresh connection
bool isTransportLoadError(CacheStoreErrorCode ec) {
    return ec == CacheStoreErrorCode::LoadConnectFailed || ec == CacheStoreErrorCode::LoadSendRequestFailed
           || ec == CacheStoreErrorCode::CallPrefillTimeout;
}

// minimal remaining budget required to start another attempt
constexpr int64_t kMinRetryBudgetMs = 100;

}  // namespace

bool TcpMessager::init(MessagerInitParams params) {
    init_params_ = params;

    load_attempt_timeout_ms_ =
        autil::EnvUtil::getEnv("CACHE_STORE_TCP_LOAD_ATTEMPT_TIMEOUT_MS", int32_t(10000));

    tcp_client_ = std::make_shared<TcpClient>();
    if (!tcp_client_->init(init_params_.io_thread_count)) {
        RTP_LLM_LOG_WARNING("messager init failed, tcp client init failed");
        return false;
    }

    tcp_server_ = std::make_shared<TcpServer>();
    if (!tcp_server_->init(
            init_params_.io_thread_count, init_params_.worker_thread_count, true, init_params_.worker_queue_size)) {
        RTP_LLM_LOG_WARNING("messager init failed, tcp server init failed");
        return false;
    }

    service_ = std::make_shared<TcpCacheStoreServiceImpl>(memory_util_,
                                                          request_block_buffer_store_,
                                                          metrics_reporter_,
                                                          timer_manager_,
                                                          locked_block_buffer_manager_,
                                                          tcp_client_,
                                                          init_params_.device_id);
    if (!tcp_server_->registerService(service_.get())) {
        RTP_LLM_LOG_WARNING("messager init failed, tcp server register service failed");
        return false;
    }

    if (!tcp_server_->start(init_params_.server_port)) {
        RTP_LLM_LOG_WARNING("messager start failed, tcp server start failed");
        return false;
    }
    RTP_LLM_LOG_INFO("tcp messager init success, server port %u, io thread count %u, worker thread count %u, "
                     "worker queue size %u, load attempt timeout %u ms",
                     init_params_.server_port,
                     init_params_.io_thread_count,
                     init_params_.worker_thread_count,
                     init_params_.worker_queue_size,
                     load_attempt_timeout_ms_);
    return true;
}

void TcpMessager::load(const std::shared_ptr<LoadRequest>&                          request,
                       const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collector) {
    // retry within request->timeout_ms against zombie channels (cached connections silently
    // killed by middleboxes): each attempt is capped by load_attempt_timeout_ms_, transport-level
    // failures invalidate the cached channel and retry on a fresh connection
    int64_t deadline_ms = autil::TimeUtility::currentTimeInMilliSeconds() + request->timeout_ms;
    doLoadAttempt(request, collector, deadline_ms);
}

void TcpMessager::doLoadAttempt(const std::shared_ptr<LoadRequest>&                          request,
                                const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collector,
                                int64_t                                                      deadline_ms) {
    int64_t remaining_ms = deadline_ms - autil::TimeUtility::currentTimeInMilliSeconds();
    if (remaining_ms <= 0) {
        RTP_LLM_LOG_WARNING("messager client load gave up, no budget left, ip %s, port %u",
                            request->ip.c_str(),
                            request->port);
        request->callback(false, CacheStoreErrorCode::CallPrefillTimeout);
        return;
    }
    uint32_t attempt_timeout_ms = (uint32_t)std::min<int64_t>(remaining_ms, load_attempt_timeout_ms_);

    auto channel = tcp_client_->getChannel(request->ip, request->port);
    if (channel == nullptr) {
        RTP_LLM_LOG_WARNING("messager client get channel failed, ip %s", request->ip.c_str());
        scheduleRetryOrFail(request, collector, deadline_ms, CacheStoreErrorCode::LoadConnectFailed, "get channel failed");
        return;
    }

    auto load_request = makeLoadRequest(request);
    if (load_request == nullptr) {
        RTP_LLM_LOG_WARNING("messager client generate load request failed");
        request->callback(false, CacheStoreErrorCode::LoadSendRequestFailed);
        return;
    }

    arpc::ANetRPCController* controller = new arpc::ANetRPCController();
    controller->SetExpireTime(attempt_timeout_ms);

    CacheLoadResponse* load_response = new CacheLoadResponse;
    auto               wrapped_callback = [this, request, collector, deadline_ms](bool success, CacheStoreErrorCode ec) {
        if (success || !isTransportLoadError(ec)) {
            request->callback(success, ec);
            return;
        }
        scheduleRetryOrFail(request, collector, deadline_ms, ec, CacheStoreErrorCodeToString(ec));
    };
    auto* closure = new TcpCacheStoreLoadServiceClosure(memory_util_,
                                                        request->request_block_buffer,
                                                        controller,
                                                        load_request,
                                                        load_response,
                                                        wrapped_callback,
                                                        collector,
                                                        init_params_.device_id);

    collector->markRequestCallBegin();
    KvCacheStoreService_Stub stub((::google::protobuf::RpcChannel*)(channel.get()),
                                  ::google::protobuf::Service::STUB_DOESNT_OWN_CHANNEL);
    stub.load(controller, load_request, load_response, closure);
}

void TcpMessager::scheduleRetryOrFail(const std::shared_ptr<LoadRequest>&                          request,
                                      const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collector,
                                      int64_t                                                      deadline_ms,
                                      CacheStoreErrorCode                                          ec,
                                      const std::string&                                           reason) {
    // transport-level failure may come from a zombie channel, drop it so the retry reconnects
    tcp_client_->invalidateChannel(request->ip, request->port);

    int64_t remaining_ms = deadline_ms - autil::TimeUtility::currentTimeInMilliSeconds();
    if (remaining_ms < kMinRetryBudgetMs) {
        RTP_LLM_LOG_WARNING("messager client load failed, no retry budget left (%ld ms), ip %s, port %u, reason %s",
                            remaining_ms,
                            request->ip.c_str(),
                            request->port,
                            reason.c_str());
        request->callback(false, ec);
        return;
    }
    RTP_LLM_LOG_WARNING("messager client load failed, retry with fresh channel, ip %s, port %u, reason %s",
                        request->ip.c_str(),
                        request->port,
                        reason.c_str());
    doLoadAttempt(request, collector, deadline_ms);
}

bool TcpMessager::generateBlockInfo(BlockBufferInfo*                    block_info,
                                    const std::shared_ptr<BlockBuffer>& block,
                                    uint32_t                            partition_count,
                                    uint32_t                            partition_id) {
    block_info->set_key(block->key);
    block_info->set_len(block->len / partition_count);  // real len
    block_info->set_addr((int64_t)(block->addr.get()) + block_info->len() * partition_id);
    return true;
}

}  // namespace rtp_llm
