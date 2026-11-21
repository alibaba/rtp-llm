// Copyright (c) 2026 Alibaba Cloud.
// SPDX-License-Identifier: Apache-2.0
#include "rtp_llm/cpp/disaggregate/cache_store/ascend/HixlMessager.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpServer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/ascend/HixlManifestService.h"
#include "rtp_llm/cpp/disaggregate/cache_store/ascend/HixlMemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/ascend/HixlManifest.h"
#include "rtp_llm/cpp/disaggregate/cache_store/ascend/HixlReadWorker.h"
#include "rtp_llm/cpp/disaggregate/cache_store/ascend/HixlTimerQueue.h"
#include "rtp_llm/cpp/disaggregate/cache_store/ascend/HixlProtocol.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <atomic>
#include <random>
#include <sstream>
#include <stdexcept>

namespace rtp_llm {
namespace {
constexpr size_t kMaxInflightLoads = 64;

std::string instanceId() {
    std::random_device random;
    std::ostringstream value;
    for (int i = 0; i < 4; ++i) {
        value << std::hex << random() << "-";
    }
    return value.str();
}

bool checked(hixl::Status status, const char* operation) {
    if (status != hixl::SUCCESS) {
        RTP_LLM_LOG_ERROR("HIXL %s failed, status=%u", operation, status);
    }
    return status == hixl::SUCCESS;
}

class HixlSdkReadDriver final: public HixlReadDriver {
public:
    explicit HixlSdkReadDriver(std::shared_ptr<HixlMemoryUtil> memory, int timeout_ms):
        memory_(std::move(memory)), timeout_ms_(timeout_ms) {}

    bool activate() override {
        return checked(aclrtSetCurrentContext(memory_->context()), "activate progress context");
    }
    bool connect(const std::string& peer) override {
        return checked(memory_->engine()->ConnectAsync(hixl::AscendString(peer.c_str()), timeout_ms_), "ConnectAsync");
    }
    HixlLinkState connection(const std::string& peer) override {
        hixl::AsyncConnectStatus state;
        if (!checked(memory_->engine()->GetAsyncConnectStatus(hixl::AscendString(peer.c_str()), state),
                     "connect status")) {
            return HixlLinkState::FAILED;
        }
        switch (state) {
            case hixl::AsyncConnectStatus::CONNECTED:
                return HixlLinkState::READY;
            case hixl::AsyncConnectStatus::NOT_CONNECT:
                return HixlLinkState::CLOSED;
            case hixl::AsyncConnectStatus::CONNECT_FAILED:
                return HixlLinkState::FAILED;
            default:
                return HixlLinkState::CONNECTING;
        }
    }
    bool read(const std::string& peer, const std::vector<HixlTransferSlice>& slices, void*& handle) override {
        std::vector<hixl::TransferOpDesc> descriptors;
        descriptors.reserve(slices.size());
        for (const auto& slice : slices) {
            descriptors.push_back({slice.local_address, slice.remote_address, slice.bytes});
        }
        return checked(
            memory_->engine()->TransferAsync(hixl::AscendString(peer.c_str()), hixl::READ, descriptors, {}, handle),
            "TransferAsync");
    }
    HixlReadState poll(void* handle) override {
        hixl::TransferStatus state;
        if (!checked(memory_->engine()->GetTransferStatus(handle, state), "transfer status")) {
            return HixlReadState::FAILED;
        }
        if (state == hixl::TransferStatus::COMPLETED) {
            return HixlReadState::COMPLETE;
        }
        return state == hixl::TransferStatus::WAITING ? HixlReadState::WAITING : HixlReadState::FAILED;
    }
    bool disconnect(const std::string& peer) override {
        return checked(memory_->engine()->DisconnectAsync(hixl::AscendString(peer.c_str()), timeout_ms_),
                       "DisconnectAsync");
    }

private:
    std::shared_ptr<HixlMemoryUtil> memory_;
    const int                       timeout_ms_;
};

template<class Request, class Response>
class RpcCall final: public google::protobuf::Closure {
public:
    RpcCall(std::shared_ptr<arpc::RPCChannelBase>      channel,
            uint32_t                                   timeout,
            std::function<void(bool, const Response&)> callback):
        channel(std::move(channel)), callback_(std::move(callback)) {
        controller.SetExpireTime(timeout);
    }
    void Run() override {
        std::unique_ptr<RpcCall> owner(this);
        callback_(!controller.Failed(), response);
    }
    std::shared_ptr<arpc::RPCChannelBase> channel;
    arpc::ANetRPCController               controller;
    Request                               request;
    Response                              response;

private:
    std::function<void(bool, const Response&)> callback_;
};

CacheStoreErrorCode readError(HixlReadResult result) {
    switch (result) {
        case HixlReadResult::OK:
            return CacheStoreErrorCode::None;
        case HixlReadResult::TIMEOUT:
            return CacheStoreErrorCode::LoadBufferTimeout;
        case HixlReadResult::CONNECT_ERROR:
            return CacheStoreErrorCode::LoadRdmaConnectFailed;
        case HixlReadResult::TRANSFER_ERROR:
            return CacheStoreErrorCode::LoadRdmaWriteFailed;
        default:
            return CacheStoreErrorCode::LoadErrorUnknown;
    }
}
}  // namespace

class HixlClientState {
public:
    std::unique_ptr<HixlLoadScheduler> admission;
    std::string                        reader;
    std::string                        engine;
    std::shared_ptr<TcpClient>         tcp;
    std::shared_ptr<HixlTimerQueue>    timers;
    std::shared_ptr<HixlMemoryUtil>    memory;
    std::unique_ptr<HixlReadProgress>  progress;
};

namespace {
class ClientRead: public std::enable_shared_from_this<ClientRead> {
public:
    ClientRead(std::shared_ptr<HixlClientState>                      state,
               std::shared_ptr<LoadRequest>                          request,
               std::shared_ptr<CacheStoreClientLoadMetricsCollector> collector):
        state_(std::move(state)),
        request_(std::move(request)),
        collector_(std::move(collector)),
        ip_(request_->ip),
        port_(request_->port),
        deadline_(std::chrono::steady_clock::now() + std::chrono::milliseconds(request_->timeout_ms)) {}

    void enqueue() {
        auto self = shared_from_this();
        if (!state_->admission->submit({ip_ + ":" + std::to_string(port_),
                                        deadline_,
                                        [self](uint64_t sequence) { self->start(sequence); },
                                        [self](HixlLoadRejection reason) {
                                            self->finish(reason == HixlLoadRejection::TIMEOUT ?
                                                             CacheStoreErrorCode::LoadBufferTimeout :
                                                             CacheStoreErrorCode::LoadErrorUnknown);
                                        }})) {
            finish(CacheStoreErrorCode::PushWorkerItemFailed);
        }
    }

private:
    void start(uint64_t sequence) {
        sequence_    = sequence;
        auto channel = state_->tcp->getChannel(ip_, port_);
        if (!channel) {
            // No RPC was sent. Retire the sequence as well, so future calls do
            // not leave a permanent hole in the source's retirement window.
            retireAndFinish(CacheStoreErrorCode::LoadConnectFailed);
            return;
        }
        const auto remaining =
            std::chrono::ceil<std::chrono::milliseconds>(deadline_ - std::chrono::steady_clock::now()).count();
        if (remaining <= 0) {
            retireAndFinish(CacheStoreErrorCode::LoadBufferTimeout);
            return;
        }
        auto  self = shared_from_this();
        auto* call = new RpcCall<hixl_proto::ManifestRequest, hixl_proto::ManifestResponse>(
            channel, remaining, [self](bool ok, const auto& response) { self->manifest(ok, response); });
        call->request.set_reader_id(state_->reader);
        call->request.set_reader_engine_id(state_->engine);
        call->request.set_transfer_id(sequence_);
        call->request.set_retired_through(state_->admission->retiredThrough(ip_ + ":" + std::to_string(port_)));
        call->request.set_request_id(request_->request_block_buffer->getRequestId());
        call->request.set_timeout_ms(remaining);
        for (const auto& item : request_->request_block_buffer->getBlocks()) {
            auto* block = call->request.add_blocks();
            block->set_key(item.first);
            block->set_bytes(item.second->len);
        }
        collector_->markRequestCallBegin();
        hixl_proto::HixlCacheStoreService_Stub stub(channel.get());
        stub.manifest(&call->controller, &call->request, &call->response, call);
    }

    void manifest(bool ok, const hixl_proto::ManifestResponse& response) {
        if (!ok || !response.has_status() || response.status() != hixl_proto::OK) {
            retireAndFinish(ok && response.status() == hixl_proto::DEADLINE_EXCEEDED ?
                       CacheStoreErrorCode::LoadBufferTimeout :
                       CacheStoreErrorCode::LoadSendRequestFailed);
            return;
        }
        std::vector<HixlTransferSlice> plan;
        try {
            plan = makeHixlReadPlan(*request_->request_block_buffer, response, *state_->memory);
        } catch (const std::invalid_argument& error) {
            RTP_LLM_LOG_ERROR("HIXL invalid manifest: %s", error.what());
            retireAndFinish(CacheStoreErrorCode::InvalidParams);
            return;
        }
        auto        self = shared_from_this();
        HixlReadJob job{response.engine_id(), std::move(plan), deadline_, request_, [self](HixlReadResult result) {
                            self->retireAndFinish(readError(result));
                        }};
        job.source_instance = response.source_instance();
        if (!state_->progress->submit(std::move(job))) {
            retireAndFinish(CacheStoreErrorCode::PushWorkerItemFailed);
        }
    }

    void retireAndFinish(CacheStoreErrorCode error) {
        if (retirement_started_.exchange(true)) {
            return;
        }
        auto self = shared_from_this();
        // Physical READ has drained (or was never submitted). Queue release
        // before handing control to the load callback.
        state_->timers->schedule(HixlTimerQueue::Clock::now(), [self] { self->release(); });
        finish(error);
    }

    void finish(CacheStoreErrorCode error) {
        collector_->markRequestCallEnd(0);
        collector_->markEnd(error == CacheStoreErrorCode::None);
        auto completed = std::move(request_);
        completed->callback(error == CacheStoreErrorCode::None, error);
        // Drop our reference, without mutating the caller's request. Physical
        // DMA has ended; retirement retries only retain endpoint metadata.
        collector_.reset();
    }

    void release() {
        auto self    = shared_from_this();
        auto channel = state_->tcp->getChannel(ip_, port_);
        if (!channel) {
            retryRelease();
            return;
        }
        auto* call = new RpcCall<hixl_proto::ReleaseRequest, hixl_proto::ReleaseResponse>(
            channel, 1000, [self](bool ok, const auto& response) {
                if (ok && response.released()) {
                    self->state_->admission->retired(self->ip_ + ":" + std::to_string(self->port_), self->sequence_);
                } else {
                    self->retryRelease();
                }
            });
        call->request.set_reader_id(state_->reader);
        call->request.set_reader_engine_id(state_->engine);
        call->request.set_transfer_id(sequence_);
        call->request.set_retired_through(state_->admission->retiredThrough(ip_ + ":" + std::to_string(port_)));
        hixl_proto::HixlCacheStoreService_Stub stub(channel.get());
        stub.release(&call->controller, &call->request, &call->response, call);
    }

    void retryRelease() {
        // A lost reply is retried idempotently. Never expire a source DMA grant
        // on a timer; unacknowledged calls continue occupying bounded admission.
        auto self = shared_from_this();
        state_->timers->schedule(HixlTimerQueue::Clock::now() + std::chrono::milliseconds(100),
                                 [self] { self->release(); });
    }

    std::shared_ptr<HixlClientState>                      state_;
    std::shared_ptr<LoadRequest>                          request_;
    std::shared_ptr<CacheStoreClientLoadMetricsCollector> collector_;
    uint64_t                                              sequence_{0};
    std::atomic<bool>                                     retirement_started_{false};
    const std::string                                     ip_;
    const uint32_t                                        port_;
    const std::chrono::steady_clock::time_point           deadline_;
};
}  // namespace

HixlMessager::HixlMessager(const CacheStoreInitParams&                     params,
                           std::shared_ptr<HixlMemoryUtil>                 memory,
                           const std::shared_ptr<RequestBlockBufferStore>& store,
                           const kmonitor::MetricsReporterPtr&             metrics):
    Messager(memory, store, metrics) {
    client_            = std::make_shared<HixlClientState>();
    client_->reader    = instanceId();
    client_->engine    = params.hixl_engine_id;
    client_->admission = std::make_unique<HixlLoadScheduler>(kMaxInflightLoads, params.queue_size);
    client_->timers    = std::make_shared<HixlTimerQueue>();
    client_->memory    = std::move(memory);
    client_->progress  = std::make_unique<HixlReadProgress>(
        std::make_unique<HixlSdkReadDriver>(client_->memory, params.rdma_connect_timeout_ms));
    grants_  = std::make_shared<HixlSourceGrants>();
    service_ = std::make_shared<HixlManifestService>(
        client_->engine, client_->reader, client_->memory, request_block_buffer_store_, grants_, client_->timers);
}

bool HixlMessager::init(MessagerInitParams params) {
    init_params_ = params;
    tcp_client_  = std::make_shared<TcpClient>();
    client_->tcp = tcp_client_;
    server_      = std::make_shared<TcpServer>();
    if (!tcp_client_->init(params.io_thread_count)
        || !server_->init(params.io_thread_count, params.worker_thread_count, true)) {
        return false;
    }
    return server_->registerService(service_.get());
}

void HixlMessager::finishRegistration() {
    client_->memory->freezeRegistrations();
    if (!client_->progress->start() || !server_->start(init_params_.server_port)) {
        throw std::runtime_error("HIXL progress/control server startup failed");
    }
    service_->setReady(true);
    client_->admission->start();
}

void HixlMessager::load(const std::shared_ptr<LoadRequest>&                          request,
                        const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collector) {
    if (request->partition_count != 1 || request->partition_id != 0) {
        request->callback(false, CacheStoreErrorCode::InvalidParams);
        return;
    }
    std::make_shared<ClientRead>(client_, request, collector)->enqueue();
}

void HixlMessager::transfer(const std::shared_ptr<TransferRequest>& request) {
    RTP_LLM_LOG_ERROR("HIXL supports the PD pull path; remote-store push is not configured");
    request->callback(false, CacheStoreErrorCode::InvalidParams, {});
}

HixlMessager::~HixlMessager() {
    service_->setReady(false);
    request_block_buffer_store_->stop();
    client_->admission->stop();
    client_->progress->stop();
    if (!client_->admission->waitForIdle(std::chrono::seconds(30))) {
        RTP_LLM_LOG_ERROR("HIXL shutdown still has %zu unacknowledged control operations",
                          client_->admission->inflight());
        std::terminate();
    }
    // Keep the release RPC available while remote readers finish their DMA.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (grants_->size() != 0) {
        if (std::chrono::steady_clock::now() >= deadline) {
            RTP_LLM_LOG_ERROR("HIXL shutdown has live source grants; retaining memory until process exit");
            std::terminate();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    server_.reset();
    service_.reset();
    client_->timers->stop();
    HixlSdkReadDriver driver(client_->memory, 1000);
    if (!driver.activate()) {
        std::terminate();
    }
    for (const auto& peer : grants_->readerEngines()) {
        if (!driver.disconnect(peer)) {
            std::terminate();
        }
        while (driver.connection(peer) != HixlLinkState::CLOSED) {
            if (std::chrono::steady_clock::now() >= deadline) {
                RTP_LLM_LOG_ERROR("HIXL source disconnect did not drain %s", peer.c_str());
                std::terminate();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    client_->memory->thawAfterDisconnect();
}
}  // namespace rtp_llm
