#include <gtest/gtest.h>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <future>
#include <limits>
#include <stdexcept>
#include <thread>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include "autil/NetUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PWorkerDecodeWrite.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PWorkerPrefillWrite.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PKeyUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PBroadcastClient.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnector.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorPrefill.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PSchedulerPrefillWrite.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorDecode.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/TcpKVCacheReceiver.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/TcpKVCacheSender.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {
namespace {

class WriteBufferConverter: public LayerBlockConverter {
public:
    mutable std::array<std::array<std::array<uint8_t, 8>, 4>, 2> bytes{};

    std::vector<BlockInfo>
    convertIndexToBuffer(int layer, const std::string&, int block, int = 1, int = 0) const override {
        BlockInfo info;
        info.addr       = bytes.at(layer).at(block).data();
        info.size_bytes = bytes.at(layer).at(block).size();
        return {info};
    }
    std::vector<std::pair<BlockInfo, size_t>> getAllBuffers() const override {
        return {};
    }
};

class WriteCudaBufferConverter: public LayerBlockConverter {
public:
    torch::Tensor bytes = torch::zeros({2, 4, 8}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

    std::vector<BlockInfo>
    convertIndexToBuffer(int layer, const std::string&, int block, int = 1, int = 0) const override {
        BlockInfo info;
        info.is_cuda      = true;
        info.device_index = bytes.get_device();
        info.addr         = bytes[layer][block].data_ptr();
        info.size_bytes   = bytes.size(2);
        return {info};
    }
    std::vector<std::pair<BlockInfo, size_t>> getAllBuffers() const override {
        return {};
    }
};

class WriteTestReceiver: public transfer::IKVCacheReceiver {
public:
    transfer::TransferTaskStore             tasks;
    int                                     registrations = 0;
    int                                     fail_after    = -1;
    bool                                    start_on_recv = false;
    std::shared_ptr<transfer::TransferTask> active;
    bool                                    regMem(const BlockInfo&, uint64_t = 0) override {
        return true;
    }
    transfer::IKVCacheRecvTaskPtr recv(const transfer::RecvRequest& request) override {
        if (registrations++ == fail_after) {
            return nullptr;
        }
        auto task = tasks.addTask(request.unique_key, request.block_info, request.deadline_ms);
        if (start_on_recv && task) {
            task->startTransfer();
            active = task;
        }
        return task;
    }
    void stealTask(const std::string& key) override {
        tasks.stealTask(key);
    }
    transfer::IKVCacheRecvTaskPtr getTask(const std::string& key) override {
        return tasks.getTask(key);
    }
};

class WriteTestSender: public transfer::IKVCacheSender {
public:
    using Callback = std::function<void(transfer::TransferErrorCode, const std::string&)>;
    explicit WriteTestSender(std::shared_ptr<WriteTestReceiver> target): target_(std::move(target)) {}
    std::vector<std::pair<transfer::SendRequest, Callback>> pending;
    std::vector<std::shared_ptr<transfer::TransferTask>>    active;
    std::vector<Callback>                                   completed_callbacks;
    bool                                                    retain_completed_callbacks = false;

    bool regMem(const BlockInfo&, uint64_t = 0) override {
        return true;
    }
    void send(const transfer::SendRequest& request, Callback callback) override {
        std::lock_guard<std::mutex> lock(mutex_);
        pending.emplace_back(request, std::move(callback));
        cv_.notify_all();
    }
    size_t pendingCount() {
        std::lock_guard<std::mutex> lock(mutex_);
        return pending.size();
    }
    bool waitPending(size_t count) {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, std::chrono::seconds(5), [&]() { return pending.size() == count; });
    }
    void begin() {
        ASSERT_TRUE(waitPending(2));
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& [request, callback] : pending) {
            auto task = target_->tasks.getTask(request.unique_key);
            ASSERT_NE(task, nullptr);
            ASSERT_TRUE(task->startTransfer());
            active.push_back(std::move(task));
        }
    }
    void finish(bool success = true, size_t count = std::numeric_limits<size_t>::max()) {
        std::lock_guard<std::mutex> lock(mutex_);
        ASSERT_EQ(active.size(), pending.size());
        count = std::min(count, pending.size());
        for (size_t i = 0; i < count; ++i) {
            auto& [request, callback] = pending[i];
            if (success) {
                for (const auto& [key, source] : request.block_info) {
                    const auto& target = active[i]->getBlockInfos().at(key);
                    ASSERT_EQ(source->blocks.size(), target->blocks.size());
                    for (size_t b = 0; b < source->blocks.size(); ++b) {
                        ASSERT_EQ(source->blocks[b].size_bytes, target->blocks[b].size_bytes);
                        std::memcpy(target->blocks[b].addr, source->blocks[b].addr, source->blocks[b].size_bytes);
                    }
                }
            }
            const auto code = success ? transfer::TransferErrorCode::OK : transfer::TransferErrorCode::UNKNOWN;
            active[i]->notifyDone(success, code, "test completion");
            callback(code, "test completion");
            if (retain_completed_callbacks) {
                completed_callbacks.push_back(std::move(callback));
            }
        }
        pending.erase(pending.begin(), pending.begin() + count);
        active.erase(active.begin(), active.begin() + count);
    }

private:
    std::mutex                         mutex_;
    std::condition_variable            cv_;
    std::shared_ptr<WriteTestReceiver> target_;
};

class BlockingWriteSender: public transfer::IKVCacheSender {
public:
    using Callback = WriteTestSender::Callback;
    bool regMem(const BlockInfo&, uint64_t = 0) override {
        return true;
    }
    void send(const transfer::SendRequest& request, Callback callback) override {
        std::unique_lock<std::mutex> lock(mutex);
        ++calls;
        requests.push_back(request);
        callbacks.push_back(std::move(callback));
        cv.notify_all();
        cv.wait_for(lock, std::chrono::seconds(5), [&]() { return released; });
        if (throw_on_send) {
            throw std::runtime_error("sender failure");
        }
    }
    bool waitEntered() {
        std::unique_lock<std::mutex> lock(mutex);
        return cv.wait_for(lock, std::chrono::seconds(5), [&]() { return calls > 0; });
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        cv.notify_all();
    }
    void complete() {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto& callback : callbacks) {
            callback(transfer::TransferErrorCode::OK, "completed");
        }
        callbacks.clear();
    }
    std::mutex                         mutex;
    std::condition_variable            cv;
    std::vector<Callback>              callbacks;
    std::vector<transfer::SendRequest> requests;
    size_t                             calls         = 0;
    bool                               released      = false;
    bool                               throw_on_send = false;
};

std::unique_ptr<P2PConnector> makeWriteDecodeConnector(const P2PConnectorWorkerConfig&             worker_config,
                                                       const std::shared_ptr<LayerBlockConverter>& converter,
                                                       const transfer::IKVCacheSenderPtr&          sender) {
    P2PConnectorConfig config;
    config.role_type                  = RoleType::DECODE;
    config.p2p_writeback_enable       = true;
    config.worker_config              = worker_config;
    auto connector                    = std::make_unique<P2PConnector>(config, converter, nullptr);
    connector->decode_                = std::make_unique<P2PConnectorDecode>(config, converter, nullptr);
    connector->decode_->write_worker_ = std::make_unique<P2PWorkerDecodeWrite>(worker_config, converter, sender);
    return connector;
}

std::unique_ptr<P2PConnector> makeWritePrefillConnector(const P2PConnectorWorkerConfig&             worker_config,
                                                        const std::shared_ptr<LayerBlockConverter>& converter,
                                                        const transfer::IKVCacheReceiverPtr&        receiver) {
    P2PConnectorConfig config;
    config.role_type            = RoleType::PREFILL;
    config.p2p_writeback_enable = true;
    config.worker_config        = worker_config;
    auto connector              = std::make_unique<P2PConnector>(config, converter, nullptr);
    connector->prefill_         = std::make_unique<P2PConnectorPrefill>(config, converter, nullptr);
    connector->prefill_->write_scheduler_ =
        std::make_unique<P2PSchedulerPrefillWrite>(config.scheduler_config, nullptr, nullptr, nullptr);
    connector->prefill_->write_worker_ = std::make_unique<P2PWorkerPrefillWrite>(worker_config, converter, receiver);
    return connector;
}

bool executeWrite(P2PConnector&                           connector,
                  const P2PConnectorBroadcastTpRequestPB& request,
                  FunctionResponsePB&                     response) {
    FunctionRequestPB function_request;
    auto*             p2p_request = function_request.mutable_p2p_request();
    *p2p_request                  = request;
    if (p2p_request->type() == QUERY_LEASE_STATUS) {
        p2p_request->set_lease_operation(WRITE);
    }
    if (connector.config_.role_type == RoleType::PREFILL) {
        switch (p2p_request->type()) {
            case WRITE:
                p2p_request->set_type(HANDLE_WRITE);
                break;
            case CANCEL_WRITE:
                p2p_request->set_type(CANCEL_HANDLE_WRITE);
                break;
            case QUERY_LEASE_STATUS:
                p2p_request->set_lease_operation(HANDLE_WRITE);
                break;
            default:
                break;
        }
    }
    std::shared_ptr<P2PSchedulerPrefillWrite::PrefillWriteRequest> registration;
    if (p2p_request->type() == HANDLE_WRITE) {
        auto& scheduler = connector.prefill_->write_scheduler_;
        std::lock_guard<std::mutex> lock(scheduler->requests_mutex_);
        auto&                       state = scheduler->requests_[p2p_request->unique_key()];
        if (!state) {
            state = std::make_shared<P2PSchedulerPrefillWrite::PrefillWriteRequest>(p2p_request->deadline_ms());
            state->registered_workers.assign(1, false);
            state->plan_digest = p2p_request->plan_digest();
        }
        registration = state;
    }
    const bool success = connector.executeFunction(function_request, response);
    if (registration) {
        std::lock_guard<std::mutex> lock(registration->mutex);
        registration->stopped = true;
    }
    return success;
}

class P2PConnectorWriteWorkerTest: public ::testing::Test {
protected:
    void SetUp() override {
        config_.topology  = test::makeTestCacheTopology(2, 2, {{0}, {1}});
        decode_connector_ = makeWriteDecodeConnector(config_, source_, sender_);
        decode_           = decode_connector_->decode_->write_worker_.get();
        ASSERT_TRUE(decode_->init());
        prefill_connector_ = makeWritePrefillConnector(config_, target_, receiver_);
        prefill_           = prefill_connector_->prefill_->write_worker_.get();
        for (int layer = 0; layer < 2; ++layer) {
            for (int block = 1; block <= 2; ++block) {
                source_->bytes[layer][block].fill(10 * layer + block);
            }
        }
    }

    P2PConnectorBroadcastTpRequestPB request() {
        P2PConnectorBroadcastTpRequestPB request;
        request.set_type(WRITE);
        request.set_unique_key("write_request");
        request.set_deadline_ms(currentTimeMs() + 60000);
        request.set_plan_digest(123);
        auto* peer = request.add_peer_workers();
        peer->set_ip("127.0.0.1");
        peer->set_cache_store_port(12345);
        for (int layer = 0; layer < 2; ++layer) {
            auto* route = request.add_routes();
            route->set_route_id(layer);
            route->set_cache_tag("group" + std::to_string(layer));
            route->set_peer_index(0);
            route->set_partition_count(1);
            route->set_slice_count(1);
            auto* blocks = route->add_layer_blocks();
            blocks->set_layer_id(layer);
            blocks->set_cache_tag(route->cache_tag());
            for (int block = 1; block <= 2; ++block) {
                blocks->add_cache_keys(100 + block);
                blocks->add_block_ids(block);
            }
        }
        return request;
    }

    P2PWorkerRoutePlan workerPlan() {
        P2PWorkerRoutePlan plan;
        plan.plan_digest = 123;
        for (int layer = 0; layer < 2; ++layer) {
            P2PWorkerRoute route;
            route.route_id  = layer;
            route.cache_tag = "group" + std::to_string(layer);
            route.dst_ip    = "127.0.0.1";
            route.dst_port  = 12345;
            auto buffer     = std::make_shared<LayerCacheBuffer>(layer, route.cache_tag);
            buffer->addBlockId(101, 1);
            buffer->addBlockId(102, 2);
            route.layer_buffers.push_back(std::move(buffer));
            plan.routes.push_back(std::move(route));
        }
        return plan;
    }

    P2PConnectorWorkerConfig              config_;
    std::shared_ptr<WriteBufferConverter> source_   = std::make_shared<WriteBufferConverter>();
    std::shared_ptr<WriteBufferConverter> target_   = std::make_shared<WriteBufferConverter>();
    std::shared_ptr<WriteTestReceiver>    receiver_ = std::make_shared<WriteTestReceiver>();
    std::shared_ptr<WriteTestSender>      sender_   = std::make_shared<WriteTestSender>(receiver_);
    P2PWorkerDecodeWrite*                 decode_   = nullptr;
    P2PWorkerPrefillWrite*                prefill_  = nullptr;
    std::unique_ptr<P2PConnector>         decode_connector_;
    std::unique_ptr<P2PConnector>         prefill_connector_;
};

struct LeaseSnapshot {
    bool sealed       = false;
    int  started_ops  = 0;
    int  finished_ops = 0;
    bool stopped      = false;
};
bool queryLease(P2PWorkerDecodeWrite& worker, const std::string& key, LeaseSnapshot& status) {
    std::lock_guard<std::mutex> lock(worker.mutex_);
    const auto it = worker.groups_.find(key);
    if (it == worker.groups_.end()) {
        status = {};
        return false;
    }
    const auto& lease = it->second->lease;
    status = {lease->isSealed(), lease->startedOps(), lease->finishedOps(), lease->isStopped()};
    return true;
}

template<class Worker>
bool queryLease(Worker& worker, const std::string& key, LeaseSnapshot& status) {
    return worker.queryLeaseStatus(key, status.sealed, status.started_ops, status.finished_ops, status.stopped);
}
static bool awaitWrite(const std::function<bool()>& predicate, int timeout = 2000) {
    const auto deadline = currentTimeMs() + timeout;
    while (!predicate() && currentTimeMs() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return predicate();
}

TEST_F(P2PConnectorWriteWorkerTest, ReservedSourceIsUsedBySendTasksAndSurvivesCancellation) {
    sender_->retain_completed_callbacks     = true;
    const auto                     deadline = currentTimeMs() + 3000;
    auto                           source   = std::make_shared<KVCacheResource>();
    std::weak_ptr<KVCacheResource> weak     = source;
    auto                           group    = decode_->holdSourceResource("internal", deadline, std::move(source));
    ASSERT_NE(group, nullptr);
    EXPECT_FALSE(weak.expired());
    auto recv =
        std::async(std::launch::async, [&]() { return prefill_->handleWrite(1, "internal", deadline, workerPlan()); });
    LeaseSnapshot status;
    ASSERT_TRUE(awaitWrite([&]() { return queryLease(*prefill_, "internal", status) && status.sealed; }));
    auto send = std::async(std::launch::async, [&]() { return decode_->write(1, "internal", deadline, workerPlan()); });
    sender_->begin();
    EXPECT_EQ(decode_->groups_.at("internal"), group);
    ASSERT_TRUE(decode_->cancelWrite("internal", deadline));
    ASSERT_TRUE(prefill_->cancelWrite("internal", deadline));
    EXPECT_TRUE(send.get().hasError());
    EXPECT_TRUE(recv.get().hasError());
    {
        std::lock_guard<std::mutex> lock(group->mutex);
        group->source_resource_.reset();
    }
    EXPECT_EQ(group->sourceResource(), nullptr);
    EXPECT_FALSE(weak.expired());
    sender_->finish(false, 1);
    EXPECT_FALSE(weak.expired());
    sender_->finish(false);
    ASSERT_TRUE(queryLease(*decode_, "internal", status));
    EXPECT_TRUE(status.stopped);
    ASSERT_TRUE(awaitWrite([&]() { return weak.expired(); }));
    EXPECT_EQ(sender_->completed_callbacks.size(), 2u);
}

TEST_F(P2PConnectorWriteWorkerTest, ReservedSourceCancellationFencesLateStart) {
    const auto                     deadline = currentTimeMs() + 3000;
    auto                           source   = std::make_shared<KVCacheResource>();
    std::weak_ptr<KVCacheResource> weak     = source;
    auto                           group    = decode_->holdSourceResource("internal", deadline, std::move(source));
    ASSERT_NE(group, nullptr);
    EXPECT_EQ(decode_->holdSourceResource("internal", deadline, std::make_shared<KVCacheResource>()), nullptr);
    ASSERT_TRUE(decode_->cancelWrite("internal", deadline));
    {
        std::lock_guard<std::mutex> lock(group->mutex);
        group->source_resource_.reset();
    }
    EXPECT_TRUE(weak.expired());
    EXPECT_TRUE(decode_->write(1, "internal", deadline, workerPlan()).hasError());
    EXPECT_EQ(sender_->pendingCount(), 0u);
    LeaseSnapshot status;
    ASSERT_TRUE(queryLease(*decode_, "internal", status));
    EXPECT_TRUE(status.sealed);
    EXPECT_TRUE(status.stopped);
    EXPECT_EQ(status.started_ops, 0);
}

TEST_F(P2PConnectorWriteWorkerTest, OriginalWorkerCallsWaitForFinalTransferResults) {
    const auto        deadline = currentTimeMs() + 3000;
    std::atomic<bool> registered{false};
    auto              recv = std::async(std::launch::async, [&]() {
        return prefill_->handleWrite(1, "internal", deadline, workerPlan(), [&]() {
            registered = true;
            return ErrorInfo::OkStatus();
        });
    });
    ASSERT_TRUE(awaitWrite([&]() { return registered.load(); }));
    EXPECT_EQ(recv.wait_for(std::chrono::milliseconds(0)), std::future_status::timeout);
    auto send = std::async(std::launch::async, [&]() { return decode_->write(1, "internal", deadline, workerPlan()); });
    sender_->begin();
    EXPECT_EQ(send.wait_for(std::chrono::milliseconds(0)), std::future_status::timeout);
    sender_->finish();
    EXPECT_TRUE(recv.get().ok());
    EXPECT_TRUE(send.get().ok());
    EXPECT_EQ(source_->bytes, target_->bytes);
    LeaseSnapshot status;
    ASSERT_TRUE(queryLease(*prefill_, "internal", status));
    EXPECT_TRUE(status.stopped);
    EXPECT_EQ(status.finished_ops, 2);
}

TEST_F(P2PConnectorWriteWorkerTest, CancellationReturnsFailureBeforePhysicalStop) {
    const auto deadline = currentTimeMs() + 3000;
    auto       recv =
        std::async(std::launch::async, [&]() { return prefill_->handleWrite(1, "internal", deadline, workerPlan()); });
    LeaseSnapshot status;
    ASSERT_TRUE(awaitWrite([&]() { return queryLease(*prefill_, "internal", status) && status.sealed; }));
    auto send = std::async(std::launch::async, [&]() { return decode_->write(1, "internal", deadline, workerPlan()); });
    sender_->begin();
    ASSERT_TRUE(prefill_->cancelWrite("internal", deadline));
    ASSERT_TRUE(decode_->cancelWrite("internal", deadline));
    EXPECT_TRUE(recv.get().hasError());
    EXPECT_TRUE(send.get().hasError());
    ASSERT_TRUE(queryLease(*prefill_, "internal", status));
    EXPECT_FALSE(status.stopped);
    ASSERT_TRUE(queryLease(*decode_, "internal", status));
    EXPECT_FALSE(status.stopped);
    sender_->finish(false);
    ASSERT_TRUE(queryLease(*prefill_, "internal", status));
    EXPECT_TRUE(status.stopped);
    ASSERT_TRUE(queryLease(*decode_, "internal", status));
    EXPECT_TRUE(status.stopped);
}

TEST_F(P2PConnectorWriteWorkerTest, WorkerRegistrationErrorRequiresSeparateStopQuery) {
    receiver_->start_on_recv = true;
    receiver_->fail_after    = 1;
    EXPECT_TRUE(prefill_->handleWrite(1, "partial", currentTimeMs() + 3000, workerPlan()).hasError());
    LeaseSnapshot status;
    ASSERT_TRUE(queryLease(*prefill_, "partial", status));
    EXPECT_FALSE(status.stopped);
    EXPECT_EQ(status.started_ops, 1);
    receiver_->active->notifyDone(true);
    ASSERT_TRUE(queryLease(*prefill_, "partial", status));
    EXPECT_TRUE(status.stopped);
    EXPECT_EQ(status.finished_ops, 1);
}

TEST_F(P2PConnectorWriteWorkerTest, DeadlineReturnsFailureAndKeepsActiveLease) {
    receiver_->start_on_recv = true;
    auto plan                = workerPlan();
    plan.routes.resize(1);
    const auto deadline = currentTimeMs() + 100;
    EXPECT_TRUE(prefill_->handleWrite(1, "expiry", deadline, plan).hasError());
    LeaseSnapshot status;
    ASSERT_TRUE(queryLease(*prefill_, "expiry", status));
    EXPECT_FALSE(status.stopped);
    EXPECT_EQ(prefill_->groups_.at("expiry")->deadline_ms, deadline);
    receiver_->active->notifyDone(true);
    ASSERT_TRUE(queryLease(*prefill_, "expiry", status));
    EXPECT_TRUE(status.stopped);
}

TEST_F(P2PConnectorWriteWorkerTest, RegistrationNotificationFailureCancelsPendingReceives) {
    EXPECT_TRUE(prefill_
                    ->handleWrite(1,
                                  "ready_fail",
                                  currentTimeMs() + 3000,
                                  workerPlan(),
                                  []() { return ErrorInfo(ErrorCode::CANCELLED, "registration callback failed"); })
                    .hasError());
    EXPECT_EQ(receiver_->tasks.getTaskCount(), 0);
    LeaseSnapshot status;
    ASSERT_TRUE(queryLease(*prefill_, "ready_fail", status));
    EXPECT_TRUE(status.stopped);
}

TEST_F(P2PConnectorWriteWorkerTest, CompletedTasksDoNotStopUnsealedGroup) {
    auto group = std::make_shared<p2p_internal::WriteTaskGroup>(currentTimeMs() + 3000);
    auto task  = std::make_shared<transfer::TransferTask>(transfer::KeyBlockInfoMap{}, group->deadline_ms);
    group->tasks.emplace("task", task);
    group->lease->onTransferStarted();
    group->watchTask("task", task);
    task->notifyDone(true);
    EXPECT_FALSE(group->lease->isStopped());
    group->lease->seal();
    EXPECT_TRUE(group->lease->isStopped());
    EXPECT_TRUE(group->waitDone().ok());
}

TEST_F(P2PConnectorWriteWorkerTest, LeaseCountsStayStableAcrossQueriesAndCleanup) {
    auto future = std::async(
        std::launch::async, [&]() { return prefill_->handleWrite(1, "counts", currentTimeMs() + 3000, workerPlan()); });
    LeaseSnapshot status;
    ASSERT_TRUE(awaitWrite([&]() { return queryLease(*prefill_, "counts", status) && status.sealed; }));
    ASSERT_TRUE(prefill_->cancelWrite("counts", currentTimeMs() + 3000));
    EXPECT_TRUE(future.get().hasError());
    for (int i = 0; i < 5; ++i) {
        prefill_->cleanup();
        ASSERT_TRUE(queryLease(*prefill_, "counts", status));
        EXPECT_TRUE(status.stopped);
        EXPECT_EQ(status.started_ops, 2);
        EXPECT_EQ(status.finished_ops, 2);
    }
}

TEST_F(P2PConnectorWriteWorkerTest, RegistrationNotificationExceptionCancelsPendingReceives) {
    const auto error =
        prefill_->handleWrite(1, "notification-throws", currentTimeMs() + 3000, workerPlan(), []() -> ErrorInfo {
            throw std::runtime_error("registration RPC failed");
        });
    EXPECT_EQ(error.code(), ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED);
    LeaseSnapshot status;
    ASSERT_TRUE(queryLease(*prefill_, "notification-throws", status));
    EXPECT_TRUE(status.stopped);
    EXPECT_EQ(status.finished_ops, status.started_ops);
}

TEST_F(P2PConnectorWriteWorkerTest, EmptyRoutesSucceedWithFullQueue) {
    auto                 blocked = std::make_shared<BlockingWriteSender>();
    P2PWorkerDecodeWrite worker(config_, source_, blocked);
    ASSERT_TRUE(worker.init(1, 1));
    auto active = std::async(std::launch::async,
                             [&]() { return worker.write(1, "active", currentTimeMs() + 3000, workerPlan()); });
    ASSERT_TRUE(blocked->waitEntered());
    const auto                     queued_deadline = currentTimeMs() + 3000;
    auto                           owner           = std::make_shared<KVCacheResource>();
    std::weak_ptr<KVCacheResource> weak            = owner;
    auto queued_group = worker.holdSourceResource("queued", queued_deadline, std::move(owner));
    ASSERT_NE(queued_group, nullptr);
    auto queued =
        std::async(std::launch::async, [&]() { return worker.write(1, "queued", queued_deadline, workerPlan()); });
    LeaseSnapshot status;
    ASSERT_TRUE(awaitWrite([&]() { return queryLease(worker, "queued", status); }));
    ASSERT_TRUE(awaitWrite([&]() { return worker.sender_pool_->getItemCount() == 1; }));
    EXPECT_TRUE(worker.write(1, "empty", currentTimeMs() + 3000, P2PWorkerRoutePlan{}).ok());
    ASSERT_TRUE(queryLease(worker, "empty", status));
    EXPECT_TRUE(status.stopped);
    EXPECT_TRUE(worker.write(1, "full", currentTimeMs() + 3000, workerPlan()).hasError());
    worker.cancelWrite("active", currentTimeMs() + 3000);
    worker.cancelWrite("queued", queued_deadline);
    ASSERT_TRUE(queryLease(worker, "queued", status));
    EXPECT_TRUE(status.stopped);
    {
        std::lock_guard<std::mutex> lock(queued_group->mutex);
        queued_group->source_resource_.reset();
    }
    EXPECT_TRUE(weak.expired());
    blocked->release();
    blocked->complete();
    EXPECT_TRUE(active.get().hasError());
    EXPECT_TRUE(queued.get().hasError());
}

TEST_F(P2PConnectorWriteWorkerTest, QueuedSendUsesRemainingTransferBudget) {
    auto                 sender = std::make_shared<BlockingWriteSender>();
    P2PWorkerDecodeWrite worker(config_, source_, sender);
    ASSERT_TRUE(worker.init(1, 1));
    const auto deadline = currentTimeMs() + 3000;
    auto active = std::async(std::launch::async, [&]() { return worker.write(1, "active", deadline, workerPlan()); });
    ASSERT_TRUE(sender->waitEntered());
    auto queued = std::async(std::launch::async, [&]() { return worker.write(1, "queued", deadline, workerPlan()); });
    ASSERT_TRUE(awaitWrite([&]() { return worker.sender_pool_->getItemCount() == 1; }));
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    worker.cancelWrite("active", deadline);
    const auto remaining = deadline - currentTimeMs();
    sender->release();
    worker.sender_pool_->stop(autil::ThreadPoolBase::STOP_AFTER_QUEUE_EMPTY);
    ASSERT_EQ(sender->requests.size(), 3u);
    for (size_t i = 1; i < sender->requests.size(); ++i) {
        EXPECT_EQ(sender->requests[i].deadline_ms, deadline);
        EXPECT_GT(sender->requests[i].timeout_ms, 0);
        EXPECT_LE(sender->requests[i].timeout_ms, remaining);
    }
    sender->complete();
    EXPECT_TRUE(active.get().hasError());
    EXPECT_TRUE(queued.get().ok());
}

TEST_F(P2PConnectorWriteWorkerTest, QueuedDeadlineExpiresWithoutSendingOrExtendingDeadline) {
    auto                 sender = std::make_shared<BlockingWriteSender>();
    P2PWorkerDecodeWrite worker(config_, source_, sender);
    ASSERT_TRUE(worker.init(1, 1));
    worker.cleanup_thread_->stop();
    const auto active_deadline = currentTimeMs() + 3000;
    auto       active =
        std::async(std::launch::async, [&]() { return worker.write(1, "active", active_deadline, workerPlan()); });
    ASSERT_TRUE(sender->waitEntered());
    const auto deadline = currentTimeMs() + 80;
    auto queued = std::async(std::launch::async, [&]() { return worker.write(1, "expired", deadline, workerPlan()); });
    EXPECT_EQ(queued.get().code(), ErrorCode::GENERATE_TIMEOUT);
    worker.cancelWrite("active", active_deadline);
    sender->release();
    worker.sender_pool_->stop(autil::ThreadPoolBase::STOP_AFTER_QUEUE_EMPTY);
    EXPECT_EQ(sender->calls, 1u);
    sender->complete();
    EXPECT_TRUE(active.get().hasError());
    LeaseSnapshot status;
    ASSERT_TRUE(queryLease(worker, "expired", status));
    EXPECT_TRUE(status.stopped);
    EXPECT_EQ(worker.groups_.at("expired")->deadline_ms, deadline);
}

TEST_F(P2PConnectorWriteWorkerTest, SenderExceptionCancelsRemainingUnits) {
    auto sender           = std::make_shared<BlockingWriteSender>();
    sender->throw_on_send = true;
    sender->release();
    P2PWorkerDecodeWrite worker(config_, source_, sender);
    ASSERT_TRUE(worker.init(1, 1));
    EXPECT_TRUE(worker.write(1, "throws", currentTimeMs() + 3000, workerPlan()).hasError());
    worker.sender_pool_->stop(autil::ThreadPoolBase::STOP_AFTER_QUEUE_EMPTY);
    EXPECT_EQ(sender->calls, 1u);
    LeaseSnapshot status;
    ASSERT_TRUE(queryLease(worker, "throws", status));
    EXPECT_TRUE(status.stopped);
    EXPECT_EQ(status.started_ops, 2);
    EXPECT_EQ(status.finished_ops, 2);
}

TEST_F(P2PConnectorWriteWorkerTest, DuplicateStartCannotReplaceRegisteredBuffers) {
    const auto deadline = currentTimeMs() + 3000;
    auto       first =
        std::async(std::launch::async, [&]() { return prefill_->handleWrite(1, "duplicate", deadline, workerPlan()); });
    LeaseSnapshot status;
    ASSERT_TRUE(awaitWrite([&]() { return queryLease(*prefill_, "duplicate", status) && status.sealed; }));
    EXPECT_TRUE(prefill_->handleWrite(1, "duplicate", deadline, workerPlan()).hasError());
    EXPECT_EQ(receiver_->registrations, 2);
    prefill_->cancelWrite("duplicate", deadline);
    EXPECT_TRUE(first.get().hasError());
}

TEST_F(P2PConnectorWriteWorkerTest, FailedDataTransferReturnsOriginalError) {
    const auto deadline = currentTimeMs() + 3000;
    auto       recv =
        std::async(std::launch::async, [&]() { return prefill_->handleWrite(1, "failure", deadline, workerPlan()); });
    LeaseSnapshot status;
    ASSERT_TRUE(awaitWrite([&]() { return queryLease(*prefill_, "failure", status) && status.sealed; }));
    auto send = std::async(std::launch::async, [&]() { return decode_->write(1, "failure", deadline, workerPlan()); });
    sender_->begin();
    sender_->finish(false);
    EXPECT_TRUE(recv.get().hasError());
    EXPECT_TRUE(send.get().hasError());
    ASSERT_TRUE(queryLease(*decode_, "failure", status));
    EXPECT_TRUE(status.stopped);
}

TEST_F(P2PConnectorWriteWorkerTest, InvalidInternalRoutesHaveNoTransferSideEffects) {
    const std::vector<std::function<void(P2PWorkerRoutePlan&)>> corruptions = {
        [](P2PWorkerRoutePlan& plan) { plan.routes[1].route_id = 0; },
        [](P2PWorkerRoutePlan& plan) { plan.routes[1].layer_buffers[0].reset(); },
        [](P2PWorkerRoutePlan& plan) { plan.routes[1].cache_tag = "unknown"; },
        [](P2PWorkerRoutePlan& plan) { plan.routes[1].layer_buffers[0]->addBlockId(102, -1); },
        [](P2PWorkerRoutePlan& plan) { plan.routes[1].partition.count = 0; },
    };
    for (size_t i = 0; i < corruptions.size(); ++i) {
        SCOPED_TRACE(i);
        auto plan = workerPlan();
        corruptions[i](plan);
        const auto deadline = currentTimeMs() + 60000;
        EXPECT_TRUE(prefill_->handleWrite(1, "invalid", deadline, plan).hasError());
        EXPECT_TRUE(decode_->write(1, "invalid", deadline, plan).hasError());
        LeaseSnapshot status;
        EXPECT_FALSE(queryLease(*prefill_, "invalid", status));
        EXPECT_FALSE(queryLease(*decode_, "invalid", status));
    }
    EXPECT_EQ(receiver_->registrations, 0);
    EXPECT_EQ(sender_->pendingCount(), 0);
}

TEST_F(P2PConnectorWriteWorkerTest, WriteKeysCannotMatchReadTasks) {
    const auto read_key  = P2PKeyUtil::makeRouteLayerKey("request", 0, "group0", 1, 123);
    const auto write_key = P2PKeyUtil::makeWriteBackRouteLayerKey("request", 0, "group0", 1, 123);
    EXPECT_NE(read_key, write_key);
    EXPECT_NE(write_key, P2PKeyUtil::makeWriteBackRouteLayerKey("request", 0, "group0", 1, 124));
}

TEST_F(P2PConnectorWriteWorkerTest, CleanupIntervalComesFromCacheStoreConfig) {
    CacheStoreConfig cache_config;
    cache_config.p2p_resource_store_timeout_check_interval_ms = 37;
    cache_config.p2p_writeback_enable                         = true;
    cache_config.p2p_writeback_timeout_ms                     = 173;
    auto config =
        P2PConnectorConfig::create(RuntimeConfig{}, cache_config, ParallelismConfig{}, PDSepConfig{}, 2, false, 0);
    EXPECT_TRUE(config.p2p_writeback_enable);
    EXPECT_EQ(config.worker_config.p2p_resource_store_timeout_check_interval_ms, 37);
    EXPECT_EQ(config.scheduler_config.p2p_resource_store_timeout_check_interval_ms, 37);
    EXPECT_EQ(config.scheduler_config.p2p_writeback_timeout_ms, 173);
}

TEST_F(P2PConnectorWriteWorkerTest, UnknownTypesDoNotCreateTasks) {
    auto               req = request();
    FunctionResponsePB response;
    for (const auto type : {QUERY_LEASE_STATUS, static_cast<P2PConnectorBroadcastType>(99)}) {
        req.set_type(type);
        EXPECT_FALSE(executeWrite(*prefill_connector_, req, response));
        EXPECT_FALSE(executeWrite(*decode_connector_, req, response));
    }
    EXPECT_EQ(receiver_->registrations, 0);
    EXPECT_EQ(sender_->pendingCount(), 0);
}

TEST_F(P2PConnectorWriteWorkerTest, OverlappingReceiveBlocksAreRejectedBeforeRegistration) {
    auto  req       = request();
    auto* duplicate = req.add_routes();
    duplicate->CopyFrom(req.routes(0));
    duplicate->set_route_id(2);
    FunctionResponsePB response;
    EXPECT_FALSE(executeWrite(*prefill_connector_, req, response));
    EXPECT_NE(response.p2p_response().error_message().find("overlap"), std::string::npos);
    EXPECT_EQ(receiver_->registrations, 0);
}

TEST_F(P2PConnectorWriteWorkerTest, InvalidMetadataCannotPartiallyRegisterOrSend) {
    using Request                                                = P2PConnectorBroadcastTpRequestPB;
    const std::vector<std::function<void(Request&)>> corruptions = {
        [](Request& req) { req.mutable_routes(1)->set_route_id(0); },
        [](Request& req) { req.mutable_routes(1)->mutable_layer_blocks(0)->set_layer_id(9); },
        [](Request& req) { req.mutable_routes(1)->mutable_layer_blocks(0)->set_cache_tag("unknown"); },
        [](Request& req) { req.mutable_routes(1)->mutable_layer_blocks(0)->clear_block_ids(); },
        [](Request& req) { req.mutable_routes(1)->mutable_layer_blocks(0)->set_cache_keys(1, 101); },
        [](Request& req) { req.mutable_routes(1)->mutable_layer_blocks(0)->set_block_ids(0, 9); },
        [](Request& req) { req.mutable_routes(1)->set_slice_count(2); },
        [](Request& req) { req.mutable_routes(1)->set_partition_count(0); },
        [](Request& req) { req.mutable_routes(1)->set_slice_mode(99); },
        [](Request& req) {
            req.mutable_routes(1)->mutable_layer_blocks(0)->set_block_ids(0, std::numeric_limits<uint32_t>::max());
        },
        [](Request& req) {
            req.mutable_routes(1)->mutable_layer_blocks(0)->set_layer_id(std::numeric_limits<uint32_t>::max());
        },
    };
    for (size_t i = 0; i < corruptions.size(); ++i) {
        SCOPED_TRACE(i);
        auto req = request();
        corruptions[i](req);
        FunctionResponsePB response;
        EXPECT_FALSE(executeWrite(*prefill_connector_, req, response));
        EXPECT_TRUE(executeWrite(*decode_connector_, req, response));
    EXPECT_NE(response.p2p_response().error_code(), ErrorCodePB::NONE_ERROR);
    }
    EXPECT_EQ(receiver_->registrations, 0);
    EXPECT_EQ(sender_->pendingCount(), 0);
}

TEST_F(P2PConnectorWriteWorkerTest, InvalidLastRouteHasNoTransferSideEffects) {
    auto req = request();
    req.mutable_routes(1)->mutable_layer_blocks(0)->set_block_ids(0, 0);
    FunctionResponsePB response;
    EXPECT_FALSE(executeWrite(*prefill_connector_, req, response));
    EXPECT_TRUE(executeWrite(*decode_connector_, req, response));
    EXPECT_NE(response.p2p_response().error_code(), ErrorCodePB::NONE_ERROR);
    EXPECT_EQ(receiver_->registrations, 0);
    EXPECT_EQ(sender_->pendingCount(), 0);
}

TEST_F(P2PConnectorWriteWorkerTest, RejectsPartitionedRoutesAndExpiredRequests) {
    auto req = request();
    req.mutable_routes(0)->set_partition_count(2);
    FunctionResponsePB response;
    EXPECT_FALSE(executeWrite(*prefill_connector_, req, response));
    EXPECT_TRUE(executeWrite(*decode_connector_, req, response));
    EXPECT_NE(response.p2p_response().error_code(), ErrorCodePB::NONE_ERROR);
    req = request();
    req.set_deadline_ms(currentTimeMs() - 1);
    EXPECT_FALSE(executeWrite(*prefill_connector_, req, response));
    EXPECT_TRUE(executeWrite(*decode_connector_, req, response));
    EXPECT_NE(response.p2p_response().error_code(), ErrorCodePB::NONE_ERROR);
    EXPECT_EQ(receiver_->registrations, 0);
}

TEST_F(P2PConnectorWriteWorkerTest, SourceCancellationSurvivesLocalPhysicalStop) {
    auto group = decode_->holdSourceResource("source-local-stopped", currentTimeMs() + 1000,
                                             std::make_shared<KVCacheResource>());
    ASSERT_NE(group, nullptr);
    group->lease->seal();
    ASSERT_TRUE(group->lease->isStopped());
    group->cancel();
    EXPECT_TRUE(group->cancelled);
    EXPECT_NE(group->sourceResource(), nullptr);
}

TEST_F(P2PConnectorWriteWorkerTest, CancellationBeforeStartRejectsDelayedBroadcast) {
    auto req = request();
    req.set_type(CANCEL_WRITE);
    FunctionResponsePB response;
    ASSERT_TRUE(executeWrite(*prefill_connector_, req, response));
    EXPECT_TRUE(response.p2p_response().lease_status().stopped());
    response.Clear();
    ASSERT_TRUE(executeWrite(*decode_connector_, req, response));
    EXPECT_FALSE(response.p2p_response().has_lease_status());
    req.set_type(WRITE);
    EXPECT_FALSE(executeWrite(*prefill_connector_, req, response));
    EXPECT_TRUE(executeWrite(*decode_connector_, req, response));
    EXPECT_NE(response.p2p_response().error_code(), ErrorCodePB::NONE_ERROR);
    EXPECT_EQ(receiver_->registrations, 0);
    EXPECT_EQ(sender_->pendingCount(), 0);
}

TEST_F(P2PConnectorWriteWorkerTest, PartialRegistrationFailureRemovesRegisteredTasks) {
    receiver_->fail_after  = 1;
    auto               req = request();
    FunctionResponsePB response;
    EXPECT_FALSE(executeWrite(*prefill_connector_, req, response));
    EXPECT_EQ(receiver_->tasks.getTaskCount(), 0);
    LeaseSnapshot lease;
    ASSERT_TRUE(queryLease(*prefill_, req.unique_key(), lease));
    EXPECT_TRUE(lease.stopped);
}
class WriteRpcEngine: public EngineBase {
public:
    explicit WriteRpcEngine(std::shared_ptr<KVCacheManager> manager): EngineBase(EngineInitParams()) {
        resource_context_.cache_manager = std::move(manager);
    }
    GenerateStreamPtr enqueue(const std::shared_ptr<GenerateInput>&) override {
        return nullptr;
    }
    void         enqueue(GenerateStreamPtr&) override {}
    absl::Status stop() override {
        return absl::OkStatus();
    }
    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>&, preRunMode) override {
        return absl::UnimplementedError("unused in write RPC tests");
    }
    KVCacheInfo getCacheStatusInfo(int64_t, bool) override {
        return {};
    }
};

class WriteRpcService: public RpcService::Service {
public:
    LocalRpcServer    local;
    std::atomic<bool> lose_start_response{false};
    std::atomic<bool> omit_control_status{false};
    grpc::Status      ExecuteFunction(grpc::ServerContext*     context,
                                      const FunctionRequestPB* request,
                                      FunctionResponsePB*      response) override {
        auto status = local.ExecuteFunction(context, request, response);
        if (request->p2p_request().type() == HANDLE_WRITE && lose_start_response && status.ok()) {
            while (!context->IsCancelled()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        } else if (request->p2p_request().type() != HANDLE_WRITE && omit_control_status) {
            response->Clear();
        }
        return status;
    }
};

class P2PConnectorWriteRpcTest: public P2PConnectorWriteWorkerTest {
protected:
    void SetUp() override {
        P2PConnectorWriteWorkerTest::SetUp();
        auto connector = std::shared_ptr<P2PConnector>(std::move(prefill_connector_));
        auto manager   = std::make_shared<KVCacheManager>(test::makeSimpleMhaCacheConfig(2, 4, 1, DataType::TYPE_FP16),
                                                        /*warmup=*/true);
        manager->p2p_connector_ = std::move(connector);
        registration_ = std::make_shared<P2PSchedulerPrefillWrite::PrefillWriteRequest>(currentTimeMs() + 60000);
        registration_->registered_workers.assign(1, false);
        registration_->plan_digest = 123;
        manager->p2p_connector_->prefill_->write_scheduler_->requests_.emplace("write_request", registration_);
        service_.local.engine_ = std::make_shared<WriteRpcEngine>(std::move(manager));
        grpc::ServerBuilder builder;
        int                 port = 0;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
        builder.RegisterService(&service_);
        server_ = builder.BuildAndStart();
        ASSERT_NE(server_, nullptr);
        address_ = "127.0.0.1:" + std::to_string(port);
        stub_    = RpcService::NewStub(grpc::CreateChannel(address_, grpc::InsecureChannelCredentials()));
        client_  = std::make_shared<P2PBroadcastClient>(std::vector<std::string>{address_});
        ASSERT_TRUE(client_->init());
    }
    void TearDown() override {
        if (server_) {
            server_->Shutdown();
            server_->Wait();
        }
        std::lock_guard<std::mutex> lock(registration_->mutex);
        registration_->stopped = true;
    }
    grpc::Status
    startRpc(const P2PConnectorBroadcastTpRequestPB& req, FunctionResponsePB& response, int timeout_ms = 2000) {
        FunctionRequestPB request;
        request.mutable_p2p_request()->CopyFrom(req);
        request.mutable_p2p_request()->set_type(HANDLE_WRITE);
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(timeout_ms));
        return stub_->ExecuteFunction(&context, request, &response);
    }
    P2PBroadcastClient::LeaseStatusResult control(const P2PConnectorBroadcastTpRequestPB& req,
                                                  P2PConnectorBroadcastType               type) {
        const auto result = type == QUERY_LEASE_STATUS ?
                                client_->queryLeaseStatusAsync(req.unique_key(), HANDLE_WRITE, 1000) :
                                client_->controlWriteAsync(req.unique_key(), type, req.deadline_ms(), 1000);
        if (!result || !result->waitDone(1000, false))
            return {};
        return P2PBroadcastClient::leaseStatus(result);
    }
    std::shared_ptr<P2PSchedulerPrefillWrite::PrefillWriteRequest> registration_;
    WriteRpcService                     service_;
    std::unique_ptr<grpc::Server>       server_;
    std::string                         address_;
    std::unique_ptr<RpcService::Stub>   stub_;
    std::shared_ptr<P2PBroadcastClient> client_;
};

TEST_F(P2PConnectorWriteRpcTest, SenderWriteAndCancelResponsesWaitForEveryPhysicalSend) {
    WriteRpcService source_service;
    auto manager = std::make_shared<KVCacheManager>(test::makeSimpleMhaCacheConfig(2, 4, 1, DataType::TYPE_FP16),
                                                     /*warmup=*/true);
    manager->p2p_connector_ = std::shared_ptr<P2PConnector>(std::move(decode_connector_));
    source_service.local.engine_ = std::make_shared<WriteRpcEngine>(std::move(manager));
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&source_service);
    auto source_server = builder.BuildAndStart();
    ASSERT_NE(source_server, nullptr);
    auto source_stub = RpcService::NewStub(grpc::CreateChannel("127.0.0.1:" + std::to_string(port),
                                                              grpc::InsecureChannelCredentials()));
    const auto req = request();
    FunctionResponsePB receive_response, send_response, cancel_response;
    auto receiving = std::async(std::launch::async, [&]() { return startRpc(req, receive_response); });
    auto call = [&](P2PConnectorBroadcastType type, FunctionResponsePB& response) {
        FunctionRequestPB function;
        function.mutable_p2p_request()->CopyFrom(req);
        function.mutable_p2p_request()->set_type(type);
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
        return source_stub->ExecuteFunction(&context, function, &response);
    };
    auto sending = std::async(std::launch::async, [&]() { return call(WRITE, send_response); });
    sender_->begin();
    auto cancelling = std::async(std::launch::async, [&]() { return call(CANCEL_WRITE, cancel_response); });
    EXPECT_EQ(sending.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
    EXPECT_EQ(cancelling.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
    sender_->finish(false, 1);
    EXPECT_EQ(sending.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
    EXPECT_EQ(cancelling.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
    sender_->finish(false);
    EXPECT_TRUE(sending.get().ok());
    EXPECT_TRUE(cancelling.get().ok());
    EXPECT_NE(send_response.p2p_response().error_code(), ErrorCodePB::NONE_ERROR);
    EXPECT_EQ(cancel_response.p2p_response().error_code(), ErrorCodePB::NONE_ERROR);
    EXPECT_FALSE(cancel_response.p2p_response().has_lease_status());
    EXPECT_FALSE(receiving.get().ok());
    source_server->Shutdown();
    source_server->Wait();
}

TEST_F(P2PConnectorWriteRpcTest, FailedStartLosesStatusAndLeaseWaitsForPhysicalStop) {
    receiver_->fail_after    = 1;
    receiver_->start_on_recv = true;
    auto               req   = request();
    FunctionResponsePB response;
    EXPECT_EQ(startRpc(req, response).error_code(), grpc::StatusCode::INTERNAL);
    EXPECT_FALSE(response.has_p2p_response());
    EXPECT_FALSE(control(req, CANCEL_HANDLE_WRITE).allStopped());
    receiver_->active->notifyDone(true);
    EXPECT_TRUE(control(req, QUERY_LEASE_STATUS).allStopped());
}

TEST_F(P2PConnectorWriteRpcTest, RpcTimeoutDoesNotProveStopped) {
    receiver_->start_on_recv = true;
    auto req                 = request();
    req.mutable_routes()->DeleteSubrange(1, 1);
    FunctionResponsePB response;
    EXPECT_EQ(startRpc(req, response, 100).error_code(), grpc::StatusCode::DEADLINE_EXCEEDED);
    EXPECT_FALSE(response.has_p2p_response());
    EXPECT_FALSE(control(req, CANCEL_HANDLE_WRITE).allStopped());
    EXPECT_FALSE(control(req, QUERY_LEASE_STATUS).allStopped());
    receiver_->active->notifyDone(true);
    EXPECT_TRUE(control(req, QUERY_LEASE_STATUS).allStopped());
}

TEST_F(P2PConnectorWriteRpcTest, UnknownLeaseCannotConfirmStoppedUntilCancelFencesLateStart) {
    auto req = request();
    EXPECT_FALSE(control(req, QUERY_LEASE_STATUS).allStopped());
    EXPECT_TRUE(control(req, CANCEL_HANDLE_WRITE).allStopped());
    FunctionResponsePB response;
    EXPECT_EQ(startRpc(req, response).error_code(), grpc::StatusCode::INTERNAL);
    EXPECT_EQ(receiver_->registrations, 0);
}

TEST_F(P2PConnectorWriteRpcTest, MissingLeaseOrFailedRankCannotConfirmStopped) {
    auto req = request();
    EXPECT_TRUE(control(req, CANCEL_HANDLE_WRITE).allStopped());
    service_.omit_control_status = true;
    EXPECT_FALSE(control(req, QUERY_LEASE_STATUS).allStopped());
    service_.omit_control_status = false;
    P2PBroadcastClient partial({address_, "127.0.0.1:1"});
    ASSERT_TRUE(partial.init());
    const auto result = partial.controlWriteAsync(req.unique_key(), CANCEL_HANDLE_WRITE, req.deadline_ms(), 100);
    ASSERT_TRUE(result);
    result->waitDone(100, false);
    EXPECT_FALSE(P2PBroadcastClient::leaseStatus(result).allStopped());
}

TEST_F(P2PConnectorWriteRpcTest, FinalResponseCarriesReceiveSuccessWithoutStatusPolling) {
    auto               req = request();
    FunctionResponsePB response;
    auto               rpc = std::async(std::launch::async, [&]() { return startRpc(req, response); });
    LeaseSnapshot      lease;
    ASSERT_TRUE(awaitWrite([&]() { return queryLease(*prefill_, req.unique_key(), lease) && lease.sealed; }));
    EXPECT_EQ(rpc.wait_for(std::chrono::milliseconds(0)), std::future_status::timeout);
    for (const auto& route : req.routes()) {
        const auto& layer = route.layer_blocks(0);
        auto        task  = receiver_->tasks.getTask(P2PKeyUtil::makeWriteBackRouteLayerKey(
            req.unique_key(), layer.layer_id(), layer.cache_tag(), route.route_id(), req.plan_digest()));
        ASSERT_NE(task, nullptr);
        task->notifyDone(true);
    }
    EXPECT_TRUE(rpc.get().ok());
    EXPECT_EQ(response.p2p_response().error_code(), ErrorCodePB::NONE_ERROR);
}

}  // namespace
}  // namespace rtp_llm
