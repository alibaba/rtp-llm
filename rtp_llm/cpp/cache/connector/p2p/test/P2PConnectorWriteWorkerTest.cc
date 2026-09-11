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
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorPrefill.h"
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
    void send(const transfer::SendRequest&, Callback callback) override {
        std::unique_lock<std::mutex> lock(mutex);
        ++calls;
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
    std::mutex              mutex;
    std::condition_variable cv;
    std::vector<Callback>   callbacks;
    size_t                  calls         = 0;
    bool                    released      = false;
    bool                    throw_on_send = false;
};

std::unique_ptr<P2PConnectorDecode> makeWriteDecodeConnector(const P2PConnectorWorkerConfig&             worker_config,
                                                             const std::shared_ptr<LayerBlockConverter>& converter,
                                                             const transfer::IKVCacheSenderPtr&          sender) {
    P2PConnectorConfig config;
    config.role_type            = RoleType::DECODE;
    config.p2p_writeback_enable = true;
    config.worker_config        = worker_config;
    auto connector              = std::make_unique<P2PConnectorDecode>(config, converter, nullptr);
    connector->write_worker_    = std::make_unique<P2PWorkerDecodeWrite>(worker_config, converter, sender);
    return connector;
}

std::unique_ptr<P2PConnectorPrefill> makeWritePrefillConnector(const P2PConnectorWorkerConfig& worker_config,
                                                               const std::shared_ptr<LayerBlockConverter>& converter,
                                                               const transfer::IKVCacheReceiverPtr&        receiver) {
    P2PConnectorConfig config;
    config.role_type            = RoleType::PREFILL;
    config.p2p_writeback_enable = true;
    config.worker_config        = worker_config;
    auto connector              = std::make_unique<P2PConnectorPrefill>(config, converter, nullptr);
    connector->write_worker_    = std::make_unique<P2PWorkerPrefillWrite>(worker_config, converter, receiver);
    return connector;
}

class P2PConnectorWriteWorkerTest: public ::testing::Test {
protected:
    void SetUp() override {
        config_.topology  = test::makeTestCacheTopology(2, 2, {{0}, {1}});
        decode_connector_ = makeWriteDecodeConnector(config_, source_, sender_);
        decode_           = decode_connector_->write_worker_.get();
        ASSERT_TRUE(decode_->init());
        prefill_connector_ = makeWritePrefillConnector(config_, target_, receiver_);
        prefill_           = prefill_connector_->write_worker_.get();
        for (int layer = 0; layer < 2; ++layer) {
            for (int block = 1; block <= 2; ++block) {
                source_->bytes[layer][block].fill(10 * layer + block);
            }
        }
    }

    P2PConnectorBroadcastTpRequestPB request() {
        P2PConnectorBroadcastTpRequestPB request;
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
    std::unique_ptr<P2PConnectorDecode>   decode_connector_;
    std::unique_ptr<P2PConnectorPrefill>  prefill_connector_;
};

TEST_F(P2PConnectorWriteWorkerTest, WorkersAcceptInternalRoutesAndReportCompletion) {
    const auto plan     = workerPlan();
    const auto deadline = currentTimeMs() + 60000;
    ASSERT_TRUE(prefill_->handleWrite(1, "internal", deadline, plan).ok());
    WriteTaskStatus status;
    ASSERT_TRUE(prefill_->queryWriteStatus("internal", status));
    EXPECT_TRUE(status.sealed);
    EXPECT_EQ(status.started_ops, 2);
    EXPECT_FALSE(status.stopped);
    EXPECT_FALSE(status.write_success);
    ASSERT_TRUE(decode_->write(1, "internal", deadline, plan).ok());
    sender_->begin();
    sender_->finish();
    ASSERT_TRUE(prefill_->queryWriteStatus("internal", status));
    EXPECT_TRUE(status.stopped);
    EXPECT_TRUE(status.write_success);
    EXPECT_EQ(status.finished_ops, 2);
    ASSERT_TRUE(decode_->queryWriteStatus("internal", status));
    EXPECT_TRUE(status.stopped);
    EXPECT_TRUE(status.write_success);
    EXPECT_EQ(source_->bytes, target_->bytes);
}

TEST_F(P2PConnectorWriteWorkerTest, WorkerControlInterfacesPreserveUnknownAndCancelledStates) {
    const auto      deadline = currentTimeMs() + 60000;
    WriteTaskStatus status;
    status.stopped = true;
    EXPECT_FALSE(prefill_->queryWriteStatus("internal", status));
    EXPECT_FALSE(status.stopped);
    status.stopped = true;
    EXPECT_FALSE(decode_->queryWriteStatus("internal", status));
    EXPECT_FALSE(status.stopped);
    EXPECT_FALSE(prefill_->cancelWrite("", deadline));
    EXPECT_FALSE(decode_->cancelWrite("", deadline));
    ASSERT_TRUE(prefill_->cancelWrite("internal", deadline));
    ASSERT_TRUE(decode_->cancelWrite("internal", deadline));
    ASSERT_TRUE(prefill_->queryWriteStatus("internal", status));
    EXPECT_TRUE(status.sealed);
    EXPECT_TRUE(status.stopped);
    EXPECT_FALSE(status.write_success);
    EXPECT_EQ(status.started_ops, 0);
    ASSERT_TRUE(decode_->queryWriteStatus("internal", status));
    EXPECT_TRUE(status.sealed);
    EXPECT_TRUE(status.stopped);
    EXPECT_FALSE(status.write_success);
    EXPECT_EQ(status.started_ops, 0);
    EXPECT_TRUE(prefill_->handleWrite(1, "internal", deadline, workerPlan()).hasError());
    EXPECT_TRUE(decode_->write(1, "internal", deadline, workerPlan()).hasError());
    EXPECT_EQ(receiver_->registrations, 0);
    EXPECT_EQ(sender_->pendingCount(), 0);
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
        WriteTaskStatus status;
        EXPECT_FALSE(prefill_->queryWriteStatus("invalid", status));
        EXPECT_FALSE(decode_->queryWriteStatus("invalid", status));
    }
    EXPECT_EQ(receiver_->registrations, 0);
    EXPECT_EQ(sender_->pendingCount(), 0);
}

TEST_F(P2PConnectorWriteWorkerTest, WorkerRegistrationErrorRequiresSeparateStopQuery) {
    receiver_->start_on_recv = true;
    receiver_->fail_after    = 1;
    EXPECT_TRUE(prefill_->handleWrite(1, "partial", currentTimeMs() + 60000, workerPlan()).hasError());
    WriteTaskStatus status;
    ASSERT_TRUE(prefill_->queryWriteStatus("partial", status));
    EXPECT_TRUE(status.error.hasError());
    EXPECT_TRUE(status.sealed);
    EXPECT_FALSE(status.stopped);
    EXPECT_EQ(status.started_ops, 1);
    ASSERT_NE(receiver_->active, nullptr);
    receiver_->active->notifyDone(false, transfer::TransferErrorCode::UNKNOWN, "test completion");
    ASSERT_TRUE(prefill_->queryWriteStatus("partial", status));
    EXPECT_TRUE(status.error.hasError());
    EXPECT_TRUE(status.stopped);
    EXPECT_FALSE(status.write_success);
    EXPECT_EQ(status.finished_ops, 1);
}

TEST_F(P2PConnectorWriteWorkerTest, LeaseCountsStayStableAcrossQueriesAndCleanup) {
    const auto plan     = workerPlan();
    const auto deadline = currentTimeMs() + 60000;
    ASSERT_TRUE(prefill_->handleWrite(1, "counted", deadline, plan).ok());
    ASSERT_TRUE(decode_->write(1, "counted", deadline, plan).ok());
    sender_->begin();
    sender_->finish(true, 1);
    WriteTaskStatus status;
    for (int i = 0; i < 3; ++i) {
        ASSERT_TRUE(prefill_->queryWriteStatus("counted", status));
        EXPECT_TRUE(status.sealed);
        EXPECT_EQ(status.started_ops, 2);
        EXPECT_EQ(status.finished_ops, 1);
        EXPECT_FALSE(status.stopped);
        EXPECT_FALSE(status.write_success);
        ASSERT_TRUE(decode_->queryWriteStatus("counted", status));
        EXPECT_TRUE(status.sealed);
        EXPECT_EQ(status.started_ops, 2);
        EXPECT_EQ(status.finished_ops, 1);
        EXPECT_FALSE(status.stopped);
        EXPECT_FALSE(status.write_success);
        prefill_->cleanup();
        decode_->cleanup();
    }
    sender_->finish();
    for (int i = 0; i < 3; ++i) {
        prefill_->cleanup();
        decode_->cleanup();
        ASSERT_TRUE(prefill_->cancelWrite("counted", deadline));
        ASSERT_TRUE(decode_->cancelWrite("counted", deadline));
        ASSERT_TRUE(prefill_->queryWriteStatus("counted", status));
        EXPECT_EQ(status.started_ops, 2);
        EXPECT_EQ(status.finished_ops, 2);
        EXPECT_TRUE(status.stopped);
        EXPECT_TRUE(status.write_success);
        ASSERT_TRUE(decode_->queryWriteStatus("counted", status));
        EXPECT_EQ(status.started_ops, 2);
        EXPECT_EQ(status.finished_ops, 2);
        EXPECT_TRUE(status.stopped);
        EXPECT_TRUE(status.write_success);
    }
    EXPECT_EQ(source_->bytes, target_->bytes);
    EXPECT_EQ(receiver_->tasks.getTaskCount(), 0);
}

TEST_F(P2PConnectorWriteWorkerTest, CancelledLeaseKeepsCountsAfterTaskCleanup) {
    const auto deadline = currentTimeMs() + 60000;
    ASSERT_TRUE(prefill_->handleWrite(1, "cancelled", deadline, workerPlan()).ok());
    ASSERT_TRUE(prefill_->cancelWrite("cancelled", deadline));
    for (int i = 0; i < 3; ++i) {
        prefill_->cleanup();
        WriteTaskStatus status;
        ASSERT_TRUE(prefill_->queryWriteStatus("cancelled", status));
        EXPECT_TRUE(status.sealed);
        EXPECT_EQ(status.started_ops, 2);
        EXPECT_EQ(status.finished_ops, 2);
        EXPECT_TRUE(status.stopped);
        EXPECT_FALSE(status.write_success);
    }
    EXPECT_EQ(receiver_->tasks.getTaskCount(), 0);
    EXPECT_TRUE(prefill_->handleWrite(1, "cancelled", deadline, workerPlan()).hasError());
}

TEST(WriteTaskGroupTest, CompletedTasksDoNotStopUnsealedGroup) {
    const auto                   deadline = currentTimeMs() + 60000;
    p2p_internal::WriteTaskGroup group(deadline);
    auto                         task = std::make_shared<transfer::TransferTask>(transfer::KeyBlockInfoMap{}, deadline);
    group.tasks.emplace("unit", task);
    group.lease->onTransferStarted();
    ASSERT_TRUE(task->startTransfer());
    task->notifyDone(true, transfer::TransferErrorCode::OK, "test completion");
    WriteTaskStatus status;
    EXPECT_FALSE(group.fillStatus(status));
    EXPECT_FALSE(status.sealed);
    EXPECT_EQ(status.started_ops, 1);
    EXPECT_EQ(status.finished_ops, 1);
    EXPECT_FALSE(status.stopped);
    EXPECT_FALSE(status.write_success);
    group.lease->seal();
    EXPECT_TRUE(group.fillStatus(status));
    EXPECT_TRUE(status.sealed);
    EXPECT_EQ(status.finished_ops, 1);
    EXPECT_TRUE(status.stopped);
    EXPECT_TRUE(status.write_success);
}

TEST_F(P2PConnectorWriteWorkerTest, RegistrationThenSendCopiesEveryLayerAndKey) {
    auto               req = request();
    FunctionResponsePB response;
    ASSERT_TRUE(prefill_connector_->processWritePerRank(req, response));
    EXPECT_TRUE(response.p2p_response().lease_status().sealed());
    EXPECT_FALSE(response.p2p_response().lease_status().stopped());
    EXPECT_FALSE(response.p2p_response().write_success());
    EXPECT_EQ(receiver_->tasks.getTaskCount(), 2);
    ASSERT_TRUE(decode_connector_->writePerRank(req, response));
    EXPECT_FALSE(response.p2p_response().lease_status().stopped());
    sender_->begin();
    sender_->finish();
    req.set_write_operation(WRITE_QUERY);
    ASSERT_TRUE(prefill_connector_->processWritePerRank(req, response));
    EXPECT_TRUE(response.p2p_response().write_success());
    EXPECT_EQ(response.p2p_response().lease_status().finished_ops(), 2);
    ASSERT_TRUE(decode_connector_->writePerRank(req, response));
    EXPECT_TRUE(response.p2p_response().write_success());
    EXPECT_EQ(source_->bytes, target_->bytes);
    prefill_->cleanup();
    decode_->cleanup();
    EXPECT_EQ(receiver_->tasks.getTaskCount(), 0);
    ASSERT_TRUE(prefill_connector_->processWritePerRank(req, response));
    EXPECT_TRUE(response.p2p_response().write_success());
    EXPECT_EQ(response.p2p_response().lease_status().started_ops(), 2);
}

TEST_F(P2PConnectorWriteWorkerTest, WriteKeysCannotMatchReadTasks) {
    const auto read_key  = P2PKeyUtil::makeRouteLayerKey("request", 0, "group0", 1, 123);
    const auto write_key = P2PKeyUtil::makeWriteBackRouteLayerKey("request", 0, "group0", 1, 123);
    EXPECT_NE(read_key, write_key);
    EXPECT_NE(write_key, P2PKeyUtil::makeWriteBackRouteLayerKey("request", 0, "group0", 1, 124));
}

TEST_F(P2PConnectorWriteWorkerTest, TcpTransferUsesExplicitSourceAndDestinationBlocks) {
    initRuntime(0, false, false, MlaOpsType::AUTO);
    ASSERT_TRUE(isRuntimeInitialized());
    auto source = std::make_shared<WriteCudaBufferConverter>();
    auto target = std::make_shared<WriteCudaBufferConverter>();
    for (int layer = 0; layer < 2; ++layer) {
        source->bytes[layer][1].fill_(10 * layer + 1);
        source->bytes[layer][2].fill_(10 * layer + 2);
    }
    at::cuda::getCurrentCUDAStream().synchronize();
    const auto port         = autil::NetUtil::randomPort();
    auto       tcp_receiver = std::make_shared<transfer::tcp::TcpKVCacheReceiver>();
    auto       tcp_sender   = std::make_shared<transfer::tcp::TcpKVCacheSender>();
    ASSERT_TRUE(tcp_receiver->init(port, 1, 1));
    ASSERT_TRUE(tcp_sender->init(1));
    auto prefill = makeWritePrefillConnector(config_, target, tcp_receiver);
    auto decode  = makeWriteDecodeConnector(config_, source, tcp_sender);
    ASSERT_TRUE(prefill->write_worker_->init());
    ASSERT_TRUE(decode->write_worker_->init());

    auto send_request = request();
    send_request.mutable_peer_workers(0)->set_cache_store_port(port);
    auto recv_request = send_request;
    recv_request.clear_peer_workers();
    for (auto& route : *recv_request.mutable_routes()) {
        route.mutable_layer_blocks(0)->set_block_ids(0, 2);
        route.mutable_layer_blocks(0)->set_block_ids(1, 3);
    }
    FunctionResponsePB recv_response;
    FunctionResponsePB send_response;
    ASSERT_TRUE(prefill->processWritePerRank(recv_request, recv_response));
    EXPECT_TRUE(recv_response.p2p_response().lease_status().sealed());
    EXPECT_FALSE(recv_response.p2p_response().lease_status().stopped());
    ASSERT_TRUE(decode->writePerRank(send_request, send_response));

    recv_request.set_write_operation(WRITE_QUERY);
    send_request.set_write_operation(WRITE_QUERY);
    const auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    do {
        ASSERT_TRUE(prefill->processWritePerRank(recv_request, recv_response));
        ASSERT_TRUE(decode->writePerRank(send_request, send_response));
        if (recv_response.p2p_response().lease_status().stopped()
            && send_response.p2p_response().lease_status().stopped()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < timeout);
    ASSERT_TRUE(send_response.p2p_response().write_success()) << send_response.DebugString();
    ASSERT_TRUE(recv_response.p2p_response().write_success()) << recv_response.DebugString();
    auto received = target->bytes.cpu();
    for (int layer = 0; layer < 2; ++layer) {
        EXPECT_TRUE(received[layer][2].eq(10 * layer + 1).all().item<bool>());
        EXPECT_TRUE(received[layer][3].eq(10 * layer + 2).all().item<bool>());
        EXPECT_TRUE(received[layer][1].eq(0).all().item<bool>());
    }
}

TEST_F(P2PConnectorWriteWorkerTest, CleanupIntervalComesFromCacheStoreConfig) {
    CacheStoreConfig cache_config;
    cache_config.p2p_resource_store_timeout_check_interval_ms = 37;
    cache_config.p2p_writeback_enable                         = true;
    auto config = P2PConnectorConfig::create(RuntimeConfig{}, cache_config, ParallelismConfig{}, PDSepConfig{}, 2);
    EXPECT_TRUE(config.p2p_writeback_enable);
    EXPECT_EQ(config.worker_config.p2p_resource_store_timeout_check_interval_ms, 37);
    EXPECT_EQ(config.scheduler_config.p2p_resource_store_timeout_check_interval_ms, 37);
}

TEST_F(P2PConnectorWriteWorkerTest, CleanupThreadCancelsExpiredPendingReceives) {
    ASSERT_TRUE(prefill_->init());
    auto req = request();
    req.set_deadline_ms(currentTimeMs() + 200);
    FunctionResponsePB response;
    ASSERT_TRUE(prefill_connector_->processWritePerRank(req, response));
    req.set_write_operation(WRITE_QUERY);
    const auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    do {
        ASSERT_TRUE(prefill_connector_->processWritePerRank(req, response));
        if (response.p2p_response().lease_status().stopped()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < timeout);
    EXPECT_TRUE(response.p2p_response().lease_status().stopped());
    EXPECT_FALSE(response.p2p_response().write_success());
    EXPECT_EQ(receiver_->tasks.getTaskCount(), 0);
}

TEST_F(P2PConnectorWriteWorkerTest, UnknownOperationsDoNotCreateTasks) {
    auto               req = request();
    FunctionResponsePB response;
    for (const auto operation : {WRITE_QUERY, static_cast<P2PWriteOperationPB>(99)}) {
        req.set_write_operation(operation);
        EXPECT_FALSE(prefill_connector_->processWritePerRank(req, response));
        EXPECT_FALSE(decode_connector_->writePerRank(req, response));
    }
    EXPECT_EQ(receiver_->registrations, 0);
    EXPECT_EQ(sender_->pendingCount(), 0);
    req.set_write_operation(WRITE_START);
    EXPECT_TRUE(prefill_connector_->processWritePerRank(req, response));
}

TEST_F(P2PConnectorWriteWorkerTest, OverlappingReceiveBlocksAreRejectedBeforeRegistration) {
    auto  req       = request();
    auto* duplicate = req.add_routes();
    duplicate->CopyFrom(req.routes(0));
    duplicate->set_route_id(2);
    FunctionResponsePB response;
    EXPECT_FALSE(prefill_connector_->processWritePerRank(req, response));
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
        EXPECT_FALSE(prefill_connector_->processWritePerRank(req, response));
        EXPECT_FALSE(decode_connector_->writePerRank(req, response));
    }
    EXPECT_EQ(receiver_->registrations, 0);
    EXPECT_EQ(sender_->pendingCount(), 0);
}

TEST_F(P2PConnectorWriteWorkerTest, InvalidLastRouteHasNoTransferSideEffects) {
    auto req = request();
    req.mutable_routes(1)->mutable_layer_blocks(0)->set_block_ids(0, 0);
    FunctionResponsePB response;
    EXPECT_FALSE(prefill_connector_->processWritePerRank(req, response));
    EXPECT_FALSE(decode_connector_->writePerRank(req, response));
    EXPECT_EQ(receiver_->registrations, 0);
    EXPECT_EQ(sender_->pendingCount(), 0);
}

TEST_F(P2PConnectorWriteWorkerTest, RejectsPartitionedRoutesAndExpiredRequests) {
    auto req = request();
    req.mutable_routes(0)->set_partition_count(2);
    FunctionResponsePB response;
    EXPECT_FALSE(prefill_connector_->processWritePerRank(req, response));
    EXPECT_FALSE(decode_connector_->writePerRank(req, response));
    req = request();
    req.set_deadline_ms(currentTimeMs() - 1);
    EXPECT_FALSE(prefill_connector_->processWritePerRank(req, response));
    EXPECT_FALSE(decode_connector_->writePerRank(req, response));
    EXPECT_EQ(receiver_->registrations, 0);
}

TEST_F(P2PConnectorWriteWorkerTest, DuplicateStartCannotReplaceRegisteredBuffers) {
    auto               req = request();
    FunctionResponsePB response;
    ASSERT_TRUE(prefill_connector_->processWritePerRank(req, response));
    EXPECT_FALSE(prefill_connector_->processWritePerRank(req, response));
    EXPECT_EQ(receiver_->registrations, 2);
    ASSERT_TRUE(decode_connector_->writePerRank(req, response));
    EXPECT_FALSE(decode_connector_->writePerRank(req, response));
    ASSERT_TRUE(sender_->waitPending(2));
    sender_->begin();
    sender_->finish();
}

TEST_F(P2PConnectorWriteWorkerTest, CancellationBeforeStartRejectsDelayedBroadcast) {
    auto req = request();
    req.set_write_operation(WRITE_CANCEL);
    FunctionResponsePB response;
    ASSERT_TRUE(prefill_connector_->processWritePerRank(req, response));
    EXPECT_TRUE(response.p2p_response().lease_status().stopped());
    EXPECT_FALSE(response.p2p_response().write_success());
    ASSERT_TRUE(decode_connector_->writePerRank(req, response));
    EXPECT_TRUE(response.p2p_response().lease_status().stopped());
    EXPECT_FALSE(response.p2p_response().write_success());
    req.set_write_operation(WRITE_START);
    EXPECT_FALSE(prefill_connector_->processWritePerRank(req, response));
    EXPECT_FALSE(decode_connector_->writePerRank(req, response));
    EXPECT_EQ(receiver_->registrations, 0);
    EXPECT_EQ(sender_->pendingCount(), 0);
}

TEST_F(P2PConnectorWriteWorkerTest, PartialRegistrationFailureRemovesRegisteredTasks) {
    receiver_->fail_after  = 1;
    auto               req = request();
    FunctionResponsePB response;
    EXPECT_FALSE(prefill_connector_->processWritePerRank(req, response));
    EXPECT_EQ(receiver_->tasks.getTaskCount(), 0);
    EXPECT_TRUE(response.p2p_response().lease_status().stopped());
    EXPECT_FALSE(response.p2p_response().write_success());
}

TEST_F(P2PConnectorWriteWorkerTest, CancelDuringTransferWaitsForCallbacks) {
    auto               req = request();
    FunctionResponsePB response;
    ASSERT_TRUE(prefill_connector_->processWritePerRank(req, response));
    ASSERT_TRUE(decode_connector_->writePerRank(req, response));
    sender_->begin();
    req.set_write_operation(WRITE_CANCEL);
    ASSERT_TRUE(prefill_connector_->processWritePerRank(req, response));
    EXPECT_FALSE(response.p2p_response().lease_status().stopped());
    ASSERT_TRUE(decode_connector_->writePerRank(req, response));
    EXPECT_FALSE(response.p2p_response().lease_status().stopped());
    EXPECT_EQ(receiver_->tasks.getTaskCount(), 0);
    sender_->finish();
    req.set_write_operation(WRITE_QUERY);
    ASSERT_TRUE(prefill_connector_->processWritePerRank(req, response));
    EXPECT_TRUE(response.p2p_response().lease_status().stopped());
    EXPECT_FALSE(response.p2p_response().write_success());
    ASSERT_TRUE(decode_connector_->writePerRank(req, response));
    EXPECT_TRUE(response.p2p_response().lease_status().stopped());
    EXPECT_FALSE(response.p2p_response().write_success());
}

TEST_F(P2PConnectorWriteWorkerTest, TimeoutCleanupCannotClaimActiveTransferStopped) {
    auto               req = request();
    FunctionResponsePB response;
    ASSERT_TRUE(prefill_connector_->processWritePerRank(req, response));
    ASSERT_TRUE(decode_connector_->writePerRank(req, response));
    sender_->begin();
    prefill_->config_.p2p_cancelled_keys_ttl_ms         = 1;
    prefill_->groups_.at(req.unique_key())->deadline_ms = currentTimeMs() - 100;
    {
        std::lock_guard<std::mutex> lock(decode_->mutex_);
        decode_->config_.p2p_cancelled_keys_ttl_ms         = 1;
        decode_->groups_.at(req.unique_key())->deadline_ms = currentTimeMs() - 100;
    }
    prefill_->cleanup();
    decode_->cleanup();
    req.set_write_operation(WRITE_QUERY);
    ASSERT_TRUE(prefill_connector_->processWritePerRank(req, response));
    EXPECT_FALSE(response.p2p_response().lease_status().stopped());
    ASSERT_TRUE(decode_connector_->writePerRank(req, response));
    EXPECT_FALSE(response.p2p_response().lease_status().stopped());
    sender_->finish();
    ASSERT_TRUE(prefill_connector_->processWritePerRank(req, response));
    EXPECT_TRUE(response.p2p_response().lease_status().stopped());
    EXPECT_FALSE(response.p2p_response().write_success());
    prefill_->cleanup();
    decode_->cleanup();
    EXPECT_TRUE(prefill_->groups_.empty());
    {
        std::lock_guard<std::mutex> lock(decode_->mutex_);
        EXPECT_TRUE(decode_->groups_.empty());
    }
}

TEST_F(P2PConnectorWriteWorkerTest, FailedDataTransferIsStoppedButUnsuccessful) {
    auto               req = request();
    FunctionResponsePB response;
    ASSERT_TRUE(prefill_connector_->processWritePerRank(req, response));
    ASSERT_TRUE(decode_connector_->writePerRank(req, response));
    sender_->begin();
    sender_->finish(false);
    req.set_write_operation(WRITE_QUERY);
    ASSERT_TRUE(prefill_connector_->processWritePerRank(req, response));
    EXPECT_TRUE(response.p2p_response().lease_status().stopped());
    EXPECT_FALSE(response.p2p_response().write_success());
    ASSERT_TRUE(decode_connector_->writePerRank(req, response));
    EXPECT_TRUE(response.p2p_response().lease_status().stopped());
    EXPECT_FALSE(response.p2p_response().write_success());
}

TEST_F(P2PConnectorWriteWorkerTest, StartReturnsWhileSenderIsBlockedAndCancelWaitsForCallback) {
    auto  sender    = std::make_shared<BlockingWriteSender>();
    auto  connector = makeWriteDecodeConnector(config_, source_, sender);
    auto& worker    = *connector->write_worker_;
    ASSERT_TRUE(worker.init(1, 1));
    auto               req = request();
    FunctionResponsePB response;
    auto               start = std::async(std::launch::async, [&]() { return connector->writePerRank(req, response); });
    ASSERT_TRUE(sender->waitEntered());
    EXPECT_EQ(start.wait_for(std::chrono::milliseconds(200)), std::future_status::ready);
    sender->release();
    ASSERT_TRUE(start.get());
    req.set_write_operation(WRITE_CANCEL);
    ASSERT_TRUE(connector->writePerRank(req, response));
    EXPECT_FALSE(response.p2p_response().lease_status().stopped());
    worker.sender_pool_->stop(autil::ThreadPoolBase::STOP_AFTER_QUEUE_EMPTY);
    sender->complete();
    req.set_write_operation(WRITE_QUERY);
    ASSERT_TRUE(connector->writePerRank(req, response));
    EXPECT_TRUE(response.p2p_response().lease_status().stopped());
    EXPECT_FALSE(response.p2p_response().write_success());
}

TEST_F(P2PConnectorWriteWorkerTest, QueueFullRejectsWithoutInlineSendAndQueuedCancelNeverSends) {
    auto  sender    = std::make_shared<BlockingWriteSender>();
    auto  connector = makeWriteDecodeConnector(config_, source_, sender);
    auto& worker    = *connector->write_worker_;
    ASSERT_TRUE(worker.init(1, 1));
    auto               active = request();
    FunctionResponsePB response;
    ASSERT_TRUE(connector->writePerRank(active, response));
    ASSERT_TRUE(sender->waitEntered());
    auto queued = request();
    queued.set_unique_key("queued");
    ASSERT_TRUE(connector->writePerRank(queued, response));
    EXPECT_FALSE(response.p2p_response().lease_status().stopped());
    EXPECT_EQ(response.p2p_response().lease_status().started_ops(), 2);
    auto rejected = request();
    rejected.set_unique_key("rejected");
    auto start = std::async(std::launch::async, [&]() { return connector->writePerRank(rejected, response); });
    EXPECT_EQ(start.wait_for(std::chrono::milliseconds(200)), std::future_status::ready);
    EXPECT_FALSE(start.get());
    rejected.set_write_operation(WRITE_QUERY);
    ASSERT_TRUE(connector->writePerRank(rejected, response));
    EXPECT_TRUE(response.p2p_response().lease_status().stopped());
    EXPECT_FALSE(response.p2p_response().write_success());
    queued.set_write_operation(WRITE_CANCEL);
    ASSERT_TRUE(connector->writePerRank(queued, response));
    EXPECT_TRUE(response.p2p_response().lease_status().stopped());
    EXPECT_FALSE(response.p2p_response().write_success());
    active.set_write_operation(WRITE_CANCEL);
    ASSERT_TRUE(connector->writePerRank(active, response));
    EXPECT_FALSE(response.p2p_response().lease_status().stopped());
    sender->release();
    worker.sender_pool_->stop(autil::ThreadPoolBase::STOP_AFTER_QUEUE_EMPTY);
    EXPECT_EQ(sender->calls, 1);
    sender->complete();
    active.set_write_operation(WRITE_QUERY);
    ASSERT_TRUE(connector->writePerRank(active, response));
    EXPECT_TRUE(response.p2p_response().lease_status().stopped());
}

TEST_F(P2PConnectorWriteWorkerTest, EmptyRoutesSucceedWithFullQueueAndKeepQueryableTerminalState) {
    auto  sender    = std::make_shared<BlockingWriteSender>();
    auto  connector = makeWriteDecodeConnector(config_, source_, sender);
    auto& worker    = *connector->write_worker_;
    ASSERT_TRUE(worker.init(1, 1));
    auto               active = request();
    FunctionResponsePB response;
    ASSERT_TRUE(connector->writePerRank(active, response));
    ASSERT_TRUE(sender->waitEntered());
    auto queued = request();
    queued.set_unique_key("queued");
    ASSERT_TRUE(connector->writePerRank(queued, response));
    EXPECT_EQ(worker.sender_pool_->getItemCount(), 1);

    auto empty = request();
    empty.set_unique_key("empty");
    empty.clear_routes();
    ASSERT_TRUE(connector->writePerRank(empty, response));
    EXPECT_EQ(response.p2p_response().error_code(), ErrorCodePB::NONE_ERROR);
    EXPECT_TRUE(response.p2p_response().write_success());
    EXPECT_TRUE(response.p2p_response().lease_status().sealed());
    EXPECT_TRUE(response.p2p_response().lease_status().stopped());
    EXPECT_EQ(response.p2p_response().lease_status().started_ops(), 0);
    EXPECT_EQ(response.p2p_response().lease_status().finished_ops(), 0);
    EXPECT_EQ(worker.sender_pool_->getItemCount(), 1);
    EXPECT_FALSE(connector->writePerRank(empty, response));

    worker.cleanup();
    empty.set_write_operation(WRITE_QUERY);
    ASSERT_TRUE(connector->writePerRank(empty, response));
    EXPECT_EQ(response.p2p_response().error_code(), ErrorCodePB::NONE_ERROR);
    EXPECT_TRUE(response.p2p_response().write_success());
    EXPECT_TRUE(response.p2p_response().lease_status().sealed());
    EXPECT_TRUE(response.p2p_response().lease_status().stopped());
    EXPECT_EQ(response.p2p_response().lease_status().started_ops(), 0);
    EXPECT_EQ(response.p2p_response().lease_status().finished_ops(), 0);

    ASSERT_TRUE(worker.cancelWrite(queued.unique_key(), queued.deadline_ms()));
    ASSERT_TRUE(worker.cancelWrite(active.unique_key(), active.deadline_ms()));
    sender->release();
    worker.sender_pool_->stop(autil::ThreadPoolBase::STOP_AFTER_QUEUE_EMPTY);
    EXPECT_EQ(sender->calls, 1);
    sender->complete();
}

TEST_F(P2PConnectorWriteWorkerTest, QueuedDeadlineExpiresWithoutSendingOrExtendingDeadline) {
    auto sender = std::make_shared<BlockingWriteSender>();
    auto config = config_;
    // Exercise the submission deadline check independently of periodic cleanup.
    config.p2p_resource_store_timeout_check_interval_ms = 60000;
    auto  connector                                     = makeWriteDecodeConnector(config, source_, sender);
    auto& worker                                        = *connector->write_worker_;
    ASSERT_TRUE(worker.init(1, 1));
    worker.cleanup_thread_->stop();
    auto               active = request();
    FunctionResponsePB response;
    ASSERT_TRUE(connector->writePerRank(active, response));
    ASSERT_TRUE(sender->waitEntered());
    auto queued = request();
    queued.set_unique_key("expired_in_queue");
    queued.set_deadline_ms(currentTimeMs() + 50);
    ASSERT_TRUE(connector->writePerRank(queued, response));
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    active.set_write_operation(WRITE_CANCEL);
    ASSERT_TRUE(connector->writePerRank(active, response));
    sender->release();
    worker.sender_pool_->stop(autil::ThreadPoolBase::STOP_AFTER_QUEUE_EMPTY);
    EXPECT_EQ(sender->calls, 1);
    sender->complete();
    queued.set_write_operation(WRITE_QUERY);
    ASSERT_TRUE(connector->writePerRank(queued, response));
    EXPECT_TRUE(response.p2p_response().lease_status().stopped());
    EXPECT_FALSE(response.p2p_response().write_success());
    EXPECT_EQ(worker.groups_.at(queued.unique_key())->deadline_ms, queued.deadline_ms());
}

TEST_F(P2PConnectorWriteWorkerTest, SenderExceptionCancelsRemainingUnits) {
    auto sender           = std::make_shared<BlockingWriteSender>();
    sender->throw_on_send = true;
    sender->release();
    auto  connector = makeWriteDecodeConnector(config_, source_, sender);
    auto& worker    = *connector->write_worker_;
    ASSERT_TRUE(worker.init(1, 1));
    auto               req = request();
    FunctionResponsePB response;
    ASSERT_TRUE(connector->writePerRank(req, response));
    worker.sender_pool_->stop(autil::ThreadPoolBase::STOP_AFTER_QUEUE_EMPTY);
    EXPECT_EQ(sender->calls, 1);
    req.set_write_operation(WRITE_QUERY);
    ASSERT_TRUE(connector->writePerRank(req, response));
    EXPECT_TRUE(response.p2p_response().lease_status().stopped());
    EXPECT_FALSE(response.p2p_response().write_success());
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
    LocalRpcServer local;
    bool           lose_start_response = false;
    bool           omit_control_status = false;
    grpc::Status   ExecuteFunction(grpc::ServerContext*     context,
                                   const FunctionRequestPB* request,
                                   FunctionResponsePB*      response) override {
        auto status = local.ExecuteFunction(context, request, response);
        if (request->p2p_request().write_operation() == WRITE_START && lose_start_response && status.ok()) {
            while (!context->IsCancelled()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        } else if (request->p2p_request().write_operation() != WRITE_START && omit_control_status) {
            response->Clear();
        }
        return status;
    }
};

class P2PConnectorWriteRpcTest: public P2PConnectorWriteWorkerTest {
protected:
    void SetUp() override {
        P2PConnectorWriteWorkerTest::SetUp();
        P2PConnectorConfig config;
        config.role_type            = RoleType::PREFILL;
        config.p2p_writeback_enable = true;
        config.worker_config        = config_;
        auto connector              = std::make_shared<P2PConnector>(config, target_, nullptr);
        connector->prefill_         = std::move(prefill_connector_);
        auto manager = std::make_shared<KVCacheManager>(test::makeSimpleMhaCacheConfig(2, 4, 1, DataType::TYPE_FP16),
                                                        /*warmup=*/true);
        manager->p2p_connector_ = std::move(connector);
        service_.local.engine_  = std::make_shared<WriteRpcEngine>(std::move(manager));
        grpc::ServerBuilder builder;
        int                 port = 0;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
        builder.RegisterService(&service_);
        server_ = builder.BuildAndStart();
        ASSERT_NE(server_, nullptr);
        address_ = "127.0.0.1:" + std::to_string(port);
        stub_    = RpcService::NewStub(grpc::CreateChannel(address_, grpc::InsecureChannelCredentials()));
        client_  = std::make_unique<P2PBroadcastClient>(std::vector<std::string>{address_});
        ASSERT_TRUE(client_->init());
    }
    void TearDown() override {
        if (server_) {
            server_->Shutdown();
            server_->Wait();
        }
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
    P2PBroadcastClient::WriteStatusResult control(const P2PConnectorBroadcastTpRequestPB& req,
                                                  P2PWriteOperationPB                     operation) {
        return client_->controlWrite(req.unique_key(), HANDLE_WRITE, operation, req.deadline_ms(), 1000);
    }
    WriteRpcService                     service_;
    std::unique_ptr<grpc::Server>       server_;
    std::string                         address_;
    std::unique_ptr<RpcService::Stub>   stub_;
    std::unique_ptr<P2PBroadcastClient> client_;
};

TEST_F(P2PConnectorWriteRpcTest, FailedStartLosesStatusAndControlMustWaitForActiveReceive) {
    receiver_->fail_after    = 1;
    receiver_->start_on_recv = true;
    auto               req   = request();
    FunctionResponsePB response;
    EXPECT_EQ(startRpc(req, response).error_code(), grpc::StatusCode::INTERNAL);
    EXPECT_FALSE(response.has_p2p_response());
    ASSERT_NE(receiver_->active, nullptr);
    EXPECT_FALSE(receiver_->active->done());
    EXPECT_FALSE(control(req, WRITE_QUERY).allStopped());
    EXPECT_FALSE(control(req, WRITE_CANCEL).allStopped());
    receiver_->active->notifyDone(true, transfer::TransferErrorCode::OK, "late completion");
    const auto stopped = control(req, WRITE_QUERY);
    EXPECT_TRUE(stopped.allStopped());
    EXPECT_FALSE(stopped.allSucceeded());
    ASSERT_EQ(stopped.ranks.size(), 1);
    EXPECT_NE(stopped.ranks[0].error_code(), ErrorCodePB::NONE_ERROR);
}

TEST_F(P2PConnectorWriteRpcTest, LostStartResponseStillRequiresCancellation) {
    service_.lose_start_response = true;
    auto               req       = request();
    FunctionResponsePB response;
    EXPECT_EQ(startRpc(req, response, 100).error_code(), grpc::StatusCode::DEADLINE_EXCEEDED);
    EXPECT_FALSE(response.has_p2p_response());
    ASSERT_EQ(receiver_->tasks.getTaskCount(), 2);
    const auto running = control(req, WRITE_QUERY);
    ASSERT_EQ(running.ranks.size(), 1);
    EXPECT_FALSE(running.allStopped());
    EXPECT_TRUE(control(req, WRITE_CANCEL).allStopped());
    EXPECT_FALSE(control(req, WRITE_QUERY).allSucceeded());
}

TEST_F(P2PConnectorWriteRpcTest, UnknownQueryIsInconclusiveAndCancelPreventsLateStart) {
    auto req = request();
    EXPECT_FALSE(control(req, WRITE_QUERY).allStopped());
    EXPECT_EQ(receiver_->registrations, 0);
    EXPECT_TRUE(control(req, WRITE_CANCEL).allStopped());
    FunctionResponsePB response;
    EXPECT_EQ(startRpc(req, response).error_code(), grpc::StatusCode::INTERNAL);
    EXPECT_EQ(receiver_->registrations, 0);
    EXPECT_TRUE(control(req, WRITE_QUERY).allStopped());
}

TEST_F(P2PConnectorWriteRpcTest, ControlWorksAfterTransferDeadlineWithoutExtendingIt) {
    auto               req = request();
    FunctionResponsePB response;
    ASSERT_TRUE(startRpc(req, response).ok());
    auto& worker = service_.local.engine_->getCacheManager()->p2p_connector_->prefill_->write_worker_;
    req.set_deadline_ms(currentTimeMs() - 1);
    worker->groups_.at(req.unique_key())->deadline_ms = req.deadline_ms();
    EXPECT_TRUE(control(req, WRITE_CANCEL).allStopped());
    EXPECT_EQ(worker->groups_.at(req.unique_key())->deadline_ms, req.deadline_ms());
    EXPECT_FALSE(control(req, WRITE_QUERY).allSucceeded());
}

TEST_F(P2PConnectorWriteRpcTest, MissingStatusOrFailedRankCannotConfirmStopped) {
    auto req = request();
    EXPECT_TRUE(control(req, WRITE_CANCEL).allStopped());
    service_.omit_control_status = true;
    EXPECT_FALSE(control(req, WRITE_QUERY).allStopped());
    service_.omit_control_status = false;
    P2PBroadcastClient partial({address_, "127.0.0.1:1"});
    ASSERT_TRUE(partial.init());
    EXPECT_FALSE(
        partial.controlWrite(req.unique_key(), HANDLE_WRITE, WRITE_CANCEL, req.deadline_ms(), 100).allStopped());
    EXPECT_FALSE(
        client_->controlWrite(req.unique_key(), HANDLE_WRITE, WRITE_START, req.deadline_ms(), 100).allStopped());
}

TEST_F(P2PConnectorWriteRpcTest, SuccessfulWriteRequiresEveryRankToReportSuccess) {
    auto               req = request();
    FunctionResponsePB response;
    ASSERT_TRUE(startRpc(req, response).ok());
    for (const auto& route : req.routes()) {
        auto key = P2PKeyUtil::makeWriteBackRouteLayerKey(
            req.unique_key(), route.layer_blocks(0).layer_id(), route.cache_tag(), route.route_id(), req.plan_digest());
        auto task = receiver_->tasks.getTask(key);
        ASSERT_NE(task, nullptr);
        ASSERT_TRUE(task->startTransfer());
        task->notifyDone(true, transfer::TransferErrorCode::OK, "completed");
    }
    const auto result = control(req, WRITE_QUERY);
    EXPECT_TRUE(result.allStopped());
    EXPECT_TRUE(result.allSucceeded());
}

}  // namespace
}  // namespace rtp_llm
