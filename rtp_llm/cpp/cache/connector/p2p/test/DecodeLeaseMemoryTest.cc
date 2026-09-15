#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <grpc++/grpc++.h>
#include <torch/torch.h>

#include "rtp_llm/cpp/cache/SingleTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorkerDecode.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"

namespace rtp_llm {
namespace test {
namespace {

torch::Tensor byteView(const BlockInfo& block) {
    const auto device = block.is_cuda ? torch::Device(torch::kCUDA, block.device_index) : torch::Device(torch::kCPU);
    return torch::from_blob(block.addr,
                            {static_cast<int64_t>(block.size_bytes)},
                            torch::TensorOptions().dtype(torch::kUInt8).device(device));
}

void writeBytes(const BlockInfo& block, uint8_t value) {
    const auto host = torch::full({static_cast<int64_t>(block.size_bytes)},
                                  value,
                                  torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    // Blocking H2D: notifyDone is issued only after the bytes have been written.
    byteView(block).copy_(host, /*non_blocking=*/false);
}

// Keep allocation, task state, lease tracking and resource retention real. Only
// transport execution is controlled: a write always precedes its completion.
class LeaseMemoryConverter: public LayerBlockConverter {
public:
    explicit LeaseMemoryConverter(std::shared_ptr<SingleTypeKVCacheAllocator> allocator):
        allocator_(std::move(allocator)) {}

    std::vector<BlockInfo>
    convertIndexToBuffer(int layer, const std::string& tag, int block, int partitions, int partition) const override {
        return allocator_->convertIndexToBufferByTag(layer, tag, block, partitions, partition);
    }
    std::vector<std::pair<BlockInfo, size_t>> getAllBuffers() const override {
        return {};
    }

private:
    std::shared_ptr<SingleTypeKVCacheAllocator> allocator_;
};

class LeaseMemoryReceiver: public transfer::IKVCacheReceiver {
public:
    explicit LeaseMemoryReceiver(size_t started_count): started_count_(started_count) {}

    bool regMem(const BlockInfo&, uint64_t) override {
        return true;
    }

    transfer::IKVCacheRecvTaskPtr recv(const transfer::RecvRequest& request) override {
        auto task = std::make_shared<transfer::TransferTask>(request.block_info, request.deadline_ms);
        std::lock_guard<std::mutex> lock(mutex_);
        if (tasks_.size() < started_count_) {
            task->startTransfer();
        }
        keys_.push_back(request.unique_key);
        tasks_.push_back(task);
        return task;
    }

    // Transport still owns submitted operations after removal from rendezvous.
    void stealTask(const std::string&) override {}

    transfer::IKVCacheRecvTaskPtr getTask(const std::string& key) override {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t i = 0; i < keys_.size(); ++i) {
            if (keys_[i] == key) {
                return tasks_[i];
            }
        }
        return nullptr;
    }

    std::vector<std::shared_ptr<transfer::TransferTask>> tasks() {
        std::lock_guard<std::mutex> lock(mutex_);
        return tasks_;
    }

    // Represents an already-submitted write arriving after logical cancellation.
    static void finishWrite(const std::shared_ptr<transfer::TransferTask>& task, uint8_t value) {
        for (const auto& [key, info] : task->getBlockInfos()) {
            for (const auto& block : info->blocks) {
                writeBytes(block, value);
            }
        }
        task->notifyDone(true);
    }

    static bool submitLateWrite(const std::shared_ptr<transfer::TransferTask>& task, uint8_t value) {
        if (!task->startTransfer()) {
            return false;
        }
        finishWrite(task, value);
        return true;
    }

private:
    const size_t                                         started_count_;
    std::mutex                                           mutex_;
    std::vector<std::string>                             keys_;
    std::vector<std::shared_ptr<transfer::TransferTask>> tasks_;
};

// The test supplies a fixed one-rank plan. Control RPCs query the worker itself,
// so the async context cannot release blocks based on a fabricated lease status.
class LeaseMemoryService: public RpcService::Service {
public:
    P2PConnectorWorkerDecode* worker = nullptr;
    P2PWorkerRoutePlan        plan;
    std::atomic<int>          lease_queries{0};

    grpc::Status
    ExecuteFunction(grpc::ServerContext*, const FunctionRequestPB* request, FunctionResponsePB* response) override {
        const auto& req = request->p2p_request();
        auto*       out = response->mutable_p2p_response();
        out->set_error_code(ErrorCodePB::NONE_ERROR);
        switch (req.type()) {
            case P2PConnectorBroadcastType::READ: {
                const auto error = worker->read(req.request_id(), req.unique_key(), req.deadline_ms(), plan);
                out->set_error_code(transErrorCodeToRPC(error.code()));
                out->set_error_message(error.ToString());
                break;
            }
            case P2PConnectorBroadcastType::CANCEL_READ:
                worker->cancelRead(req.unique_key(), req.request_deadline_ms());
                break;
            case P2PConnectorBroadcastType::QUERY_LEASE_STATUS: {
                bool sealed = false, stopped = false;
                int  started = 0, finished = 0;
                worker->queryLeaseStatus(req.unique_key(), sealed, started, finished, stopped);
                auto* lease = out->mutable_lease_status();
                lease->set_sealed(sealed);
                lease->set_started_ops(started);
                lease->set_finished_ops(finished);
                lease->set_stopped(stopped);
                ++lease_queries;
                break;
            }
            default:
                return {grpc::StatusCode::INVALID_ARGUMENT, "unexpected lease test RPC"};
        }
        return grpc::Status::OK;
    }
};

class DecodeLeaseMemoryTest: public ::testing::TestWithParam<int> {
protected:
    static bool waitUntil(const std::function<bool()>& predicate) {
        const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!predicate()) {
            if (std::chrono::steady_clock::now() >= end) {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return true;
    }

    void SetUp() override {
        config_    = test::makeSimpleMhaCacheConfig(2, 4, 4, DataType::TYPE_FP16, 2, 8);
        allocator_ = std::make_shared<SingleTypeKVCacheAllocator>(config_, AllocationType::DEVICE);
        ASSERT_TRUE(allocator_->init());
        pool_ = allocator_->getDeviceBlockPool();
        ASSERT_NE(pool_, nullptr);
        initial_free_ = pool_->freeBlocksNum();
        ASSERT_GT(initial_free_, 1u);

        receiver_                          = std::make_shared<LeaseMemoryReceiver>(GetParam());
        auto                     converter = std::make_shared<LeaseMemoryConverter>(allocator_);
        P2PConnectorWorkerConfig worker_config;
        worker_config.tp_size       = 1;
        worker_config.tp_rank       = 0;
        worker_config.layer_all_num = 2;
        worker_         = std::make_unique<P2PConnectorWorkerDecode>(worker_config, converter, nullptr, receiver_);
        service_.worker = worker_.get();
        grpc::ServerBuilder builder;
        int                 port = 0;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
        builder.RegisterService(&service_);
        server_ = builder.BuildAndStart();
        ASSERT_NE(server_, nullptr);
        client_ = std::make_shared<P2PBroadcastClient>(std::vector<std::string>{"127.0.0.1:" + std::to_string(port)});
        ASSERT_TRUE(client_->init());
        control_pool_ = std::make_shared<autil::LockFreeThreadPool>(1, 16, nullptr, "LeaseMemoryTest");
        ASSERT_TRUE(control_pool_->start());
        checker_ = std::make_unique<P2PConnectorAsyncReadContextChecker>();
        ASSERT_TRUE(checker_->init(nullptr, client_, control_pool_));
    }

    void TearDown() override {
        // Drain the handler even if a fatal assertion interrupted the scenario.
        if (worker_) {
            worker_->cancelRead(key_);
        }
        if (receiver_) {
            for (const auto& task : receiver_->tasks()) {
                task->notifyDone(false, transfer::TransferErrorCode::CANCELLED);
            }
        }
        if (checker_) {
            checker_->stop();
        }
        if (control_pool_) {
            control_pool_->stop();
        }
        checker_.reset();
        if (server_) {
            server_->Shutdown();
            server_->Wait();
        }
    }

    KVCacheResourcePtr allocate(size_t count) {
        const auto blocks = pool_->malloc(count);
        if (!blocks) {
            return nullptr;
        }
        pool_->incRef(*blocks);
        KVCacheResource source;
        source.initGroups(config_.topologyPtr());
        source.mutableBlockIds(0).assign(*blocks);
        for (const auto block : *blocks) {
            source.cacheKeys().push_back(1000 + block);
        }
        auto resource = allocator_->incrKVCacheRef(source, source.cacheKeys(), false);
        pool_->decRef(*blocks);
        return resource;
    }

    std::vector<BlockInfo> buffers(BlockIdxType block, int layer) const {
        return allocator_->convertIndexToBuffer(layer, block);
    }

    void fill(const KVCacheResource& resource, uint8_t value) {
        for (const auto block : resource.blocks(0)) {
            for (int layer = 0; layer < 2; ++layer) {
                for (const auto& buffer : buffers(block, layer)) {
                    writeBytes(buffer, value);
                }
            }
        }
    }

    void expectBytes(BlockIdxType block, int layer, uint8_t value) const {
        const auto views = buffers(block, layer);
        ASSERT_FALSE(views.empty());
        for (size_t i = 0; i < views.size(); ++i) {
            SCOPED_TRACE(::testing::Message() << "block=" << block << " layer=" << layer << " buffer=" << i);
            ASSERT_NE(views[i].addr, nullptr);
            ASSERT_GT(views[i].size_bytes, 0u);
            const auto  host  = byteView(views[i]).cpu();
            const auto* bytes = host.data_ptr<uint8_t>();
            EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + views[i].size_bytes),
                      std::vector<uint8_t>(views[i].size_bytes, value));
        }
    }

    CacheConfig                                          config_;
    std::shared_ptr<SingleTypeKVCacheAllocator>          allocator_;
    DeviceBlockPoolPtr                                   pool_;
    size_t                                               initial_free_ = 0;
    const std::string                                    key_          = "cancel_memory_reuse";
    std::shared_ptr<LeaseMemoryReceiver>                 receiver_;
    std::unique_ptr<P2PConnectorWorkerDecode>            worker_;
    LeaseMemoryService                                   service_;
    std::unique_ptr<grpc::Server>                        server_;
    std::shared_ptr<P2PBroadcastClient>                  client_;
    std::shared_ptr<autil::LockFreeThreadPool>           control_pool_;
    std::unique_ptr<P2PConnectorAsyncReadContextChecker> checker_;
};

TEST_P(DecodeLeaseMemoryTest, CancelRetainsBlockUntilLastWriteAndProtectsReusedBytes) {
    auto request_a = allocate(1);
    ASSERT_NE(request_a, nullptr);
    const auto block_a = request_a->blocks(0).at(0);
    EXPECT_EQ(pool_->refCount(block_a), 1u);
    fill(*request_a, 0x11);
    auto connector_ref = allocator_->incrKVCacheRef(*request_a, request_a->cacheKeys(), true);
    ASSERT_NE(connector_ref, nullptr);
    EXPECT_EQ(pool_->refCount(block_a), 2u);

    P2PWorkerRoute route;
    route.route_id  = 0;
    route.cache_tag = config_.topologyPtr()->groups()[0].tag;
    route.partition = {1, 0};
    for (int layer = 0; layer < 2; ++layer) {
        auto buffer = std::make_shared<LayerCacheBuffer>(layer, route.cache_tag);
        buffer->addBlockId(request_a->cacheKeys()[0], block_a);
        route.layer_buffers.push_back(buffer);
    }
    service_.plan.routes.push_back(std::move(route));

    const auto                          deadline = currentTimeMs() + 30000;
    P2PBroadcastClient::BroadcastParams params;
    params.request_id          = 1;
    params.unique_key          = key_;
    params.deadline_ms         = deadline;
    params.request_deadline_ms = deadline;
    params.type                = P2PConnectorBroadcastType::READ;
    auto read                  = client_->broadcast(std::move(params));
    ASSERT_NE(read, nullptr);
    auto prefill = std::make_shared<DecodeLoadHelper::Result>();
    prefill->response.mutable_payload()->set_has_first_generate_token(true);
    prefill->complete(true);
    auto context =
        std::make_shared<P2PConnectorAsyncReadContext>(connector_ref,
                                                       key_,
                                                       std::make_shared<DecodeSchedulerMetricsCollector>(nullptr),
                                                       30000,
                                                       false,
                                                       deadline,
                                                       deadline);
    ASSERT_TRUE(context->beginKickoff());
    context->setCallResults(read, prefill);
    connector_ref.reset();
    checker_->addContext(context);
    ASSERT_TRUE(waitUntil([&] { return receiver_->tasks().size() == 2; }));
    const auto tasks = receiver_->tasks();

    context->cancel(client_);
    ASSERT_TRUE(waitUntil([&] { return context->done(); }));
    EXPECT_EQ(context->errorInfo().code(), ErrorCode::P2P_CONNECTOR_WORKER_READ_CANCELLED);
    // Simulate the cancelled stream dropping its request reference. Only the
    // production async context may now keep the block alive.
    request_a.reset();
    EXPECT_EQ(pool_->refCount(block_a), 1u);
    std::weak_ptr<P2PConnectorAsyncReadContext> weak_context = context;
    context.reset();

    KVCacheResourcePtr request_b;
    if (GetParam() > 0) {
        ASSERT_TRUE(waitUntil([&] { return service_.lease_queries.load() > 0; }));
        EXPECT_FALSE(weak_context.expired());
        EXPECT_EQ(pool_->refCount(block_a), 1u);
        EXPECT_EQ(pool_->freeBlocksNum(), initial_free_ - 1);
        request_b = allocate(initial_free_ - 1);
        ASSERT_NE(request_b, nullptr);
        EXPECT_EQ(std::count(request_b->blocks(0).begin(), request_b->blocks(0).end(), block_a), 0);
        fill(*request_b, 0xB2);
        EXPECT_FALSE(pool_->malloc(1).has_value());

        for (int i = 0; i < GetParam(); ++i) {
            ASSERT_FALSE(tasks[i]->done());
            EXPECT_EQ(pool_->refCount(block_a), 1u);
            // Complete only after the assertions above; no timing-dependent
            // sleep decides whether the delayed writer has run.
            std::thread writer([&, i] { LeaseMemoryReceiver::finishWrite(tasks[i], 0xA0 + i); });
            writer.join();
            expectBytes(block_a, i, 0xA0 + i);
            for (const auto block : request_b->blocks(0)) {
                EXPECT_EQ(pool_->refCount(block), 1u);
                expectBytes(block, 0, 0xB2);
                expectBytes(block, 1, 0xB2);
            }
            if (i + 1 < GetParam()) {
                const auto queries = service_.lease_queries.load();
                ASSERT_TRUE(waitUntil([&] { return service_.lease_queries.load() > queries; }));
                EXPECT_FALSE(weak_context.expired());
                EXPECT_EQ(pool_->refCount(block_a), 1u);
                EXPECT_EQ(pool_->freeBlocksNum(), 0u);
            }
        }
    }

    ASSERT_TRUE(waitUntil([&] { return weak_context.expired() && pool_->refCount(block_a) == 0; }));
    EXPECT_EQ(pool_->freeBlocksNum(), request_b ? 1u : initial_free_);
    for (int layer = 0; layer < GetParam(); ++layer) {
        expectBytes(block_a, layer, 0xA0 + layer);
    }
    for (int layer = GetParam(); layer < 2; ++layer) {
        EXPECT_TRUE(tasks[layer]->done());
        expectBytes(block_a, layer, 0x11);  // PENDING tasks never wrote.
    }

    // Occupy every free block, guaranteeing the old physical block is reused.
    auto request_c = allocate(pool_->freeBlocksNum());
    ASSERT_NE(request_c, nullptr);
    ASSERT_EQ(std::count(request_c->blocks(0).begin(), request_c->blocks(0).end(), block_a), 1);
    fill(*request_c, 0xC3);
    EXPECT_EQ(pool_->refCount(block_a), 1u);
    EXPECT_TRUE(worker_->cancelRead(key_));
    const auto late_read = worker_->read(2, key_, currentTimeMs() + 1000, service_.plan);
    EXPECT_EQ(late_read.code(), ErrorCode::P2P_CONNECTOR_WORKER_READ_CANCELLED);
    EXPECT_EQ(receiver_->tasks().size(), 2u);
    for (const auto& task : tasks) {
        EXPECT_FALSE(LeaseMemoryReceiver::submitLateWrite(task, 0xEE));
        task->cancel();
        task->notifyDone(true);  // A duplicate completion must not release C's ref.
    }
    for (const auto block : request_c->blocks(0)) {
        EXPECT_EQ(pool_->refCount(block), 1u);
        expectBytes(block, 0, 0xC3);
        expectBytes(block, 1, 0xC3);
    }
    EXPECT_EQ(pool_->freeBlocksNum(), 0u);
    const auto reused_blocks = request_c->blocks(0);
    request_c.reset();
    for (const auto block : reused_blocks) {
        EXPECT_EQ(pool_->refCount(block), 0u);
    }
    if (request_b) {
        const auto other_blocks = request_b->blocks(0);
        for (const auto block : other_blocks) {
            EXPECT_EQ(pool_->refCount(block), 1u);
            expectBytes(block, 0, 0xB2);
            expectBytes(block, 1, 0xB2);
        }
        request_b.reset();
        for (const auto block : other_blocks) {
            EXPECT_EQ(pool_->refCount(block), 0u);
        }
    }
    EXPECT_EQ(pool_->freeBlocksNum(), initial_free_);
}

INSTANTIATE_TEST_SUITE_P(CancelStages,
                         DecodeLeaseMemoryTest,
                         ::testing::Values(0, 1, 2),
                         [](const ::testing::TestParamInfo<int>& info) {
                             return (std::vector<std::string>{"AllPending", "Mixed", "AllTransferring"})[info.param];
                         });

}  // namespace
}  // namespace test
}  // namespace rtp_llm
