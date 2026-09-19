#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <thread>
#include <cuda_runtime.h>

#include "autil/EnvUtil.h"
#include "autil/NetUtil.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/cpp/cache/SingleTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConverterImpl.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnector.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorDecode.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorPrefill.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PSchedulerDecodeWrite.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PSchedulerPrefillWrite.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PWorkerDecodeWrite.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PWorkerPrefillWrite.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/TcpKVCacheReceiver.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/TcpKVCacheSender.h"
#include "rtp_llm/cpp/cache/test/BlockTreeCacheAllocatorTestHelper.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServerNew2.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestBase.h"

namespace rtp_llm {
namespace {

class WritebackEngine: public EngineBase {
public:
    explicit WritebackEngine(std::shared_ptr<KVCacheManager> manager): EngineBase(EngineInitParams()) {
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
        return absl::UnimplementedError("unused");
    }
    KVCacheInfo getCacheStatusInfo(int64_t, bool) override {
        return {};
    }
};

class WritebackService: public RpcService::Service {
public:
    LocalRpcServer       local;
    PrefillRpcServerNew2 prefill;
    std::atomic<int>     handshakes{0};
    grpc::Status         ExecuteFunction(grpc::ServerContext*     context,
                                         const FunctionRequestPB* request,
                                         FunctionResponsePB*      response) override {
        return local.ExecuteFunction(context, request, response);
    }
    grpc::Status StartWrite(grpc::ServerContext*                   context,
                            const P2PConnectorStartWriteRequestPB* request,
                            P2PConnectorStartWriteResponsePB*      response) override {
        ++handshakes;
        return prefill.StartWrite(context, request, response);
    }
};

class GatedTcpSender: public transfer::IKVCacheSender {
public:
    GatedTcpSender(): sender_(std::make_shared<transfer::tcp::TcpKVCacheSender>()) {}
    bool regMem(const BlockInfo& info, uint64_t size) override {
        return sender_->regMem(info, size);
    }
    void send(const transfer::SendRequest&                                         request,
              std::function<void(transfer::TransferErrorCode, const std::string&)> callback) override {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            ++started;
            cv_.wait(lock, [this]() { return released_; });
        }
        sender_->send(request, std::move(callback));
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true;
        cv_.notify_all();
    }
    std::shared_ptr<transfer::tcp::TcpKVCacheSender> sender_;
    std::atomic<int>                                 started{0};

private:
    std::mutex              mutex_;
    std::condition_variable cv_;
    bool                    released_{false};
};

class DecodeWritebackIntegrationTest: public DeviceTestBase {
protected:
    using Allocator = test::BlockTreeCacheTestAllocator<SingleTypeKVCacheAllocator>;
    autil::EnvGuard                                    perf_{"PERF_TEST", "1"};
    CacheConfig                                        config_;
    std::shared_ptr<Allocator>                         source_, target_;
    std::shared_ptr<KVCacheManager>                    decode_, prefill_;
    std::shared_ptr<GatedTcpSender>                    sender_;
    std::shared_ptr<transfer::tcp::TcpKVCacheReceiver> receiver_;
    WritebackService                                   decode_service_, prefill_service_;
    std::vector<std::unique_ptr<grpc::Server>>         servers_;
    int                                                prefill_port_{0};
    GenerateStreamPtr                                  stream_;
    CacheKeysType                                      keys_;
    BlockIndicesType                                   source_blocks_;
    std::vector<int>                                   tokens_{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};

    static bool await(const std::function<bool()>& predicate, int timeout_ms = 5000) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (!predicate() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return predicate();
    }

    std::string launch(WritebackService& service, int& port) {
        grpc::ServerBuilder builder;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
        builder.RegisterService(&service);
        servers_.push_back(builder.BuildAndStart());
        EXPECT_NE(servers_.back(), nullptr);
        return "127.0.0.1:" + std::to_string(port);
    }

    std::shared_ptr<KVCacheManager>
    makeManager(RoleType role, const std::shared_ptr<Allocator>& allocator, bool enabled = true) {
        PDSepConfig pd;
        pd.role_type       = role;
        pd.decode_entrance = role == RoleType::DECODE;
        CacheStoreConfig store;
        store.p2p_writeback_enable = enabled;
        auto manager               = std::make_shared<KVCacheManager>(config_,
                                                        false,
                                                        nullptr,
                                                        KVCacheConfig{},
                                                        ParallelismConfig{},
                                                        RuntimeConfig{},
                                                        SpeculativeExecutionConfig{},
                                                        pd,
                                                        store);
        manager->allocator_        = allocator;
        manager->block_tree_cache_ = allocator->blockTreeCache();
        return manager;
    }

    void SetUp() override {
        DeviceTestBase::SetUp();
        config_ = test::makeSimpleMhaCacheConfig(2, 16, 4, DataType::TYPE_FP16, 1, 2);
        source_ = std::make_shared<Allocator>(config_);
        target_ = std::make_shared<Allocator>(config_);
        ASSERT_TRUE(source_->init());
        ASSERT_TRUE(target_->init());
        decode_                          = makeManager(RoleType::DECODE, source_);
        prefill_                         = makeManager(RoleType::PREFILL, target_);
        decode_service_.local.engine_    = std::make_shared<WritebackEngine>(decode_);
        prefill_service_.local.engine_   = std::make_shared<WritebackEngine>(prefill_);
        prefill_service_.prefill.engine_ = prefill_service_.local.engine_;
        int        decode_port           = 0;
        const auto decode_addr           = launch(decode_service_, decode_port);
        const auto prefill_addr          = launch(prefill_service_, prefill_port_);
        const auto transfer_port         = autil::NetUtil::randomPort();
        receiver_                        = std::make_shared<transfer::tcp::TcpKVCacheReceiver>();
        sender_                          = std::make_shared<GatedTcpSender>();
        ASSERT_TRUE(receiver_->init(transfer_port, 1, 1));
        ASSERT_TRUE(sender_->sender_->init(1));

        for (auto role : {RoleType::DECODE, RoleType::PREFILL}) {
            auto               manager   = role == RoleType::DECODE ? decode_ : prefill_;
            auto               converter = std::make_shared<LayerBlockConverterImpl>(manager->allocator_);
            const auto&        address   = role == RoleType::DECODE ? decode_addr : prefill_addr;
            P2PConnectorConfig connector_config;
            connector_config.role_type                             = role;
            connector_config.p2p_writeback_enable                  = true;
            auto& worker                                           = connector_config.worker_config;
            worker.topology                                        = config_.topologyPtr();
            worker.layer_all_num                                   = config_.layer_num;
            worker.p2p_resource_store_timeout_check_interval_ms    = 5;
            auto& scheduler                                        = connector_config.scheduler_config;
            scheduler.topology                                     = config_.topologyPtr();
            scheduler.role_type                                    = role;
            scheduler.parallelism_config.role_type                 = role;
            scheduler.worker_grpc_addrs                            = {address};
            scheduler.p2p_worker_addrs                             = {"127.0.0.1:" + std::to_string(transfer_port) + ":"
                                                                      + std::to_string(prefill_port_)};
            scheduler.p2p_resource_store_timeout_check_interval_ms = 5;
            scheduler.p2p_cancel_broadcast_timeout_ms              = 100;
            scheduler.p2p_writeback_timeout_ms                     = 3000;
            auto client = std::make_shared<P2PBroadcastClient>(std::vector<std::string>{address}, 100);
            ASSERT_TRUE(client->init());
            auto connector = std::make_shared<P2PConnector>(connector_config, converter, nullptr, manager->allocator_);
            if (role == RoleType::DECODE) {
                auto entry              = std::make_unique<P2PConnectorDecode>(connector_config, converter, nullptr);
                entry->write_worker_    = std::make_unique<P2PWorkerDecodeWrite>(worker, converter, sender_);
                entry->write_scheduler_ = std::make_unique<P2PSchedulerDecodeWrite>(scheduler, client);
                ASSERT_TRUE(entry->write_worker_->init());
                ASSERT_TRUE(entry->write_scheduler_->init());
                connector->decode_ = std::move(entry);
            } else {
                auto entry = std::make_unique<P2PConnectorPrefill>(connector_config, converter, nullptr, target_);
                entry->write_worker_ = std::make_unique<P2PWorkerPrefillWrite>(worker, converter, receiver_);
                entry->write_scheduler_ =
                    std::make_unique<P2PSchedulerPrefillWrite>(scheduler, target_, nullptr, client);
                ASSERT_TRUE(entry->write_worker_->init());
                ASSERT_TRUE(entry->write_scheduler_->init());
                connector->prefill_ = std::move(entry);
            }
            manager->p2p_connector_ = std::move(connector);
        }
        keys_ = calculateCacheKeys(tokens_.data(), 12, 4);
        ASSERT_TRUE(test::seedCompleteBlockTreePath(target_, {keys_.front()}).success);
    }

    void TearDown() override {
        if (sender_)
            sender_->release();
        if (decode_)
            decode_->stopWriteback();
        if (prefill_)
            prefill_->stopWriteback();
        stream_.reset();
        for (auto& server : servers_) {
            server->Shutdown();
            server->Wait();
        }
        DeviceTestBase::TearDown();
    }

    void prepare(int token_count = 13, int ready_count = 12) {
        auto input              = std::make_shared<GenerateInput>();
        input->input_ids        = torch::tensor(std::vector<int>{1, 2, 3, 4}, torch::kInt32);
        auto gc                 = std::make_shared<GenerateConfig>();
        gc->max_new_tokens      = 32;
        gc->unique_key          = "finished-writeback";
        gc->reuse_cache         = true;
        gc->enable_device_cache = true;
        gc->role_addrs.emplace_back(RoleType::PREFILL, "127.0.0.1", 0, prefill_port_);
        input->generate_config = gc;
        ResourceContext resources;
        resources.cache_manager = decode_;
        resources.reuse_cache   = true;
        resources.role_type     = RoleType::DECODE;
        ModelConfig model;
        model.max_seq_len                  = 64;
        model.vocab_size                   = 128;
        model.attn_config.tokens_per_block = 4;
        stream_ = std::make_shared<NormalGenerateStream>(input, model, RuntimeConfig{}, resources, nullptr);
        stream_->setPrefillTpSize(1);
        stream_->setPrefillCpSize(1);
        stream_->generate_status_->status = StreamState::RUNNING;
        ASSERT_TRUE(stream_->streamCacheResource().initKVBlock().ok());
        auto new_tokens =
            torch::tensor(std::vector<int>(tokens_.begin() + 4, tokens_.begin() + token_count), torch::kInt32)
                .reshape({1, token_count - 4});
        StreamUpdateInfo update{new_tokens, token_count - 4};
        update.kv_ready_token_count = ready_count;
        stream_->update(update);
        ASSERT_FALSE(stream_->hasError());
        ASSERT_TRUE(stream_->incrKVBlock().ok());
        source_blocks_ = stream_->streamCacheResource().kvCache().blocks(0, 0);
        for (int layer = 0; layer < 2; ++layer) {
            for (size_t i = 0; i < source_blocks_.size(); ++i) {
                for (const auto& block : source_->convertIndexToBufferByTag(
                         layer, config_.topology().groups().front().tag, source_blocks_[i], 1, 0)) {
                    ASSERT_EQ(cudaMemset(block.addr, 10 * layer + i + 1, block.size_bytes), cudaSuccess);
                }
            }
        }
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        stream_->generate_status_->status = StreamState::FINISHED;
    }

    void checkWritebackAfterSpeculativeStop(bool eos) {
        autil::EnvGuard perf_scope("PERF_TEST", "0");
        prepare();
        stream_->generate_status_->status = StreamState::RUNNING;
        if (eos) {
            stream_->special_tokens_.eos_token_id = 14;
        } else {
            stream_->generate_input_->generate_config->stop_words_list = {{13, 14}};
        }
        auto output    = std::make_shared<SpeculativeExecutorStreamOutput>();
        output->tokens = torch::zeros({1, 2}, torch::kInt32);
        stream_->setSPOutputBuffer(output);
        StreamSpecUpdateInfo update{torch::tensor({{14, 15, 16, 17}}, torch::kInt32),
                                    4, -1, torch::Tensor(), torch::Tensor()};
        update.kv_ready_token_count = 16;
        stream_->specUpdate(update);
        ASSERT_FALSE(stream_->hasError());
        ASSERT_EQ(stream_->seqLength(), 14);
        EXPECT_EQ(stream_->writebackKVReadyTokenCount(), 13);

        ASSERT_EQ(stream_->moveToNext(), StreamState::FINISHED);
        ASSERT_TRUE(await([&]() { return sender_->started > 0; }));
        auto write = context();
        ASSERT_NE(write, nullptr);
        EXPECT_EQ(write->resource_->cacheKeys(), keys_);
        sender_->release();
        ASSERT_TRUE(await([&]() { return drained(); }));
        EXPECT_TRUE(write->success()) << write->errorInfo().ToString();
        EXPECT_EQ(target_->devicePrefixBlocksForTest(keys_), 3u);
    }

    P2PConnectorAsyncWriteContextChecker& checker() {
        return decode_->p2p_connector_->decode_->write_scheduler_->checker_;
    }
    std::shared_ptr<P2PConnectorAsyncWriteContext> context() {
        std::lock_guard<std::mutex> lock(checker().mutex_);
        auto                        it = checker().contexts_.find("finished-writeback");
        return it == checker().contexts_.end() ? nullptr : it->second;
    }
    bool drained() {
        return checker().inflightContextCount() == 0
               && prefill_->p2p_connector_->prefill_->write_scheduler_->checker_.inflightContextCount() == 0;
    }
    size_t sourceRefs(size_t index) {
        return source_->blockTreeCache()->groupSets().front()->devicePools().front()->refCount(source_blocks_[index]);
    }
};

TEST_F(DecodeWritebackIntegrationTest, FinishedStreamReleasesBeforeTcpCopyAndNextRequestReusesSuffix) {
    prepare();
    stream_->generate_status_->status = StreamState::RUNNING;
    stream_->reportEvent(StreamEvents::GenerateDone);
    ASSERT_EQ(stream_->moveToNext(), StreamState::FINISHED);
    ASSERT_TRUE(await([&]() { return sender_->started > 0; }));
    auto write = context();
    ASSERT_NE(write, nullptr);
    EXPECT_TRUE(write->resourceHoldPending());
    EXPECT_EQ(sourceRefs(1), 2u);
    EXPECT_EQ(stream_->streamCacheResource().curBlocksNum(), 0);
    stream_->generate_input_->generate_config->role_addrs.clear();
    stream_->completeTokenIdsPtr()->data(0)[4] = 99;
    stream_.reset();
    EXPECT_EQ(write->resource_->cacheKeys(), keys_);
    EXPECT_EQ(write->resource_->blocks(0).size(), 3u);
    sender_->release();
    ASSERT_TRUE(await([&]() { return drained(); }));
    EXPECT_TRUE(write->success()) << write->errorInfo().ToString();
    EXPECT_EQ(sourceRefs(1), 1u);

    auto reused = std::make_shared<BatchKVCacheResource>();
    reused->resetBatchSize(1);
    reused->initGroups(config_.topologyPtr());
    auto ids               = std::make_shared<CompleteTokenIds>(1, 1, 64, 4);
    auto input             = std::make_shared<GenerateInput>();
    input->input_ids       = torch::tensor(tokens_, torch::kInt32);
    input->generate_config = std::make_shared<GenerateConfig>();
    ids->init(input);
    MallocInfo info;
    info.batch_kv_cache_resource = reused;
    info.complete_token_ids      = ids;
    info.reuse_cache             = true;
    info.enable_cache_lookup     = true;
    auto result                  = prefill_->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, 12);
    for (int layer = 0; layer < 2; ++layer) {
        for (size_t i = 1; i < 3; ++i) {
            for (const auto& block :
                 target_->convertIndexToBufferByTag(
                     layer, config_.topology().groups().front().tag, reused->blocks(0, 0)[i], 1, 0)) {
                std::vector<uint8_t> bytes(block.size_bytes);
                ASSERT_EQ(cudaMemcpy(bytes.data(), block.addr, bytes.size(), cudaMemcpyDeviceToHost), cudaSuccess);
                EXPECT_TRUE(
                    std::all_of(bytes.begin(), bytes.end(), [&](uint8_t b) { return b == 10 * layer + i + 1; }));
            }
        }
    }
    prefill_->free(FreeInfo{reused, ids});
}

TEST_F(DecodeWritebackIntegrationTest, LastSampleAtBlockBoundaryIsExcluded) {
    prepare(12, 11);
    stream_->releaseResource();
    ASSERT_TRUE(await([&]() { return sender_->started > 0; }));
    auto write = context();
    ASSERT_NE(write, nullptr);
    EXPECT_EQ(write->resource_->cacheKeys().size(), 2u);
    sender_->release();
    ASSERT_TRUE(await([&]() { return drained(); }));
    EXPECT_EQ(target_->devicePrefixBlocksForTest(keys_), 2u);
}

TEST_F(DecodeWritebackIntegrationTest, SpeculativeStopWordsKeepCompletedSuffixWriteback) {
    checkWritebackAfterSpeculativeStop(false);
}

TEST_F(DecodeWritebackIntegrationTest, SpeculativeEosKeepsCompletedSuffixWriteback) {
    checkWritebackAfterSpeculativeStop(true);
}

TEST_F(DecodeWritebackIntegrationTest, MissingKvRangeNeverSubmitsWriteback) {
    prepare(13, -1);
    EXPECT_EQ(stream_->writebackKVReadyTokenCount(), 0);
    stream_->releaseResource();
    EXPECT_TRUE(drained());
    EXPECT_EQ(prefill_service_.handshakes, 0);
    EXPECT_EQ(sourceRefs(1), 1u);
}

TEST_F(DecodeWritebackIntegrationTest, DisabledWritebackPreservesLocalRelease) {
    auto disabled            = makeManager(RoleType::DECODE, source_, false);
    disabled->p2p_connector_ = decode_->p2p_connector_;
    decode_                  = std::move(disabled);
    prepare();
    stream_->releaseResource();
    EXPECT_TRUE(drained());
    EXPECT_EQ(prefill_service_.handshakes, 0);
    EXPECT_EQ(sourceRefs(1), 1u);
}

TEST_F(DecodeWritebackIntegrationTest, MissingRoutingPreservesLocalRelease) {
    prepare();
    stream_->generate_input_->generate_config->role_addrs.clear();
    stream_->releaseResource();
    EXPECT_TRUE(drained());
    EXPECT_EQ(prefill_service_.handshakes, 0);
    EXPECT_EQ(sourceRefs(1), 1u);
}

TEST_F(DecodeWritebackIntegrationTest, IncompleteResponseBlockDoesNotSubmit) {
    prepare(8, 7);
    stream_->releaseResource();
    EXPECT_TRUE(drained());
    EXPECT_EQ(prefill_service_.handshakes, 0);
}

TEST_F(DecodeWritebackIntegrationTest, HandshakeFailureDoesNotChangeFinishedRequest) {
    prefill_service_.prefill.engine_.reset();
    prepare();
    stream_->releaseResource();
    ASSERT_TRUE(await([&]() { return prefill_service_.handshakes > 0 && drained(); }));
    EXPECT_EQ(sender_->started, 0);
    EXPECT_EQ(sourceRefs(1), 1u);
    EXPECT_FALSE(stream_->hasError());
    EXPECT_EQ(stream_->getStatus(), StreamState::FINISHED);
}

TEST_F(DecodeWritebackIntegrationTest, NoTransferReleasesSourceHold) {
    ASSERT_TRUE(test::seedCompleteBlockTreePath(target_, keys_).success);
    prepare();
    stream_->releaseResource();
    ASSERT_TRUE(await([&]() { return prefill_service_.handshakes > 0 && drained(); }));
    EXPECT_EQ(sender_->started, 0);
    EXPECT_EQ(sourceRefs(1), 1u);
}

TEST_F(DecodeWritebackIntegrationTest, CancelledStreamDoesNotTriggerWriteback) {
    prepare();
    stream_->reportError(ErrorCode::CANCELLED, "test cancellation");
    stream_->releaseResource();
    EXPECT_TRUE(drained());
    EXPECT_EQ(prefill_service_.handshakes, 0);
}

TEST_F(DecodeWritebackIntegrationTest, DeadlineKeepsSourcePinnedUntilSenderStops) {
    prepare();
    stream_->releaseResource();
    ASSERT_TRUE(await([&]() { return sender_->started > 0; }));
    auto write = context();
    ASSERT_NE(write, nullptr);
    ASSERT_TRUE(await([&]() { return write->done(); }));
    EXPECT_FALSE(write->success());
    EXPECT_TRUE(write->resourceHoldPending());
    EXPECT_EQ(sourceRefs(1), 2u);
    sender_->release();
    ASSERT_TRUE(await([&]() { return drained(); }));
    EXPECT_EQ(sourceRefs(1), 1u);
    EXPECT_EQ(target_->devicePrefixBlocksForTest(keys_), 1u);
}

TEST_F(DecodeWritebackIntegrationTest, StopDrainsWithWorkerRpcStillAvailable) {
    prepare();
    stream_->releaseResource();
    ASSERT_TRUE(await([&]() { return sender_->started > 0; }));
    auto write    = context();
    auto stopping = std::async(std::launch::async, [&]() { decode_->stopWriteback(); });
    EXPECT_TRUE(await([&]() { return write->done(); }));
    EXPECT_EQ(stopping.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
    EXPECT_EQ(sourceRefs(1), 2u);
    sender_->release();
    EXPECT_EQ(stopping.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_FALSE(write->resourceHoldPending());
    EXPECT_EQ(sourceRefs(1), 1u);
}

}  // namespace
}  // namespace rtp_llm
