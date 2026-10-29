#include <gtest/gtest.h>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <numeric>
#include <thread>
#include <tuple>
#include <cuda_runtime.h>

#include "autil/EnvUtil.h"
#include "autil/NetUtil.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/cpp/cache/CoordinatorCacheManager.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
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
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"
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
    PrefillRpcServer prefill;
    DecodeRpcServer      decode;
    std::atomic<int>     handshakes{0};
    grpc::Status         ExecuteFunction(grpc::ServerContext*     context,
                                         const FunctionRequestPB* request,
                                         FunctionResponsePB*      response) override {
        return local.ExecuteFunction(context, request, response);
    }
    grpc::Status StartWriteBack(grpc::ServerContext*                   context,
                            const P2PConnectorStartWriteBackRequestPB* request,
                            P2PConnectorStartWriteBackResponsePB*      response) override {
        ++handshakes;
        return prefill.StartWriteBack(context, request, response);
    }

    grpc::Status CancelWriteBack(grpc::ServerContext*                    context,
                             const P2PConnectorCancelWriteBackRequestPB* request,
                             P2PConnectorStartWriteBackResponsePB*       response) override {
        return prefill.CancelWriteBack(context, request, response);
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
            offered_blocks.fetch_add(request.block_info.size());
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
    std::atomic<size_t>                              offered_blocks{0};

private:
    std::mutex              mutex_;
    std::condition_variable cv_;
    bool                    released_{false};
};

template<typename AllocatorType>
class DecodeWritebackIntegrationTestBase: public DeviceTestBase {
protected:
    using Allocator = test::BlockTreeCacheTestAllocator<AllocatorType>;
    autil::EnvGuard                                    perf_{"PERF_TEST", "1"};
    CacheConfig                                        config_;
    CacheConfig                                        prefill_config_;
    std::shared_ptr<Allocator>                         source_, target_;
    std::shared_ptr<KVCacheManager>                    decode_, prefill_;
    std::shared_ptr<GatedTcpSender>                    sender_;
    std::shared_ptr<transfer::tcp::TcpKVCacheReceiver> receiver_;
    WritebackService                                   decode_service_, prefill_service_;
    std::vector<std::unique_ptr<grpc::Server>>         servers_;
    std::vector<std::shared_ptr<Allocator>>            peer_allocators_;
    std::vector<std::shared_ptr<Allocator>>            decode_peer_allocators_;
    std::vector<std::shared_ptr<KVCacheManager>>       peer_managers_;
    std::vector<std::unique_ptr<WritebackService>>     peer_services_;
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
    makeManager(RoleType                          role,
                const std::shared_ptr<Allocator>& allocator,
                bool                              enabled          = true,
                double                            kv_reserve_ratio = CacheStoreConfig{}.p2p_writeback_kv_reserve_ratio,
                const KVCacheConfig&              kv_cache_config  = {}) {
        PDSepConfig pd;
        pd.role_type = role;
        CacheStoreConfig store;
        store.p2p_writeback_enable           = enabled;
        store.p2p_writeback_kv_reserve_ratio = kv_reserve_ratio;
        auto manager =
            std::make_shared<KVCacheManager>(role == RoleType::PREFILL ? prefill_config_ : config_,
                                             false,
                                             nullptr,
                                             kv_cache_config,
                                             role == RoleType::DECODE ? parallelismFor(role) : ParallelismConfig{},
                                             RuntimeConfig{},
                                             SpeculativeExecutionConfig{},
                                             pd,
                                             store);
        manager->coordinator_manager_ = allocator;
        manager->block_tree_cache_    = allocator->blockTreeCache();
        manager->cp_slot_mapper_      = allocator->cpSlotMapper();
        return manager;
    }

    virtual CacheConfig cacheConfig(RoleType) const {
        return test::makeSimpleMhaCacheConfig(2, 16, 4, DataType::TYPE_FP16, 1, 2);
    }

    virtual int prefillCpSize() const {
        return 1;
    }

    virtual int decodeTpSize() const {
        return 1;
    }

    ParallelismConfig parallelismFor(RoleType role) const {
        ParallelismConfig parallelism;
        parallelism.role_type = role;
        parallelism.tp_size = role == RoleType::PREFILL ? prefillCpSize() : decodeTpSize();
        if (prefillCpSize() > 1) {
            parallelism.prefill_cp_config.kv_cache_sharded = true;
            parallelism.prefill_cp_config.prefill_cp_size  = prefillCpSize();
            parallelism.prefill_cp_config.method =
                role == RoleType::PREFILL ? CPRotateMethod::ALL_GATHER : CPRotateMethod::PREFILL_CP;
        }
        return parallelism;
    }

    void SetUp() override {
        DeviceTestBase::SetUp();
        config_ = cacheConfig(RoleType::DECODE);
        config_.cache_key_hash_seed = 42;
        prefill_config_ = cacheConfig(RoleType::PREFILL);
        prefill_config_.cache_key_hash_seed = config_.cache_key_hash_seed;
        source_ = std::make_shared<Allocator>(config_);
        target_ = std::make_shared<Allocator>(prefill_config_);
        ASSERT_TRUE(source_->init());
        ASSERT_TRUE(target_->init());
        if (prefillCpSize() > 1) {
            target_->setCPSlotMapper(std::make_shared<CPSlotMapper>(0, prefillCpSize(), config_.seq_size_per_block));
        }
        decode_                          = makeManager(RoleType::DECODE, source_);
        prefill_                         = makeManager(RoleType::PREFILL, target_);
        decode_service_.local.engine_    = std::make_shared<WritebackEngine>(decode_);
        decode_service_.decode.engine_   = decode_service_.local.engine_;
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
        std::vector<std::string> decode_grpc_addrs{decode_addr};
        std::vector<std::string> prefill_grpc_addrs{prefill_addr};
        std::vector<std::string> prefill_worker_addrs{"127.0.0.1:" + std::to_string(transfer_port) + ":"
                                                      + std::to_string(prefill_port_)};
        for (int rank = 1; rank < prefillCpSize(); ++rank) {
            auto allocator = std::make_shared<Allocator>(prefill_config_);
            ASSERT_TRUE(allocator->init());
            allocator->setCPSlotMapper(
                std::make_shared<CPSlotMapper>(rank, prefillCpSize(), config_.seq_size_per_block));
            auto manager           = makeManager(RoleType::PREFILL, allocator);
            auto service           = std::make_unique<WritebackService>();
            service->local.engine_ = std::make_shared<WritebackEngine>(manager);
            int rpc_port           = 0;
            prefill_grpc_addrs.push_back(launch(*service, rpc_port));
            const int tcp_port = autil::NetUtil::randomPort();
            auto      receiver = std::make_shared<transfer::tcp::TcpKVCacheReceiver>();
            ASSERT_TRUE(receiver->init(tcp_port, 1, 1));
            prefill_worker_addrs.push_back("127.0.0.1:" + std::to_string(tcp_port) + ":" + std::to_string(rpc_port));
            P2PConnectorConfig connector_config;
            connector_config.role_type                          = RoleType::PREFILL;
            connector_config.tp_rank                            = rank;
            connector_config.p2p_writeback_enable               = true;
            auto& worker                                        = connector_config.worker_config;
            worker.topology                                     = prefill_config_.topologyPtr();
            worker.layer_all_num                                = config_.layer_num;
            worker.tp_rank                                      = rank;
            worker.tp_size                                      = prefillCpSize();
            worker.cp_size                                      = prefillCpSize();
            worker.p2p_writeback_cleanup_interval_ms = 5;
            auto converter                                      = std::make_shared<LayerBlockConverterImpl>(allocator);
            auto connector = std::make_shared<P2PConnector>(connector_config, converter, nullptr, allocator);
            auto entry     = std::make_unique<P2PConnectorPrefill>(connector_config, converter, nullptr, allocator);
            entry->write_worker_ = std::make_unique<P2PWorkerPrefillWrite>(worker, converter, receiver);
            entry->write_registration_client_ =
                std::make_shared<P2PBroadcastClient>(std::vector<std::string>{prefill_addr}, 100);
            ASSERT_TRUE(entry->write_worker_->init());
            ASSERT_TRUE(entry->write_registration_client_->init());
            connector->prefill_     = std::move(entry);
            manager->p2p_connector_ = std::move(connector);
            peer_allocators_.push_back(std::move(allocator));
            peer_managers_.push_back(std::move(manager));
            peer_services_.push_back(std::move(service));
        }
        for (int rank = 1; rank < decodeTpSize(); ++rank) {
            auto allocator = std::make_shared<Allocator>(config_);
            ASSERT_TRUE(allocator->init());
            auto manager = makeManager(RoleType::DECODE, allocator);
            auto service = std::make_unique<WritebackService>();
            service->local.engine_ = std::make_shared<WritebackEngine>(manager);
            int rpc_port = 0;
            decode_grpc_addrs.push_back(launch(*service, rpc_port));
            P2PConnectorConfig connector_config;
            connector_config.role_type = RoleType::DECODE;
            connector_config.tp_rank = rank;
            connector_config.p2p_writeback_enable = true;
            auto& worker = connector_config.worker_config;
            worker.topology = config_.topologyPtr();
            worker.layer_all_num = config_.layer_num;
            worker.tp_rank = rank;
            worker.tp_size = decodeTpSize();
            worker.cache_key_tokens_per_block = config_.seq_size_per_block;
            worker.p2p_writeback_cleanup_interval_ms = 5;
            auto converter = std::make_shared<LayerBlockConverterImpl>(allocator);
            auto connector = std::make_shared<P2PConnector>(connector_config, converter, nullptr, allocator);
            auto entry = std::make_unique<P2PConnectorDecode>(connector_config, converter, nullptr);
            entry->write_worker_ = std::make_unique<P2PWorkerDecodeWrite>(worker, converter, sender_);
            ASSERT_TRUE(entry->write_worker_->init());
            connector->decode_ = std::move(entry);
            manager->p2p_connector_ = std::move(connector);
            decode_peer_allocators_.push_back(std::move(allocator));
            peer_managers_.push_back(std::move(manager));
            peer_services_.push_back(std::move(service));
        }
        // Warm the synchronous D2H path before starting the writeback deadline.
        auto warmup = torch::zeros({4096}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
        std::array<std::array<char, 1024>, 4> host{};
        std::vector<transfer::tcp::CopyTask>  copies;
        for (size_t i = 0; i < host.size(); ++i) {
            copies.push_back({warmup.data_ptr<uint8_t>() + i * 1024, 1024, host[i].data(), warmup.get_device()});
        }
        ASSERT_TRUE(sender_->sender_->cuda_copy_util_->batchCopyToHost(copies));

        for (auto role : {RoleType::DECODE, RoleType::PREFILL}) {
            auto               manager   = role == RoleType::DECODE ? decode_ : prefill_;
            auto               converter = std::make_shared<LayerBlockConverterImpl>(manager->coordinator_manager_);
            const auto&        local_config = role == RoleType::DECODE ? config_ : prefill_config_;
            P2PConnectorConfig connector_config;
            connector_config.role_type                             = role;
            connector_config.p2p_writeback_enable                  = true;
            auto& worker                                           = connector_config.worker_config;
            worker.topology                                        = local_config.topologyPtr();
            worker.cache_key_tokens_per_block                      = config_.seq_size_per_block;
            worker.layer_all_num                                   = config_.layer_num;
            worker.tp_size                                         = role == RoleType::PREFILL ? prefillCpSize() : decodeTpSize();
            worker.cp_size                                         = role == RoleType::PREFILL ? prefillCpSize() : 1;
            worker.p2p_writeback_cleanup_interval_ms    = 5;
            auto& scheduler                                        = connector_config.scheduler_config;
            scheduler.topology                                     = local_config.topologyPtr();
            scheduler.cache_key_tokens_per_block                           = config_.seq_size_per_block;
            scheduler.cache_key_hash_seed = config_.cache_key_hash_seed;
            scheduler.role_type                                    = role;
            scheduler.parallelism_config                           = parallelismFor(role);
            scheduler.worker_grpc_addrs =
                role == RoleType::PREFILL ? prefill_grpc_addrs : decode_grpc_addrs;
            scheduler.worker_addrs                                 = prefill_worker_addrs;
            scheduler.p2p_cancel_broadcast_timeout_ms              = 100;
            scheduler.p2p_writeback_timeout_ms                     = 3000;
            auto client = std::make_shared<P2PBroadcastClient>(scheduler.worker_grpc_addrs, 100);
            ASSERT_TRUE(client->init());
            auto connector = std::make_shared<P2PConnector>(connector_config, converter, nullptr, manager->coordinator_manager_);
            if (role == RoleType::DECODE) {
                auto entry              = std::make_unique<P2PConnectorDecode>(connector_config, converter, nullptr);
                entry->write_worker_    = std::make_unique<P2PWorkerDecodeWrite>(worker, converter, sender_);
                entry->write_scheduler_ =
                    std::make_unique<P2PSchedulerDecodeWrite>(scheduler, client, *entry->write_worker_);
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
        keys_ = calculateCacheKeys(tokens_.data(), 12, 4, config_.cache_key_hash_seed);
        if (prefillCpSize() == 1) {
            ASSERT_TRUE(test::seedCompleteBlockTreePath(target_, {keys_.front()}).success);
        }
    }

    void TearDown() override {
        if (sender_)
            sender_->release();
        if (decode_)
            decode_->stopWriteback();
        if (prefill_)
            prefill_->stopWriteback();
        for (const auto& manager : peer_managers_) {
            manager->stopWriteback();
        }
        stream_.reset();
        for (auto& server : servers_) {
            server->Shutdown();
            server->Wait();
        }
        DeviceTestBase::TearDown();
    }

    void prepare(int  token_count         = 13,
                 int  ready_count         = 12,
                 int  input_length        = 4,
                 bool reuse_cache         = true,
                 bool request_reuse_cache = true,
                 bool enable_device_cache = true) {
        auto input = std::make_shared<GenerateInput>();
        input->input_ids =
            torch::tensor(std::vector<int>(tokens_.begin(), tokens_.begin() + input_length), torch::kInt32);
        auto gc                 = std::make_shared<GenerateConfig>();
        gc->max_new_tokens      = 32;
        gc->unique_key          = "finished-writeback";
        gc->reuse_cache         = request_reuse_cache;
        gc->enable_device_cache = true;
        gc->role_addrs.emplace_back(RoleType::PREFILL, "127.0.0.1", 0, prefill_port_);
        input->generate_config = gc;
        ResourceContext resources;
        resources.cache_manager       = decode_;
        resources.reuse_cache         = reuse_cache;
        resources.enable_device_cache = enable_device_cache;
        resources.role_type           = RoleType::DECODE;
        ModelConfig model;
        model.max_seq_len                  = 64;
        model.vocab_size                   = 128;
        model.attn_config.tokens_per_block = 4;
        stream_ = std::make_shared<NormalGenerateStream>(input, model, RuntimeConfig{}, resources, nullptr);
        stream_->setPrefillTpSize(prefillCpSize());
        stream_->setPrefillCpSize(prefillCpSize());
        stream_->generate_status_->status = StreamState::RUNNING;
        ASSERT_TRUE(stream_->streamCacheResource().initKVBlock().ok());
        auto new_tokens = torch::tensor(std::vector<int>(tokens_.begin() + input_length, tokens_.begin() + token_count),
                                        torch::kInt32)
                              .reshape({1, token_count - input_length});
        StreamUpdateInfo update{new_tokens, token_count - input_length};
        update.kv_ready_token_count = ready_count;
        stream_->update(update);
        ASSERT_FALSE(stream_->hasError());
        ASSERT_TRUE(stream_->incrKVBlock().ok());
        source_blocks_ = stream_->streamCacheResource().kvCache().blocks(0, config_.topology().groups().front().tag);
        for (const auto& group : config_.topology().groups()) {
            const auto& blocks = stream_->streamCacheResource().kvCache().blocks(0, group.tag);
            for (int layer : config_.topology().layerIdsForGroup(group.tag)) {
                for (size_t i = 0; i < blocks.size(); ++i) {
                    if (isNullBlockIdx(blocks[i])) {
                        continue;
                    }
                    for (const auto& block : source_->convertIndexToBuffer(layer, group.tag, blocks[i], 1, 0)) {
                        ASSERT_EQ(cudaMemset(block.addr, 10 * layer + i + 1, block.size_bytes), cudaSuccess);
                    }
                }
            }
        }
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        stream_->generate_status_->status = StreamState::FINISHED;
    }

    void checkWritebackAfterFinalStepCrossesBoundary() {
        prepare(7, 6);
        ASSERT_EQ(stream_->cacheKeys().size(), 1u);
        stream_->generate_status_->status = StreamState::RUNNING;
        StreamUpdateInfo update{torch::tensor({{8, 9}}, torch::kInt32), 2};
        update.kv_ready_token_count = 8;
        stream_->update(update);
        ASSERT_EQ(stream_->seqLength(), 9);
        ASSERT_EQ(stream_->writebackKVReadyTokenCount(), 8);
        ASSERT_EQ(stream_->cacheKeys().size(), 1u);
        stream_->reportEvent(StreamEvents::GenerateDone);
        ASSERT_EQ(stream_->moveToNext(), StreamState::FINISHED);
        ASSERT_TRUE(await([&]() { return sender_->started > 0; }));
        auto write = context();
        ASSERT_NE(write, nullptr);
        ASSERT_NE(writeSource(), nullptr);
        EXPECT_EQ(writeSource()->cacheKeys(), (CacheKeysType{keys_[0], keys_[1]}));
        sender_->release();
        ASSERT_TRUE(await([&]() { return drained(); }));
        ASSERT_TRUE(write->success()) << write->errorInfo().ToString();
        const CacheKeysType expected_keys = prefillCpSize() == 1 ? CacheKeysType{keys_[0], keys_[1]} :
                                                                 CacheKeysType{keys_[1]};
        EXPECT_EQ(target_->devicePrefixBlocksForTest(expected_keys), expected_keys.size());
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
        ASSERT_NE(writeSource(), nullptr);
        EXPECT_EQ(writeSource()->cacheKeys(), keys_);
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
    KVCacheResourcePtr writeSource() {
        auto&                       worker = *decode_->p2p_connector_->decode_->write_worker_;
        std::lock_guard<std::mutex> lock(worker.mutex_);
        auto                        it = worker.groups_.find("finished-writeback");
        return it == worker.groups_.end() ? nullptr : it->second->sourceResource();
    }
    bool drained() {
        return !writeSource() && checker().inflightContextCount() == 0
               && prefill_->p2p_connector_->prefill_->write_scheduler_->inflightContextCount() == 0;
    }
    size_t sourceRefs(size_t index) {
        const auto pool = source_->blockTreeCache()->groupSets().front()->devicePools().front();
        return pool->isAllocated(source_blocks_[index]) ? pool->refCount(source_blocks_[index]) : 0;
    }
};

class DecodeWritebackIntegrationTest: public DecodeWritebackIntegrationTestBase<CoordinatorCacheManager> {};

class DecodeWritebackCachePolicyIntegrationTest:
    public DecodeWritebackIntegrationTest,
    public ::testing::WithParamInterface<std::tuple<bool, bool, bool>> {};

class DecodeWritebackIndependentPoolsIntegrationTest:
    public DecodeWritebackIntegrationTestBase<CoordinatorCacheManager> {
protected:
    CacheConfig cacheConfig(RoleType) const override {
        auto       config                  = test::makeSimpleMhaCacheConfig(2, 16, 4, DataType::TYPE_FP16, 1, 2);
        const auto full                    = test::makeMhaSpec("full", 4, DataType::TYPE_FP16, 1, 2);
        const auto swa                     = test::makeMhaSpec("swa", 4, DataType::TYPE_FP16, 1, 2);
        auto       policy                  = defaultCacheGroupPolicy(CacheGroupType::SWA);
        policy.sliding_window_size         = 8;
        config.fromGroupedSpecs({full, swa},
                                {{0}, {1}},
                                {CacheGroupType::FULL, CacheGroupType::SWA},
                                {"full", "swa"},
                                {defaultCacheGroupPolicy(CacheGroupType::FULL), policy});
        config.setGroupBlockLayout({"full", "swa"}, {8, 32}, {full->block_size_bytes(), swa->block_size_bytes()}, {0, 0});
        return config;
    }
};

class DecodeWritebackCpIntegrationTest: public DecodeWritebackIntegrationTestBase<CoordinatorCacheManager> {
protected:
    int prefillCpSize() const override {
        return 2;
    }
};

TEST_F(DecodeWritebackIntegrationTest, FinalStepCrossesBoundaryWithoutAnotherAllocation) {
    checkWritebackAfterFinalStepCrossesBoundary();
}

TEST_F(DecodeWritebackCpIntegrationTest, FinalStepCrossesBoundaryWithoutAnotherAllocation) {
    checkWritebackAfterFinalStepCrossesBoundary();
}

TEST_F(DecodeWritebackCpIntegrationTest, TcpScatterPublishesCanonicalPrefixAndReusesBothRankBuffers) {
    prepare();
    stream_->generate_status_->status = StreamState::RUNNING;
    stream_->reportEvent(StreamEvents::GenerateDone);
    ASSERT_EQ(stream_->moveToNext(), StreamState::FINISHED);
    ASSERT_TRUE(await([&]() { return sender_->started > 0; }));
    auto write = context();
    ASSERT_NE(write, nullptr);
    EXPECT_EQ(target_->devicePrefixBlocksForTest({keys_[1]}), 0u);
    sender_->release();
    ASSERT_TRUE(await([&]() { return drained(); }));
    ASSERT_TRUE(write->success()) << write->errorInfo().ToString();
    EXPECT_EQ(target_->devicePrefixBlocksForTest({keys_[1]}), 1u);
    EXPECT_EQ(target_->devicePrefixBlocksForTest({keys_[1], keys_[2]}), 1u);

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
    EXPECT_EQ(result.reuse_len, 8);
    const auto first_use_tokens = result.reuse_accounting.unaccountedBlocks(result.reuse_len / 8) * 8;
    EXPECT_EQ(first_use_tokens, 8u);
    ASSERT_EQ(peer_allocators_.size(), 1u);
    EXPECT_EQ(peer_allocators_.front()->devicePrefixBlocksForTest({keys_[1]}), 0u);
    const std::array<std::shared_ptr<Allocator>, 2> ranks{target_, peer_allocators_.front()};
    for (size_t rank = 0; rank < ranks.size(); ++rank) {
        SCOPED_TRACE(rank);
        for (int layer = 0; layer < 2; ++layer) {
            for (const auto& block : ranks[rank]->convertIndexToBuffer(
                     layer, config_.topology().groups().front().tag, reused->blocks(0, config_.topology().groups().front().tag).front(), 1, 0)) {
                std::vector<uint8_t> bytes(block.size_bytes);
                ASSERT_EQ(cudaMemcpy(bytes.data(), block.addr, bytes.size(), cudaMemcpyDeviceToHost), cudaSuccess);
                EXPECT_TRUE(std::all_of(
                    bytes.begin(), bytes.end(), [&](uint8_t value) { return value == 10 * layer + rank + 1; }));
            }
        }
    }
    InsertInfo insert_info{reused, ids, false, Tier::DEVICE};
    insert_info.account_for_prefill = true;
    size_t resident_prefix_length = 0;
    prefill_->insertIntoCache(insert_info, resident_prefix_length);
    const auto cache = target_->blockTreeCacheOwner();
    auto after_commit = cache->match({keys_[1]});
    EXPECT_EQ(after_commit.matched_device_blocks, 1u);
    EXPECT_EQ(after_commit.reuse_accounting.unaccountedBlocks(1), 0u);
    EXPECT_EQ(result.reuse_accounting.unaccountedBlocks(1), 1u);
    block_tree_cache_test::releaseRequestRefsForTest(*cache, after_commit.matched_device_resources);
    prefill_->free(FreeInfo{reused, ids});
}

class DecodeWritebackCpTpIntegrationTest:
    public DecodeWritebackCpIntegrationTest,
    public ::testing::WithParamInterface<int> {
protected:
    int decodeTpSize() const override {
        return 2;
    }

    CacheConfig cacheConfig(RoleType role) const override {
        auto config = DecodeWritebackCpIntegrationTest::cacheConfig(role);
        auto parallelism = parallelismFor(role);
        AttentionConfigs attention;
        attention.kv_head_num = GetParam();
        attention.size_per_head = 2;
        KVCacheSpecDesc desc;
        desc.tag = "default";
        desc.cache_type = KVCacheSpecType::MultiHeadAttention;
        desc.dtype = DataType::TYPE_FP16;
        SpecBuildContext build;
        build.seq_size_per_block = 4;
        build.attn_config = &attention;
        build.parallelism_config = &parallelism;
        auto spec = MHAKVCacheSpec::build(desc, build);
        config.fromGroupedSpecs({spec}, {{0, 1}}, {CacheGroupType::FULL}, {desc.tag});
        config.setGroupBlockLayout({desc.tag}, {16}, {spec->block_size_bytes()}, {0});
        return config;
    }
};

TEST_P(DecodeWritebackCpTpIntegrationTest, TcpScatterMergesHeadsAndReusesBothPrefillRankBuffers) {
    ASSERT_TRUE(decode_->writebackEnabled());
    prepare();
    ASSERT_EQ(decode_peer_allocators_.size(), 1u);
    const int head_shards = std::gcd(GetParam(), decodeTpSize());
    const int local_heads = GetParam() / head_shards;
    const std::array<std::shared_ptr<Allocator>, 2> sources{source_, decode_peer_allocators_.front()};
    for (int rank = 0; rank < decodeTpSize(); ++rank) {
        for (int layer = 0; layer < 2; ++layer) {
            for (size_t position = 0; position < source_blocks_.size(); ++position) {
                for (int head = 0; head < local_heads; ++head) {
                    const int global_head = rank / (decodeTpSize() / head_shards) * local_heads + head;
                    const auto blocks = sources[rank]->convertIndexToBuffer(
                        layer, "default", source_blocks_[position], local_heads, head);
                    ASSERT_EQ(blocks.size(), 2u);
                    for (size_t kv = 0; kv < blocks.size(); ++kv) {
                        const int value = 64 * kv + 16 * layer + 4 * position + global_head + 1;
                        ASSERT_EQ(cudaMemset(blocks[kv].addr, value, blocks[kv].size_bytes), cudaSuccess);
                    }
                }
            }
        }
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    stream_->generate_status_->status = StreamState::RUNNING;
    stream_->reportEvent(StreamEvents::GenerateDone);
    ASSERT_EQ(stream_->moveToNext(), StreamState::FINISHED);
    ASSERT_TRUE(await([&]() { return sender_->started >= head_shards; }));
    auto write = context();
    ASSERT_NE(write, nullptr);
    EXPECT_EQ(target_->devicePrefixBlocksForTest({keys_[1]}), 0u);
    sender_->release();
    ASSERT_TRUE(await([&]() { return drained(); }));
    ASSERT_TRUE(write->success()) << write->errorInfo().ToString();
    EXPECT_EQ(target_->devicePrefixBlocksForTest({keys_[1]}), 1u);

    auto reused = std::make_shared<BatchKVCacheResource>();
    reused->resetBatchSize(1);
    reused->initGroups(prefill_config_.topologyPtr());
    auto ids = std::make_shared<CompleteTokenIds>(1, 1, 64, 4);
    auto input = std::make_shared<GenerateInput>();
    input->input_ids = torch::tensor(tokens_, torch::kInt32);
    input->generate_config = std::make_shared<GenerateConfig>();
    ids->init(input);
    MallocInfo info;
    info.batch_kv_cache_resource = reused;
    info.complete_token_ids = ids;
    info.reuse_cache = true;
    info.enable_cache_lookup = true;
    const auto result = prefill_->malloc(info);
    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.reuse_len, 8);
    ASSERT_EQ(peer_allocators_.size(), 1u);
    const std::array<std::shared_ptr<Allocator>, 2> targets{target_, peer_allocators_.front()};
    for (int rank = 0; rank < prefillCpSize(); ++rank) {
        for (int layer = 0; layer < 2; ++layer) {
            for (int head = 0; head < GetParam(); ++head) {
                const auto blocks = targets[rank]->convertIndexToBuffer(
                    layer, "default", reused->blocks(0, "default").front(), GetParam(), head);
                ASSERT_EQ(blocks.size(), 2u);
                for (size_t kv = 0; kv < blocks.size(); ++kv) {
                    const uint8_t expected = 64 * kv + 16 * layer + 4 * rank + head + 1;
                    std::vector<uint8_t> bytes(blocks[kv].size_bytes);
                    ASSERT_EQ(cudaMemcpy(bytes.data(), blocks[kv].addr, bytes.size(), cudaMemcpyDeviceToHost), cudaSuccess);
                    EXPECT_EQ(bytes, std::vector<uint8_t>(bytes.size(), expected))
                        << "rank=" << rank << " layer=" << layer << " head=" << head << " kv=" << kv;
                }
            }
        }
    }
    prefill_->free(FreeInfo{reused, ids});
}

INSTANTIATE_TEST_SUITE_P(ReplicatedAndShardedHeads, DecodeWritebackCpTpIntegrationTest, ::testing::Values(1, 4));

class DecodeWritebackLinearCpIntegrationTest: public DecodeWritebackCpIntegrationTest {
protected:
    CacheConfig cacheConfig(RoleType role) const override {
        auto config = DecodeWritebackCpIntegrationTest::cacheConfig(role);
        const auto spec = test::makeLinearSpec("default", 4, DataType::TYPE_FP16, 1, 2);
        config.fromGroupedSpecs({spec}, {{0, 1}}, {CacheGroupType::LINEAR}, {"default"});
        config.setGroupBlockLayout({"default"}, {16}, {spec->block_size_bytes()}, {0});
        config.linear_step = 1;
        return config;
    }
};

TEST_F(DecodeWritebackLinearCpIntegrationTest, FinalStepReplicatesStateAndPublishesCanonicalPrefix) {
    checkWritebackAfterFinalStepCrossesBoundary();
}

class DecodeWritebackHybridIntegrationTest:
    public DecodeWritebackIntegrationTestBase<CoordinatorCacheManager>,
    public ::testing::WithParamInterface<int> {
protected:
    CacheConfig cacheConfig(RoleType) const override {
        auto config        = test::makeFullSwaLinearCacheConfig(GetParam() != 1);
        config.linear_step = 1;
        if (GetParam() == 2) {
            auto groups                          = config.topology().groups();
            groups[1].policy.sliding_window_size = 16;
            groups[1].policy.active_tail_blocks  = 4;
            config.setTopology(std::move(groups), config.topology().layers());
        } else if (GetParam() == 3 || GetParam() == 4) {
            const bool linear = GetParam() == 3;
            auto       group  = config.topology().groups()[linear ? 2 : 1];
            KVCacheSpecPtr spec;
            if (linear) {
                spec = test::makeLinearSpec(group.tag, 4, DataType::TYPE_FP16, 1, 2);
            } else {
                spec = test::makeMhaSpec(group.tag, 4, DataType::TYPE_FP16, 1, 2);
            }
            config.fromGroupedSpecs({spec}, {{0, 1, 2}}, {group.policy.group_type}, {group.tag}, {group.policy});
        }
        std::vector<uint32_t> blocks(config.groupNums(), 128);
        std::vector<size_t>   strides, scales;
        for (const auto& group : config.topology().groups()) {
            strides.push_back(group.kvBlockStrideBytes());
            scales.push_back(group.kvScaleStrideBytes());
        }
        config.setGroupBlockLayout(config.groupTags(), blocks, strides, scales);
        return config;
    }
};

TEST_P(DecodeWritebackHybridIntegrationTest, TransfersReusableGroupsAndPreservesSparseHoles) {
    tokens_ = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
    keys_   = calculateCacheKeys(tokens_.data(), 20, 4, config_.cache_key_hash_seed);
    prepare(21, 20);
    // Keep only the configured active tail among complete source blocks.
    for (const auto& group : source_->cacheGroups()) {
        if (group->policy().group_type == CacheGroupType::FULL) {
            continue;
        }
        const size_t keep = group->activeTailBlocks() == 0 ? keys_.size() : group->activeTailBlocks();
        auto&        ids = stream_->streamCacheResource().kvCache().cacheResource(0).mutableBlockIds(group->tag());
        for (size_t i = 0; i + keep < keys_.size(); ++i) {
            const auto block = ids.blocks()[i];
            if (!isNullBlockIdx(block)) {
                group->unreference({block});
                ids.setAt(i, NULL_BLOCK_IDX);
            }
        }
    }
    stream_->generate_status_->status = StreamState::RUNNING;
    stream_->reportEvent(StreamEvents::GenerateDone);
    ASSERT_EQ(stream_->moveToNext(), StreamState::FINISHED);
    ASSERT_TRUE(await([&]() { return sender_->started > 0; }));
    auto write = context();
    ASSERT_NE(write, nullptr);
    ASSERT_NE(writeSource(), nullptr);
    EXPECT_EQ(writeSource()->cacheKeys(), keys_);
    sender_->release();
    ASSERT_TRUE(await([&]() { return drained(); }));
    ASSERT_TRUE(write->success()) << write->errorInfo().ToString();

    const auto cache = target_->blockTreeCacheOwner();
    const auto path  = cache->tree()->findNode(keys_);
    ASSERT_EQ(path.size(), keys_.size());
    for (const auto& group_set : cache->groupSets()) {
        const auto&  base  = config_.group(group_set->groupTags().front());
        const size_t count = base.policy.active_tail_blocks == 0 ? keys_.size() : base.policy.active_tail_blocks;
        const size_t first = std::max<size_t>(1, keys_.size() - std::min(count, keys_.size()));
        const auto   gid   = group_set->groupSetId();
        for (size_t i = 1; i < first; ++i) {
            EXPECT_TRUE(path[i]->group_set_resources[gid].is_empty());
        }
        for (size_t i = first; i < keys_.size(); ++i) {
            const auto& blocks = path[i]->group_set_resources[gid].device_blocks;
            ASSERT_EQ(blocks.size(), group_set->groupTags().size());
            for (size_t member = 0; member < blocks.size(); ++member) {
                const auto& group = config_.group(group_set->groupTags()[member]);
                for (int layer : config_.topology().layerIdsForGroup(group.tag)) {
                    for (const auto& block :
                         target_->convertIndexToBuffer(layer, group.tag, blocks[member], 1, 0)) {
                        std::vector<uint8_t> bytes(block.size_bytes);
                        ASSERT_EQ(cudaMemcpy(bytes.data(), block.addr, bytes.size(), cudaMemcpyDeviceToHost),
                                  cudaSuccess);
                        EXPECT_TRUE(std::all_of(
                            bytes.begin(), bytes.end(), [&](uint8_t b) { return b == 10 * layer + i + 1; }));
                    }
                }
            }
        }
    }
    EXPECT_EQ(cache->groupSets().size(), GetParam() >= 3 ? 1u : (GetParam() == 1 ? 2u : 3u));
    auto match = cache->match(keys_);
    EXPECT_EQ(match.matched_device_blocks, keys_.size());
    EXPECT_EQ(match.reuse_accounting.unaccountedBlocks(match.matched_device_blocks), keys_.size() - 1);
    block_tree_cache_test::releaseRequestRefsForTest(*cache, match.matched_device_resources);
    auto repeated_match = cache->match(keys_);
    EXPECT_EQ(repeated_match.reuse_accounting.unaccountedBlocks(repeated_match.matched_device_blocks), keys_.size() - 1);
    block_tree_cache_test::releaseRequestRefsForTest(*cache, repeated_match.matched_device_resources);

    auto input = std::make_shared<GenerateInput>();
    input->input_ids = torch::tensor(tokens_, torch::kInt32);
    input->generate_config = std::make_shared<GenerateConfig>();
    ModelConfig model_config;
    model_config.max_seq_len = 64;
    model_config.vocab_size = 128;
    ResourceContext resource_context;
    resource_context.cache_manager = prefill_;
    resource_context.role_type = RoleType::PREFILL;
    resource_context.reuse_cache = true;
    auto reused_stream =
        std::make_shared<NormalGenerateStream>(input, model_config, RuntimeConfig{}, resource_context, nullptr);
    ASSERT_TRUE(reused_stream->initKVBlock().ok());
    EXPECT_EQ(reused_stream->initialReuseLength(), 20);
    EXPECT_EQ(reused_stream->writebackFirstUseLength(), 16);
    reused_stream->generate_status_->status = StreamState::FINISHED;
    reused_stream->releaseResource();
    auto after_commit = cache->match(keys_);
    EXPECT_EQ(after_commit.matched_device_blocks, keys_.size());
    EXPECT_EQ(after_commit.reuse_accounting.unaccountedBlocks(after_commit.matched_device_blocks), 0u);
    block_tree_cache_test::releaseRequestRefsForTest(*cache, after_commit.matched_device_resources);
}

INSTANTIATE_TEST_SUITE_P(FullSwaLinear, DecodeWritebackHybridIntegrationTest, ::testing::Values(0, 1, 2, 3, 4));

TEST_F(DecodeWritebackIntegrationTest, FinishedStreamReleasesBeforeTcpCopyAndNextRequestReusesSuffix) {
    prepare();
    stream_->generate_status_->status = StreamState::RUNNING;
    stream_->reportEvent(StreamEvents::GenerateDone);
    ASSERT_EQ(stream_->moveToNext(), StreamState::FINISHED);
    ASSERT_TRUE(await([&]() { return sender_->started > 0; }));
    auto write = context();
    ASSERT_NE(write, nullptr);
    EXPECT_TRUE(write->resourceHoldPending());
    EXPECT_EQ(sourceRefs(0), 1u);
    EXPECT_EQ(sourceRefs(1), 2u);
    EXPECT_EQ(stream_->streamCacheResource().curBlocksNum(), 0);
    stream_->generate_input_->generate_config->role_addrs.clear();
    stream_->completeTokenIdsPtr()->data(0)[4] = 99;
    stream_.reset();
    ASSERT_NE(writeSource(), nullptr);
    EXPECT_EQ(writeSource()->cacheKeys(), keys_);
    EXPECT_EQ(writeSource()->blocks("default").size(), 3u);
    EXPECT_TRUE(isNullBlockIdx(writeSource()->blocks("default")[0]));
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
    const auto first_use_tokens = result.reuse_accounting.unaccountedBlocks(result.reuse_len / 4) * 4;
    EXPECT_EQ(first_use_tokens, 8u);
    for (int layer = 0; layer < 2; ++layer) {
        for (size_t i = 1; i < 3; ++i) {
            for (const auto& block :
                 target_->convertIndexToBuffer(
                     layer, config_.topology().groups().front().tag, reused->blocks(0, "default")[i], 1, 0)) {
                std::vector<uint8_t> bytes(block.size_bytes);
                ASSERT_EQ(cudaMemcpy(bytes.data(), block.addr, bytes.size(), cudaMemcpyDeviceToHost), cudaSuccess);
                EXPECT_TRUE(
                    std::all_of(bytes.begin(), bytes.end(), [&](uint8_t b) { return b == 10 * layer + i + 1; }));
            }
        }
    }
    prefill_->free(FreeInfo{reused, ids});
}

TEST_P(DecodeWritebackCachePolicyIntegrationTest, LocalCachePolicyDoesNotGateWritebackOrPrefillReuse) {
    const auto [reuse_cache, request_reuse_cache, enable_device_cache] = GetParam();
    KVCacheConfig kv_cache_config;
    kv_cache_config.reuse_cache         = reuse_cache;
    kv_cache_config.enable_device_cache = enable_device_cache;
    auto manager                        = makeManager(
        RoleType::DECODE, source_, true, CacheStoreConfig{}.p2p_writeback_kv_reserve_ratio, kv_cache_config);
    manager->p2p_connector_ = decode_->p2p_connector_;
    decode_                 = std::move(manager);
    ASSERT_TRUE(decode_->writebackEnabled());
    prepare(13, 12, 4, reuse_cache, request_reuse_cache, enable_device_cache);
    const bool stored_locally = reuse_cache && request_reuse_cache && enable_device_cache;
    EXPECT_EQ(stream_->streamCacheResource().reuseCache(), reuse_cache && request_reuse_cache);
    stream_->generate_status_->status = StreamState::RUNNING;
    stream_->reportEvent(StreamEvents::GenerateDone);
    ASSERT_EQ(stream_->moveToNext(), StreamState::FINISHED);
    ASSERT_TRUE(await([&] { return sender_->started > 0; }));
    auto write = context();
    ASSERT_NE(write, nullptr);
    EXPECT_TRUE(write->resourceHoldPending());
    EXPECT_EQ(stream_->streamCacheResource().curBlocksNum(), 0);
    EXPECT_EQ(sourceRefs(0), stored_locally ? 1u : 0u);
    EXPECT_EQ(sourceRefs(1), stored_locally ? 2u : 1u);
    EXPECT_EQ(source_->devicePrefixBlocksForTest(keys_), stored_locally ? keys_.size() : 0u);
    stream_.reset();
    sender_->release();
    ASSERT_TRUE(await([&] { return drained(); }));
    ASSERT_TRUE(write->success()) << write->errorInfo().ToString();
    EXPECT_EQ(sourceRefs(1), stored_locally ? 1u : 0u);
    EXPECT_EQ(target_->devicePrefixBlocksForTest(keys_), keys_.size());

    auto input             = std::make_shared<GenerateInput>();
    input->input_ids       = torch::tensor(tokens_, torch::kInt32);
    input->generate_config = std::make_shared<GenerateConfig>();
    ModelConfig model;
    model.max_seq_len                  = 64;
    model.vocab_size                   = 128;
    model.attn_config.tokens_per_block = 4;
    ResourceContext resources;
    resources.cache_manager = prefill_;
    resources.role_type     = RoleType::PREFILL;
    resources.reuse_cache   = true;
    auto reused_stream      = std::make_shared<NormalGenerateStream>(input, model, RuntimeConfig{}, resources, nullptr);
    ASSERT_TRUE(reused_stream->initKVBlock().ok());
    EXPECT_EQ(reused_stream->initialReuseLength(), 12);
    EXPECT_EQ(reused_stream->writebackFirstUseLength(), 8);
    const auto& reused = reused_stream->streamCacheResource().kvCache();
    for (int layer = 0; layer < 2; ++layer) {
        for (size_t position = 1; position < keys_.size(); ++position) {
            for (const auto& block :
                 target_->convertIndexToBuffer(layer, "default", reused.blocks(0, "default")[position], 1, 0)) {
                std::vector<uint8_t> bytes(block.size_bytes);
                ASSERT_EQ(cudaMemcpy(bytes.data(), block.addr, bytes.size(), cudaMemcpyDeviceToHost), cudaSuccess);
                EXPECT_TRUE(std::all_of(
                    bytes.begin(), bytes.end(), [&](uint8_t value) { return value == 10 * layer + position + 1; }));
            }
        }
    }
    reused_stream->generate_status_->status = StreamState::FINISHED;
    reused_stream->releaseResource();
}

INSTANTIATE_TEST_SUITE_P(CachePolicy,
                         DecodeWritebackCachePolicyIntegrationTest,
                         ::testing::Combine(::testing::Bool(), ::testing::Bool(), ::testing::Bool()));

TEST_F(DecodeWritebackIntegrationTest, WritebackDiscountStartsAfterPrefillCommitThroughRealStreamOutputs) {
    prepare();
    stream_->generate_status_->status = StreamState::RUNNING;
    stream_->reportEvent(StreamEvents::GenerateDone);
    ASSERT_EQ(stream_->moveToNext(), StreamState::FINISHED);
    ASSERT_TRUE(await([&]() { return sender_->started > 0; }));
    sender_->release();
    ASSERT_TRUE(await([&]() { return drained(); }));

    const auto make_prefill = [&]() {
        auto input                             = std::make_shared<GenerateInput>();
        input->input_ids                       = torch::tensor(tokens_, torch::kInt32);
        input->generate_config                 = std::make_shared<GenerateConfig>();
        input->generate_config->aux_info       = true;
        input->generate_config->is_streaming   = true;
        input->generate_config->max_new_tokens = 2;
        ModelConfig model_config;
        model_config.max_seq_len = 64;
        model_config.vocab_size  = 128;
        ResourceContext resource_context;
        resource_context.cache_manager = prefill_;
        resource_context.role_type     = RoleType::PREFILL;
        resource_context.reuse_cache   = true;
        return std::make_shared<NormalGenerateStream>(input, model_config, RuntimeConfig{}, resource_context, nullptr);
    };
    const auto emit_chunk = [&](const std::shared_ptr<NormalGenerateStream>& request, int first_use_tokens) {
        EXPECT_EQ(request->initialReuseLength(), 12);
        EXPECT_EQ(request->writebackFirstUseLength(), first_use_tokens);
        request->update({.new_tokens = torch::tensor({{42}}, torch::kInt32), .num_new_tokens = 1});
        auto output = request->nextOutput(1000);
        ASSERT_TRUE(output.ok());
        ASSERT_EQ(output.value().generate_outputs.size(), 1u);
        const auto& aux = output.value().generate_outputs.front().aux_info;
        EXPECT_EQ(aux.reuse_len, 12 - first_use_tokens);
        EXPECT_EQ(aux.writeback_first_use_len, first_use_tokens);
    };

    auto cancelled = make_prefill();
    ASSERT_TRUE(cancelled->initKVBlock().ok());
    EXPECT_EQ(cancelled->writebackFirstUseLength(), 8);
    cancelled->reportError(ErrorCode::GENERATE_TIMEOUT, "prefill cancelled before cache commit");
    cancelled->generate_status_->status = StreamState::FINISHED;
    cancelled->releaseResource();

    auto first = make_prefill();
    auto concurrent = make_prefill();
    ASSERT_TRUE(first->initKVBlock().ok());
    ASSERT_TRUE(concurrent->initKVBlock().ok());
    emit_chunk(first, 8);
    emit_chunk(concurrent, 8);
    emit_chunk(first, 8);
    first->generate_status_->status = StreamState::FINISHED;
    first->releaseResource();

    auto subsequent = make_prefill();
    ASSERT_TRUE(subsequent->initKVBlock().ok());
    emit_chunk(subsequent, 0);
    emit_chunk(concurrent, 8);
    emit_chunk(subsequent, 0);
    concurrent->generate_status_->status = StreamState::FINISHED;
    concurrent->releaseResource();
    subsequent->generate_status_->status = StreamState::FINISHED;
    subsequent->releaseResource();
}

TEST_F(DecodeWritebackIntegrationTest, ReceiverPullsOnlyMissingSuffixFromCandidateSourceBlocks) {
    ASSERT_TRUE(test::seedCompleteBlockTreePath(target_, {keys_[0], keys_[1]}).success);
    prepare();
    stream_->releaseResource();
    ASSERT_TRUE(await([&] { return sender_->started > 0; }));
    auto write = context();
    ASSERT_NE(write, nullptr);
    EXPECT_TRUE(write->resourceHoldPending());
    sender_->release();
    ASSERT_TRUE(await([&] { return drained(); }));
    ASSERT_TRUE(write->success()) << write->errorInfo().ToString();
    EXPECT_EQ(sender_->offered_blocks, 2u);
    EXPECT_EQ(target_->devicePrefixBlocksForTest(keys_), 3u);
    auto resource = std::make_shared<BatchKVCacheResource>();
    resource->resetBatchSize(1);
    resource->initGroups(config_.topologyPtr());
    auto ids               = std::make_shared<CompleteTokenIds>(1, 1, 64, 4);
    auto input             = std::make_shared<GenerateInput>();
    input->input_ids       = torch::tensor(tokens_, torch::kInt32);
    input->generate_config = std::make_shared<GenerateConfig>();
    ids->init(input);
    MallocInfo info;
    info.batch_kv_cache_resource = resource;
    info.complete_token_ids      = ids;
    info.reuse_cache             = true;
    info.enable_cache_lookup     = true;
    const auto result            = prefill_->malloc(info);
    ASSERT_TRUE(result.success);
    ASSERT_EQ(result.reuse_len, 12);
    for (int layer = 0; layer < 2; ++layer) {
        for (const auto& block : target_->convertIndexToBuffer(
                 layer, config_.topology().groups().front().tag, resource->blocks(0, config_.topology().groups().front().tag)[2], 1, 0)) {
            std::vector<uint8_t> bytes(block.size_bytes);
            ASSERT_EQ(cudaMemcpy(bytes.data(), block.addr, bytes.size(), cudaMemcpyDeviceToHost), cudaSuccess);
            EXPECT_TRUE(std::all_of(bytes.begin(), bytes.end(), [&](uint8_t b) { return b == 10 * layer + 3; }));
        }
    }
    prefill_->free(FreeInfo{resource, ids});
}

TEST_F(DecodeWritebackIntegrationTest, LastSampleAtBlockBoundaryIsExcluded) {
    prepare(12, 11);
    stream_->releaseResource();
    ASSERT_TRUE(await([&]() { return sender_->started > 0; }));
    auto write = context();
    ASSERT_NE(write, nullptr);
    ASSERT_NE(writeSource(), nullptr);
    EXPECT_EQ(writeSource()->cacheKeys().size(), 2u);
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
    ASSERT_FALSE(decode_->writebackEnabled());
    prepare();
    stream_->releaseResource();
    EXPECT_TRUE(drained());
    EXPECT_EQ(prefill_service_.handshakes, 0);
    EXPECT_EQ(sourceRefs(1), 1u);
}

TEST_F(DecodeWritebackIntegrationTest, NonDecodeRolePreservesLocalReleaseWithoutWriteback) {
    auto prefill_role            = makeManager(RoleType::PREFILL, source_);
    prefill_role->p2p_connector_ = decode_->p2p_connector_;
    decode_                      = std::move(prefill_role);
    ASSERT_FALSE(decode_->writebackEnabled());
    prepare();
    stream_->releaseResource();
    EXPECT_TRUE(drained());
    EXPECT_EQ(prefill_service_.handshakes, 0);
    EXPECT_EQ(sourceRefs(1), 1u);
}

TEST_F(DecodeWritebackIntegrationTest, KvReserveSkipsWritebackAndPreservesLocalRelease) {
    auto reserved            = makeManager(RoleType::DECODE, source_, true, /*kv_reserve_ratio=*/1.0);
    reserved->p2p_connector_ = decode_->p2p_connector_;
    decode_                  = std::move(reserved);
    prepare();
    stream_->releaseResource();
    EXPECT_TRUE(drained());
    EXPECT_EQ(prefill_service_.handshakes, 0);
    EXPECT_EQ(sourceRefs(1), 1u);
}

TEST_F(DecodeWritebackIntegrationTest, KvReserveDoesNotCreditPromptHeldByAnotherRequest) {
    auto reserved            = makeManager(RoleType::DECODE, source_, true, /*kv_reserve_ratio=*/0.7);
    reserved->p2p_connector_ = decode_->p2p_connector_;
    decode_                  = std::move(reserved);
    tokens_.resize(29);
    std::iota(tokens_.begin(), tokens_.end(), 1);
    keys_ = calculateCacheKeys(tokens_.data(), 28, 4, config_.cache_key_hash_seed);
    ASSERT_TRUE(test::seedCompleteBlockTreePath(target_, CacheKeysType(keys_.begin(), keys_.begin() + 5)).success);
    prepare(29, 28, 20);
    ASSERT_EQ(source_blocks_.size(), 8u);
    const auto             pool = source_->cacheGroups().front()->blockPool();
    const BlockIndicesType prompt_blocks(source_blocks_.begin(), source_blocks_.begin() + 5);
    pool->incRef(prompt_blocks);
    const auto other_request =
        std::shared_ptr<void>(nullptr, [pool, prompt_blocks](void*) { pool->decRef(prompt_blocks); });
    ASSERT_EQ(pool->totalBlocksNum(), 15u);
    ASSERT_EQ(pool->availableBlocksNum(), 7u);
    EXPECT_EQ(sourceRefs(0), 2u);

    stream_->releaseResource();

    EXPECT_TRUE(drained());
    EXPECT_EQ(prefill_service_.handshakes, 0);
    EXPECT_EQ(stream_->streamCacheResource().curBlocksNum(), 0);
    EXPECT_EQ(pool->availableBlocksNum(), 10u);
    EXPECT_EQ(sourceRefs(0), 2u);
    EXPECT_EQ(sourceRefs(5), 1u);
}

TEST_F(DecodeWritebackIntegrationTest, KvReserveCreditsReleasedLongPromptBeforeWriteback) {
    auto reserved            = makeManager(RoleType::DECODE, source_, true, /*kv_reserve_ratio=*/0.7);
    reserved->p2p_connector_ = decode_->p2p_connector_;
    decode_                  = std::move(reserved);
    tokens_.resize(29);
    std::iota(tokens_.begin(), tokens_.end(), 1);
    keys_ = calculateCacheKeys(tokens_.data(), 28, 4, config_.cache_key_hash_seed);
    ASSERT_TRUE(test::seedCompleteBlockTreePath(target_, CacheKeysType(keys_.begin(), keys_.begin() + 5)).success);
    prepare(29, 28, 20);
    ASSERT_EQ(source_blocks_.size(), 8u);
    const auto pool = source_->cacheGroups().front()->blockPool();
    ASSERT_EQ(pool->totalBlocksNum(), 15u);
    ASSERT_EQ(pool->availableBlocksNum(), 7u);

    stream_->releaseResource();

    ASSERT_TRUE(await([&]() { return sender_->started > 0; }));
    const auto write = context();
    ASSERT_NE(write, nullptr);
    EXPECT_EQ(stream_->streamCacheResource().curBlocksNum(), 0);
    EXPECT_EQ(pool->availableBlocksNum(), 13u);
    EXPECT_EQ(sourceRefs(0), 1u);
    EXPECT_EQ(sourceRefs(5), 2u);
    sender_->release();
    ASSERT_TRUE(await([&]() { return drained(); }));
    EXPECT_TRUE(write->success()) << write->errorInfo().ToString();
    EXPECT_EQ(target_->devicePrefixBlocksForTest(keys_), 7u);
}

TEST_F(DecodeWritebackIndependentPoolsIntegrationTest, KvReserveChecksEachPoolBeforeHoldingSuffix) {
    auto reserved            = makeManager(RoleType::DECODE, source_, true, /*kv_reserve_ratio=*/0.875);
    reserved->p2p_connector_ = decode_->p2p_connector_;
    decode_                  = std::move(reserved);
    prepare();
    const auto& pools = source_->groupBlockPools();
    ASSERT_EQ(pools.size(), 2u);
    ASSERT_EQ(pools[0]->totalBlocksNum(), 7u);
    ASSERT_EQ(pools[1]->totalBlocksNum(), 31u);
    ASSERT_EQ(pools[0]->availableBlocksNum(), 3u);
    ASSERT_GT(source_->availableBlocksNum(), 29u);

    stream_->releaseResource();

    EXPECT_TRUE(drained());
    EXPECT_EQ(prefill_service_.handshakes, 0);
    EXPECT_EQ(stream_->streamCacheResource().curBlocksNum(), 0);
    EXPECT_EQ(pools[0]->availableBlocksNum(), pools[0]->totalBlocksNum());
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

TEST_F(DecodeWritebackIntegrationTest, UnfinishedStreamWithReuseDisabledDoesNotTriggerWriteback) {
    prepare(13, 12, 4, false);
    stream_->generate_status_->status = StreamState::RUNNING;
    stream_->releaseResource();
    EXPECT_TRUE(drained());
    EXPECT_EQ(prefill_service_.handshakes, 0);
    EXPECT_EQ(sourceRefs(1), 0u);
}

TEST_F(DecodeWritebackIntegrationTest, FailedFinishedStreamWithReuseDisabledDoesNotTriggerWriteback) {
    prepare(13, 12, 4, false);
    stream_->reportError(ErrorCode::EXECUTION_EXCEPTION, "test writeback exclusion");
    stream_->generate_status_->status = StreamState::FINISHED;
    stream_->releaseResource();
    EXPECT_TRUE(drained());
    EXPECT_EQ(prefill_service_.handshakes, 0);
    EXPECT_EQ(sourceRefs(1), 0u);
}

TEST_F(DecodeWritebackIntegrationTest, FakeStreamWithReuseDisabledDoesNotTriggerWriteback) {
    prepare(13, 12, 4, false);
    stream_->setIsFakeStream(true);
    stream_->releaseResource();
    EXPECT_TRUE(drained());
    EXPECT_EQ(prefill_service_.handshakes, 0);
    EXPECT_EQ(sourceRefs(1), 0u);
}

TEST_F(DecodeWritebackIntegrationTest, BeamStreamWithReuseDisabledDoesNotTriggerWriteback) {
    prepare(13, 12, 4, false);
    stream_->generateConfig()->num_beams = 2;
    ASSERT_TRUE(stream_->hasNumBeams());
    stream_->releaseResource();
    EXPECT_TRUE(drained());
    EXPECT_EQ(prefill_service_.handshakes, 0);
    EXPECT_EQ(sourceRefs(1), 0u);
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
