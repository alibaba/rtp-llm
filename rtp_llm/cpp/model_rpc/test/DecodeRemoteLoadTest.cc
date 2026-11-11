#include <gtest/gtest.h>

#include <chrono>
#include <functional>
#include <future>
#include <mutex>
#include <vector>

#include "autil/NetUtil.h"
#include "rtp_llm/cpp/cache/LinearKVCacheSpec.h"
#include "rtp_llm/cpp/cache/MLAKVCacheSpec.h"
#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"

namespace rtp_llm {
namespace {

class CacheOnlyEngine: public EngineBase {
public:
    explicit CacheOnlyEngine(const CacheConfig& config): EngineBase(EngineInitParams()) {
        // No model or GPU cache allocation is needed to test RPC dispatch.
        resource_context_.cache_manager = std::make_shared<KVCacheManager>(config, true);
    }

    std::shared_ptr<GenerateStream> enqueue(const std::shared_ptr<GenerateInput>&) override {
        return nullptr;
    }
    void         enqueue(std::shared_ptr<GenerateStream>&) override {}
    absl::Status stop() override {
        return absl::OkStatus();
    }
    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>&, preRunMode) override {
        return absl::UnimplementedError("unused in test");
    }
    KVCacheInfo getCacheStatusInfo(int64_t, bool) override {
        return {};
    }
};

class RecordingCacheStore: public NormalCacheStore {
public:
    struct Call {
        std::string ip;
        int64_t     timeout_ms;
        int         partition_count;
        int         partition_id;
    };

    std::shared_ptr<LoadContext> loadBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>&,
                                             const std::string&           ip,
                                             uint32_t                     port,
                                             uint32_t                     rdma_port,
                                             int64_t                      timeout_ms,
                                             LoadContext::CheckCancelFunc cancel,
                                             int                          partition_count,
                                             int                          partition_id) override {
        calls.push_back({ip, timeout_ms, partition_count, partition_id});
        if (on_load) {
            on_load();
        }
        auto context = std::make_shared<LoadContext>(shared_from_this(), true);
        // Complete at the CacheStore boundary without real transfers. The
        // production PB decoder, loadCache and wait/retry path still execute.
        context->load({}, ip, port, rdma_port, timeout_ms, cancel, partition_count, partition_id);
        context->error_info_ = error;
        return context;
    }

    std::vector<Call>     calls;
    ErrorInfo             error = ErrorInfo::OkStatus();
    std::function<void()> on_load;
};

class ContextService: public RpcService::Service {
public:
    grpc::Status GetWorkerStatus(grpc::ServerContext* context, const StatusVersionPB*, WorkerStatusPB*) override {
        std::function<void(grpc::ServerContext*)> current;
        {
            std::lock_guard<std::mutex> lock(mutex);
            current = operation;
        }
        current(context);
        return grpc::Status::OK;
    }

    std::mutex                                mutex;
    std::function<void(grpc::ServerContext*)> operation;
};

class DecodeRemoteLoadTest: public testing::Test {
protected:
    void SetUp() override {
        CacheConfig config{};
        config.dtype                                                     = DataType::TYPE_FP16;
        config.use_mla                                                   = true;
        auto mla                                                         = std::make_shared<MLAKVCacheSpec>();
        mla->dtype                                                       = DataType::TYPE_FP16;
        mla->local_head_num_kv                                           = 1;
        mla->kv_lora_rank                                                = 16;
        mla->rope_head_dim                                               = 16;
        config.cache_specs                                               = {mla, std::make_shared<LinearKVCacheSpec>()};
        server_.engine_                                                  = std::make_shared<CacheOnlyEngine>(config);
        server_.maga_init_params_.model_config_.num_layers               = 0;
        server_.maga_init_params_.parallelism_config.tp_size             = 1;
        server_.maga_init_params_.parallelism_config.dp_rank             = 3;
        server_.maga_init_params_.pd_sep_config.rdma_connect_retry_times = 3;
        server_.resource_.workers                                        = {"self"};
        // Any accidental fallback to loopback RPC fails instead of hiding a
        // regression behind a healthy test service.
        server_.resource_.grpc_workers = {"127.0.0.1:1"};

        store_ = std::make_shared<RecordingCacheStore>();
        CacheStoreInitParams params;
        params.listen_port   = autil::NetUtil::randomPort();
        params.rdma_mode     = false;
        params.enable_metric = false;
        ASSERT_TRUE(store_->init(params));
        server_.resource_.cache_store = store_;
        for (int rank = 0; rank < 8; ++rank) {
            peers_.push_back("127.0.0." + std::to_string(rank + 1) + ":10000:10001");
        }
        for (int group = 0; group < 7; ++group) {
            auto ids = std::make_shared<BlockIds>();
            ids->assign(BlockIndicesType{group * 10 + 1, -1, group * 10 + 3});
            groups_.push_back(ids);
        }

        grpc::ServerBuilder builder;
        int                 port = 0;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
        builder.RegisterService(&context_service_);
        context_server_ = builder.BuildAndStart();
        ASSERT_TRUE(context_server_);
        stub_ = RpcService::NewStub(
            grpc::CreateChannel("127.0.0.1:" + std::to_string(port), grpc::InsecureChannelCredentials()));
    }

    void TearDown() override {
        if (context_server_) {
            context_server_->Shutdown();
        }
    }

    ErrorInfo invoke(int cp_size = 1, bool cancel_before = false, bool cancel_during = false) {
        auto done      = std::make_shared<std::promise<ErrorInfo>>();
        auto result    = done->get_future();
        auto operation = [this, cp_size, cancel_before, cancel_during, done](grpc::ServerContext* context) {
            if (cancel_before) {
                context->TryCancel();
            }
            if (cancel_during) {
                store_->on_load = [context] { context->TryCancel(); };
            } else {
                store_->on_load = nullptr;
            }
            DecodeRpcContext             rpc{nullptr};
            kmonitor::MetricsReporterPtr reporter;
            auto                         meta = std::make_shared<RpcServerRuntimeMeta>();
            DecodeGenerateContext        generate(rpc, 5000, context, reporter, meta);
            generate.peer_addrs = peers_;
            DecodeRpcServer::LoadKVCacheContext load(
                5353, key_, peers_, keys_, groups_, 2, 5000, 1, 0, context, cp_size);
            auto error = server_.loadCacheAsyncForTp(generate, load);
            if (error.ok()) {
                EXPECT_GE(generate.stat_info.load_cache_min_rt_us, 0);
                EXPECT_EQ(generate.stat_info.load_cache_min_rt_us, generate.stat_info.load_cache_max_rt_us);
                EXPECT_EQ(generate.stat_info.load_cache_polling_cost_us, 0);
            }
            done->set_value(error);
        };
        {
            std::lock_guard<std::mutex> lock(context_service_.mutex);
            context_service_.operation = std::move(operation);
        }
        grpc::ClientContext caller;
        caller.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
        WorkerStatusPB response;
        auto           status = stub_->GetWorkerStatus(&caller, StatusVersionPB(), &response);
        EXPECT_TRUE(status.ok()
                    || ((cancel_before || cancel_during) && status.error_code() == grpc::StatusCode::CANCELLED))
            << status.error_message();
        if (result.wait_for(std::chrono::seconds(1)) != std::future_status::ready) {
            ADD_FAILURE() << "native dispatch did not complete";
            return ErrorInfo(ErrorCode::UNKNOWN_ERROR, "test RPC did not complete");
        }
        return result.get();
    }

    DecodeRpcServer                      server_;
    std::shared_ptr<RecordingCacheStore> store_;
    std::vector<std::string>             peers_;
    const std::string                    key_ = "remote-load-test";
    const std::vector<CacheKeyType>      keys_{101, 202, 303};
    GroupBlockIds                        groups_;
    ContextService                       context_service_;
    std::unique_ptr<grpc::Server>        context_server_;
    std::unique_ptr<RpcService::Stub>    stub_;
};

TEST_F(DecodeRemoteLoadTest, LoadsAsymmetricHybridWithoutLoopbackRpc) {
    EXPECT_TRUE(invoke().ok());
    ASSERT_EQ(store_->calls.size(), peers_.size());
    for (size_t i = 0; i < peers_.size(); ++i) {
        EXPECT_EQ(store_->calls[i].ip, "127.0.0." + std::to_string(i + 1));
        EXPECT_EQ(store_->calls[i].timeout_ms, 5000);
        EXPECT_EQ(store_->calls[i].partition_count, 1);
        EXPECT_EQ(store_->calls[i].partition_id, 0);
    }
    EXPECT_TRUE(server_.resource_.rpc_pool.connection_pool_.empty());
}

TEST_F(DecodeRemoteLoadTest, PreservesReplicatedMlaAndCpPeerSelection) {
    server_.engine_->resourceContext().cache_manager->config_.cache_specs.resize(1);
    EXPECT_TRUE(invoke().ok());
    ASSERT_EQ(store_->calls.size(), 1);
    EXPECT_EQ(store_->calls[0].ip, "127.0.0.1");
    store_->calls.clear();
    EXPECT_TRUE(invoke(8).ok());
    EXPECT_EQ(store_->calls.size(), 8);
}

TEST_F(DecodeRemoteLoadTest, LoadsNonMlaPeersLocally) {
    server_.engine_->resourceContext().cache_manager->config_.use_mla = false;
    EXPECT_TRUE(invoke().ok());
    EXPECT_EQ(store_->calls.size(), 8);
    EXPECT_TRUE(server_.resource_.rpc_pool.connection_pool_.empty());
}

TEST_F(DecodeRemoteLoadTest, PreservesGroupedBlockTablesAndRequestMetadata) {
    DecodeRpcServer::LoadKVCacheContext load(5353, key_, peers_, keys_, groups_, 2, 5000, 1, 0, nullptr, 8);
    for (bool mla : {false, true}) {
        const auto request = mla ? server_.constructRemoteLoadRequestForMla(load, 0, peers_) :
                                   server_.constructRemoteLoadRequest(load, 0, peers_);
        EXPECT_EQ(request.request_id(), 5353);
        EXPECT_EQ(request.request_key(), key_);
        EXPECT_EQ(request.dp_rank(), 3);
        EXPECT_EQ(request.timeout_ms(), 5000);
        EXPECT_EQ(request.prefill_cp_size(), 8);
        EXPECT_EQ(request.partition_count(), 1);
        EXPECT_EQ(request.partition_id(), 0);
        // The existing wire request leaves this field at its default.
        EXPECT_EQ(request.reuse_block_size(), 0);
        EXPECT_EQ(std::vector<std::string>(request.peer_addrs().begin(), request.peer_addrs().end()), peers_);
        EXPECT_EQ(std::vector<CacheKeyType>(request.cache_keys().begin(), request.cache_keys().end()), keys_);
        ASSERT_EQ(request.group_block_ids_size(), groups_.size());
        for (int group = 0; group < request.group_block_ids_size(); ++group) {
            const auto& values = request.group_block_ids(group).values();
            EXPECT_EQ(BlockIndicesType(values.begin(), values.end()), groups_[group]->blocks());
        }
    }
}

TEST_F(DecodeRemoteLoadTest, PreservesNativeLoaderFailures) {
    for (auto code :
         {ErrorCode::CACHE_STORE_LOAD_RDMA_WRITE_FAILED, ErrorCode::LOAD_CACHE_TIMEOUT, ErrorCode::CANCELLED}) {
        store_->calls.clear();
        store_->error    = ErrorInfo(code, "injected loader failure");
        const auto error = invoke();
        EXPECT_EQ(error.code(), code);
        EXPECT_EQ(error.ToString(), "injected loader failure");
        EXPECT_EQ(store_->calls.size(), 8);
    }
    EXPECT_TRUE(server_.resource_.rpc_pool.connection_pool_.empty());
}

TEST_F(DecodeRemoteLoadTest, DoesNotRestartExhaustedTp1ConnectionRetries) {
    peers_.resize(1);
    store_->error = ErrorInfo(ErrorCode::CACHE_STORE_LOAD_RDMA_CONNECT_FAILED, "connection retries exhausted");
    EXPECT_EQ(invoke().code(), ErrorCode::CACHE_STORE_LOAD_RDMA_CONNECT_FAILED);
    EXPECT_EQ(store_->calls.size(), 1);
}

TEST_F(DecodeRemoteLoadTest, ObservesParentCancellationBeforeAndAfterLocalLoad) {
    EXPECT_EQ(invoke(1, true).code(), ErrorCode::CANCELLED);
    EXPECT_TRUE(store_->calls.empty());
    EXPECT_EQ(invoke(1, false, true).code(), ErrorCode::CANCELLED);
    EXPECT_TRUE(server_.resource_.rpc_pool.connection_pool_.empty());
}

TEST_F(DecodeRemoteLoadTest, KeepsMultipleRanksOnTheRemotePath) {
    server_.resource_.workers.push_back("remote");
    server_.resource_.grpc_workers.push_back("127.0.0.1:1");
    EXPECT_EQ(invoke().code(), ErrorCode::LOAD_KV_CACHE_FAILED);
    EXPECT_TRUE(store_->calls.empty());
    EXPECT_FALSE(server_.resource_.rpc_pool.connection_pool_.empty());
}

}  // namespace
}  // namespace rtp_llm
