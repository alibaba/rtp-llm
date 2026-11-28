#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeReclaimer.h"
#include "rtp_llm/cpp/cache/block_tree_cache/test/CompletionRpcTestServer.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/KVCMMockTestBase.h"

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <thread>
#include <unistd.h>

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"

namespace rtp_llm {
namespace {
using namespace std::chrono_literals;
using Code = kv_cache_manager::ClientErrorCode;

template<typename Predicate>
bool waitForRecovery(Predicate completed) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!completed()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return true;
}

class NativeFileFactory final: public kvcm::ClientFactory {
public:
    NativeFileFactory(std::string storage, kv_cache_manager::Locations locations):
        storage_(std::move(storage)), locations_(std::move(locations)) {}
    std::unique_ptr<kv_cache_manager::MetaClient>
    createMetaClient(const std::string&, const kv_cache_manager::InitParams&) const override {
        auto client = std::make_unique<::testing::NiceMock<kv_cache_manager::MockMetaClient>>();
        ON_CALL(*client, GetStorageConfig()).WillByDefault(::testing::ReturnRef(storage_));
        ON_CALL(*client, MatchLocation(_, _, _, _, _, _, _))
            .WillByDefault(Invoke([this](const auto&, auto, const auto& keys, const auto&, const auto&, auto, const auto&) {
                auto locations = locations_;
                locations.resize(keys.size());
                return std::make_pair(Code::ER_OK, std::move(locations));
            }));
        return client;
    }
private:
    std::string storage_;
    kv_cache_manager::Locations locations_;
};

constexpr size_t kBlockBytes = 2 * 1024 * 1024;
constexpr size_t kPoolBlocks = 8;

BackendEnvironment nativeEnvironment() {
    BackendEnvironment result;
    result.cache_config = test::makeSimpleMhaCacheConfig(1, kPoolBlocks + 1, 8, DataType::TYPE_FP16, 1, 65536);
    initializeEnvironmentPools(result, "native_sdk_timeout_smoke", kPoolBlocks);
    return result;
}

class FilePayload {
public:
    FilePayload() {
        const char* root = std::getenv("TEST_TMPDIR");
        path = std::string(root ? root : "/tmp") + "/kvcm-timeout-payload-XXXXXX";
        const int fd = mkstemp(path.data());
        if (fd < 0)
            throw std::runtime_error("mkstemp failed");
        if (::ftruncate(fd, kBlockBytes) != 0) {
            ::close(fd);
            ::unlink(path.c_str());
            throw std::runtime_error("create SDK file payload failed");
        }
        ::close(fd);
    }
    ~FilePayload() { unlink(path.c_str()); }
    std::string uri() const {
        return "file://smoke" + path + "?size=" + std::to_string(kBlockBytes) + "&blkid=0";
    }
    std::string storage() const {
        return R"([{"type":"file","global_unique_name":"smoke","storage_spec":{"root_path":"/tmp/","key_count_per_file":32}}])";
    }
    std::string path;
};

struct NativeWorker {
    BackendEnvironment environment{nativeEnvironment()};
    std::shared_ptr<kvcm::ClientWrapper> client;
    std::shared_ptr<KVCMStorageBackend> backend;
    std::shared_ptr<KVCacheManager> manager;

    bool init(const FilePayload& file, int rank, int ranks, std::shared_ptr<BroadcastManager> broadcast = nullptr) {
        kv_cache_manager::Locations locations(1);
        for (int peer = 0; peer < ranks; ++peer)
            locations.front().push_back({"tp" + std::to_string(peer) + "_Fdefault", file.uri()});
        client = std::make_shared<kvcm::ClientWrapper>(std::make_unique<NativeFileFactory>(
            file.storage(), std::move(locations)));
        KVCacheConfig config;
        config.kvcm_server_address = "metadata-fixture";
        config.kvcm_model_sdk_config = R"([{"type":"file"}])";
        config.kvcm_asyncwrapper_thread_num = 1;
        config.kvcm_asyncwrapper_queue_size = 256;
        config.kvcm_storage_thread_num = 1;
        config.kvcm_get_timeout_ms = 30000;
        config.kvcm_put_timeout_ms = 30000;
        config.kvcm_task_timeout_ms = 30000;
        config.block_tree_frozen_force_release_timeout_ms = 0;
        auto parallel = singleRankConfig();
        parallel.tp_size = ranks;
        parallel.tp_rank = rank;
        parallel.world_rank = rank;
        RuntimeConfig runtime;
        runtime.model_name = "native_timeout_smoke";
        backend = std::make_shared<KVCMStorageBackend>(environment.cache_config, config, runtime, parallel,
            SpeculativeExecutionConfig{}, broadcast, false, nullptr, client);
        manager = std::make_shared<KVCacheManager>(environment.cache_config, false, nullptr, config, parallel);
        auto full = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{environment.device_pool}, nullptr, nullptr);
        full->initialize(0, std::make_shared<CacheConfig>(environment.cache_config), {"default"});
        BlockTreeCacheConfig cache_config;
        cache_config.enable_remote_cache = true;
        block_tree_cache_test::RecoveryTestOptions recovery;
        recovery.rank = rank;
        recovery.interval_ms = 10;
        recovery.force_release_timeout_ms = config.block_tree_frozen_force_release_timeout_ms;
        manager->block_tree_cache_ = block_tree_cache_test::makeBlockTreeCacheForTest(
            {full}, cache_config, backend, std::move(broadcast), nullptr, recovery);
        return manager->block_tree_cache_ != nullptr;
    }

    StorageRequest request(std::chrono::milliseconds timeout, const std::vector<BlockIdxType>& blocks) const {
        CacheKeysType keys(blocks.size());
        std::iota(keys.begin(), keys.end(), 101);
        auto value = makeStorageRequest(environment, std::move(keys), 0, blocks);
        value.transfer_task.emplace(std::vector<TransferDescriptor>{}, timeout);
        return value;
    }
    std::shared_ptr<StorageBackendMatchMeta> matched(const std::vector<BlockIdxType>& blocks) {
        auto result = match(*backend, request(30s, blocks));
        EXPECT_TRUE(result.success);
        EXPECT_EQ(result.matched_blocks_num, blocks.size());
        return result.match_meta;
    }
};

class KVCMTimeoutRecoverySmokeTest: public ::testing::Test {
protected:
    autil::EnvGuard publication_{"RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0"};
    autil::EnvGuard async_{"RTP_LLM_STREAM_ASYNC", "0"};
};

TEST_F(KVCMTimeoutRecoverySmokeTest, ReadRejectsTraceAlreadyRegisteredByBlockTransfer) {
    FilePayload file;
    NativeWorker worker;
    ASSERT_TRUE(worker.init(file, 0, 1));
    auto& registry = *worker.manager->block_tree_cache_->transfer_dispatcher_->reclaimer_;
    ASSERT_TRUE(registry.registerOnWorkerTaskWithTraceID("71"));

    RemoteOperationRequestPB read;
    read.set_op(REMOTE_OPERATION_READ);
    read.set_trace_id("71");
    RemoteOperationResponsePB response;
    EXPECT_FALSE(worker.backend->execute(read, response));
    EXPECT_EQ(response.transfer_status(), REMOTE_TRANSFER_STATUS_FAILED);
    EXPECT_EQ(registry.localState("71"), BlockTreeReclaimer::State::RUNNING);

    read.set_trace_id("72");
    ASSERT_TRUE(worker.backend->execute(read, response));
    EXPECT_EQ(registry.localState("72"), BlockTreeReclaimer::State::BUFFER_SAFE);
}

TEST_F(KVCMTimeoutRecoverySmokeTest, LostNativeReadReplyRecoversThroughWorkerQueryAndForget) {
    FilePayload file;
    NativeWorker owner, peer;
    std::atomic<size_t> queries_to_safe_worker{0};
    std::atomic<bool> peer_query_received{false};
    std::atomic<bool> allow_peer_proof{false};
    std::promise<std::string> lost_read_trace;
    auto trace_future = lost_read_trace.get_future();
    auto handle = [&](size_t rank, const FunctionRequestPB& request, FunctionResponsePB& response) {
        if (request.has_completion_query()) {
            if (rank == 0) {
                ++queries_to_safe_worker;
            } else {
                peer_query_received = true;
                if (!allow_peer_proof)
                    return grpc::Status(grpc::StatusCode::UNAVAILABLE, "completion proof withheld");
            }
        }
        auto& worker = rank == 0 ? owner : peer;
        if (!worker.manager->executeFunction(request, response))
            return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "unhandled request");
        if (rank == 1 && request.has_remote_request() && request.remote_request().op() == REMOTE_OPERATION_READ
            && response.remote_response().transfer_status() == REMOTE_TRANSFER_STATUS_SUCCESS) {
            lost_read_trace.set_value(request.remote_request().trace_id());
            return grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "lost read reply");
        }
        return grpc::Status::OK;
    };
    block_tree_cache_test::CompletionRpcTestServer server0(0, handle), server1(1, handle);
    ASSERT_TRUE(server0.start());
    ASSERT_TRUE(server1.start());
    auto broadcast = std::make_shared<BroadcastManager>(std::vector<std::string>{server0.address(), server1.address()});
    ASSERT_TRUE(broadcast->init());
    ASSERT_TRUE(owner.init(file, 0, 2, broadcast));
    ASSERT_TRUE(peer.init(file, 1, 2));
    ScopedReferencedBlocks target(owner.environment.device_pool, 1);
    ScopedReferencedBlocks replica(peer.environment.device_pool, 1);
    ASSERT_EQ(target.get(), replica.get());
    auto meta = owner.matched(target.get());
    ASSERT_NE(meta, nullptr);
    ASSERT_EQ(readResult(*owner.backend, owner.request(10s, target.get()), meta).code(), ErrorCode::DEADLINE_EXCEEDED);
    ASSERT_EQ(trace_future.wait_for(0s), std::future_status::ready);
    const auto trace = trace_future.get();
    ASSERT_FALSE(trace.empty());
    ASSERT_TRUE(waitForRecovery([&] { return peer_query_received.load(); }));
    ASSERT_TRUE(owner.environment.device_pool->isFrozen(target.get().front()));
    auto& reclaimer = owner.manager->block_tree_cache_->reclaimer();
    auto& peer_reclaimer = peer.manager->block_tree_cache_->reclaimer();
    EXPECT_EQ(reclaimer.localState(trace), BlockTreeReclaimer::State::BUFFER_SAFE);
    EXPECT_EQ(peer_reclaimer.localState(trace), BlockTreeReclaimer::State::BUFFER_SAFE);
    allow_peer_proof = true;
    ASSERT_TRUE(waitForRecovery([&] {
        return !owner.environment.device_pool->isFrozen(target.get().front())
               && reclaimer.localState(trace) == BlockTreeReclaimer::State::UNKNOWN
               && peer_reclaimer.localState(trace) == BlockTreeReclaimer::State::UNKNOWN;
    }));
    EXPECT_EQ(queries_to_safe_worker.load(), 0u);
}

}
}
