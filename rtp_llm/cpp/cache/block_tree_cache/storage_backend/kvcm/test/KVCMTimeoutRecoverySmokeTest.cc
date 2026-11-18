// Native SDK + file I/O + production queues/RPC/recovery, without loading a model.
// Only metadata discovery is a fixture. Payload results and completion proofs
// always come from the selected KVCM SDK, never from the test.
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/test/KVCMMockTestBase.h"

#include <cuda_runtime_api.h>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <unordered_set>

#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/FrozenTaskRecovery.h"
#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/kvcm/SdkBufferSafety.h"

namespace rtp_llm {
namespace {
using namespace std::chrono_literals;
using Code = kv_cache_manager::ClientErrorCode;

template<typename Predicate>
bool waitFor(Predicate predicate, std::chrono::milliseconds budget = 5s) {
    const auto deadline = std::chrono::steady_clock::now() + budget;
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline)
            return false;
        std::this_thread::sleep_for(2ms);
    }
    return true;
}

struct SdkObservation {
    std::atomic<size_t> started{0};
    std::atomic<size_t> finished{0};
    std::atomic<int> last_code{static_cast<int>(Code::ER_OK)};
    std::atomic<int64_t> last_duration_us{0};
    std::mutex mutex;
    std::unordered_set<const void*> entered_buffers;
    bool entered(const void* address) {
        std::lock_guard<std::mutex> lock(mutex);
        return entered_buffers.count(address) != 0;
    }
    void resetBlocks() {
        std::lock_guard<std::mutex> lock(mutex);
        entered_buffers.clear();
    }
};

class ObservedNativeClient final: public kv_cache_manager::TransferClient {
public:
    ObservedNativeClient(std::unique_ptr<TransferClient> native, std::shared_ptr<SdkObservation> observation):
        native_(std::move(native)), observation_(std::move(observation)) {}
    Code LoadKvCaches(const kv_cache_manager::UriStrVec& uris, const kv_cache_manager::BlockBuffers& buffers,
                      std::shared_ptr<kv_cache_manager::TransferTraceInfo> trace) override {
        {
            std::lock_guard<std::mutex> lock(observation_->mutex);
            for (const auto& buffer : buffers)
                for (const auto& iov : buffer.iovs)
                    observation_->entered_buffers.insert(iov.base);
        }
        const auto begin = std::chrono::steady_clock::now();
        ++observation_->started;
        const auto result = native_->LoadKvCaches(uris, buffers, std::move(trace));
        observation_->last_duration_us.store(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - begin).count());
        observation_->last_code.store(static_cast<int>(result));
        ++observation_->finished;
        return result;
    }
    std::pair<Code, kv_cache_manager::UriStrVec>
    SaveKvCaches(const kv_cache_manager::UriStrVec& uris, const kv_cache_manager::BlockBuffers& buffers,
                 std::shared_ptr<kv_cache_manager::TransferTraceInfo> trace) override {
        return native_->SaveKvCaches(uris, buffers, std::move(trace));
    }
private:
    Code Init(const std::string&, const kv_cache_manager::InitParams&) override { return Code::ER_INVALID_PARAMS; }
    std::unique_ptr<TransferClient> native_;
    std::shared_ptr<SdkObservation> observation_;
};

// Metadata returns a fixed existing file location; payload client construction
// and all file reads use the production factory and selected SDK shared object.
class NativeFileFactory final: public kvcm::ClientFactory {
public:
    NativeFileFactory(std::string storage, kv_cache_manager::Locations locations,
                       std::shared_ptr<SdkObservation> observation):
        storage_(std::move(storage)), locations_(std::move(locations)), observation_(std::move(observation)) {}
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
    std::unique_ptr<kv_cache_manager::TransferClient>
    createTransferClient(const std::string& config, const kv_cache_manager::InitParams& params) const override {
        auto native = kvcm::ClientFactory::createTransferClient(config, params);
        if (!native)
            return nullptr;
        return std::make_unique<ObservedNativeClient>(std::move(native), observation_);
    }
private:
    std::string storage_;
    kv_cache_manager::Locations locations_;
    std::shared_ptr<SdkObservation> observation_;
};

constexpr size_t kBlockBytes = 2 * 1024 * 1024;
constexpr size_t kBatchBlocks = 32;
constexpr size_t kPoolBlocks = 128;

BackendEnvironment nativeEnvironment() {
    BackendEnvironment result;
    result.cache_config = test::makeSimpleMhaCacheConfig(1, kPoolBlocks + 1, 8, DataType::TYPE_FP16, 1, 65536);
    initializeEnvironmentPools(result, "native_sdk_timeout_smoke", kPoolBlocks);
    return result;
}

class FilePayload {
public:
    FilePayload(): bytes(kBlockBytes) {
        const char* root = std::getenv("TEST_TMPDIR");
        path = std::string(root ? root : "/tmp") + "/kvcm-timeout-payload-XXXXXX";
        const int fd = mkstemp(path.data());
        if (fd < 0)
            throw std::runtime_error("mkstemp failed");
        for (size_t i = 0; i < bytes.size(); ++i)
            bytes[i] = static_cast<unsigned char>((i * 17 + 31) & 0xff);
        for (size_t block = 0; block < kBatchBlocks; ++block) {
            size_t written = 0;
            while (written < bytes.size()) {
                const auto count = ::write(fd, bytes.data() + written, bytes.size() - written);
                if (count <= 0) {
                    ::close(fd);
                    throw std::runtime_error("seed SDK file failed");
                }
                written += count;
            }
        }
        ::close(fd);
    }
    ~FilePayload() { unlink(path.c_str()); }
    std::string uri(size_t block) const {
        return "file://smoke" + path + "?size=" + std::to_string(bytes.size()) + "&blkid=" + std::to_string(block);
    }
    std::string storage() const {
        return R"([{"type":"file","global_unique_name":"smoke","storage_spec":{"root_path":"/tmp/","key_count_per_file":32}}])";
    }
    std::string path;
    std::vector<unsigned char> bytes;
};

struct NativeWorker {
    BackendEnvironment environment{nativeEnvironment()};
    std::shared_ptr<SdkObservation> observation{std::make_shared<SdkObservation>()};
    std::shared_ptr<kvcm::ClientWrapper> client;
    std::shared_ptr<KVCMStorageBackend> backend;
    std::shared_ptr<KVCacheManager> manager;
    ~NativeWorker() {
        if (manager)
            manager->stopFrozenTaskRecovery();
        if (backend)
            backend->shutdown();
    }

    bool init(const FilePayload& file, int rank = 0, int ranks = 1,
               std::shared_ptr<BroadcastManager> broadcast = nullptr, int sdk_timeout_ms = 30000) {
        kv_cache_manager::Locations locations;
        for (size_t block = 0; block < kBatchBlocks; ++block) {
            kv_cache_manager::Location location;
            for (int peer = 0; peer < ranks; ++peer)
                location.push_back({"tp" + std::to_string(peer) + "_Fdefault", file.uri(block)});
            locations.push_back(std::move(location));
        }
        client = std::make_shared<kvcm::ClientWrapper>(std::make_unique<NativeFileFactory>(
            file.storage(), std::move(locations), observation));
        KVCacheConfig config;
        config.kvcm_server_address = "metadata-fixture";
        config.kvcm_model_sdk_config = R"([{"type":"file"}])";
        config.kvcm_asyncwrapper_thread_num = 1;
        config.kvcm_asyncwrapper_queue_size = 256;
        config.kvcm_storage_thread_num = 1;
        config.kvcm_get_timeout_ms = sdk_timeout_ms;
        config.kvcm_put_timeout_ms = 30000;
        config.kvcm_task_timeout_ms = 30000;
        config.block_tree_frozen_recovery_interval_ms = 10;
        config.block_tree_frozen_recovery_query_timeout_ms = 1000;
        auto parallel = singleRankConfig();
        parallel.tp_size = ranks;
        parallel.tp_rank = rank;
        parallel.world_rank = rank;
        RuntimeConfig runtime;
        runtime.model_name = "native_timeout_smoke";
        backend = std::make_shared<KVCMStorageBackend>(environment.cache_config, config, runtime, parallel,
            SpeculativeExecutionConfig{}, broadcast, false, nullptr, client);
        if (!initSingleRank(*backend, environment))
            return false;
        manager = std::make_shared<KVCacheManager>(environment.cache_config, false, nullptr, config, parallel);
        auto full = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{environment.device_pool}, nullptr, nullptr);
        full->initialize(0, environment.cache_config.topologyPtr(), {"default"});
        manager->block_tree_cache_ = block_tree_cache_test::makeBlockTreeCacheForTest({full});
        manager->block_tree_cache_->storage_backend_ = backend;
        manager->transfer_broadcast_manager_ = std::move(broadcast);
        manager->completion_cleanup_ = std::make_shared<TransferCompletionCleanup>();
        backend->setCompletionCleanup(manager->completion_cleanup_);
        return true;
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
    std::future<ErrorInfo> readAsync(std::shared_ptr<StorageBackendMatchMeta> meta,
                                    std::chrono::milliseconds timeout, const std::vector<BlockIdxType>& blocks) {
        auto promise = std::make_shared<std::promise<ErrorInfo>>();
        auto future = promise->get_future();
        backend->read(request(timeout, blocks), std::move(meta),
                      [promise](ErrorInfo result) { promise->set_value(std::move(result)); });
        return future;
    }
    void expectPayload(const FilePayload& file, const std::vector<BlockIdxType>& blocks) const {
        std::vector<unsigned char> actual(file.bytes.size());
        for (const auto block : blocks) {
            const auto* data = blockBase(environment, "default", block);
            ASSERT_EQ(cudaMemcpy(actual.data(), data, actual.size(), cudaMemcpyDeviceToHost), cudaSuccess);
            EXPECT_EQ(actual, file.bytes) << "block=" << block;
        }
    }
};

class NativeWorkerServer final: public RpcService::Service {
public:
    explicit NativeWorkerServer(NativeWorker& worker): worker_(worker) {}
    ~NativeWorkerServer() {
        if (server_) {
            server_->Shutdown(std::chrono::system_clock::now() + 2s);
            server_->Wait();
        }
    }
    bool start() {
        grpc::ServerBuilder builder;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port_);
        builder.RegisterService(this);
        server_ = builder.BuildAndStart();
        return server_ && port_ != 0;
    }
    std::string address() const { return "127.0.0.1:" + std::to_string(port_); }
    grpc::Status ExecuteFunction(grpc::ServerContext*, const FunctionRequestPB* request,
                                 FunctionResponsePB* response) override {
        return worker_.manager->executeFunction(*request, *response) ? grpc::Status::OK :
            grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "unhandled function");
    }
private:
    NativeWorker& worker_;
    std::unique_ptr<grpc::Server> server_;
    int port_{0};
};

class KVCMTimeoutRecoverySmokeTest: public ::testing::Test {
protected:
    autil::EnvGuard publication_{"RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH", "0"};
    autil::EnvGuard async_{"RTP_LLM_STREAM_ASYNC", "0"};
};

template<typename SdkCode>
bool isSdkTimeout(SdkCode code) {
    if (code == SdkCode::ER_SDK_TIMEOUT)
        return true;
    if constexpr (kvcm::detail::HasExplicitBufferSafeCodes<SdkCode>::value)
        return code == SdkCode::ER_SDK_TIMEOUT_BUFFER_SAFE;
    return false;
}

TEST_F(KVCMTimeoutRecoverySmokeTest, ExpiredBeforeSubmissionAndLongDeadlineControl) {
    FilePayload file;
    NativeWorker worker;
    ASSERT_EQ(worker.environment.cache_config.blockSizeBytesForGroup("default"), kBlockBytes);
    ASSERT_TRUE(worker.init(file));
    ScopedReferencedBlocks targets(worker.environment.device_pool, kBatchBlocks);
    auto meta = worker.matched(targets.get());
    ASSERT_NE(meta, nullptr);
    const auto result = readResult(*worker.backend, worker.request(0ms, targets.get()), meta);
    EXPECT_EQ(result.code(), ErrorCode::DEADLINE_EXCEEDED);
    EXPECT_EQ(worker.observation->started.load(), 0u);
    EXPECT_TRUE(worker.manager->frozenTasks().empty());
    EXPECT_TRUE(readResult(*worker.backend, worker.request(30s, targets.get()), meta).ok());
    worker.expectPayload(file, targets.get());
    EXPECT_TRUE(worker.manager->frozenTasks().empty());
}

TEST_F(KVCMTimeoutRecoverySmokeTest, TaskDeadlineAfterNativeIoCompletesDoesNotFreezeSafeBuffers) {
    FilePayload file;
    NativeWorker worker;
    ASSERT_TRUE(worker.init(file));
    ScopedReferencedBlocks targets(worker.environment.device_pool, kBatchBlocks);
    auto meta = worker.matched(targets.get());
    ASSERT_NE(meta, nullptr);
    bool exercised = false;
    for (int budget : {1, 2, 5, 10, 20, 50}) {
        const auto before = worker.observation->started.load();
        const auto result = readResult(*worker.backend,
            worker.request(std::chrono::milliseconds(budget), targets.get()), meta);
        ASSERT_TRUE(waitForBackendOperationsForTest(*worker.backend));
        EXPECT_TRUE(worker.manager->frozenTasks().empty());
        if (result.code() == ErrorCode::DEADLINE_EXCEEDED
            && worker.observation->started.load() > before
            && worker.observation->last_code.load() == static_cast<int>(Code::ER_OK)) {
            RecordProperty("task_budget_after_sdk_io_ms", budget);
            RecordProperty("native_sdk_duration_us", std::to_string(worker.observation->last_duration_us.load()));
            exercised = true;
            break;
        }
    }
    EXPECT_TRUE(exercised) << "no task deadline expired after a successful native SDK call";
    for (const auto block : targets.get()) {
        EXPECT_FALSE(worker.environment.device_pool->isFrozen(block));
        EXPECT_EQ(worker.environment.device_pool->refCount(block), 1u);
    }
    ASSERT_TRUE(readResult(*worker.backend, worker.request(30s, targets.get()), meta).ok());
    worker.expectPayload(file, targets.get());
}

TEST_F(KVCMTimeoutRecoverySmokeTest, NaturalBackendQueuePressureExpiresBeforeIoAndRestoresReferences) {
    FilePayload file;
    NativeWorker worker;
    ASSERT_TRUE(worker.init(file));
    ScopedReferencedBlocks targets(worker.environment.device_pool, 96);
    const std::vector<BlockIdxType> one{targets.get().front()};
    auto meta = worker.matched(one);
    ASSERT_NE(meta, nullptr);
    size_t queue_peak = 0;
    size_t expired_before_io = 0;
    for (int budget : {1, 2, 5}) {
        std::vector<std::future<ErrorInfo>> futures;
        for (const auto block : targets.get()) {
            futures.push_back(worker.readAsync(meta, std::chrono::milliseconds(budget), {block}));
            queue_peak = std::max(queue_peak, worker.backend->executor_->queueSizes().load);
        }
        for (auto& future : futures) {
            ASSERT_EQ(future.wait_for(30s), std::future_status::ready);
            const auto result = future.get();
            if (result.code() == ErrorCode::DEADLINE_EXCEEDED
                && result.ToString().find("storage read deadline exceeded before I/O") != std::string::npos)
                ++expired_before_io;
        }
        ASSERT_TRUE(waitForBackendOperationsForTest(*worker.backend));
        if (expired_before_io && queue_peak)
            break;
    }
    RecordProperty("backend_queue_peak", static_cast<int>(queue_peak));
    RecordProperty("backend_queue_expired_before_io", static_cast<int>(expired_before_io));
    RecordProperty("sdk_calls", static_cast<int>(worker.observation->started.load()));
    EXPECT_GT(queue_peak, 0u) << "no backend queue pressure was observed";
    EXPECT_GT(expired_before_io, 0u) << "1/2/5ms budgets did not exercise backend queue expiry";
    EXPECT_TRUE(worker.manager->frozenTasks().empty());
    for (const auto block : targets.get()) {
        EXPECT_FALSE(worker.environment.device_pool->isFrozen(block));
        EXPECT_EQ(worker.environment.device_pool->refCount(block), 1u);
    }
    EXPECT_TRUE(readResult(*worker.backend, worker.request(30s, one), meta).ok());
    worker.expectPayload(file, one);
}

TEST_F(KVCMTimeoutRecoverySmokeTest, NaturalWorkerPayloadQueueExpiresWithoutSubmittingSdk) {
    FilePayload file;
    NativeWorker worker;
    ASSERT_TRUE(worker.init(file, 1, 2));
    ScopedReferencedBlocks targets(worker.environment.device_pool, 96);
    size_t queue_peak = 0;
    size_t safe_timeouts = 0;
    for (int budget : {1, 2, 5}) {
        worker.observation->resetBlocks();
        std::vector<std::future<std::pair<bool, RemoteOperationResponsePB>>> futures;
        for (const auto block : targets.get()) {
            RemoteOperationRequestPB request;
            request.set_op(REMOTE_OPERATION_READ);
            request.set_trace_id("worker_queue_" + std::to_string(budget) + "_" + std::to_string(block));
            request.set_timeout_ms(budget);
            request.add_group_tags("default");
            request.add_block_ids(block);
            request.add_uris(file.uri(0));
            futures.push_back(std::async(std::launch::async, [backend = worker.backend, request] {
                RemoteOperationResponsePB response;
                const bool result = backend->execute(request, response);
                return std::make_pair(result, response);
            }));
            queue_peak = std::max(queue_peak, worker.backend->queueSizes().load);
        }
        for (size_t i = 0; i < futures.size(); ++i) {
            ASSERT_EQ(futures[i].wait_for(30s), std::future_status::ready);
            const auto result = futures[i].get();
            if (!result.first && result.second.transfer_status() == REMOTE_TRANSFER_STATUS_TIMEOUT
                && result.second.payload_buffer_safe() && !result.second.sdk_failed()
                && !worker.observation->entered(blockBase(worker.environment, "default", targets.get()[i])))
                ++safe_timeouts;
        }
        RecordProperty("worker_queue_budget_ms", budget);
        if (safe_timeouts && queue_peak)
            break;
    }
    RecordProperty("worker_queue_peak", static_cast<int>(queue_peak));
    RecordProperty("worker_queue_expired_without_sdk", static_cast<int>(safe_timeouts));
    RecordProperty("sdk_calls", static_cast<int>(worker.observation->started.load()));
    EXPECT_GT(queue_peak, 0u);
    EXPECT_GT(safe_timeouts, 0u) << "1/2/5ms budgets did not exercise worker payload queue expiry";
    EXPECT_TRUE(worker.manager->frozenTasks().empty());
    for (const auto block : targets.get()) {
        EXPECT_FALSE(worker.environment.device_pool->isFrozen(block));
        EXPECT_EQ(worker.environment.device_pool->refCount(block), 1u);
    }
    RemoteOperationRequestPB followup;
    followup.set_op(REMOTE_OPERATION_READ);
    followup.set_trace_id("worker_queue_followup");
    followup.set_timeout_ms(30000);
    followup.add_group_tags("default");
    followup.add_block_ids(targets.get().front());
    followup.add_uris(file.uri(0));
    RemoteOperationResponsePB response;
    ASSERT_TRUE(worker.backend->execute(followup, response));
    EXPECT_TRUE(response.payload_buffer_safe());
    worker.expectPayload(file, {targets.get().front()});
}

TEST_F(KVCMTimeoutRecoverySmokeTest, NaturalRpcDeadlineDuringNativeSdkIoAutomaticallyUnfreezes) {
    FilePayload file;
    NativeWorker worker0, worker1;
    NativeWorkerServer server0(worker0), server1(worker1);
    ASSERT_TRUE(server0.start());
    ASSERT_TRUE(server1.start());
    auto broadcast = std::make_shared<BroadcastManager>(std::vector<std::string>{server0.address(), server1.address()});
    ASSERT_TRUE(broadcast->init());
    ASSERT_TRUE(worker0.init(file, 0, 2, broadcast));
    ASSERT_TRUE(worker1.init(file, 1, 2));
    ScopedReferencedBlocks targets0(worker0.environment.device_pool, kBatchBlocks);
    ScopedReferencedBlocks targets1(worker1.environment.device_pool, kBatchBlocks);
    ASSERT_EQ(targets0.get(), targets1.get());
    auto meta = worker0.matched(targets0.get());
    ASSERT_NE(meta, nullptr);
    ASSERT_TRUE(worker0.manager->startFrozenTaskRecovery());
    bool exercised_inflight_timeout = false;
    size_t frozen_count = 0;
    for (int budget : {1, 2, 5, 10, 20, 50}) {
        auto timed = worker0.readAsync(meta, std::chrono::milliseconds(budget), targets0.get());
        ASSERT_EQ(timed.wait_for(30s), std::future_status::ready);
        const auto result = timed.get();
        const bool sdk_inflight = worker0.observation->started.load() > worker0.observation->finished.load()
                                 || worker1.observation->started.load() > worker1.observation->finished.load();
        const auto records = worker0.manager->frozenTasks();
        RecordProperty("rpc_budget_ms", budget);
        RecordProperty("rpc_sdk_inflight", sdk_inflight ? 1 : 0);
        frozen_count += records.size();
        for (const auto& record : records) {
            EXPECT_EQ(record->execution().expected_rank_count, 2u);
            EXPECT_GT(record->snapshot().target_block_count, 0u);
            ASSERT_TRUE(waitFor([&] { return record->snapshot().state == FrozenTask::State::RELEASED; }, 30s));
            const auto proof = record->recoverySnapshot().last_query;
            ASSERT_NE(proof, nullptr);
            EXPECT_TRUE(proof->all_safe);
            EXPECT_EQ(proof->ranks.size(), 2u);
        }
        ASSERT_TRUE(waitFor([&] { return worker0.manager->frozenTasks().empty(); }, 30s));
        ASSERT_TRUE(waitFor([&] {
            return worker0.observation->started.load() == worker0.observation->finished.load()
                   && worker1.observation->started.load() == worker1.observation->finished.load();
        }, 30s));
        ASSERT_TRUE(waitForBackendOperationsForTest(*worker0.backend));
        if (result.code() == ErrorCode::DEADLINE_EXCEEDED && sdk_inflight && !records.empty()) {
            exercised_inflight_timeout = true;
            break;
        }
    }
    RecordProperty("rpc_frozen_and_released", static_cast<int>(frozen_count));
    RecordProperty("rank0_sdk_calls", static_cast<int>(worker0.observation->started.load()));
    RecordProperty("rank1_sdk_calls", static_cast<int>(worker1.observation->started.load()));
    RecordProperty("rank0_native_sdk_duration_us", std::to_string(worker0.observation->last_duration_us.load()));
    RecordProperty("rank1_native_sdk_duration_us", std::to_string(worker1.observation->last_duration_us.load()));
    EXPECT_TRUE(exercised_inflight_timeout)
        << "no real SDK-in-flight RPC timeout in 1/2/5/10/20/50ms sweep; frozen=" << frozen_count;
    for (const auto block : targets0.get()) {
        EXPECT_FALSE(worker0.environment.device_pool->isFrozen(block));
        EXPECT_EQ(worker0.environment.device_pool->refCount(block), 1u);
    }
    ASSERT_TRUE(readResult(*worker0.backend, worker0.request(30s, targets0.get()), meta).ok());
    worker0.expectPayload(file, targets0.get());
    worker1.expectPayload(file, targets1.get());
}

TEST_F(KVCMTimeoutRecoverySmokeTest, NativeSdkDeadlineHonorsItsReportedBufferSafety) {
    FilePayload file;
    bool exercised_sdk_timeout = false;
    for (int budget : {1, 2, 5}) {
        NativeWorker worker;
        ASSERT_TRUE(worker.init(file, 0, 1, nullptr, budget));
        ScopedReferencedBlocks targets(worker.environment.device_pool, kBatchBlocks);
        auto meta = worker.matched(targets.get());
        ASSERT_NE(meta, nullptr);
        ASSERT_TRUE(worker.manager->startFrozenTaskRecovery());
        const auto result = readResult(*worker.backend, worker.request(30s, targets.get()), meta);
        const auto code = static_cast<Code>(worker.observation->last_code.load());
        ASSERT_GT(worker.observation->started.load(), 0u);
        if (result.ok())
            continue;
        // A failure with no SDK timeout evidence does not satisfy this smoke.
        ASSERT_TRUE(isSdkTimeout(code)) << "unexpected native SDK error=" << static_cast<int>(code);
        exercised_sdk_timeout = true;
        RecordProperty("sdk_get_timeout_ms", budget);
        RecordProperty("sdk_error_code", static_cast<int>(code));
        RecordProperty("sdk_buffer_safe", kvcm::sdkCodeProvesBufferSafety(code) ? 1 : 0);
        const auto records = worker.manager->frozenTasks();
        if (kvcm::sdkCodeProvesBufferSafety(code)) {
            EXPECT_TRUE(records.empty());
            for (const auto block : targets.get())
                EXPECT_FALSE(worker.environment.device_pool->isFrozen(block));
        } else {
            ASSERT_EQ(records.size(), 1u);
            const auto record = records.front();
            ASSERT_TRUE(waitFor([&] { return record->recoverySnapshot().query_count >= 2; }));
            EXPECT_EQ(record->snapshot().state, FrozenTask::State::FROZEN);
            // SDK destruction drains real pending work, but cannot fabricate
            // a late completion notification in RTP's registry.
            worker.client->shutdown();
            const auto previous_queries = record->recoverySnapshot().query_count;
            ASSERT_TRUE(waitFor([&] { return record->recoverySnapshot().query_count >= previous_queries + 2; }));
            EXPECT_EQ(record->snapshot().state, FrozenTask::State::FROZEN);
            EXPECT_FALSE(record->recoverySnapshot().last_query->all_safe);
        }
        break;
    }
    EXPECT_TRUE(exercised_sdk_timeout) << "native 64MiB SDK read did not exceed 1/2/5ms deadlines";
}

}  // namespace
}  // namespace rtp_llm
