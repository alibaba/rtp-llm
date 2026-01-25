// Copyright (c) RTP-LLM

#include <csignal>
#include <chrono>
#include <execinfo.h>
#include <thread>
#include <unistd.h>

#include <grpcpp/alarm.h>

#include "gtest/gtest.h"
#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/connector/memory/KVCacheMemoryConnector.h"
#include "rtp_llm/cpp/cache/connector/memory/MemoryBlockCache.h"
#include "rtp_llm/cpp/cache/connector/memory/test/mock/TestRpcService.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/SingleTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/model_rpc/TpBroadcastManager.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/EplbConfig.h"
#include "rtp_llm/cpp/config/ModelConfig.h"

namespace rtp_llm::test {

namespace {

// When bazel runs tests, a SIGSEGV in the test binary often ends up as a core dump of the bazel bash wrapper.
// Install a lightweight handler to print a backtrace to stderr so the real crash site is visible in test.log.
void crashBacktraceHandler(int sig) {
    void*  stack[128];
    size_t n = static_cast<size_t>(::backtrace(stack, 128));
    {
        constexpr char kHeader[] = "\n=== RTP-LLM unit test crash backtrace ===\n";
        const auto     rc        = ::write(STDERR_FILENO, kHeader, sizeof(kHeader) - 1);
        (void)rc;
    }
    ::backtrace_symbols_fd(stack, static_cast<int>(n), STDERR_FILENO);
    {
        constexpr char kFooter[] = "\n=== end backtrace ===\n";
        const auto     rc        = ::write(STDERR_FILENO, kFooter, sizeof(kFooter) - 1);
        (void)rc;
    }
    ::_exit(128 + sig);
}

struct CrashHandlerInstaller {
    CrashHandlerInstaller() {
        std::signal(SIGSEGV, crashBacktraceHandler);
        std::signal(SIGABRT, crashBacktraceHandler);
    }
};

static CrashHandlerInstaller g_crash_handler_installer;

}  // namespace

// --------------------------------- MemoryConnectorAsyncContextTest ---------------------------------

class MemoryConnectorAsyncContextTest: public ::testing::Test {
protected:
    // NOTE: This test file needs a "completed" TPBroadcastResult without running real RPCs.
    // We achieve this by scheduling a grpc::Alarm event onto each worker's CompletionQueue,
    // then calling TPBroadcastResult::waitDone() once to finalize its internal success flag.
    using MemoryBroadcastResultT = rtp_llm::TPBroadcastResult<FunctionRequestPB, FunctionResponsePB>;
    using MemoryWorkerCtxT       = typename MemoryBroadcastResultT::WorkerRpcContext;

    static std::shared_ptr<MemoryBroadcastResultT>
    makeCompletedBroadcastResult(const std::vector<std::shared_ptr<MemoryWorkerCtxT>>& workers) {
        // TPBroadcastResult::waitDone() may call TryCancel() on all contexts when any status is not OK.
        for (const auto& w : workers) {
            if (w && !w->client_context) {
                w->client_context = std::make_shared<grpc::ClientContext>();
            }
        }

        auto result = std::make_shared<MemoryBroadcastResultT>(workers);

        // Post one event per worker so TPBroadcastResult::waitDone() can finish immediately.
        std::vector<std::unique_ptr<grpc::Alarm>> alarms;
        alarms.reserve(workers.size());
        for (size_t i = 0; i < workers.size(); ++i) {
            alarms.emplace_back(std::make_unique<grpc::Alarm>());
            alarms.back()->Set(&workers[i]->completion_queue,
                               std::chrono::system_clock::now(),
                               reinterpret_cast<void*>(static_cast<intptr_t>(i)));
        }

        result->waitDone();
        return result;
    }
};

TEST_F(MemoryConnectorAsyncContextTest, success_ReturnFalse_WhenBroadcastResultNotSuccess) {
    auto worker0            = std::make_shared<MemoryWorkerCtxT>();
    worker0->client_context = std::make_shared<grpc::ClientContext>();
    worker0->status         = grpc::Status(grpc::StatusCode::CANCELLED, "cancelled");
    worker0->response.mutable_mem_response()->set_success(true);

    auto result = makeCompletedBroadcastResult({worker0});

    auto ctx = std::make_shared<rtp_llm::MemoryConnectorAsyncContext>(/*done_callback=*/nullptr);
    ctx->setBroadcastResult(result);
    EXPECT_FALSE(ctx->success());
}

TEST_F(MemoryConnectorAsyncContextTest, success_ReturnFalse_WhenAnyResponseMissingMemResponse) {
    auto worker0            = std::make_shared<MemoryWorkerCtxT>();  // default: no mem_response
    worker0->client_context = std::make_shared<grpc::ClientContext>();
    worker0->status         = grpc::Status::OK;
    auto result             = makeCompletedBroadcastResult({worker0});

    auto ctx = std::make_shared<rtp_llm::MemoryConnectorAsyncContext>(/*done_callback=*/nullptr);
    ctx->setBroadcastResult(result);
    EXPECT_FALSE(ctx->success());
}

TEST_F(MemoryConnectorAsyncContextTest, success_ReturnFalse_WhenAnyMemResponseFailed) {
    auto worker0            = std::make_shared<MemoryWorkerCtxT>();
    worker0->client_context = std::make_shared<grpc::ClientContext>();
    worker0->status         = grpc::Status::OK;
    worker0->response.mutable_mem_response()->set_success(true);
    auto worker1            = std::make_shared<MemoryWorkerCtxT>();
    worker1->client_context = std::make_shared<grpc::ClientContext>();
    worker1->status         = grpc::Status::OK;
    worker1->response.mutable_mem_response()->set_success(false);

    auto result = makeCompletedBroadcastResult({worker0, worker1});

    auto ctx = std::make_shared<rtp_llm::MemoryConnectorAsyncContext>(/*done_callback=*/nullptr);
    ctx->setBroadcastResult(result);
    EXPECT_FALSE(ctx->success());
}

TEST_F(MemoryConnectorAsyncContextTest, success_ReturnTrue_WhenAllResponsesSuccess) {
    auto worker0            = std::make_shared<MemoryWorkerCtxT>();
    worker0->client_context = std::make_shared<grpc::ClientContext>();
    worker0->status         = grpc::Status::OK;
    worker0->response.mutable_mem_response()->set_success(true);
    auto worker1            = std::make_shared<MemoryWorkerCtxT>();
    worker1->client_context = std::make_shared<grpc::ClientContext>();
    worker1->status         = grpc::Status::OK;
    worker1->response.mutable_mem_response()->set_success(true);

    auto result = makeCompletedBroadcastResult({worker0, worker1});

    auto ctx = std::make_shared<rtp_llm::MemoryConnectorAsyncContext>(/*done_callback=*/nullptr);
    ctx->setBroadcastResult(result);
    EXPECT_TRUE(ctx->success());
}

TEST_F(MemoryConnectorAsyncContextTest, waitDone_ReturnVoid_WhenBroadcastResultNullAndCallbackCalledOnce) {
    int  callback_cnt = 0;
    bool last_ok      = true;
    auto cb           = [&](bool ok) {
        callback_cnt++;
        last_ok = ok;
    };

    auto ctx = std::make_shared<rtp_llm::MemoryConnectorAsyncContext>(/*done_callback=*/cb);
    ctx->setBroadcastResult(nullptr);
    EXPECT_FALSE(ctx->done());
    EXPECT_FALSE(ctx->success());

    ctx->waitDone();
    EXPECT_TRUE(ctx->done());
    EXPECT_EQ(callback_cnt, 1);
    EXPECT_FALSE(last_ok);

    // Second call should be no-op.
    ctx->waitDone();
    EXPECT_EQ(callback_cnt, 1);
    EXPECT_TRUE(ctx->done());
}

TEST_F(MemoryConnectorAsyncContextTest, waitDone_BlocksUntilBroadcastResultReady_ThenCallbackOnce) {
    std::atomic<int>  callback_cnt{0};
    std::atomic<bool> last_ok{true};
    auto              cb = [&](bool ok) {
        callback_cnt.fetch_add(1);
        last_ok.store(ok);
    };

    auto ctx = std::make_shared<rtp_llm::MemoryConnectorAsyncContext>(cb);
    EXPECT_FALSE(ctx->done());

    std::thread t([&]() { ctx->waitDone(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    EXPECT_FALSE(ctx->done());
    EXPECT_EQ(callback_cnt.load(), 0);

    ctx->setBroadcastResult(nullptr);
    t.join();
    EXPECT_TRUE(ctx->done());
    EXPECT_EQ(callback_cnt.load(), 1);
    EXPECT_FALSE(last_ok.load());
}

TEST_F(MemoryConnectorAsyncContextTest, waitDone_ReturnVoid_WhenBroadcastResultNonNullAndCallbackReceivesSuccess) {
    // Empty worker contexts => TPBroadcastResult::waitDone() returns immediately and sets all_request_success_ = true.
    auto result = std::make_shared<MemoryBroadcastResultT>(std::vector<std::shared_ptr<MemoryWorkerCtxT>>{});

    int  callback_cnt = 0;
    bool last_ok      = false;
    auto cb           = [&](bool ok) {
        callback_cnt++;
        last_ok = ok;
    };

    auto ctx = std::make_shared<rtp_llm::MemoryConnectorAsyncContext>(cb);
    ctx->setBroadcastResult(result);
    EXPECT_FALSE(ctx->done());
    EXPECT_FALSE(ctx->success());  // default all_request_success_ is false before waitDone().

    ctx->waitDone();
    EXPECT_TRUE(ctx->done());
    EXPECT_TRUE(ctx->success());
    EXPECT_EQ(callback_cnt, 1);
    EXPECT_TRUE(last_ok);

    // Second call should be no-op.
    ctx->waitDone();
    EXPECT_EQ(callback_cnt, 1);
    EXPECT_TRUE(ctx->done());
}

TEST_F(MemoryConnectorAsyncContextTest, waitDone_ReturnVoid_WhenBroadcastResultNonNullAndDoneCallbackNull) {
    auto result = std::make_shared<MemoryBroadcastResultT>(std::vector<std::shared_ptr<MemoryWorkerCtxT>>{});
    auto ctx    = std::make_shared<rtp_llm::MemoryConnectorAsyncContext>(/*done_callback=*/nullptr);
    ctx->setBroadcastResult(result);

    EXPECT_FALSE(ctx->done());
    EXPECT_FALSE(ctx->success());

    ctx->waitDone();
    EXPECT_TRUE(ctx->done());
    EXPECT_TRUE(ctx->success());
}

TEST_F(MemoryConnectorAsyncContextTest, waitDone_IsIdempotent_CallbackOnlyOnce) {
    int  callback_cnt = 0;
    bool last_ok      = false;
    auto cb           = [&](bool ok) {
        callback_cnt++;
        last_ok = ok;
    };

    // Use empty worker contexts: TPBroadcastResult::waitDone() completes immediately and marks success.
    auto result = std::make_shared<MemoryBroadcastResultT>(std::vector<std::shared_ptr<MemoryWorkerCtxT>>{});

    auto ctx = std::make_shared<rtp_llm::MemoryConnectorAsyncContext>(cb);
    ctx->setBroadcastResult(result);
    ctx->waitDone();
    ctx->waitDone();

    EXPECT_TRUE(ctx->done());
    EXPECT_TRUE(ctx->success());
    EXPECT_EQ(callback_cnt, 1);
    EXPECT_TRUE(last_ok);
}

// --------------------------------- KVCacheMemoryConnectorTest ---------------------------------

class TestReadMeta: public rtp_llm::KVCacheConnector::Meta {
public:
    TestReadMeta(int start_block_index, int size): start_block_index_(start_block_index), size_(size) {}
    ~TestReadMeta() override = default;

public:
    std::pair<int, int> blockRange() const override {
        return {start_block_index_, size_};
    }

private:
    int start_block_index_{0};
    int size_{0};
};

class KVCacheMemoryConnectorTest: public ::testing::Test {
protected:
    void SetUp() override {
        device_ = createDevice();

        cache_config_ = createMockCacheConfig();
        allocator_    = std::make_shared<SingleTypeKVCacheAllocator>(cache_config_, device_, AllocationType::DEVICE);
        ASSERT_TRUE(allocator_->init());

        const int server_num = 4;
        startRpcServer(server_num);

        connector_ = std::make_shared<KVCacheMemoryConnector>(
            cache_config_, kv_cache_config_, allocator_, device_, server_addrs_);
        ASSERT_TRUE(connector_->init());
    }

    DeviceBase*                                 device_{nullptr};
    CacheConfig                                 cache_config_;
    KVCacheConfig                               kv_cache_config_;
    std::shared_ptr<KVCacheAllocator>           allocator_;
    std::shared_ptr<KVCacheMemoryConnector>     connector_;
    std::vector<std::unique_ptr<TestRpcServer>> servers_;
    std::vector<std::string>                    server_addrs_;

private:
    DeviceBase* createDevice() const {
        ParallelismConfig           parallelism_config;
        ModelConfig                 model_config;
        EPLBConfig                  eplb_config;
        FMHAConfig                  fmha_config;
        DeviceResourceConfig        device_resource_config;
        MoeConfig                   moe_config;
        SpeculativeExecutionConfig  sp_config;
        MiscellaneousConfig         misc_config;
        ProfilingDebugLoggingConfig profiling_debug_logging_config;
        HWKernelConfig              hw_kernel_config;
        ConcurrencyConfig           concurrency_config;
        FfnDisAggregateConfig       ffn_disaggregate_config;
        RuntimeConfig               runtime_config;

        // Keep tests stable on shared GPUs with low free memory:
        // - device_reserve_memory_bytes=0 => use DeviceFactory default (-512MB), i.e. reserve (free - 512MB)
        // - host_reserve_memory_bytes=0  => don't reserve pinned host memory
        device_resource_config.device_reserve_memory_bytes = 0;
        device_resource_config.host_reserve_memory_bytes   = 0;

        DeviceFactory::initDevices(parallelism_config,
                                   model_config,
                                   eplb_config,
                                   fmha_config,
                                   device_resource_config,
                                   moe_config,
                                   sp_config,
                                   misc_config,
                                   profiling_debug_logging_config,
                                   hw_kernel_config,
                                   concurrency_config,
                                   ffn_disaggregate_config,
                                   runtime_config);
        return DeviceFactory::getDefaultDevice();
    }
    CacheConfig createMockCacheConfig(int layer_num = 4, int block_num = 10, int seq_size_per_block = 8) {
        constexpr int kTestMemoryCacheSizeMb      = 64;
        constexpr int kTestMemoryCacheSyncTimeout = 1000;

        CacheConfig config;
        config.layer_num                              = layer_num;
        config.layer_all_num                          = layer_num;
        config.block_num                              = block_num;
        config.seq_size_per_block                     = seq_size_per_block;
        kv_cache_config_.memory_cache_size_mb         = kTestMemoryCacheSizeMb;
        kv_cache_config_.memory_cache_sync_timeout_ms = kTestMemoryCacheSyncTimeout;

        auto mha_spec       = std::make_shared<MHAKVCacheSpec>();
        mha_spec->layer_num = layer_num;
        // mha_spec->block_nums         = block_num;
        mha_spec->local_head_num_kv  = 8;
        mha_spec->size_per_head      = 128;
        mha_spec->seq_size_per_block = seq_size_per_block;
        mha_spec->dtype              = rtp_llm::DataType::TYPE_FP16;
        mha_spec->type               = KVCacheType::MultiHeadAttention;
        config.cache_specs.push_back(mha_spec);
        // Keep CacheConfig sizes consistent with the business definition:
        // block_size_bytes = "one cacheKey across all layers" total bytes (kv + scale).
        config.dtype              = mha_spec->dtype;
        config.block_stride_bytes = mha_spec->block_size_bytes();  // one-layer bytes for one logical block
        config.block_size_bytes   = static_cast<size_t>(layer_num) * config.block_stride_bytes;

        std::vector<int> layer_ids(layer_num);
        for (int i = 0; i < layer_num; ++i) {
            layer_ids[i] = i;
        }
        config.layer_ids.push_back(layer_ids);
        // SingleTypeKVCacheAllocator::init() expects global_layer_ids[0] to exist.
        // In these unit tests we only have one "model group", so keep it consistent with layer_ids.
        config.global_layer_ids.push_back(layer_ids);

        return config;
    }
    void startRpcServer(int server_num) {
        for (int i = 0; i < server_num; ++i) {
            auto service = std::make_unique<TestRpcService>();
            auto server  = std::make_unique<TestRpcServer>(std::move(service));
            ASSERT_TRUE(server->start());
            server_addrs_.push_back("127.0.0.1:" + std::to_string(server->listenPort()));
            servers_.push_back(std::move(server));
        }
    }
    void setBufferContent(const BufferPtr& buffer, char c) const {
        setBufferContent(buffer, 0, buffer->sizeBytes(), c);
    }
    void setBufferContent(const BufferPtr& buffer, size_t offset, size_t byte_len, char c) const {
        auto addr = static_cast<char*>(buffer->dataWithOffset(offset));
        if (buffer->where() == MemoryType::MEMORY_GPU) {
            check_cuda_value(cudaMemset(addr, c, byte_len));
        } else {
            memset(addr, c, byte_len);
        }
    }

    // Byte-accurate helpers (do NOT use dataWithOffset(), which is element-based).
    // These are used to validate copyCache multi-layer offsets precisely.
    void setBufferBytes(const BufferPtr& buffer, size_t byte_offset, size_t byte_len, char c) const {
        auto base = static_cast<char*>(buffer->data());
        ASSERT_NE(base, nullptr);
        auto addr = base + byte_offset;
        if (buffer->where() == MemoryType::MEMORY_GPU) {
            check_cuda_value(cudaMemset(addr, c, byte_len));
        } else {
            memset(addr, c, byte_len);
        }
    }

    void verifyBufferBytesEq(const BufferPtr& buffer, size_t byte_offset, size_t byte_len, char expected) const {
        std::shared_ptr<void> data;
        auto                  base = static_cast<char*>(buffer->data());
        ASSERT_NE(base, nullptr);
        auto addr = base + byte_offset;
        if (buffer->where() == MemoryType::MEMORY_GPU) {
            data = std::shared_ptr<void>(malloc(byte_len), ::free);
            check_cuda_value(cudaMemcpy(data.get(), addr, byte_len, cudaMemcpyDeviceToHost));
        } else {
            data = std::shared_ptr<void>(addr, [](void*) {});
        }
        auto   ptr      = static_cast<const unsigned char*>(data.get());
        size_t mismatch = 0;
        for (; mismatch < byte_len; ++mismatch) {
            if (ptr[mismatch] != static_cast<unsigned char>(expected)) {
                break;
            }
        }
        ASSERT_EQ(mismatch, byte_len) << "mismatch at byte offset " << mismatch << " expect '" << expected << "' got 0x"
                                      << std::hex << static_cast<int>(ptr[mismatch]) << std::dec;
    }
    void verifyBufferContent(const BufferPtr& buffer, char c) const {
        verifyBufferContent(buffer, 0, buffer->sizeBytes(), c);
    }
    void verifyBufferContent(const BufferPtr& buffer, size_t offset, size_t byte_len, char c) const {
        std::shared_ptr<void> data;
        if (buffer->where() == MemoryType::MEMORY_GPU) {
            auto buffer_addr = static_cast<char*>(buffer->dataWithOffset(offset));
            data             = std::shared_ptr<void>(malloc(byte_len), ::free);
            check_cuda_value(cudaMemcpy(data.get(), buffer_addr, byte_len, cudaMemcpyDeviceToHost));
        } else {
            auto buffer_addr = static_cast<char*>(buffer->dataWithOffset(offset));
            data             = std::shared_ptr<void>(buffer_addr, [](void*) {});
        }
        auto   data_ptr = static_cast<char*>(data.get());
        size_t mismatch = 0;
        bool   ok       = true;
        for (; mismatch < byte_len; ++mismatch) {
            if (data_ptr[mismatch] != c) {
                ok = false;
                break;
            }
        }
        ASSERT_TRUE(ok) << "mismatch at byte offset " << mismatch << " expect '" << c << "' got 0x" << std::hex
                        << static_cast<int>(static_cast<unsigned char>(data_ptr[mismatch])) << std::dec;
    }
    void verifyGpuBufferContent(const std::vector<KVCacheMemoryConnector::LayerBlock>& gpu_layer_blocks) const {
        for (const auto& layer_block : gpu_layer_blocks) {
            const auto gpu_buf = allocator_->convertIndexToBuffer(layer_block.layer_id, layer_block.block_id);
            ASSERT_NE(gpu_buf.kv_addr, nullptr);
            // ASSERT_NE(gpu_buf.v_addr, nullptr);
            verifyBufferContent(gpu_buf.kv_addr, 'k' + layer_block.layer_id);
        }
    }
    void verifyCpuBufferContent(const std::vector<KVCacheMemoryConnector::LayerBlock>& gpu_layer_blocks,
                                int                                                    mem_block_index,
                                size_t                                                 mem_block_size) const {
        auto pool = requireExistingBlockPool(mem_block_size);
        ASSERT_NE(pool, nullptr);
        auto mem_buffer = pool->convertIndexToBuffer(0, mem_block_index);
        ASSERT_NE(mem_buffer.kv_addr, nullptr);
        // ASSERT_NE(mem_buffer.v_addr, nullptr);

        size_t offset = 0;
        for (const auto& layer_block : gpu_layer_blocks) {
            const auto gpu_buf = allocator_->convertIndexToBuffer(layer_block.layer_id, layer_block.block_id);
            ASSERT_NE(gpu_buf.kv_addr, nullptr);
            // ASSERT_NE(gpu_buf.v_addr, nullptr);

            char expected_k = 'k' + layer_block.layer_id;

            verifyBufferContent(mem_buffer.kv_addr, offset, gpu_buf.kv_addr->sizeBytes(), expected_k);
            offset += gpu_buf.kv_addr->sizeBytes();
        }
    }
    void prepareBufferContent(const std::vector<KVCacheMemoryConnector::LayerBlock>& gpu_layer_blocks,
                              int&                                                   mem_block_index,
                              size_t&                                                mem_block_size,
                              bool                                                   fill_gpu,
                              bool                                                   fill_cpu) const {
        // std::vector<KVCacheMemoryConnector::LayerBlock> gpu_layer_blocks{
        //     {/*layer_id*/0, /*block_id*/1},
        //     {/*layer_id*/1, /*block_id*/2},
        //     {/*layer_id*/2, /*block_id*/2},
        // };
        size_t total = 0;
        for (const auto& layer_block : gpu_layer_blocks) {
            const auto gpu_buf = allocator_->convertIndexToBuffer(layer_block.layer_id, layer_block.block_id);
            ASSERT_NE(gpu_buf.kv_addr, nullptr);
            // ASSERT_NE(gpu_buf.v_addr, nullptr);
            if (fill_gpu) {
                setBufferContent(gpu_buf.kv_addr, 'k' + layer_block.layer_id);
            }
            total += gpu_buf.kv_addr->sizeBytes();
        }

        // 申请memory block
        auto pool = ensureBlockPool(total);
        ASSERT_NE(pool, nullptr);
        auto mem_blocks = pool->malloc(1);
        ASSERT_EQ(mem_blocks.size(), 1u);
        auto malloced_mem_block_index = mem_blocks[0];
        auto mem_buffer               = pool->convertIndexToBuffer(0, malloced_mem_block_index);
        ASSERT_NE(mem_buffer.kv_addr, nullptr);
        EXPECT_EQ(mem_buffer.kv_addr->sizeBytes(), total);

        // 给mem_buffer填充数据
        if (fill_cpu) {
            size_t offset = 0;
            for (const auto& layer_block : gpu_layer_blocks) {
                const auto gpu_buf = allocator_->convertIndexToBuffer(layer_block.layer_id, layer_block.block_id);
                ASSERT_NE(gpu_buf.kv_addr, nullptr);
                // ASSERT_NE(gpu_buf.v_addr, nullptr);
                setBufferContent(mem_buffer.kv_addr, offset, gpu_buf.kv_addr->sizeBytes(), 'k' + layer_block.layer_id);
                offset += gpu_buf.kv_addr->sizeBytes();
            }
        }

        mem_block_index = malloced_mem_block_index;
        mem_block_size  = total;
    }
    void addOneCopyInfoToPb(MemoryOperationRequestPB&                              req,
                            const std::vector<KVCacheMemoryConnector::LayerBlock>& gpu_layer_blocks,
                            int                                                    mem_block_index,
                            size_t                                                 mem_block_size) const {
        auto* gb = req.add_gpu_blocks();
        for (const auto& layer_block : gpu_layer_blocks) {
            auto* lb = gb->add_layer_blocks();
            lb->set_layer_id(layer_block.layer_id);
            lb->set_block_id(layer_block.block_id);
        }
        req.add_mem_block_ids(mem_block_index);
        req.add_mem_block_sizes(mem_block_size);
    }
    LayerBlockIds makeLayerBlockIds(const std::vector<std::vector<int>>& per_layer_block_indices,
                                    size_t                               cache_keys_num) const {
        LayerBlockIds lbs;
        const size_t  layer_num = cache_config_.layer_num;
        lbs.reserve(layer_num);

        for (size_t layer = 0; layer < layer_num; ++layer) {
            auto  ptr    = std::make_shared<BlockIds>();
            auto& blocks = ptr->blocks();
            if (layer < per_layer_block_indices.size()) {
                blocks = per_layer_block_indices[layer];
            }
            if (blocks.size() < cache_keys_num) {
                blocks.resize(cache_keys_num, NULL_BLOCK_IDX);
            }
            lbs.emplace_back(std::move(ptr));
        }
        return lbs;
    }
    std::shared_ptr<KVCacheResource> makeCacheResource(const std::vector<int64_t>&          cache_keys,
                                                       const std::vector<std::vector<int>>& per_layer_block_indices,
                                                       size_t                               reuse_len = 0) const {
        auto res             = std::make_shared<KVCacheResource>();
        res->cache_keys      = cache_keys;
        res->layer_block_ids = makeLayerBlockIds(per_layer_block_indices, cache_keys.size());
        res->setReuseBlocksNum(reuse_len);
        return res;
    }
    std::vector<BlockIdxType> putItemsToCache(const std::vector<int64_t>& keys, size_t mem_block_size) const {
        std::vector<BlockIdxType> block_indices;
        if (keys.empty()) {
            return block_indices;
        }

        auto pool = ensureBlockPool(mem_block_size);
        if (!pool) {
            return block_indices;
        }

        for (size_t i = 0; i < keys.size(); ++i) {
            auto blocks = pool->malloc(1);  // will increase request ref
            if (blocks.size() != 1u) {
                ADD_FAILURE() << "malloc memory block failed, block_size=" << mem_block_size;
                break;
            }
            const BlockIdxType block_idx = blocks[0];
            block_indices.push_back(block_idx);

            MemoryBlockCache::CacheItem item;
            item.cache_key   = keys[i];
            item.block_index = block_idx;
            item.block_size  = mem_block_size;
            item.is_resident = false;
            connector_->block_cache_->put(item);

            pool->blockCacheReference({block_idx});

            // malloc会增加request ref, 所以这里需要requestFree减少request ref
            pool->requestFree({block_idx});
        }

        return block_indices;
    }
    std::shared_ptr<BlockPool> ensureBlockPool(size_t block_size) const {
        auto pool = connector_->getBlockPool(block_size);
        if (!pool) {
            // createBlockPool now requires an explicit pool size in MB and does NOT auto-register.
            const size_t one_mb        = 1024 * 1024;
            const size_t min_pool_size = (block_size + one_mb - 1) / one_mb;  // at least 1 block
            pool                       = connector_->createBlockPool(block_size, std::max<size_t>(1, min_pool_size));
            if (pool) {
                std::lock_guard<std::shared_mutex> lock(connector_->pool_mutex_);
                connector_->block_pools_[block_size] = pool;
            }
        }
        if (!pool) {
            ADD_FAILURE() << "failed to create block pool, block_size=" << block_size;
        }
        return pool;
    }
    std::shared_ptr<BlockPool> requireExistingBlockPool(size_t block_size) const {
        auto pool = connector_->getBlockPool(block_size);
        if (!pool) {
            ADD_FAILURE() << "expected block pool exists, block_size=" << block_size;
        }
        return pool;
    }
};

TEST_F(KVCacheMemoryConnectorTest, init_ReturnFalse_NoWorkerAddrs) {
    // 构造空的 worker 地址，TpBroadcastManager::init() 会失败；业务代码使用 RTP_LLM_CHECK，
    // 因此这里期望抛出 std::runtime_error。
    std::vector<std::string> empty_addrs;
    auto                     conn =
        std::make_shared<KVCacheMemoryConnector>(cache_config_, kv_cache_config_, allocator_, device_, empty_addrs);
    EXPECT_THROW(conn->init(), std::runtime_error);
}

TEST_F(KVCacheMemoryConnectorTest, init_ReturnFalse_WhenMemoryCacheSizeMbZero) {
    auto kv_cfg                         = kv_cache_config_;
    kv_cfg.memory_cache_size_mb         = 0;
    kv_cfg.memory_cache_sync_timeout_ms = 1000;

    auto conn = std::make_shared<KVCacheMemoryConnector>(cache_config_, kv_cfg, allocator_, device_, server_addrs_);
    EXPECT_THROW(conn->init(), std::runtime_error);
    // Init fails early, nothing should be created.
    EXPECT_EQ(conn->block_cache_, nullptr);
    EXPECT_EQ(conn->tp_broadcast_manager_, nullptr);
    EXPECT_EQ(conn->wait_done_thread_pool_, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, init_ReturnFalse_WhenMemoryCacheSyncTimeoutMsZero) {
    auto kv_cfg                         = kv_cache_config_;
    kv_cfg.memory_cache_size_mb         = 64;
    kv_cfg.memory_cache_sync_timeout_ms = 0;

    auto conn = std::make_shared<KVCacheMemoryConnector>(cache_config_, kv_cfg, allocator_, device_, server_addrs_);
    EXPECT_THROW(conn->init(), std::runtime_error);
    // Init fails early, nothing should be created.
    EXPECT_EQ(conn->block_cache_, nullptr);
    EXPECT_EQ(conn->tp_broadcast_manager_, nullptr);
    EXPECT_EQ(conn->wait_done_thread_pool_, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, init_ReturnFalse_WhenBlockSizeBytesZero) {
    auto cfg             = cache_config_;
    cfg.block_size_bytes = 0;

    auto kv_cfg                         = kv_cache_config_;
    kv_cfg.memory_cache_size_mb         = 64;
    kv_cfg.memory_cache_sync_timeout_ms = 1000;

    // initBlockPool() checks block_size_bytes > 0
    auto conn = std::make_shared<KVCacheMemoryConnector>(cfg, kv_cfg, allocator_, device_, server_addrs_);
    EXPECT_THROW(conn->init(), std::runtime_error);
}

TEST_F(KVCacheMemoryConnectorTest, init_ReturnFalse_WhenPoolTooSmallForBlockSize) {
    auto cfg = cache_config_;
    // Make sure pool_size_mb * 1MB / block_size_bytes == 0 -> createBlockPool() should fail with CHECK.
    cfg.block_size_bytes = 2 * 1024 * 1024;  // 2MB

    auto kv_cfg                         = kv_cache_config_;
    kv_cfg.memory_cache_size_mb         = 1;     // 1MB
    kv_cfg.memory_cache_sync_timeout_ms = 1000;  // valid

    auto conn = std::make_shared<KVCacheMemoryConnector>(cfg, kv_cfg, allocator_, device_, server_addrs_);
    EXPECT_THROW(conn->init(), std::runtime_error);
}

TEST_F(KVCacheMemoryConnectorTest, init_ReturnTrue_WithWorkerAddrs) {
    // 使用有效的 worker 地址，init 应成功并正确设置 manager
    auto conn =
        std::make_shared<KVCacheMemoryConnector>(cache_config_, kv_cache_config_, allocator_, device_, server_addrs_);
    auto ok = conn->init();
    EXPECT_TRUE(ok);
    ASSERT_NE(conn->block_cache_, nullptr);
    ASSERT_NE(conn->tp_broadcast_manager_, nullptr);
    EXPECT_EQ(conn->tp_broadcast_manager_->workerNum(), server_addrs_.size());
}

TEST_F(KVCacheMemoryConnectorTest, initBlockPool_Throw_WhenMemoryCacheSizeMbZero) {
    auto kv_cfg                         = kv_cache_config_;
    kv_cfg.memory_cache_size_mb         = 0;
    kv_cfg.memory_cache_sync_timeout_ms = 1000;

    auto conn = std::make_shared<KVCacheMemoryConnector>(cache_config_, kv_cfg, allocator_, device_, server_addrs_);
    EXPECT_THROW(conn->initBlockPool(), std::runtime_error);
}

TEST_F(KVCacheMemoryConnectorTest, initBlockPool_Throw_WhenBlockSizeBytesZero) {
    auto cfg             = cache_config_;
    cfg.block_size_bytes = 0;

    auto kv_cfg                         = kv_cache_config_;
    kv_cfg.memory_cache_size_mb         = 64;
    kv_cfg.memory_cache_sync_timeout_ms = 1000;

    auto conn = std::make_shared<KVCacheMemoryConnector>(cfg, kv_cfg, allocator_, device_, server_addrs_);
    EXPECT_THROW(conn->initBlockPool(), std::runtime_error);
}

TEST_F(KVCacheMemoryConnectorTest, initBlockPool_Throw_WhenCreateBlockPoolFails) {
    auto cfg = cache_config_;
    // Force createBlockPool() to compute block_num=0:
    // block_num = pool_size_mb * 1MB / block_size_bytes.
    cfg.block_size_bytes = 2 * 1024 * 1024;  // 2MB

    auto kv_cfg                         = kv_cache_config_;
    kv_cfg.memory_cache_size_mb         = 1;     // 1MB
    kv_cfg.memory_cache_sync_timeout_ms = 1000;  // not used by initBlockPool but keep valid

    auto conn = std::make_shared<KVCacheMemoryConnector>(cfg, kv_cfg, allocator_, device_, server_addrs_);
    EXPECT_THROW(conn->initBlockPool(), std::runtime_error);
}

TEST_F(KVCacheMemoryConnectorTest, initBlockPool_ReturnTrue_AndRegistersPool) {
    auto kv_cfg                         = kv_cache_config_;
    kv_cfg.memory_cache_size_mb         = 64;
    kv_cfg.memory_cache_sync_timeout_ms = 1000;  // not used by initBlockPool but keep valid

    auto conn = std::make_shared<KVCacheMemoryConnector>(cache_config_, kv_cfg, allocator_, device_, server_addrs_);
    EXPECT_NO_THROW(conn->initBlockPool());
    auto pool = conn->getBlockPool(cache_config_.block_size_bytes);
    ASSERT_NE(pool, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, asyncMatch_ReturnNull_WhenGpuReuseLenGEKeysSize) {
    const size_t                  N = 3;
    std::vector<int64_t>          cache_keys{70001, 70002, 70003};
    std::vector<std::vector<int>> lbs_vec{{1, 1, 1}, {2, 2, 2}};
    auto                          res = makeCacheResource(cache_keys, lbs_vec, /*reuse_len=*/N);

    // Even if memory has matches, asyncMatch should skip when gpu reuse covers all keys.
    const auto   buf      = allocator_->convertIndexToBuffer(0, 0);
    const size_t mem_size = buf.kv_addr->sizeBytes();
    putItemsToCache(cache_keys, mem_size);

    auto match_ctx = connector_->asyncMatch(res, /*meta=*/nullptr);
    EXPECT_EQ(match_ctx, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, asyncMatch_ReturnNull_WhenNoPrefixMatched) {
    std::vector<int64_t>          cache_keys{71001, 71002};
    std::vector<std::vector<int>> lbs_vec{{1, 1}};
    auto                          res = makeCacheResource(cache_keys, lbs_vec, /*reuse_len=*/0);

    // No cache prefill => matched_num == 0
    auto match_ctx = connector_->asyncMatch(res, /*meta=*/nullptr);
    EXPECT_EQ(match_ctx, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, asyncMatch_ReturnMatchedNum_WhenPrefixMatchedAndStopAtFirstMiss) {
    std::vector<int64_t>          cache_keys{72001, 72002, 72003};
    std::vector<std::vector<int>> lbs_vec{{1, 1, 1}};
    auto                          res = makeCacheResource(cache_keys, lbs_vec, /*reuse_len=*/0);

    const auto   buf      = allocator_->convertIndexToBuffer(0, 0);
    const size_t mem_size = buf.kv_addr->sizeBytes();

    // Only prefill first 2 keys in cache; 3rd miss => matched_num should be 2.
    putItemsToCache({cache_keys[0], cache_keys[1]}, mem_size);

    auto match_ctx = connector_->asyncMatch(res, /*meta=*/nullptr);
    ASSERT_NE(match_ctx, nullptr);
    EXPECT_TRUE(match_ctx->done());
    EXPECT_TRUE(match_ctx->success());
    EXPECT_EQ(match_ctx->matchedBlockCount(), 2u);
    EXPECT_EQ(match_ctx->connectorType(), rtp_llm::KVCacheConnector::ConnectorType::Memory);
}

TEST_F(KVCacheMemoryConnectorTest, asyncRead_ReturnNull_OnInvalidInputs) {
    // resource is nullptr
    auto ctx_null = connector_->asyncRead(nullptr, nullptr, nullptr);
    EXPECT_EQ(ctx_null, nullptr);

    // empty cache_keys
    auto res_empty_keys = makeCacheResource({}, {{1}});
    auto ctx1           = connector_->asyncRead(res_empty_keys, nullptr, nullptr);
    EXPECT_EQ(ctx1, nullptr);

    // empty layer_block_ids
    auto res_empty_lbs        = std::make_shared<KVCacheResource>();
    res_empty_lbs->cache_keys = CacheKeysType{1};
    res_empty_lbs->layer_block_ids.clear();
    auto ctx2 = connector_->asyncRead(res_empty_lbs, nullptr, nullptr);
    EXPECT_EQ(ctx2, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, asyncRead_ReturnNull_WhenReuseLenGEKeys) {
    const size_t                  N = 3;
    std::vector<int64_t>          cache_keys{10001, 10002, 10003};
    std::vector<std::vector<int>> lbs_vec{{1, 1, 1}, {2, 2, 2}};
    auto                          res = makeCacheResource(cache_keys, lbs_vec, N);

    // With reuse_len == keys size, asyncMatch should skip and there is nothing to read.
    auto match_ctx = connector_->asyncMatch(res, nullptr);
    EXPECT_EQ(match_ctx, nullptr);
    EXPECT_EQ(res->reuseBlocksNum(), N);
}

TEST_F(KVCacheMemoryConnectorTest, asyncRead_ReturnNull_WhenPlanEmpty) {
    // Simulate mismatch between match result and current cache state:
    // asyncRead does NOT call asyncMatch any more, so it relies on match_context + meta.
    // Here cache has no items, so buildCopyPlanForRead should fail and asyncRead returns nullptr.
    std::vector<int64_t>          cache_keys{20001, 20002};
    std::vector<std::vector<int>> lbs_vec{{3, 3}, {4, 4}};
    auto                          res = makeCacheResource(cache_keys, lbs_vec);

    class TestMatchContext: public rtp_llm::KVCacheConnector::AsyncMatchContext {
    public:
        explicit TestMatchContext(size_t matched): matched_(matched) {}
        bool done() const override {
            return true;
        }
        bool success() const override {
            return true;
        }
        size_t matchedBlockCount() const override {
            return matched_;
        }
        KVCacheConnector::ConnectorType connectorType() const override {
            return KVCacheConnector::ConnectorType::Memory;
        }

    private:
        size_t matched_{0};
    };

    auto match_ctx = std::make_shared<TestMatchContext>(/*matched=*/1);
    auto meta      = std::make_shared<TestReadMeta>(/*start_block_index=*/0, /*size=*/1);
    auto ctx       = connector_->asyncRead(res, meta, match_ctx);
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, asyncRead_ReturnNull_WhenSendCopyPlanFails_NoWorkers) {
    // 有命中项，但没有 worker，发送失败应返回 nullptr
    std::vector<int64_t> cache_keys{30001, 30002};

    const auto   buf      = allocator_->convertIndexToBuffer(0, 0);
    const size_t mem_size = buf.kv_addr->sizeBytes();

    auto block_indices = putItemsToCache(cache_keys, mem_size);
    ASSERT_EQ(block_indices.size(), cache_keys.size());

    std::vector<std::vector<int>> lbs_vec{{10, 11}, {20, 21}};
    auto                          res = makeCacheResource(cache_keys, lbs_vec);

    connector_->tp_broadcast_manager_->worker_addrs_.clear();
    auto match_ctx = connector_->asyncMatch(res, nullptr);
    ASSERT_NE(match_ctx, nullptr);
    auto meta = std::make_shared<TestReadMeta>(/*start_block_index=*/0, /*size=*/(int)match_ctx->matchedBlockCount());
    auto ctx  = connector_->asyncRead(res, meta, match_ctx);
    ASSERT_NE(ctx, nullptr);
    auto mem_ctx = std::dynamic_pointer_cast<rtp_llm::MemoryConnectorAsyncContext>(ctx);
    ASSERT_NE(mem_ctx, nullptr);
    mem_ctx->waitDone();
    EXPECT_FALSE(ctx->success());
}

TEST_F(KVCacheMemoryConnectorTest, asyncRead_Success_IncrementsReuseLen_ByMatchedPrefix) {
    // 初始 reuse_len=1, 内存全部命中 => mem_match_len=3，最终 reuse_len=3
    std::vector<int64_t> cache_keys{40001, 40002, 40003};

    const auto   buf      = allocator_->convertIndexToBuffer(0, 0);
    const size_t mem_size = buf.kv_addr->sizeBytes();

    auto block_indices = putItemsToCache(cache_keys, mem_size);
    ASSERT_EQ(block_indices.size(), cache_keys.size());

    std::vector<std::vector<int>> lbs_vec{
        {101, 102, 103},  // layer0
        {201, 202, 203},  // layer1
    };
    auto res = makeCacheResource(cache_keys, lbs_vec, 1);

    auto match_ctx = connector_->asyncMatch(res, nullptr);
    ASSERT_NE(match_ctx, nullptr);
    const int reuse_num = static_cast<int>(res->reuseBlocksNum());
    const int read_num  = static_cast<int>(match_ctx->matchedBlockCount()) - reuse_num;
    ASSERT_GT(read_num, 0);
    auto meta = std::make_shared<TestReadMeta>(reuse_num, read_num);
    auto ctx  = connector_->asyncRead(res, meta, match_ctx);
    ASSERT_NE(ctx, nullptr);
    auto mem_ctx = std::dynamic_pointer_cast<rtp_llm::MemoryConnectorAsyncContext>(ctx);
    ASSERT_NE(mem_ctx, nullptr);
    mem_ctx->waitDone();
    EXPECT_TRUE(ctx->success());
    EXPECT_EQ(res->reuseBlocksNum(), 3u);
}

TEST_F(KVCacheMemoryConnectorTest, asyncRead_FailureOnMemResponse_NoReuseLenIncrement) {
    // 构造部分 rank mem_response 失败，最终 AsyncContext->success() 应为 false，reuse_len 不增加
    std::vector<std::unique_ptr<TestRpcServer>> servers;
    std::vector<std::string>                    addrs;
    for (int i = 0; i < 4; ++i) {
        auto service = std::make_unique<TestRpcService>();
        service->setMemResponseSuccess(i % 2 == 0);  // 只有偶数 rank 成功
        auto server = std::make_unique<TestRpcServer>(std::move(service));
        ASSERT_TRUE(server->start());
        addrs.push_back("127.0.0.1:" + std::to_string(server->listenPort()));
        servers.push_back(std::move(server));
    }
    auto broadcast_manager = std::make_shared<TpBroadcastManager>(addrs);
    ASSERT_TRUE(broadcast_manager->init());
    connector_->tp_broadcast_manager_ = broadcast_manager;

    std::vector<int64_t> cache_keys{50001, 50002};

    const auto   buf      = allocator_->convertIndexToBuffer(0, 0);
    const size_t mem_size = buf.kv_addr->sizeBytes();

    auto block_indices = putItemsToCache(cache_keys, mem_size);
    ASSERT_EQ(block_indices.size(), cache_keys.size());

    std::vector<std::vector<int>> lbs_vec{{11, 12}, {21, 22}};
    auto                          res = makeCacheResource(cache_keys, lbs_vec);

    auto match_ctx = connector_->asyncMatch(res, nullptr);
    ASSERT_NE(match_ctx, nullptr);
    auto meta = std::make_shared<TestReadMeta>(/*start_block_index=*/0, /*size=*/(int)match_ctx->matchedBlockCount());
    auto ctx  = connector_->asyncRead(res, meta, match_ctx);
    ASSERT_NE(ctx, nullptr);
    auto mem_ctx = std::dynamic_pointer_cast<rtp_llm::MemoryConnectorAsyncContext>(ctx);
    ASSERT_NE(mem_ctx, nullptr);
    mem_ctx->waitDone();
    EXPECT_FALSE(ctx->success());
    EXPECT_EQ(res->reuseBlocksNum(), 0u);

    connector_->tp_broadcast_manager_.reset();
    for (auto& s : servers) {
        s->shutdown();
    }
    servers.clear();
    addrs.clear();
}

TEST_F(KVCacheMemoryConnectorTest, asyncRead_FailureOnRpcStatus_NoReuseLenIncrement) {
    // 构造部分 rank RPC 状态失败，最终 AsyncContext->success() 应为 false，reuse_len 不增加
    std::vector<std::unique_ptr<TestRpcServer>> servers;
    std::vector<std::string>                    addrs;
    for (int i = 0; i < 4; ++i) {
        auto service = std::make_unique<TestRpcService>();
        if (i % 2 == 0) {
            service->setRpcResponseStatus(::grpc::Status(::grpc::StatusCode::UNAVAILABLE, "down"));
        }
        auto server = std::make_unique<TestRpcServer>(std::move(service));
        ASSERT_TRUE(server->start());
        addrs.push_back("127.0.0.1:" + std::to_string(server->listenPort()));
        servers.push_back(std::move(server));
    }
    auto broadcast_manager = std::make_shared<TpBroadcastManager>(addrs);
    ASSERT_TRUE(broadcast_manager->init());
    connector_->tp_broadcast_manager_ = broadcast_manager;

    std::vector<int64_t> cache_keys{60001, 60002};

    const auto   buf      = allocator_->convertIndexToBuffer(0, 0);
    const size_t mem_size = buf.kv_addr->sizeBytes();

    auto block_indices = putItemsToCache(cache_keys, mem_size);
    ASSERT_EQ(block_indices.size(), cache_keys.size());

    std::vector<std::vector<int>> lbs_vec{{31, 32}, {41, 42}};
    auto                          res = makeCacheResource(cache_keys, lbs_vec);

    auto match_ctx = connector_->asyncMatch(res, nullptr);
    ASSERT_NE(match_ctx, nullptr);
    auto meta = std::make_shared<TestReadMeta>(/*start_block_index=*/0, /*size=*/(int)match_ctx->matchedBlockCount());
    auto ctx  = connector_->asyncRead(res, meta, match_ctx);
    ASSERT_NE(ctx, nullptr);
    auto mem_ctx = std::dynamic_pointer_cast<rtp_llm::MemoryConnectorAsyncContext>(ctx);
    ASSERT_NE(mem_ctx, nullptr);
    mem_ctx->waitDone();
    EXPECT_FALSE(ctx->success());
    EXPECT_EQ(res->reuseBlocksNum(), 0u);

    connector_->tp_broadcast_manager_.reset();
    for (auto& s : servers) {
        s->shutdown();
    }
    servers.clear();
    addrs.clear();
}

TEST_F(KVCacheMemoryConnectorTest, asyncRead_ReturnNull_WhenThreadPoolFull) {
    // asyncRead is executed asynchronously in thread pool; the API should return a non-null context immediately.
    // Use a blocked worker thread so the send+wait task is queued (simulating "busy/full" pool).
    auto old_pool = connector_->wait_done_thread_pool_;

    connector_->wait_done_thread_pool_ = std::make_shared<autil::LockFreeThreadPool>(/*thread_num=*/1,
                                                                                     /*queue_size=*/10,
                                                                                     /*thread_init_func=*/nullptr,
                                                                                     /*name=*/"AsyncReadFullTP");
    ASSERT_TRUE(connector_->wait_done_thread_pool_->start());

    std::atomic<bool> block_worker{true};
    ASSERT_EQ(connector_->wait_done_thread_pool_->pushTask([&]() {
        while (block_worker.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }),
              autil::ThreadPoolBase::ERROR_NONE);

    std::vector<int64_t> cache_keys{70001, 70002};
    const auto           buf      = allocator_->convertIndexToBuffer(0, 0);
    const size_t         mem_size = buf.kv_addr->sizeBytes();
    putItemsToCache(cache_keys, mem_size);
    auto res       = makeCacheResource(cache_keys, {{1, 2}, {3, 4}});
    auto match_ctx = connector_->asyncMatch(res, nullptr);
    ASSERT_NE(match_ctx, nullptr);
    auto meta = std::make_shared<TestReadMeta>(/*start_block_index=*/0, /*size=*/(int)match_ctx->matchedBlockCount());

    // Force failure when the queued task eventually runs.
    connector_->tp_broadcast_manager_->worker_addrs_.clear();
    auto ctx = connector_->asyncRead(res, meta, match_ctx);
    ASSERT_NE(ctx, nullptr);

    block_worker.store(false);
    auto mem_ctx = std::dynamic_pointer_cast<rtp_llm::MemoryConnectorAsyncContext>(ctx);
    ASSERT_NE(mem_ctx, nullptr);
    mem_ctx->waitDone();
    EXPECT_FALSE(ctx->success());

    connector_->wait_done_thread_pool_->stop();
    connector_->wait_done_thread_pool_.reset();
    connector_->wait_done_thread_pool_ = old_pool;
}

TEST_F(KVCacheMemoryConnectorTest, asyncWrite_ReturnNull_OnInvalidInputs) {
    // resource is nullptr
    auto ctx_null = connector_->asyncWrite(nullptr, nullptr);
    EXPECT_EQ(ctx_null, nullptr);

    // empty cache_keys
    auto res_empty_keys = makeCacheResource({}, {{1}});
    auto ctx1           = connector_->asyncWrite(res_empty_keys, nullptr);
    EXPECT_EQ(ctx1, nullptr);

    // empty layer_block_ids
    auto res_empty_lbs        = std::make_shared<KVCacheResource>();
    res_empty_lbs->cache_keys = CacheKeysType{1};
    res_empty_lbs->layer_block_ids.clear();
    auto ctx2 = connector_->asyncWrite(res_empty_lbs, nullptr);
    EXPECT_EQ(ctx2, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, asyncWrite_ReturnNull_WhenAllKeysInCache) {
    // 两个 key 均已在内存缓存中
    const int                     layer0        = 0;
    const int                     gpu_block_idx = 1;
    std::vector<int64_t>          cache_keys{10, 11};
    std::vector<std::vector<int>> lbs_vec{{gpu_block_idx, gpu_block_idx}};
    auto                          res = makeCacheResource(cache_keys, lbs_vec);

    const auto buf = allocator_->convertIndexToBuffer(layer0, gpu_block_idx);
    ASSERT_NE(buf.kv_addr, nullptr);
    // ASSERT_NE(buf.v_addr, nullptr);
    const size_t total = buf.kv_addr->sizeBytes();

    // 预置到 cache
    auto block_indices = putItemsToCache(cache_keys, total);
    ASSERT_EQ(block_indices.size(), cache_keys.size());

    const size_t cache_size_before = connector_->block_cache_->size();

    auto ctx = connector_->asyncWrite(res, nullptr);
    ASSERT_EQ(ctx, nullptr);
    EXPECT_EQ(connector_->block_cache_->size(), cache_size_before);
}

TEST_F(KVCacheMemoryConnectorTest, asyncWrite_ReturnSuccess_WhenPrefixInCacheOnlyWriteSuffix) {
    const int                     layer0 = 0;
    std::vector<int64_t>          cache_keys{60001, 60002, 60003};
    std::vector<std::vector<int>> lbs_vec{{1, 2, 3}};
    auto                          res = makeCacheResource(cache_keys, lbs_vec);

    const auto buf = allocator_->convertIndexToBuffer(layer0, /*block_id=*/1);
    ASSERT_NE(buf.kv_addr, nullptr);
    const size_t total = buf.kv_addr->sizeBytes();

    // Pre-insert only the first key, so cpu_matched_num should be 1 and only suffix gets written.
    auto pre_blocks = putItemsToCache({cache_keys[0]}, total);
    ASSERT_EQ(pre_blocks.size(), 1u);
    ASSERT_TRUE(connector_->block_cache_->contains(cache_keys[0]));
    const size_t cache_before = connector_->block_cache_->size();

    auto ctx = connector_->asyncWrite(res, nullptr);
    ASSERT_NE(ctx, nullptr);
    auto mem_ctx = std::dynamic_pointer_cast<rtp_llm::MemoryConnectorAsyncContext>(ctx);
    ASSERT_NE(mem_ctx, nullptr);
    mem_ctx->waitDone();
    EXPECT_TRUE(ctx->success());

    // Only 2 new items inserted.
    EXPECT_GE(connector_->block_cache_->size(), cache_before + 2);
    EXPECT_TRUE(connector_->block_cache_->contains(cache_keys[0]));
    EXPECT_TRUE(connector_->block_cache_->contains(cache_keys[1]));
    EXPECT_TRUE(connector_->block_cache_->contains(cache_keys[2]));
}

TEST_F(KVCacheMemoryConnectorTest, asyncWrite_ReturnSuccess_WhenKeyInsertedDuringWriteDone) {
    // Delay RPC so we can insert a key into block_cache_ while asyncWrite is in flight.
    std::vector<std::unique_ptr<TestRpcServer>> servers;
    std::vector<std::string>                    addrs;
    for (int i = 0; i < 2; ++i) {
        auto service = std::make_unique<TestRpcService>();
        service->setSleepMillis(200);
        auto server = std::make_unique<TestRpcServer>(std::move(service));
        ASSERT_TRUE(server->start());
        addrs.push_back("127.0.0.1:" + std::to_string(server->listenPort()));
        servers.push_back(std::move(server));
    }
    auto broadcast_manager = std::make_shared<TpBroadcastManager>(addrs);
    ASSERT_TRUE(broadcast_manager->init());
    connector_->tp_broadcast_manager_ = broadcast_manager;

    std::vector<int64_t>          cache_keys{61001, 61002};
    std::vector<std::vector<int>> lbs_vec{{1, 2}};
    auto                          res = makeCacheResource(cache_keys, lbs_vec);

    // Ensure block pools exist for the exact total_bytes computed by buildCopyPlanForWrite():
    // it sums kv + scale only for layers whose gpu block id is NOT NULL for that key.
    auto totalBytesForKeyIndex = [&](size_t key_index) -> size_t {
        size_t      total           = 0;
        const auto& layer_block_ids = res->layerBlocks();
        for (size_t layer = 0; layer < layer_block_ids.size(); ++layer) {
            const int gpu_block_idx = layer_block_ids.at(layer)->blocks().at(key_index);
            if (gpu_block_idx == NULL_BLOCK_IDX) {
                continue;
            }
            const auto buffers = allocator_->convertIndexToBuffer(static_cast<int>(layer), gpu_block_idx);
            if (buffers.kv_addr) {
                total += buffers.kv_addr->sizeBytes();
            }
            if (buffers.kv_scale_addr) {
                total += buffers.kv_scale_addr->sizeBytes();
            }
        }
        return total;
    };
    const size_t total_key0 = totalBytesForKeyIndex(/*key_index=*/0);
    const size_t total_key1 = totalBytesForKeyIndex(/*key_index=*/1);
    ASSERT_GT(total_key0, 0u);
    ASSERT_GT(total_key1, 0u);
    ASSERT_NE(ensureBlockPool(total_key0), nullptr);
    ASSERT_NE(ensureBlockPool(total_key1), nullptr);

    auto ctx = connector_->asyncWrite(res, nullptr);
    ASSERT_NE(ctx, nullptr);

    // While in flight, insert the first key so write_done should skip inserting it.
    auto pre_blocks = putItemsToCache({cache_keys[0]}, total_key0);
    ASSERT_EQ(pre_blocks.size(), 1u);
    ASSERT_TRUE(connector_->block_cache_->contains(cache_keys[0]));

    auto mem_ctx = std::dynamic_pointer_cast<rtp_llm::MemoryConnectorAsyncContext>(ctx);
    ASSERT_NE(mem_ctx, nullptr);
    mem_ctx->waitDone();
    EXPECT_TRUE(ctx->success());

    EXPECT_TRUE(connector_->block_cache_->contains(cache_keys[0]));
    EXPECT_TRUE(connector_->block_cache_->contains(cache_keys[1]));

    connector_->tp_broadcast_manager_.reset();
    for (auto& s : servers) {
        s->shutdown();
    }
    servers.clear();
    addrs.clear();
}

TEST_F(KVCacheMemoryConnectorTest, asyncWrite_ReturnNull_WhenBuildPlanEmpty) {
    // 所有 layer 对于第一个未命中 key 的 blockIdx 都为 NULL，导致 plan 为空
    std::vector<int64_t> cache_keys{100, 101};
    // 2 层，全部 NULL
    std::vector<std::vector<int>> lbs_vec{{NULL_BLOCK_IDX, NULL_BLOCK_IDX}, {NULL_BLOCK_IDX, NULL_BLOCK_IDX}};
    auto                          res = makeCacheResource(cache_keys, lbs_vec);

    auto ctx = connector_->asyncWrite(res, nullptr);
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, asyncWrite_ReturnNull_WhenSendCopyPlanFails_NoWorkers) {
    // 合法的 plan，但由于没有 worker，sendCopyPlan 返回空并触发回滚
    const int                     gpu_block_idx = 2;
    std::vector<int64_t>          cache_keys{1, 2};
    std::vector<std::vector<int>> lbs_vec{{gpu_block_idx, gpu_block_idx}};
    auto                          res = makeCacheResource(cache_keys, lbs_vec);

    // Ensure block pool exists for the total_bytes buildCopyPlanForWrite() will compute:
    // only layer0 is set in lbs_vec, other layers are NULL in makeLayerBlockIds().
    size_t      total           = 0;
    const auto& layer_block_ids = res->layerBlocks();
    for (size_t layer = 0; layer < layer_block_ids.size(); ++layer) {
        const int block_id = layer_block_ids.at(layer)->blocks().at(0);
        if (block_id == NULL_BLOCK_IDX) {
            continue;
        }
        const auto buffers = allocator_->convertIndexToBuffer(static_cast<int>(layer), block_id);
        if (buffers.kv_addr) {
            total += buffers.kv_addr->sizeBytes();
        }
        if (buffers.kv_scale_addr) {
            total += buffers.kv_scale_addr->sizeBytes();
        }
    }
    ASSERT_GT(total, 0u);
    ASSERT_NE(ensureBlockPool(total), nullptr);

    connector_->tp_broadcast_manager_->worker_addrs_.clear();
    auto ctx = connector_->asyncWrite(res, nullptr);
    // Business behavior: asyncWrite returns a context, but will complete with failure when no workers.
    ASSERT_NE(ctx, nullptr);
    auto mem_ctx = std::dynamic_pointer_cast<rtp_llm::MemoryConnectorAsyncContext>(ctx);
    ASSERT_NE(mem_ctx, nullptr);
    mem_ctx->waitDone();
    EXPECT_FALSE(ctx->success());
    // Failure should not insert cache entries.
    EXPECT_FALSE(connector_->block_cache_->contains(cache_keys[0]));
    EXPECT_FALSE(connector_->block_cache_->contains(cache_keys[1]));
}

TEST_F(KVCacheMemoryConnectorTest, asyncWrite_Success_AddsToBlockCache_AndKeepsMemBlocks) {
    // 默认 RPC 服务均返回 OK + mem success
    const int                     layer0        = 0;
    const int                     gpu_block_idx = 2;
    const size_t                  N             = 2;
    std::vector<int64_t>          cache_keys{200, 201};
    std::vector<std::vector<int>> lbs_vec{{gpu_block_idx, gpu_block_idx + 1}};
    auto                          res = makeCacheResource(cache_keys, lbs_vec);

    const auto buf = allocator_->convertIndexToBuffer(layer0, gpu_block_idx);
    ASSERT_NE(buf.kv_addr, nullptr);
    // ASSERT_NE(buf.v_addr, nullptr);
    const size_t total = buf.kv_addr->sizeBytes();
    auto         pool  = ensureBlockPool(total);
    ASSERT_NE(pool, nullptr);
    const size_t free_before  = pool->freeBlocksNum();
    const size_t cache_before = connector_->block_cache_->size();

    auto ctx = connector_->asyncWrite(res, nullptr);
    ASSERT_NE(ctx, nullptr);
    auto mem_ctx = std::dynamic_pointer_cast<rtp_llm::MemoryConnectorAsyncContext>(ctx);
    ASSERT_NE(mem_ctx, nullptr);
    mem_ctx->waitDone();
    EXPECT_TRUE(ctx->success());

    // block_cache 中应新增 N 个条目
    EXPECT_EQ(connector_->block_cache_->size(), cache_before + N);
    for (auto key : cache_keys) {
        EXPECT_TRUE(connector_->block_cache_->contains(key));
    }
    // 对应大小的 pool 空闲块减少 N（分配后未释放，缓存驻留）
    EXPECT_EQ(pool->freeBlocksNum(), free_before - N);
}

TEST_F(KVCacheMemoryConnectorTest, asyncWrite_FailureOnMemResponse_FreesAllocatedBlocks_NoCacheInsert) {
    // 构造部分 rank mem_response 失败，最终 AsyncContext->success() 应为 false
    std::vector<std::unique_ptr<TestRpcServer>> servers;
    std::vector<std::string>                    addrs;
    for (int i = 0; i < 4; ++i) {
        auto service = std::make_unique<TestRpcService>();
        service->setMemResponseSuccess(i % 2 == 0);  // 只有偶数 rank 成功
        auto server = std::make_unique<TestRpcServer>(std::move(service));
        ASSERT_TRUE(server->start());
        addrs.push_back("127.0.0.1:" + std::to_string(server->listenPort()));
        servers.push_back(std::move(server));
    }
    auto broadcast_manager = std::make_shared<TpBroadcastManager>(addrs);
    ASSERT_TRUE(broadcast_manager->init());
    connector_->tp_broadcast_manager_ = broadcast_manager;

    const int                     layer0        = 0;
    const int                     gpu_block_idx = 1;
    std::vector<int64_t>          cache_keys{301, 302};
    std::vector<std::vector<int>> lbs_vec{{gpu_block_idx, gpu_block_idx}};
    auto                          res = makeCacheResource(cache_keys, lbs_vec);

    const auto buf = allocator_->convertIndexToBuffer(layer0, gpu_block_idx);
    ASSERT_NE(buf.kv_addr, nullptr);
    // ASSERT_NE(buf.v_addr, nullptr);
    const size_t total = buf.kv_addr->sizeBytes();
    auto         pool  = ensureBlockPool(total);
    ASSERT_NE(pool, nullptr);
    const size_t free_before  = pool->freeBlocksNum();
    const size_t cache_before = connector_->block_cache_->size();

    auto ctx = connector_->asyncWrite(res, nullptr);
    ASSERT_NE(ctx, nullptr);
    auto mem_ctx = std::dynamic_pointer_cast<rtp_llm::MemoryConnectorAsyncContext>(ctx);
    ASSERT_NE(mem_ctx, nullptr);
    mem_ctx->waitDone();
    EXPECT_FALSE(ctx->success());
    // 应未插入缓存
    EXPECT_EQ(connector_->block_cache_->size(), cache_before);
    // 分配的块应被回收
    EXPECT_EQ(pool->freeBlocksNum(), free_before);

    connector_->tp_broadcast_manager_.reset();
    for (auto& s : servers) {
        s->shutdown();
    }
    servers.clear();
    addrs.clear();
}

TEST_F(KVCacheMemoryConnectorTest, asyncWrite_ReturnNull_WhenThreadPoolFull) {
    // asyncWrite is executed asynchronously in thread pool; the API should return a non-null context immediately.
    // Use a blocked worker thread so the send+wait task is queued (simulating "busy/full" pool).
    auto old_pool = connector_->wait_done_thread_pool_;

    connector_->wait_done_thread_pool_ = std::make_shared<autil::LockFreeThreadPool>(/*thread_num=*/1,
                                                                                     /*queue_size=*/10,
                                                                                     /*thread_init_func=*/nullptr,
                                                                                     /*name=*/"AsyncWriteFullTP");
    ASSERT_TRUE(connector_->wait_done_thread_pool_->start());

    std::atomic<bool> block_worker{true};
    ASSERT_EQ(connector_->wait_done_thread_pool_->pushTask([&]() {
        while (block_worker.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }),
              autil::ThreadPoolBase::ERROR_NONE);

    const int                     layer0 = 0;
    std::vector<int64_t>          cache_keys{71001, 71002, 71003};
    std::vector<std::vector<int>> lbs_vec{{1, 2, 3}};
    auto                          res = makeCacheResource(cache_keys, lbs_vec);

    // Pre-insert one key so cpu_matched_num < cache_keys.size() and it reaches the thread-pool-full check.
    const auto   buf   = allocator_->convertIndexToBuffer(layer0, /*block_id=*/1);
    const size_t total = buf.kv_addr->sizeBytes();
    auto         pool  = ensureBlockPool(total);
    (void)putItemsToCache({cache_keys[0]}, total);

    // Force failure when the queued task eventually runs.
    connector_->tp_broadcast_manager_->worker_addrs_.clear();
    auto ctx = connector_->asyncWrite(res, nullptr);
    ASSERT_NE(ctx, nullptr);

    block_worker.store(false);
    auto mem_ctx = std::dynamic_pointer_cast<rtp_llm::MemoryConnectorAsyncContext>(ctx);
    ASSERT_NE(mem_ctx, nullptr);
    mem_ctx->waitDone();
    EXPECT_FALSE(ctx->success());

    connector_->wait_done_thread_pool_->stop();
    connector_->wait_done_thread_pool_.reset();
    connector_->wait_done_thread_pool_ = old_pool;
}

TEST_F(KVCacheMemoryConnectorTest, sendCopyPlan_ReturnNull_NoWorkers) {
    // 模拟没有worker
    connector_->tp_broadcast_manager_->worker_addrs_.clear();

    std::vector<KVCacheMemoryConnector::CopyInfoPerKey> infos;
    auto result = connector_->sendCopyPlan(infos, KVCacheMemoryConnector::CopyDirection::H2D);
    EXPECT_EQ(result, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, sendCopyPlan_ReturnContext_AllRanksSuccess) {
    const int  layer_id      = 0;
    const int  gpu_block_idx = 2;
    const auto gpu_buf       = allocator_->convertIndexToBuffer(layer_id, gpu_block_idx);
    ASSERT_NE(gpu_buf.kv_addr, nullptr);
    // ASSERT_NE(gpu_buf.v_addr, nullptr);
    const size_t total = gpu_buf.kv_addr->sizeBytes();

    KVCacheMemoryConnector::CopyInfoPerKey info;
    info.cache_key        = 1;
    info.mem_block_index  = 1;
    info.mem_block_size   = total;
    info.gpu_layer_blocks = {KVCacheMemoryConnector::LayerBlock{layer_id, gpu_block_idx}};
    std::vector<KVCacheMemoryConnector::CopyInfoPerKey> infos{info};

    auto result = connector_->sendCopyPlan(infos, KVCacheMemoryConnector::CopyDirection::H2D);
    ASSERT_NE(result, nullptr);
    result->waitDone();
    EXPECT_TRUE(result->success());
    const auto responses = result->responses();
    EXPECT_EQ(responses.size(), server_addrs_.size());
    for (const auto& response : responses) {
        EXPECT_TRUE(response.has_mem_response());
        EXPECT_TRUE(response.mem_response().success());
    }
}

TEST_F(KVCacheMemoryConnectorTest, sendCopyPlan_ReturnContext_PartialRanksFail) {
    std::vector<std::unique_ptr<TestRpcServer>> servers;
    std::vector<std::string>                    addrs;
    for (int i = 0; i < 4; ++i) {
        auto service = std::make_unique<TestRpcService>();
        // 只有偶数rank response成功
        service->setMemResponseSuccess(i % 2 == 0);
        auto server = std::make_unique<TestRpcServer>(std::move(service));
        ASSERT_TRUE(server->start());
        addrs.push_back("127.0.0.1:" + std::to_string(server->listenPort()));
        servers.push_back(std::move(server));
    }
    auto broadcast_manager = std::make_shared<TpBroadcastManager>(addrs);
    ASSERT_TRUE(broadcast_manager->init());
    connector_->tp_broadcast_manager_ = broadcast_manager;

    const int  layer_id      = 0;
    const int  gpu_block_idx = 2;
    const auto gpu_buf       = allocator_->convertIndexToBuffer(layer_id, gpu_block_idx);
    ASSERT_NE(gpu_buf.kv_addr, nullptr);
    // ASSERT_NE(gpu_buf.v_addr, nullptr);
    const size_t total = gpu_buf.kv_addr->sizeBytes();

    KVCacheMemoryConnector::CopyInfoPerKey info;
    info.cache_key        = 2;
    info.mem_block_index  = 1;
    info.mem_block_size   = total;
    info.gpu_layer_blocks = {KVCacheMemoryConnector::LayerBlock{layer_id, gpu_block_idx}};
    std::vector<KVCacheMemoryConnector::CopyInfoPerKey> infos{info};

    auto result = connector_->sendCopyPlan(infos, KVCacheMemoryConnector::CopyDirection::H2D);
    ASSERT_NE(result, nullptr);

    result->waitDone();
    EXPECT_TRUE(result->success());
    const auto responses = result->responses();
    EXPECT_EQ(responses.size(), addrs.size());
    for (size_t i = 0; i < responses.size(); ++i) {
        EXPECT_TRUE(responses[i].has_mem_response());
        if (i % 2 == 0) {
            EXPECT_TRUE(responses[i].mem_response().success());
        } else {
            EXPECT_FALSE(responses[i].mem_response().success());
        }
    }
    connector_->tp_broadcast_manager_.reset();
    for (auto& server : servers) {
        server->shutdown();
    }
    servers.clear();
    addrs.clear();
}

TEST_F(KVCacheMemoryConnectorTest, sendCopyPlan_ReturnContext_RpcStatusError) {
    std::vector<std::unique_ptr<TestRpcServer>> servers;
    std::vector<std::string>                    addrs;
    for (int i = 0; i < 4; ++i) {
        auto service = std::make_unique<TestRpcService>();
        // 只有偶数rank返回rpc状态失败
        if (i % 2 == 0) {
            service->setRpcResponseStatus(::grpc::Status(::grpc::StatusCode::UNAVAILABLE, "down"));
        }
        auto server = std::make_unique<TestRpcServer>(std::move(service));
        ASSERT_TRUE(server->start());
        addrs.push_back("127.0.0.1:" + std::to_string(server->listenPort()));
        servers.push_back(std::move(server));
    }
    auto broadcast_manager = std::make_shared<TpBroadcastManager>(addrs);
    ASSERT_TRUE(broadcast_manager->init());
    connector_->tp_broadcast_manager_ = broadcast_manager;

    const int  layer_id      = 0;
    const int  gpu_block_idx = 2;
    const auto gpu_buf       = allocator_->convertIndexToBuffer(layer_id, gpu_block_idx);
    ASSERT_NE(gpu_buf.kv_addr, nullptr);
    // ASSERT_NE(gpu_buf.v_addr, nullptr);
    const size_t total = gpu_buf.kv_addr->sizeBytes();

    KVCacheMemoryConnector::CopyInfoPerKey info;
    info.cache_key        = 3;
    info.mem_block_index  = 1;
    info.mem_block_size   = total;
    info.gpu_layer_blocks = {KVCacheMemoryConnector::LayerBlock{layer_id, gpu_block_idx}};
    std::vector<KVCacheMemoryConnector::CopyInfoPerKey> infos{info};

    auto result = connector_->sendCopyPlan(infos, KVCacheMemoryConnector::CopyDirection::H2D);
    ASSERT_NE(result, nullptr);
    result->waitDone();
    EXPECT_FALSE(result->success());
}

TEST_F(KVCacheMemoryConnectorTest, copyCache_ReturnFalse_CountMismatch) {
    MemoryOperationRequestPB req;
    auto*                    gb = req.add_gpu_blocks();
    auto*                    lb = gb->add_layer_blocks();
    lb->set_layer_id(0);
    lb->set_block_id(1);
    // Intentionally do not add mem_block_ids or mem_block_sizes to trigger mismatch
    req.set_copy_direction(MemoryOperationRequestPB::H2D);

    MemoryOperationResponsePB resp;
    auto                      ok = connector_->copyCache(req, resp);
    EXPECT_FALSE(ok);
    EXPECT_FALSE(resp.success());
}

TEST_F(KVCacheMemoryConnectorTest, copyCache_ReturnFalse_InvalidMemBlockId) {
    const int  layer_id      = 0;
    const int  gpu_block_idx = 1;
    const auto gpu_buf       = allocator_->convertIndexToBuffer(layer_id, gpu_block_idx);
    ASSERT_NE(gpu_buf.kv_addr, nullptr);
    // ASSERT_NE(gpu_buf.v_addr, nullptr);
    const size_t total = gpu_buf.kv_addr->sizeBytes();

    MemoryOperationRequestPB req;
    auto*                    gb = req.add_gpu_blocks();
    auto*                    lb = gb->add_layer_blocks();
    lb->set_layer_id(layer_id);
    lb->set_block_id(gpu_block_idx);
    req.add_mem_block_ids(-1);  // invalid mem block id
    req.add_mem_block_sizes(static_cast<int64_t>(total));
    req.set_copy_direction(MemoryOperationRequestPB::H2D);

    MemoryOperationResponsePB resp;
    auto                      ok = connector_->copyCache(req, resp);
    EXPECT_FALSE(ok);
    EXPECT_FALSE(resp.success());
}

TEST_F(KVCacheMemoryConnectorTest, copyCache_ReturnFalse_InvalidMemBlockSize) {
    const int  layer_id      = 0;
    const int  gpu_block_idx = 1;
    const auto gpu_buf       = allocator_->convertIndexToBuffer(layer_id, gpu_block_idx);
    ASSERT_NE(gpu_buf.kv_addr, nullptr);
    // ASSERT_NE(gpu_buf.v_addr, nullptr);

    MemoryOperationRequestPB req;
    auto*                    gb = req.add_gpu_blocks();
    auto*                    lb = gb->add_layer_blocks();
    lb->set_layer_id(layer_id);
    lb->set_block_id(gpu_block_idx);
    req.add_mem_block_ids(1);
    req.add_mem_block_sizes(0);  // invalid mem block size
    req.set_copy_direction(MemoryOperationRequestPB::H2D);

    MemoryOperationResponsePB resp;
    auto                      ok = connector_->copyCache(req, resp);
    EXPECT_FALSE(ok);
    EXPECT_FALSE(resp.success());
}

TEST_F(KVCacheMemoryConnectorTest, copyCache_ReturnFalse_InvalidLayerId_BuildCopyPlanFailed) {
    const int  valid_layer   = 0;
    const int  gpu_block_idx = 1;
    const auto gpu_buf       = allocator_->convertIndexToBuffer(valid_layer, gpu_block_idx);
    ASSERT_NE(gpu_buf.kv_addr, nullptr);
    // ASSERT_NE(gpu_buf.v_addr, nullptr);
    const size_t total = gpu_buf.kv_addr->sizeBytes();

    auto pool = ensureBlockPool(total);
    ASSERT_NE(pool, nullptr);
    auto mem_blocks = pool->malloc(1);
    ASSERT_EQ(mem_blocks.size(), 1u);
    const int mem_block_index = mem_blocks[0];

    MemoryOperationRequestPB req;
    auto*                    gb = req.add_gpu_blocks();
    auto*                    lb = gb->add_layer_blocks();
    lb->set_layer_id(cache_config_.layer_num);  // out of range
    lb->set_block_id(gpu_block_idx);
    req.add_mem_block_ids(mem_block_index);
    req.add_mem_block_sizes(static_cast<int64_t>(total));
    req.set_copy_direction(MemoryOperationRequestPB::H2D);

    MemoryOperationResponsePB resp;
    auto                      ok = connector_->copyCache(req, resp);
    EXPECT_FALSE(ok);
    EXPECT_FALSE(resp.success());
}

TEST_F(KVCacheMemoryConnectorTest, copyCache_ReturnTrue_H2D_SingleLayer) {
    const int  layer_id      = 0;
    const int  gpu_block_idx = 2;
    const auto gpu_buf       = allocator_->convertIndexToBuffer(layer_id, gpu_block_idx);
    ASSERT_NE(gpu_buf.kv_addr, nullptr);
    // ASSERT_NE(gpu_buf.v_addr, nullptr);
    const size_t total = gpu_buf.kv_addr->sizeBytes();

    // H2D 路径需要预先存在 mem pool 与有效 block
    auto pool = ensureBlockPool(total);
    ASSERT_NE(pool, nullptr);
    auto mem_blocks = pool->malloc(1);
    ASSERT_EQ(mem_blocks.size(), 1u);
    const int mem_block_index = mem_blocks[0];
    auto      mem_buffer      = pool->convertIndexToBuffer(0, mem_block_index);
    ASSERT_NE(mem_buffer.kv_addr, nullptr);
    EXPECT_EQ(mem_buffer.kv_addr->sizeBytes(), total);

    // 给mem_buffer填充数据
    size_t offset = 0;
    setBufferContent(mem_buffer.kv_addr, offset, gpu_buf.kv_addr->sizeBytes(), 'a');
    offset += gpu_buf.kv_addr->sizeBytes();
    // setBufferContent(mem_buffer.kv_addr, offset, gpu_buf.v_addr->sizeBytes(), 'b');

    MemoryOperationRequestPB req;
    auto*                    gb = req.add_gpu_blocks();
    auto*                    lb = gb->add_layer_blocks();
    lb->set_layer_id(layer_id);
    lb->set_block_id(gpu_block_idx);
    req.add_mem_block_ids(mem_block_index);
    req.add_mem_block_sizes(static_cast<int64_t>(total));
    req.set_copy_direction(MemoryOperationRequestPB::H2D);

    MemoryOperationResponsePB resp;
    auto                      ok = connector_->copyCache(req, resp);
    EXPECT_TRUE(ok);
    EXPECT_TRUE(resp.success());

    // H2D, 验证数据是否拷贝成功
    verifyBufferContent(gpu_buf.kv_addr, 'a');
}

TEST_F(KVCacheMemoryConnectorTest, copyCache_ReturnTrue_H2D_MultiLayer) {
    // 创建两个block_size不同的memory buffer
    std::vector<KVCacheMemoryConnector::LayerBlock> gpu_layer_blocks1{
        {/*layer_id*/ 0, /*block_id*/ 1},
        {/*layer_id*/ 1, /*block_id*/ 2},
    };
    int    mem_block_index1 = -1;
    size_t mem_block_size1  = 0;
    prepareBufferContent(gpu_layer_blocks1, mem_block_index1, mem_block_size1, /*fill_gpu=*/false, /*fill_cpu=*/true);
    ASSERT_NE(mem_block_index1, -1);
    ASSERT_NE(mem_block_size1, 0);

    std::vector<KVCacheMemoryConnector::LayerBlock> gpu_layer_blocks2{
        {/*layer_id*/ 0, /*block_id*/ 1},
        {/*layer_id*/ 1, /*block_id*/ 2},
        {/*layer_id*/ 2, /*block_id*/ 2},
    };
    int    mem_block_index2 = -1;
    size_t mem_block_size2  = 0;
    prepareBufferContent(gpu_layer_blocks2, mem_block_index2, mem_block_size2, /*fill_gpu=*/false, /*fill_cpu=*/true);
    ASSERT_NE(mem_block_index2, -1);
    ASSERT_NE(mem_block_size2, 0);

    MemoryOperationRequestPB req;
    req.set_copy_direction(MemoryOperationRequestPB::H2D);
    addOneCopyInfoToPb(req, gpu_layer_blocks1, mem_block_index1, mem_block_size1);
    addOneCopyInfoToPb(req, gpu_layer_blocks2, mem_block_index2, mem_block_size2);

    MemoryOperationResponsePB resp;
    auto                      ok = connector_->copyCache(req, resp);
    EXPECT_TRUE(ok);
    EXPECT_TRUE(resp.success());

    // H2D, 验证数据是否拷贝成功
    verifyGpuBufferContent(gpu_layer_blocks1);
    verifyGpuBufferContent(gpu_layer_blocks2);
}

TEST_F(KVCacheMemoryConnectorTest, copyCache_ReturnTrue_D2H_SingleLayer) {
    const int  layer_id      = 0;
    const int  gpu_block_idx = 3;
    const auto gpu_buf       = allocator_->convertIndexToBuffer(layer_id, gpu_block_idx);
    ASSERT_NE(gpu_buf.kv_addr, nullptr);
    // ASSERT_NE(gpu_buf.v_addr, nullptr);
    const size_t total = gpu_buf.kv_addr->sizeBytes();

    // 给gpu_buf填充数据
    setBufferContent(gpu_buf.kv_addr, 'a');
    // setBufferContent(gpu_buf.v_addr, 'b');

    // 为确保索引有效，仍然预先创建并分配一个块
    auto pool = ensureBlockPool(total);
    ASSERT_NE(pool, nullptr);
    auto mem_blocks = pool->malloc(1);
    ASSERT_EQ(mem_blocks.size(), 1u);
    const int mem_block_index = mem_blocks[0];
    auto      mem_buffer      = pool->convertIndexToBuffer(0, mem_block_index);
    ASSERT_NE(mem_buffer.kv_addr, nullptr);
    EXPECT_EQ(mem_buffer.kv_addr->sizeBytes(), total);

    MemoryOperationRequestPB req;
    auto*                    gb = req.add_gpu_blocks();
    auto*                    lb = gb->add_layer_blocks();
    lb->set_layer_id(layer_id);
    lb->set_block_id(gpu_block_idx);
    req.add_mem_block_ids(mem_block_index);
    req.add_mem_block_sizes(static_cast<int64_t>(total));
    req.set_copy_direction(MemoryOperationRequestPB::D2H);

    MemoryOperationResponsePB resp;
    auto                      ok = connector_->copyCache(req, resp);
    EXPECT_TRUE(ok);
    EXPECT_TRUE(resp.success());

    // D2H, 验证数据是否拷贝成功
    verifyBufferContent(mem_buffer.kv_addr, 0, gpu_buf.kv_addr->sizeBytes(), 'a');
}

TEST_F(KVCacheMemoryConnectorTest, copyCache_ReturnTrue_D2H_MultiLayer) {
    std::vector<KVCacheMemoryConnector::LayerBlock> gpu_layer_blocks1{
        {/*layer_id*/ 0, /*block_id*/ 1},
        {/*layer_id*/ 1, /*block_id*/ 2},
    };
    int    mem_block_index1 = -1;
    size_t mem_block_size1  = 0;
    prepareBufferContent(gpu_layer_blocks1, mem_block_index1, mem_block_size1, /*fill_gpu=*/true, /*fill_cpu=*/false);
    ASSERT_NE(mem_block_index1, -1);
    ASSERT_NE(mem_block_size1, 0);

    std::vector<KVCacheMemoryConnector::LayerBlock> gpu_layer_blocks2{
        {/*layer_id*/ 0, /*block_id*/ 1},
        {/*layer_id*/ 1, /*block_id*/ 2},
        {/*layer_id*/ 2, /*block_id*/ 2},
    };
    int    mem_block_index2 = -1;
    size_t mem_block_size2  = 0;
    prepareBufferContent(gpu_layer_blocks2, mem_block_index2, mem_block_size2, /*fill_gpu=*/true, /*fill_cpu=*/false);
    ASSERT_NE(mem_block_index2, -1);
    ASSERT_NE(mem_block_size2, 0);

    MemoryOperationRequestPB req;
    req.set_copy_direction(MemoryOperationRequestPB::D2H);
    addOneCopyInfoToPb(req, gpu_layer_blocks1, mem_block_index1, mem_block_size1);
    addOneCopyInfoToPb(req, gpu_layer_blocks2, mem_block_index2, mem_block_size2);

    MemoryOperationResponsePB resp;
    auto                      ok = connector_->copyCache(req, resp);
    EXPECT_TRUE(ok);
    EXPECT_TRUE(resp.success());

    // D2H, 验证数据是否拷贝成功
    verifyCpuBufferContent(gpu_layer_blocks1, mem_block_index1, mem_block_size1);
    verifyCpuBufferContent(gpu_layer_blocks2, mem_block_index2, mem_block_size2);
}

// Regression test: verifies multi-layer D2H copy uses correct byte offsets even when memory buffer dtype size != 1.
// This would fail if prepareCopyBuffers mistakenly treats bytes as elements in Buffer::slice().
TEST_F(KVCacheMemoryConnectorTest, copyCache_D2H_MultiLayer_ValidatesByteOffsets) {
    std::vector<KVCacheMemoryConnector::LayerBlock> gpu_layer_blocks{
        {/*layer_id*/ 0, /*block_id*/ 1},
        {/*layer_id*/ 1, /*block_id*/ 2},
    };

    // Fill GPU source with distinct byte patterns.
    for (const auto& lb : gpu_layer_blocks) {
        const auto gpu_buf = allocator_->convertIndexToBuffer(lb.layer_id, lb.block_id);
        ASSERT_NE(gpu_buf.kv_addr, nullptr);
        setBufferBytes(
            gpu_buf.kv_addr, /*byte_offset=*/0, gpu_buf.kv_addr->sizeBytes(), static_cast<char>('k' + lb.layer_id));
    }

    // Allocate one memory block sized to hold both layers' kv bytes contiguously.
    size_t total_bytes = 0;
    for (const auto& lb : gpu_layer_blocks) {
        const auto gpu_buf = allocator_->convertIndexToBuffer(lb.layer_id, lb.block_id);
        total_bytes += gpu_buf.kv_addr->sizeBytes();
    }
    ASSERT_GT(total_bytes, 0u);

    auto pool = ensureBlockPool(total_bytes);
    ASSERT_NE(pool, nullptr);
    auto mem_blocks = pool->malloc(1);
    ASSERT_EQ(mem_blocks.size(), 1u);
    const int mem_block_index = mem_blocks[0];

    auto mem_buffer = pool->convertIndexToBuffer(0, mem_block_index);
    ASSERT_NE(mem_buffer.kv_addr, nullptr);
    EXPECT_EQ(mem_buffer.kv_addr->sizeBytes(), total_bytes);
    setBufferBytes(mem_buffer.kv_addr, /*byte_offset=*/0, total_bytes, 0);

    MemoryOperationRequestPB req;
    req.set_copy_direction(MemoryOperationRequestPB::D2H);
    addOneCopyInfoToPb(req, gpu_layer_blocks, mem_block_index, total_bytes);

    MemoryOperationResponsePB resp;
    const auto                ok = connector_->copyCache(req, resp);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(resp.success());

    // Validate two segments land at correct byte offsets.
    size_t byte_off = 0;
    for (const auto& lb : gpu_layer_blocks) {
        const auto gpu_buf = allocator_->convertIndexToBuffer(lb.layer_id, lb.block_id);
        const auto bytes   = gpu_buf.kv_addr->sizeBytes();
        verifyBufferBytesEq(mem_buffer.kv_addr, byte_off, bytes, static_cast<char>('k' + lb.layer_id));
        byte_off += bytes;
    }
}

}  // namespace rtp_llm::test

int main(int argc, char** argv) {
    rtp_llm::initLogger();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
