// Copyright (c) RTP-LLM

#include <csignal>
#include <chrono>
#include <execinfo.h>
#include <thread>
#include <unistd.h>

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

class MemoryConnectorAsyncContextTest: public ::testing::Test {};

TEST_F(MemoryConnectorAsyncContextTest, success_ReturnFalse_WhenBroadcastResultNotSuccess) {
    using ResultT = rtp_llm::TPBroadcastResult<BroadcastTpRequestPB, BroadcastTpResponsePB>;
    using CtxT    = typename ResultT::WorkerRpcContext;

    auto worker0 = std::make_shared<CtxT>();
    worker0->response.mutable_mem_response()->set_success(true);

    auto result                  = std::make_shared<ResultT>(std::vector<std::shared_ptr<CtxT>>{worker0});
    result->all_request_success_ = false;

    auto ctx = std::make_shared<rtp_llm::MemoryConnectorAsyncContext>(result, /*done_callback=*/nullptr);
    // Avoid blocking in ~MemoryConnectorAsyncContext()->waitDone() since this TPBroadcastResult can never complete.
    ctx->already_done_ = true;
    EXPECT_FALSE(ctx->success());
}

TEST_F(MemoryConnectorAsyncContextTest, success_ReturnFalse_WhenAnyResponseMissingMemResponse) {
    using ResultT = rtp_llm::TPBroadcastResult<BroadcastTpRequestPB, BroadcastTpResponsePB>;
    using CtxT    = typename ResultT::WorkerRpcContext;

    auto worker0                 = std::make_shared<CtxT>();  // default: no mem_response
    auto result                  = std::make_shared<ResultT>(std::vector<std::shared_ptr<CtxT>>{worker0});
    result->all_request_success_ = true;

    auto ctx = std::make_shared<rtp_llm::MemoryConnectorAsyncContext>(result, /*done_callback=*/nullptr);
    // Avoid blocking in ~MemoryConnectorAsyncContext()->waitDone() since this TPBroadcastResult can never complete.
    ctx->already_done_ = true;
    EXPECT_FALSE(ctx->success());
}

TEST_F(MemoryConnectorAsyncContextTest, success_ReturnFalse_WhenAnyMemResponseFailed) {
    using ResultT = rtp_llm::TPBroadcastResult<BroadcastTpRequestPB, BroadcastTpResponsePB>;
    using CtxT    = typename ResultT::WorkerRpcContext;

    auto worker0 = std::make_shared<CtxT>();
    worker0->response.mutable_mem_response()->set_success(true);
    auto worker1 = std::make_shared<CtxT>();
    worker1->response.mutable_mem_response()->set_success(false);

    auto result                  = std::make_shared<ResultT>(std::vector<std::shared_ptr<CtxT>>{worker0, worker1});
    result->all_request_success_ = true;

    auto ctx = std::make_shared<rtp_llm::MemoryConnectorAsyncContext>(result, /*done_callback=*/nullptr);
    // Avoid blocking in ~MemoryConnectorAsyncContext()->waitDone() since this TPBroadcastResult can never complete.
    ctx->already_done_ = true;
    EXPECT_FALSE(ctx->success());
}

TEST_F(MemoryConnectorAsyncContextTest, success_ReturnTrue_WhenAllResponsesSuccess) {
    using ResultT = rtp_llm::TPBroadcastResult<BroadcastTpRequestPB, BroadcastTpResponsePB>;
    using CtxT    = typename ResultT::WorkerRpcContext;

    auto worker0 = std::make_shared<CtxT>();
    worker0->response.mutable_mem_response()->set_success(true);
    auto worker1 = std::make_shared<CtxT>();
    worker1->response.mutable_mem_response()->set_success(true);

    auto result                  = std::make_shared<ResultT>(std::vector<std::shared_ptr<CtxT>>{worker0, worker1});
    result->all_request_success_ = true;

    auto ctx = std::make_shared<rtp_llm::MemoryConnectorAsyncContext>(result, /*done_callback=*/nullptr);
    // Avoid blocking in ~MemoryConnectorAsyncContext()->waitDone() since this TPBroadcastResult can never complete.
    ctx->already_done_ = true;
    EXPECT_TRUE(ctx->success());
}

TEST_F(MemoryConnectorAsyncContextTest, waitDone_ReturnVoid_WhenBroadcastResultNullAndCallbackCalledOnce) {
    int  callback_cnt = 0;
    bool last_ok      = true;
    auto cb           = [&](bool ok) {
        callback_cnt++;
        last_ok = ok;
    };

    auto ctx = std::make_shared<rtp_llm::MemoryConnectorAsyncContext>(
        /*broadcast_result=*/nullptr, /*done_callback=*/cb);
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

TEST_F(MemoryConnectorAsyncContextTest, waitDone_ReturnVoid_WhenBroadcastResultNonNullAndCallbackReceivesSuccess) {
    using ResultT = rtp_llm::TPBroadcastResult<BroadcastTpRequestPB, BroadcastTpResponsePB>;
    using CtxT    = typename ResultT::WorkerRpcContext;

    // Empty worker contexts => TPBroadcastResult::waitDone() returns immediately and sets all_request_success_ = true.
    auto result = std::make_shared<ResultT>(std::vector<std::shared_ptr<CtxT>>{});

    int  callback_cnt = 0;
    bool last_ok      = false;
    auto cb           = [&](bool ok) {
        callback_cnt++;
        last_ok = ok;
    };

    auto ctx = std::make_shared<rtp_llm::MemoryConnectorAsyncContext>(result, cb);
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
    using ResultT = rtp_llm::TPBroadcastResult<BroadcastTpRequestPB, BroadcastTpResponsePB>;
    using CtxT    = typename ResultT::WorkerRpcContext;

    auto result = std::make_shared<ResultT>(std::vector<std::shared_ptr<CtxT>>{});
    auto ctx    = std::make_shared<rtp_llm::MemoryConnectorAsyncContext>(result, /*done_callback=*/nullptr);

    EXPECT_FALSE(ctx->done());
    EXPECT_FALSE(ctx->success());

    ctx->waitDone();
    EXPECT_TRUE(ctx->done());
    EXPECT_TRUE(ctx->success());
}

TEST_F(MemoryConnectorAsyncContextTest, waitDone_ReturnVoid_WhenAlreadyDone_ReturnsEarlyWithoutCallback) {
    using ResultT = rtp_llm::TPBroadcastResult<BroadcastTpRequestPB, BroadcastTpResponsePB>;
    using CtxT    = typename ResultT::WorkerRpcContext;

    auto result = std::make_shared<ResultT>(std::vector<std::shared_ptr<CtxT>>{});

    int  callback_cnt = 0;
    auto cb           = [&](bool /*ok*/) { callback_cnt++; };

    auto ctx           = std::make_shared<rtp_llm::MemoryConnectorAsyncContext>(result, cb);
    ctx->already_done_ = true;

    ctx->waitDone();
    EXPECT_EQ(callback_cnt, 0);
    EXPECT_TRUE(ctx->done());
    EXPECT_FALSE(ctx->success());                // broadcast_result_ was not waited.
    EXPECT_FALSE(result->all_request_success_);  // TPBroadcastResult::waitDone() not invoked.
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
        config.block_size = static_cast<int>(mha_spec->block_size());

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
    void addOneCopyInfoToPb(MemoryBroadcastTpRequestPB&                            req,
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
            pool = connector_->createBlockPool(block_size);
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
    // 构造空的 worker 地址，init 应返回 false，tp_broadcast_manager_ 应为空
    std::vector<std::string> empty_addrs;
    auto                     conn =
        std::make_shared<KVCacheMemoryConnector>(cache_config_, kv_cache_config_, allocator_, device_, empty_addrs);
    auto ok = conn->init();
    EXPECT_FALSE(ok);
    ASSERT_NE(conn->block_cache_, nullptr);
    EXPECT_EQ(conn->tp_broadcast_manager_, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, init_ReturnFalse_WhenMemoryCacheSizeMbZero) {
    auto kv_cfg                         = kv_cache_config_;
    kv_cfg.memory_cache_size_mb         = 0;
    kv_cfg.memory_cache_sync_timeout_ms = 1000;

    auto conn = std::make_shared<KVCacheMemoryConnector>(cache_config_, kv_cfg, allocator_, device_, server_addrs_);
    auto ok   = conn->init();
    EXPECT_FALSE(ok);
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
    auto ok   = conn->init();
    EXPECT_FALSE(ok);
    // Init fails early, nothing should be created.
    EXPECT_EQ(conn->block_cache_, nullptr);
    EXPECT_EQ(conn->tp_broadcast_manager_, nullptr);
    EXPECT_EQ(conn->wait_done_thread_pool_, nullptr);
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

TEST_F(KVCacheMemoryConnectorTest, init_Reinit_ClearsBlockPools_And_ResetsBlockCache) {
    // 预先创建一个 block pool，并向 block_cache_ 放入条目
    const size_t block_size                       = 4096;
    kv_cache_config_.memory_cache_size_mb         = 64;
    kv_cache_config_.memory_cache_sync_timeout_ms = 1000;
    auto pool                                     = ensureBlockPool(block_size);
    ASSERT_NE(pool, nullptr);
    auto blocks = pool->malloc(1);
    ASSERT_EQ(blocks.size(), 1u);

    MemoryBlockCache::CacheItem item;
    item.cache_key   = 123456;
    item.block_index = blocks[0];
    item.block_size  = block_size;
    connector_->block_cache_->put(item);
    ASSERT_TRUE(connector_->block_cache_->contains(item.cache_key));

    // 重新 init，应清空 block_pools_ 并重置 block_cache_
    auto ok = connector_->init();
    EXPECT_TRUE(ok);
    EXPECT_EQ(connector_->tp_broadcast_manager_->workerNum(), server_addrs_.size());

    // 原 block pool 不应再可见
    auto pool_after = connector_->getBlockPool(block_size);
    EXPECT_EQ(pool_after, nullptr);
    // block_cache_ 应被重置为空
    EXPECT_EQ(connector_->block_cache_->size(), 0u);
    EXPECT_FALSE(connector_->block_cache_->contains(item.cache_key));
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
    res_empty_lbs->layer_block_ids.clear();  // make it truly invalid for KVCacheMemoryConnector::checkKVCacheResource
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
    EXPECT_EQ(ctx, nullptr);
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
    // Make thread pool full, asyncRead should fail early before building copy plan / sending RPC.
    auto old_pool = connector_->wait_done_thread_pool_;

    connector_->wait_done_thread_pool_ = std::make_shared<autil::LockFreeThreadPool>(/*thread_num=*/1,
                                                                                     /*queue_size=*/1,
                                                                                     /*thread_init_func=*/nullptr,
                                                                                     /*name=*/"AsyncReadFullTP");
    ASSERT_TRUE(connector_->wait_done_thread_pool_->start());

    std::atomic<bool> block_worker{true};
    std::atomic<bool> first_task_started{false};
    ASSERT_EQ(connector_->wait_done_thread_pool_->pushTask([&]() {
        first_task_started.store(true);
        while (block_worker.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }),
              autil::ThreadPoolBase::ERROR_NONE);
    const auto started_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < started_deadline && !first_task_started.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(first_task_started.load());
    (void)connector_->wait_done_thread_pool_->pushTask([&]() {
        while (block_worker.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    const auto full_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < full_deadline && !connector_->isThreadPoolFull()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(connector_->isThreadPoolFull());

    std::vector<int64_t> cache_keys{70001, 70002};
    const auto           buf      = allocator_->convertIndexToBuffer(0, 0);
    const size_t         mem_size = buf.kv_addr->sizeBytes();
    putItemsToCache(cache_keys, mem_size);
    auto res       = makeCacheResource(cache_keys, {{1, 2}, {3, 4}});
    auto match_ctx = connector_->asyncMatch(res, nullptr);
    ASSERT_NE(match_ctx, nullptr);
    auto meta = std::make_shared<TestReadMeta>(/*start_block_index=*/0, /*size=*/(int)match_ctx->matchedBlockCount());
    auto ctx  = connector_->asyncRead(res, meta, match_ctx);
    EXPECT_EQ(ctx, nullptr);

    block_worker.store(false);
    connector_->wait_done_thread_pool_->stop();
    connector_->wait_done_thread_pool_.reset();
    connector_->wait_done_thread_pool_ = old_pool;
}

TEST_F(KVCacheMemoryConnectorTest, buildCopyPlanForRead_ReturnEmpty_WhenNoMatch) {
    // block_cache_ 无命中，返回空
    std::vector<int64_t>          cache_keys{1, 2};
    std::vector<std::vector<int>> lbs_vec{{10, 11}, {20, 21}};
    auto                          lbs  = makeLayerBlockIds(lbs_vec, cache_keys.size());
    auto                          plan = connector_->buildCopyPlanForRead(
        cache_keys, lbs, /*start_read_block_index=*/0, /*read_block_num=*/static_cast<int>(cache_keys.size()));
    EXPECT_TRUE(plan.empty());
}

TEST_F(KVCacheMemoryConnectorTest, buildCopyPlanForRead_PrefixMatch_StopsOnFirstMiss) {
    // 仅前缀两个 key 命中，第三个未命中，应在未命中处停止
    std::vector<int64_t> cache_keys{10, 11, 12};
    // 预置命中项
    const auto   buf      = allocator_->convertIndexToBuffer(0, 0);
    const size_t mem_size = buf.kv_addr->sizeBytes();

    auto block_indices = putItemsToCache({10, 11}, mem_size);
    ASSERT_EQ(block_indices.size(), 2u);

    ASSERT_TRUE(connector_->block_cache_->contains(10));
    ASSERT_TRUE(connector_->block_cache_->contains(11));
    ASSERT_FALSE(connector_->block_cache_->contains(12));

    // 两层，不同的 block 索引
    std::vector<std::vector<int>> lbs_vec{
        {100, 101, 102},  // layer0
        {200, 201, 202},  // layer1
    };
    auto lbs = makeLayerBlockIds(lbs_vec, cache_keys.size());

    // 注意：buildCopyPlanForRead 的真实签名是 (start_read_block_index, read_block_num)，并且“范围内任一 key
    // 未命中”会导致返回空。 这里单独验证 miss（只请求 miss 的那个 key），以及 prefix 范围内的成功。
    auto plan_miss =
        connector_->buildCopyPlanForRead(cache_keys, lbs, /*start_read_block_index=*/2, /*read_block_num=*/1);
    EXPECT_TRUE(plan_miss.empty());

    auto plan = connector_->buildCopyPlanForRead(cache_keys, lbs, /*start_read_block_index=*/0, /*read_block_num=*/2);
    ASSERT_EQ(plan.size(), 2u);

    // key=10
    EXPECT_EQ(plan[0].cache_key, 10u);
    EXPECT_EQ(plan[0].mem_block_size, mem_size);
    EXPECT_EQ(plan[0].mem_block_index, block_indices[0]);
    ASSERT_EQ(plan[0].gpu_layer_blocks.size(), 2u);
    EXPECT_EQ(plan[0].gpu_layer_blocks[0].layer_id, 0);
    EXPECT_EQ(plan[0].gpu_layer_blocks[0].block_id, 100);
    EXPECT_EQ(plan[0].gpu_layer_blocks[1].layer_id, 1);
    EXPECT_EQ(plan[0].gpu_layer_blocks[1].block_id, 200);

    // key=11
    EXPECT_EQ(plan[1].cache_key, 11u);
    EXPECT_EQ(plan[1].mem_block_size, mem_size);
    EXPECT_EQ(plan[1].mem_block_index, block_indices[1]);
    ASSERT_EQ(plan[1].gpu_layer_blocks.size(), 2u);
    EXPECT_EQ(plan[1].gpu_layer_blocks[0].layer_id, 0);
    EXPECT_EQ(plan[1].gpu_layer_blocks[0].block_id, 101);
    EXPECT_EQ(plan[1].gpu_layer_blocks[1].layer_id, 1);
    EXPECT_EQ(plan[1].gpu_layer_blocks[1].block_id, 201);
}

TEST_F(KVCacheMemoryConnectorTest, buildCopyPlanForRead_RespectsGpuReuseLen) {
    // 三个 key 均命中，但 gpu_reuse_len=2，仅返回最后一个
    std::vector<int64_t> cache_keys{30, 31, 32};
    const auto           buf      = allocator_->convertIndexToBuffer(0, 0);
    const size_t         mem_size = buf.kv_addr->sizeBytes();

    auto block_indices = putItemsToCache(cache_keys, mem_size);
    ASSERT_EQ(block_indices.size(), cache_keys.size());
    ASSERT_TRUE(connector_->block_cache_->contains(30));
    ASSERT_TRUE(connector_->block_cache_->contains(31));
    ASSERT_TRUE(connector_->block_cache_->contains(32));

    std::vector<std::vector<int>> lbs_vec{
        {10, 11, 12},  // layer0
        {20, 21, 22},  // layer1
    };
    auto lbs  = makeLayerBlockIds(lbs_vec, cache_keys.size());
    auto plan = connector_->buildCopyPlanForRead(cache_keys, lbs, /*start_read_block_index=*/2, /*read_block_num=*/1);
    ASSERT_EQ(plan.size(), 1u);
    EXPECT_EQ(plan[0].cache_key, 32u);
    EXPECT_EQ(plan[0].mem_block_size, mem_size);
    EXPECT_EQ(plan[0].mem_block_index, block_indices[2]);
    ASSERT_EQ(plan[0].gpu_layer_blocks.size(), 2u);
    EXPECT_EQ(plan[0].gpu_layer_blocks[0].block_id, 12);
    EXPECT_EQ(plan[0].gpu_layer_blocks[1].block_id, 22);
}

TEST_F(KVCacheMemoryConnectorTest, buildCopyPlanForRead_SkipsNullLayerBlocks) {
    // 命中一个 key，其中一层为 NULL_BLOCK_IDX，应被过滤
    std::vector<int64_t> cache_keys{40};
    const auto           buf      = allocator_->convertIndexToBuffer(0, 0);
    const size_t         mem_size = buf.kv_addr->sizeBytes();

    auto block_indices = putItemsToCache(cache_keys, mem_size);
    ASSERT_EQ(block_indices.size(), cache_keys.size());

    ASSERT_TRUE(connector_->block_cache_->contains(40));

    std::vector<std::vector<int>> lbs_vec{
        {7},               // layer0
        {NULL_BLOCK_IDX},  // layer1 should be skipped
    };
    auto lbs  = makeLayerBlockIds(lbs_vec, cache_keys.size());
    auto plan = connector_->buildCopyPlanForRead(cache_keys, lbs, /*start_read_block_index=*/0, /*read_block_num=*/1);
    ASSERT_EQ(plan.size(), 1u);
    EXPECT_EQ(plan[0].cache_key, 40u);
    EXPECT_EQ(plan[0].mem_block_size, mem_size);
    EXPECT_EQ(plan[0].mem_block_index, block_indices[0]);
    ASSERT_EQ(plan[0].gpu_layer_blocks.size(), 1u);
    EXPECT_EQ(plan[0].gpu_layer_blocks[0].layer_id, 0);
    EXPECT_EQ(plan[0].gpu_layer_blocks[0].block_id, 7);
}

TEST_F(KVCacheMemoryConnectorTest, buildCopyPlanForRead_AllLayersEmpty_StillReturnsMemInfo) {
    // 无层（layer_num=0），命中一个 key，仍应返回包含 mem 信息的条目且 gpu_layer_blocks 为空
    std::vector<int64_t> cache_keys{50};
    const auto           buf      = allocator_->convertIndexToBuffer(0, 0);
    const size_t         mem_size = buf.kv_addr->sizeBytes();

    auto block_indices = putItemsToCache(cache_keys, mem_size);
    ASSERT_EQ(block_indices.size(), cache_keys.size());

    ASSERT_TRUE(connector_->block_cache_->contains(50));

    std::vector<std::vector<int>> empty_layers;
    auto                          lbs = makeLayerBlockIds(empty_layers, cache_keys.size());
    auto plan = connector_->buildCopyPlanForRead(cache_keys, lbs, /*start_read_block_index=*/0, /*read_block_num=*/1);
    ASSERT_EQ(plan.size(), 1u);
    EXPECT_EQ(plan[0].cache_key, 50u);
    EXPECT_EQ(plan[0].mem_block_size, mem_size);
    EXPECT_EQ(plan[0].mem_block_index, block_indices[0]);
    EXPECT_TRUE(plan[0].gpu_layer_blocks.empty());
}

TEST_F(KVCacheMemoryConnectorTest, buildCopyPlanForRead_UsesMemBlockSizeFromCache) {
    // 验证 mem_block_size 与 cache 中 put 的值一致，而非 GPU buffer 大小
    std::vector<int64_t> cache_keys{60};

    const auto   buf      = allocator_->convertIndexToBuffer(0, 0);
    const size_t mem_size = buf.kv_addr->sizeBytes() * 2;  // custom size

    auto block_indices = putItemsToCache(cache_keys, mem_size);
    ASSERT_EQ(block_indices.size(), cache_keys.size());

    ASSERT_TRUE(connector_->block_cache_->contains(60));

    std::vector<std::vector<int>> lbs_vec{
        {1},
    };
    auto lbs  = makeLayerBlockIds(lbs_vec, cache_keys.size());
    auto plan = connector_->buildCopyPlanForRead(cache_keys, lbs, /*start_read_block_index=*/0, /*read_block_num=*/1);
    ASSERT_EQ(plan.size(), 1u);
    EXPECT_EQ(plan[0].cache_key, 60u);
    EXPECT_EQ(plan[0].mem_block_size, mem_size);
    EXPECT_EQ(plan[0].mem_block_index, block_indices[0]);
    ASSERT_EQ(plan[0].gpu_layer_blocks.size(), 1u);
    EXPECT_EQ(plan[0].gpu_layer_blocks[0].layer_id, 0);
    EXPECT_EQ(plan[0].gpu_layer_blocks[0].block_id, 1);
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
    res_empty_lbs->layer_block_ids.clear();  // make it truly invalid for KVCacheMemoryConnector::checkKVCacheResource
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
    auto         pool  = ensureBlockPool(total);
    ASSERT_NE(pool, nullptr);

    const size_t free_before  = pool->freeBlocksNum();
    const size_t cache_before = connector_->block_cache_->size();

    // Pre-insert only the first key, so cpu_matched_num should be 1 and only suffix gets written.
    auto pre_blocks = putItemsToCache({cache_keys[0]}, total);
    ASSERT_EQ(pre_blocks.size(), 1u);
    ASSERT_TRUE(connector_->block_cache_->contains(cache_keys[0]));

    auto ctx = connector_->asyncWrite(res, nullptr);
    ASSERT_NE(ctx, nullptr);
    auto mem_ctx = std::dynamic_pointer_cast<rtp_llm::MemoryConnectorAsyncContext>(ctx);
    ASSERT_NE(mem_ctx, nullptr);
    mem_ctx->waitDone();
    EXPECT_TRUE(ctx->success());

    // Only 2 new items inserted.
    EXPECT_EQ(connector_->block_cache_->size(), cache_before + 3);
    EXPECT_TRUE(connector_->block_cache_->contains(cache_keys[0]));
    EXPECT_TRUE(connector_->block_cache_->contains(cache_keys[1]));
    EXPECT_TRUE(connector_->block_cache_->contains(cache_keys[2]));
    // Net: we kept 3 cached blocks in pool.
    EXPECT_EQ(pool->freeBlocksNum(), free_before - 3);
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

    const int                     layer0 = 0;
    std::vector<int64_t>          cache_keys{61001, 61002};
    std::vector<std::vector<int>> lbs_vec{{1, 2}};
    auto                          res = makeCacheResource(cache_keys, lbs_vec);

    const auto buf = allocator_->convertIndexToBuffer(layer0, /*block_id=*/1);
    ASSERT_NE(buf.kv_addr, nullptr);
    const size_t total = buf.kv_addr->sizeBytes();
    auto         pool  = ensureBlockPool(total);
    ASSERT_NE(pool, nullptr);
    const size_t free_before = pool->freeBlocksNum();

    auto ctx = connector_->asyncWrite(res, nullptr);
    ASSERT_NE(ctx, nullptr);

    // While in flight, insert the first key so write_done should skip inserting it.
    auto pre_blocks = putItemsToCache({cache_keys[0]}, total);
    ASSERT_EQ(pre_blocks.size(), 1u);
    ASSERT_TRUE(connector_->block_cache_->contains(cache_keys[0]));

    auto mem_ctx = std::dynamic_pointer_cast<rtp_llm::MemoryConnectorAsyncContext>(ctx);
    ASSERT_NE(mem_ctx, nullptr);
    mem_ctx->waitDone();
    EXPECT_TRUE(ctx->success());

    EXPECT_TRUE(connector_->block_cache_->contains(cache_keys[0]));
    EXPECT_TRUE(connector_->block_cache_->contains(cache_keys[1]));
    // Only 2 cached blocks should remain in pool.
    EXPECT_EQ(pool->freeBlocksNum(), free_before - 2);

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
    const int                     layer0        = 0;
    const int                     gpu_block_idx = 2;
    std::vector<int64_t>          cache_keys{1, 2};
    std::vector<std::vector<int>> lbs_vec{{gpu_block_idx, gpu_block_idx}};
    auto                          res = makeCacheResource(cache_keys, lbs_vec);

    const auto buf = allocator_->convertIndexToBuffer(layer0, gpu_block_idx);
    ASSERT_NE(buf.kv_addr, nullptr);
    // ASSERT_NE(buf.v_addr, nullptr);
    const size_t total = buf.kv_addr->sizeBytes();
    auto         pool  = ensureBlockPool(total);
    ASSERT_NE(pool, nullptr);
    const size_t free_before = pool->freeBlocksNum();

    connector_->tp_broadcast_manager_->worker_addrs_.clear();
    auto ctx = connector_->asyncWrite(res, nullptr);
    EXPECT_EQ(ctx, nullptr);
    // 分配的块应被释放
    EXPECT_EQ(pool->freeBlocksNum(), free_before);
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
    // Make thread pool full, asyncWrite should fail early before building write plan / allocating blocks.
    auto old_pool = connector_->wait_done_thread_pool_;

    connector_->wait_done_thread_pool_ = std::make_shared<autil::LockFreeThreadPool>(/*thread_num=*/1,
                                                                                     /*queue_size=*/1,
                                                                                     /*thread_init_func=*/nullptr,
                                                                                     /*name=*/"AsyncWriteFullTP");
    ASSERT_TRUE(connector_->wait_done_thread_pool_->start());

    std::atomic<bool> block_worker{true};
    std::atomic<bool> first_task_started{false};
    ASSERT_EQ(connector_->wait_done_thread_pool_->pushTask([&]() {
        first_task_started.store(true);
        while (block_worker.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }),
              autil::ThreadPoolBase::ERROR_NONE);
    const auto started_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < started_deadline && !first_task_started.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(first_task_started.load());
    (void)connector_->wait_done_thread_pool_->pushTask([&]() {
        while (block_worker.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    const auto full_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < full_deadline && !connector_->isThreadPoolFull()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(connector_->isThreadPoolFull());

    const int                     layer0 = 0;
    std::vector<int64_t>          cache_keys{71001, 71002, 71003};
    std::vector<std::vector<int>> lbs_vec{{1, 2, 3}};
    auto                          res = makeCacheResource(cache_keys, lbs_vec);

    // Pre-insert one key so cpu_matched_num < cache_keys.size() and it reaches the thread-pool-full check.
    const auto   buf      = allocator_->convertIndexToBuffer(layer0, /*block_id=*/1);
    const size_t total    = buf.kv_addr->sizeBytes();
    auto         pool     = ensureBlockPool(total);
    const size_t free_bef = pool->freeBlocksNum();
    (void)putItemsToCache({cache_keys[0]}, total);

    auto ctx = connector_->asyncWrite(res, nullptr);
    EXPECT_EQ(ctx, nullptr);
    // Should not allocate any new blocks when failing early.
    EXPECT_EQ(pool->freeBlocksNum(), free_bef - 1);

    block_worker.store(false);
    connector_->wait_done_thread_pool_->stop();
    connector_->wait_done_thread_pool_.reset();
    connector_->wait_done_thread_pool_ = old_pool;
}

TEST_F(KVCacheMemoryConnectorTest, buildCopyPlanForWrite_ReturnEmpty_WhenMatchLenEqualsSize) {
    const size_t         N = 3;
    std::vector<int64_t> cache_keys{0, 1, 2};
    ASSERT_EQ(cache_keys.size(), N);
    // 使用 1 层，索引有效，但 match_len == size，应返回空
    std::vector<std::vector<int>> per_layer_block_indices = {
        {1, 1, 1},
    };
    auto lbs  = makeLayerBlockIds(per_layer_block_indices, cache_keys.size());
    auto plan = connector_->buildCopyPlanForWrite(cache_keys, lbs, /*match_len=*/N);
    EXPECT_TRUE(plan.empty());
}

TEST_F(KVCacheMemoryConnectorTest, buildCopyPlanForWrite_ReturnPlan_SingleLayer_AllValid) {
    const size_t         N = 3;
    std::vector<int64_t> cache_keys{100, 101, 102};
    // 单层，全部有效
    const int                     layer0                  = 0;
    const int                     gpu_block_idx           = 1;
    std::vector<std::vector<int>> per_layer_block_indices = {
        {gpu_block_idx, gpu_block_idx, gpu_block_idx},  // layer0
    };
    auto lbs = makeLayerBlockIds(per_layer_block_indices, cache_keys.size());

    const auto buf0 = allocator_->convertIndexToBuffer(layer0, gpu_block_idx);
    ASSERT_NE(buf0.kv_addr, nullptr);
    // ASSERT_NE(buf0.v_addr, nullptr);
    const size_t total = buf0.kv_addr->sizeBytes();

    auto plan = connector_->buildCopyPlanForWrite(cache_keys, lbs, /*match_len=*/0);
    ASSERT_EQ(plan.size(), N);
    // 校验每个条目
    for (size_t i = 0; i < N; ++i) {
        const auto& copy_info = plan[i];
        EXPECT_EQ(copy_info.cache_key, cache_keys[i]);
        EXPECT_EQ(copy_info.mem_block_size, total);
        ASSERT_EQ(copy_info.gpu_layer_blocks.size(), 1u);
        EXPECT_EQ(copy_info.gpu_layer_blocks[0].layer_id, layer0);
        EXPECT_EQ(copy_info.gpu_layer_blocks[0].block_id, gpu_block_idx);
        EXPECT_GE(copy_info.mem_block_index, 0);

        const auto pool = connector_->getBlockPool(copy_info.mem_block_size);
        ASSERT_NE(pool, nullptr);
        auto ref_count = pool->all_ref_counter_.getRefCounterUnchecked(copy_info.mem_block_index);
        ASSERT_EQ(ref_count, 1);
    }

    // 释放分配的内存块
    std::vector<int> mem_indices;
    mem_indices.reserve(plan.size());
    for (const auto& copy_info : plan) {
        mem_indices.push_back(copy_info.mem_block_index);
    }
    auto pool = connector_->getBlockPool(total);
    ASSERT_NE(pool, nullptr);
    EXPECT_TRUE(connector_->freeBlocks(pool, mem_indices));
}

TEST_F(KVCacheMemoryConnectorTest, buildCopyPlanForWrite_ReturnPlan_MultiLayer_SomeNull) {
    const size_t         N = 3;
    std::vector<int64_t> cache_keys{10, 11, 12};
    const int            layer0 = 0, layer1 = 1;
    // layer0 全部有效，layer1 只有中间一个有效
    std::vector<std::vector<int>> per_layer_block_indices = {
        {1, 1, 1},                           // layer0
        {NULL_BLOCK_IDX, 2, NULL_BLOCK_IDX}  // layer1
    };
    auto lbs = makeLayerBlockIds(per_layer_block_indices, cache_keys.size());

    const auto b0 = allocator_->convertIndexToBuffer(layer0, 1);
    const auto b1 = allocator_->convertIndexToBuffer(layer1, 2);
    ASSERT_NE(b0.kv_addr, nullptr);
    // ASSERT_NE(b0.v_addr, nullptr);
    ASSERT_NE(b1.kv_addr, nullptr);
    // ASSERT_NE(b1.v_addr, nullptr);
    const size_t t0 = b0.kv_addr->sizeBytes();
    const size_t t1 = b1.kv_addr->sizeBytes();

    auto plan = connector_->buildCopyPlanForWrite(cache_keys, lbs, /*match_len=*/0);
    ASSERT_EQ(plan.size(), N);
    // key0: 只有 layer0
    EXPECT_EQ(plan[0].mem_block_size, t0);
    ASSERT_EQ(plan[0].gpu_layer_blocks.size(), 1u);
    // key1: 两层
    EXPECT_EQ(plan[1].mem_block_size, t0 + t1);
    ASSERT_EQ(plan[1].gpu_layer_blocks.size(), 2u);
    // key2: 只有 layer0
    EXPECT_EQ(plan[2].mem_block_size, t0);
    ASSERT_EQ(plan[2].gpu_layer_blocks.size(), 1u);
    for (const auto& copy_info : plan) {
        const auto pool = connector_->getBlockPool(copy_info.mem_block_size);
        ASSERT_NE(pool, nullptr);
        auto ref_count = pool->all_ref_counter_.getRefCounterUnchecked(copy_info.mem_block_index);
        ASSERT_EQ(ref_count, 1);
    }

    // 释放分配的内存块（注意每个条目可能对应不同的 pool）
    for (const auto& copy_info : plan) {
        auto pool = connector_->getBlockPool(copy_info.mem_block_size);
        ASSERT_NE(pool, nullptr);
        EXPECT_TRUE(connector_->freeBlocks(pool, {copy_info.mem_block_index}));
    }
}

TEST_F(KVCacheMemoryConnectorTest, buildCopyPlanForWrite_ReturnEmptyAndCleanup_OnInvalidGpuBlocks) {
    // 前两个 key 有效，第三个 key 所有 layer 的 blockIdx 均为 NULL，触发失败并清理之前分配
    std::vector<int64_t> cache_keys{1, 2, 3};
    // 使用 4 层，提升 block_size 唯一性
    std::vector<std::vector<int>> per_layer_block_indices = {
        {1, 1, NULL_BLOCK_IDX},  // layer0
        {1, 1, NULL_BLOCK_IDX},  // layer1
        {1, 1, NULL_BLOCK_IDX},  // layer2
        {1, 1, NULL_BLOCK_IDX},  // layer3
    };
    auto lbs = makeLayerBlockIds(per_layer_block_indices, cache_keys.size());

    // 计算前两个 key 的 block_size（4 层）
    size_t single_layer_total = 0;
    {
        const auto buf = allocator_->convertIndexToBuffer(0, 1);
        ASSERT_NE(buf.kv_addr, nullptr);
        // ASSERT_NE(buf.v_addr, nullptr);
        single_layer_total = buf.kv_addr->sizeBytes();
    }
    const size_t total4 = single_layer_total * 4;
    auto         pool   = ensureBlockPool(total4);
    ASSERT_NE(pool, nullptr);
    const size_t free_before = pool->freeBlocksNum();

    auto plan = connector_->buildCopyPlanForWrite(cache_keys, lbs, /*match_len=*/0);
    // 构建应失败并清理已分配
    EXPECT_TRUE(plan.empty());
    EXPECT_EQ(pool->freeBlocksNum(), free_before);
}

TEST_F(KVCacheMemoryConnectorTest, buildCopyPlanForWrite_ReturnPartialPlan_OnMallocFailure) {
    // 使用单层，预先占用 pool 使得仅剩 keep_free 个可用块
    const size_t         keep_free = 2;
    const size_t         N         = keep_free + 3;  // 使得中途分配失败
    std::vector<int64_t> cache_keys(N);
    for (size_t i = 0; i < N; ++i) {
        cache_keys[i] = i + 1000;
    }
    std::vector<std::vector<int>> per_layer_block_indices = {
        std::vector<int>(N, 2),  // layer0 全部使用 block 2
    };
    auto lbs = makeLayerBlockIds(per_layer_block_indices, cache_keys.size());

    const auto buf = allocator_->convertIndexToBuffer(0, 2);
    ASSERT_NE(buf.kv_addr, nullptr);
    // ASSERT_NE(buf.v_addr, nullptr);
    const size_t total = buf.kv_addr->sizeBytes();
    auto         pool  = ensureBlockPool(total);
    ASSERT_NE(pool, nullptr);
    const size_t free_now = pool->freeBlocksNum();
    ASSERT_GT(free_now, keep_free);
    const size_t prealloc = free_now - keep_free;
    auto         occupied = pool->malloc(static_cast<int>(prealloc));
    ASSERT_EQ(occupied.size(), prealloc);

    auto plan = connector_->buildCopyPlanForWrite(cache_keys, lbs, /*match_len=*/0);
    // 由于可用块仅剩 keep_free，应只构建 keep_free 条
    ASSERT_EQ(plan.size(), keep_free);
    for (const auto& copy_info : plan) {
        EXPECT_EQ(copy_info.mem_block_size, total);
        EXPECT_GE(copy_info.mem_block_index, 0);
        ASSERT_EQ(copy_info.gpu_layer_blocks.size(), 1u);

        const auto pool = connector_->getBlockPool(copy_info.mem_block_size);
        ASSERT_NE(pool, nullptr);
        auto ref_count = pool->all_ref_counter_.getRefCounterUnchecked(copy_info.mem_block_index);
        ASSERT_EQ(ref_count, 1);
    }

    // 清理：释放 plan 和预占用的块
    std::vector<int> plan_blocks;
    for (const auto& copy_info : plan) {
        plan_blocks.push_back(copy_info.mem_block_index);
    }
    EXPECT_TRUE(connector_->freeBlocks(pool, plan_blocks));
    pool->requestFree(occupied);
    EXPECT_EQ(pool->freeBlocksNum(), free_now);
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
    MemoryBroadcastTpRequestPB req;
    auto*                      gb = req.add_gpu_blocks();
    auto*                      lb = gb->add_layer_blocks();
    lb->set_layer_id(0);
    lb->set_block_id(1);
    // Intentionally do not add mem_block_ids or mem_block_sizes to trigger mismatch
    req.set_copy_direction(MemoryBroadcastTpRequestPB::H2D);

    MemoryBroadcastTpResponsePB resp;
    auto                        ok = connector_->copyCache(req, resp);
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

    MemoryBroadcastTpRequestPB req;
    auto*                      gb = req.add_gpu_blocks();
    auto*                      lb = gb->add_layer_blocks();
    lb->set_layer_id(layer_id);
    lb->set_block_id(gpu_block_idx);
    req.add_mem_block_ids(-1);  // invalid mem block id
    req.add_mem_block_sizes(static_cast<int64_t>(total));
    req.set_copy_direction(MemoryBroadcastTpRequestPB::H2D);

    MemoryBroadcastTpResponsePB resp;
    auto                        ok = connector_->copyCache(req, resp);
    EXPECT_FALSE(ok);
    EXPECT_FALSE(resp.success());
}

TEST_F(KVCacheMemoryConnectorTest, copyCache_ReturnFalse_InvalidMemBlockSize) {
    const int  layer_id      = 0;
    const int  gpu_block_idx = 1;
    const auto gpu_buf       = allocator_->convertIndexToBuffer(layer_id, gpu_block_idx);
    ASSERT_NE(gpu_buf.kv_addr, nullptr);
    // ASSERT_NE(gpu_buf.v_addr, nullptr);

    MemoryBroadcastTpRequestPB req;
    auto*                      gb = req.add_gpu_blocks();
    auto*                      lb = gb->add_layer_blocks();
    lb->set_layer_id(layer_id);
    lb->set_block_id(gpu_block_idx);
    req.add_mem_block_ids(1);
    req.add_mem_block_sizes(0);  // invalid mem block size
    req.set_copy_direction(MemoryBroadcastTpRequestPB::H2D);

    MemoryBroadcastTpResponsePB resp;
    auto                        ok = connector_->copyCache(req, resp);
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

    MemoryBroadcastTpRequestPB req;
    auto*                      gb = req.add_gpu_blocks();
    auto*                      lb = gb->add_layer_blocks();
    lb->set_layer_id(cache_config_.layer_num);  // out of range
    lb->set_block_id(gpu_block_idx);
    req.add_mem_block_ids(mem_block_index);
    req.add_mem_block_sizes(static_cast<int64_t>(total));
    req.set_copy_direction(MemoryBroadcastTpRequestPB::H2D);

    MemoryBroadcastTpResponsePB resp;
    auto                        ok = connector_->copyCache(req, resp);
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

    MemoryBroadcastTpRequestPB req;
    auto*                      gb = req.add_gpu_blocks();
    auto*                      lb = gb->add_layer_blocks();
    lb->set_layer_id(layer_id);
    lb->set_block_id(gpu_block_idx);
    req.add_mem_block_ids(mem_block_index);
    req.add_mem_block_sizes(static_cast<int64_t>(total));
    req.set_copy_direction(MemoryBroadcastTpRequestPB::H2D);

    MemoryBroadcastTpResponsePB resp;
    auto                        ok = connector_->copyCache(req, resp);
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

    MemoryBroadcastTpRequestPB req;
    req.set_copy_direction(MemoryBroadcastTpRequestPB::H2D);
    addOneCopyInfoToPb(req, gpu_layer_blocks1, mem_block_index1, mem_block_size1);
    addOneCopyInfoToPb(req, gpu_layer_blocks2, mem_block_index2, mem_block_size2);

    MemoryBroadcastTpResponsePB resp;
    auto                        ok = connector_->copyCache(req, resp);
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

    MemoryBroadcastTpRequestPB req;
    auto*                      gb = req.add_gpu_blocks();
    auto*                      lb = gb->add_layer_blocks();
    lb->set_layer_id(layer_id);
    lb->set_block_id(gpu_block_idx);
    req.add_mem_block_ids(mem_block_index);
    req.add_mem_block_sizes(static_cast<int64_t>(total));
    req.set_copy_direction(MemoryBroadcastTpRequestPB::D2H);

    MemoryBroadcastTpResponsePB resp;
    auto                        ok = connector_->copyCache(req, resp);
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

    MemoryBroadcastTpRequestPB req;
    req.set_copy_direction(MemoryBroadcastTpRequestPB::D2H);
    addOneCopyInfoToPb(req, gpu_layer_blocks1, mem_block_index1, mem_block_size1);
    addOneCopyInfoToPb(req, gpu_layer_blocks2, mem_block_index2, mem_block_size2);

    MemoryBroadcastTpResponsePB resp;
    auto                        ok = connector_->copyCache(req, resp);
    EXPECT_TRUE(ok);
    EXPECT_TRUE(resp.success());

    // D2H, 验证数据是否拷贝成功
    verifyCpuBufferContent(gpu_layer_blocks1, mem_block_index1, mem_block_size1);
    verifyCpuBufferContent(gpu_layer_blocks2, mem_block_index2, mem_block_size2);
}

TEST_F(KVCacheMemoryConnectorTest, prepareCopyBuffers_ReturnFalse_NoMemPool_H2D) {
    // H2D 路径下不创建内存池，应返回 false
    const int  layer_id      = 0;
    const int  gpu_block_idx = 1;
    const auto gpu_buf       = allocator_->convertIndexToBuffer(layer_id, gpu_block_idx);
    ASSERT_NE(gpu_buf.kv_addr, nullptr);
    // ASSERT_NE(gpu_buf.v_addr, nullptr);
    const size_t total_bytes = gpu_buf.kv_addr->sizeBytes();

    std::vector<KVCacheMemoryConnector::LayerBlock> gpu_indices{{layer_id, gpu_block_idx}};
    std::vector<rtp_llm::BufferPtr>                 dst, src;
    const int                                       mem_block_index = 1;  // 未创建 pool，任意值
    auto                                            ok              = connector_->prepareCopyBuffers(
        gpu_indices, mem_block_index, total_bytes, KVCacheMemoryConnector::CopyDirection::H2D, dst, src);
    EXPECT_FALSE(ok);
    EXPECT_TRUE(dst.empty());
    EXPECT_TRUE(src.empty());
}

TEST_F(KVCacheMemoryConnectorTest, prepareCopyBuffers_ReturnFalse_GetMemBufferFailed) {
    // 无法模拟 convertIndexToBuffer 失败
}

TEST_F(KVCacheMemoryConnectorTest, prepareCopyBuffers_ReturnFalse_InvalidGpuBlockIdx) {
    const int  layer_id      = 0;
    const int  gpu_block_idx = 1;
    const auto gpu_buf       = allocator_->convertIndexToBuffer(layer_id, gpu_block_idx);
    ASSERT_NE(gpu_buf.kv_addr, nullptr);
    // ASSERT_NE(gpu_buf.v_addr, nullptr);
    const size_t total_bytes = gpu_buf.kv_addr->sizeBytes();

    auto pool = ensureBlockPool(total_bytes);
    ASSERT_NE(pool, nullptr);
    auto mem_blocks = pool->malloc(1);
    ASSERT_EQ(mem_blocks.size(), 1u);
    const int mem_block_index = mem_blocks[0];

    // 使用 NULL_BLOCK_IDX 触发实现中的校验失败
    std::vector<KVCacheMemoryConnector::LayerBlock> gpu_layer_blocks{{layer_id, NULL_BLOCK_IDX}};
    std::vector<rtp_llm::BufferPtr>                 dst, src;
    auto                                            ok = connector_->prepareCopyBuffers(
        gpu_layer_blocks, mem_block_index, total_bytes, KVCacheMemoryConnector::CopyDirection::H2D, dst, src);
    EXPECT_FALSE(ok);
    EXPECT_TRUE(dst.empty());
    EXPECT_TRUE(src.empty());
}

TEST_F(KVCacheMemoryConnectorTest, prepareCopyBuffers_ReturnFalse_InvalidLayerId) {
    const int  valid_layer   = 0;
    const int  gpu_block_idx = 1;
    const auto gpu_buf       = allocator_->convertIndexToBuffer(valid_layer, gpu_block_idx);
    ASSERT_NE(gpu_buf.kv_addr, nullptr);
    // ASSERT_NE(gpu_buf.v_addr, nullptr);
    const size_t total_bytes = gpu_buf.kv_addr->sizeBytes();

    // 创建内存池并分配一个块，保证不是因为 mem_pool/mem_buffer 失败
    auto pool = ensureBlockPool(total_bytes);
    ASSERT_NE(pool, nullptr);
    auto mem_blocks = pool->malloc(1);
    ASSERT_EQ(mem_blocks.size(), 1u);
    const int mem_block_index = mem_blocks[0];

    {
        // 使用非法 layer_id 触发实现中的校验失败, layer_id < 0
        std::vector<KVCacheMemoryConnector::LayerBlock> gpu_layer_blocks{{-1, gpu_block_idx}};
        std::vector<rtp_llm::BufferPtr>                 dst, src;
        auto                                            ok = connector_->prepareCopyBuffers(
            gpu_layer_blocks, mem_block_index, total_bytes, KVCacheMemoryConnector::CopyDirection::H2D, dst, src);
        EXPECT_FALSE(ok);
        EXPECT_TRUE(dst.empty());
        EXPECT_TRUE(src.empty());
    }
    {
        // 使用非法 layer_id 触发实现中的校验失败, layer_id >= cache_config_.layer_num
        std::vector<KVCacheMemoryConnector::LayerBlock> gpu_layer_blocks{
            {static_cast<int>(cache_config_.layer_num), gpu_block_idx}};
        std::vector<rtp_llm::BufferPtr> dst, src;
        auto                            ok = connector_->prepareCopyBuffers(
            gpu_layer_blocks, mem_block_index, total_bytes, KVCacheMemoryConnector::CopyDirection::H2D, dst, src);
        EXPECT_FALSE(ok);
        EXPECT_TRUE(dst.empty());
        EXPECT_TRUE(src.empty());
    }
}

TEST_F(KVCacheMemoryConnectorTest, prepareCopyBuffers_ReturnTrue_H2D_SingleLayerWithKVBuffer) {
    const int  layer_id      = 0;
    const int  gpu_block_idx = 2;
    const auto gpu_buf       = allocator_->convertIndexToBuffer(layer_id, gpu_block_idx);
    ASSERT_NE(gpu_buf.kv_addr, nullptr);
    // ASSERT_NE(gpu_buf.v_addr, nullptr);
    const size_t k_bytes = gpu_buf.kv_addr->sizeBytes();
    // const size_t v_bytes = gpu_buf.v_addr->sizeBytes();
    const size_t total = k_bytes;

    // 先创建内存池并分配一个块
    auto pool = ensureBlockPool(total);
    ASSERT_NE(pool, nullptr);
    auto mem_blocks = pool->malloc(1);
    ASSERT_EQ(mem_blocks.size(), 1u);
    const int mem_block_index = mem_blocks[0];

    std::vector<KVCacheMemoryConnector::LayerBlock> gpu_indices{{layer_id, gpu_block_idx}};
    std::vector<rtp_llm::BufferPtr>                 dst, src;
    auto                                            ok = connector_->prepareCopyBuffers(
        gpu_indices, mem_block_index, total, KVCacheMemoryConnector::CopyDirection::H2D, dst, src);
    ASSERT_TRUE(ok);

    // 预期两段（K/V）
    ASSERT_EQ(src.size(), 1u);
    ASSERT_EQ(dst.size(), 1u);
    // H2D: src 为 CPU buffer，dst 为 GPU buffer
    for (const auto& buffer : src) {
        EXPECT_EQ(buffer->where(), MemoryType::MEMORY_CPU);
    }
    for (const auto& buffer : dst) {
        EXPECT_EQ(buffer->where(), MemoryType::MEMORY_GPU);
    }
    EXPECT_EQ(src[0]->sizeBytes(), k_bytes);
    EXPECT_EQ(dst[0]->sizeBytes(), k_bytes);
}

TEST_F(KVCacheMemoryConnectorTest, prepareCopyBuffers_ReturnTrue_H2D_MultiLayerWithKVBuffer) {
    const int  layer0         = 0;
    const int  layer1         = 1;
    const int  gpu_block_idx1 = 1;
    const int  gpu_block_idx2 = 2;
    const auto buf0           = allocator_->convertIndexToBuffer(layer0, gpu_block_idx1);
    const auto buf1           = allocator_->convertIndexToBuffer(layer1, gpu_block_idx2);
    ASSERT_NE(buf0.kv_addr, nullptr);
    // ASSERT_NE(buf0.v_addr, nullptr);
    ASSERT_NE(buf1.kv_addr, nullptr);
    // ASSERT_NE(buf1.v_addr, nullptr);
    const size_t total = buf0.kv_addr->sizeBytes() + buf1.kv_addr->sizeBytes();

    auto pool = ensureBlockPool(total);
    ASSERT_NE(pool, nullptr);
    auto mem_blocks = pool->malloc(1);
    ASSERT_EQ(mem_blocks.size(), 1u);
    const int mem_block_index = mem_blocks[0];
    auto      mem_buffer      = pool->convertIndexToBuffer(0, mem_block_index);
    ASSERT_NE(mem_buffer.kv_addr, nullptr);
    // ASSERT_NE(mem_buffer.v_addr, nullptr);
    EXPECT_EQ(mem_buffer.kv_addr->sizeBytes(), total);
    // EXPECT_EQ(mem_buffer.v_addr->sizeBytes(), total);

    std::vector<KVCacheMemoryConnector::LayerBlock> gpu_indices{{layer0, gpu_block_idx1}, {layer1, gpu_block_idx2}};
    std::vector<rtp_llm::BufferPtr>                 dst, src;
    auto                                            ok = connector_->prepareCopyBuffers(
        gpu_indices, mem_block_index, total, KVCacheMemoryConnector::CopyDirection::H2D, dst, src);
    ASSERT_TRUE(ok);

    // 预期四段（K/V）
    EXPECT_EQ(src.size(), 2u);
    EXPECT_EQ(dst.size(), 2u);
    // H2D: src 为 CPU buffer，dst 为 GPU buffer
    for (const auto& buffer : src) {
        EXPECT_EQ(buffer->where(), MemoryType::MEMORY_CPU);
    }
    for (const auto& buffer : dst) {
        EXPECT_EQ(buffer->where(), MemoryType::MEMORY_GPU);
    }
    EXPECT_EQ(src[0]->sizeBytes(), buf0.kv_addr->sizeBytes());
    EXPECT_EQ(src[1]->sizeBytes(), buf1.kv_addr->sizeBytes());
    EXPECT_EQ(dst[0]->sizeBytes(), buf0.kv_addr->sizeBytes());
    EXPECT_EQ(dst[1]->sizeBytes(), buf1.kv_addr->sizeBytes());
}

TEST_F(KVCacheMemoryConnectorTest, prepareCopyBuffers_ReturnTrue_D2H_SingleLayer_MemPoolNotCreated) {
    const int  layer_id      = 0;
    const int  gpu_block_idx = 3;
    const auto gpu_buf       = allocator_->convertIndexToBuffer(layer_id, gpu_block_idx);
    ASSERT_NE(gpu_buf.kv_addr, nullptr);
    // ASSERT_NE(gpu_buf.v_addr, nullptr);
    const size_t k_bytes = gpu_buf.kv_addr->sizeBytes();
    // const size_t v_bytes = gpu_buf.v_addr->sizeBytes();
    const size_t total = k_bytes;

    // D2H 路径：不提前创建内存池
    auto pool = connector_->getBlockPool(total);
    ASSERT_EQ(pool, nullptr);

    int                                             mem_block_index = 1;
    std::vector<KVCacheMemoryConnector::LayerBlock> gpu_indices{{layer_id, gpu_block_idx}};
    std::vector<rtp_llm::BufferPtr>                 dst, src;
    auto                                            ok = connector_->prepareCopyBuffers(
        gpu_indices, mem_block_index, total, KVCacheMemoryConnector::CopyDirection::D2H, dst, src);
    ASSERT_TRUE(ok);

    // 验证内存池是否创建成功
    ASSERT_NE(connector_->getBlockPool(total), nullptr);

    // 预期两段（K/V）
    ASSERT_EQ(src.size(), 1u);
    ASSERT_EQ(dst.size(), 1u);
    // D2H: src 为 GPU buffer，dst 为 CPU buffer
    for (const auto& buffer : src) {
        EXPECT_EQ(buffer->where(), MemoryType::MEMORY_GPU);
    }
    for (const auto& buffer : dst) {
        EXPECT_EQ(buffer->where(), MemoryType::MEMORY_CPU);
    }
    EXPECT_EQ(src[0]->sizeBytes(), k_bytes);
    EXPECT_EQ(dst[0]->sizeBytes(), k_bytes);
}

TEST_F(KVCacheMemoryConnectorTest, prepareCopyBuffers_ReturnTrue_D2H_SingleLayerWithKVBuffer) {
    const int  layer_id      = 0;
    const int  gpu_block_idx = 3;
    const auto gpu_buf       = allocator_->convertIndexToBuffer(layer_id, gpu_block_idx);
    ASSERT_NE(gpu_buf.kv_addr, nullptr);
    // ASSERT_NE(gpu_buf.v_addr, nullptr);
    const size_t k_bytes = gpu_buf.kv_addr->sizeBytes();
    // const size_t v_bytes = gpu_buf.v_addr->sizeBytes();
    const size_t total = k_bytes;

    // D2H 路径：内部会创建内存池，但为了稳定，提前创建并分配
    auto pool = ensureBlockPool(total);
    ASSERT_NE(pool, nullptr);
    auto mem_blocks = pool->malloc(1);
    ASSERT_EQ(mem_blocks.size(), 1u);
    const int mem_block_index = mem_blocks[0];

    std::vector<KVCacheMemoryConnector::LayerBlock> gpu_indices{{layer_id, gpu_block_idx}};
    std::vector<rtp_llm::BufferPtr>                 dst, src;
    auto                                            ok = connector_->prepareCopyBuffers(
        gpu_indices, mem_block_index, total, KVCacheMemoryConnector::CopyDirection::D2H, dst, src);
    ASSERT_TRUE(ok);

    // 预期两段（K/V）
    ASSERT_EQ(src.size(), 1u);
    ASSERT_EQ(dst.size(), 1u);
    // D2H: src 为 GPU buffer，dst 为 CPU buffer
    for (const auto& buffer : src) {
        EXPECT_EQ(buffer->where(), MemoryType::MEMORY_GPU);
    }
    for (const auto& buffer : dst) {
        EXPECT_EQ(buffer->where(), MemoryType::MEMORY_CPU);
    }
    EXPECT_EQ(src[0]->sizeBytes(), k_bytes);
    EXPECT_EQ(dst[0]->sizeBytes(), k_bytes);
}

TEST_F(KVCacheMemoryConnectorTest, prepareCopyBuffers_ReturnTrue_D2H_MultiLayerWithKVBuffer) {
    const int  layer0         = 0;
    const int  layer1         = 1;
    const int  gpu_block_idx1 = 1;
    const int  gpu_block_idx2 = 2;
    const auto buf0           = allocator_->convertIndexToBuffer(layer0, gpu_block_idx1);
    const auto buf1           = allocator_->convertIndexToBuffer(layer1, gpu_block_idx2);
    ASSERT_NE(buf0.kv_addr, nullptr);
    // ASSERT_NE(buf0.v_addr, nullptr);
    ASSERT_NE(buf1.kv_addr, nullptr);
    // ASSERT_NE(buf1.v_addr, nullptr);
    const size_t total = buf0.kv_addr->sizeBytes() + buf1.kv_addr->sizeBytes();

    auto pool = ensureBlockPool(total);
    ASSERT_NE(pool, nullptr);
    auto mem_blocks = pool->malloc(1);
    ASSERT_EQ(mem_blocks.size(), 1u);
    const int mem_block_index = mem_blocks[0];
    auto      mem_buffer      = pool->convertIndexToBuffer(0, mem_block_index);
    ASSERT_NE(mem_buffer.kv_addr, nullptr);
    // ASSERT_NE(mem_buffer.v_addr, nullptr);
    EXPECT_EQ(mem_buffer.kv_addr->sizeBytes(), total);
    // EXPECT_EQ(mem_buffer.v_addr->sizeBytes(), total);

    std::vector<KVCacheMemoryConnector::LayerBlock> gpu_indices{{layer0, gpu_block_idx1}, {layer1, gpu_block_idx2}};
    std::vector<rtp_llm::BufferPtr>                 dst, src;
    auto                                            ok = connector_->prepareCopyBuffers(
        gpu_indices, mem_block_index, total, KVCacheMemoryConnector::CopyDirection::D2H, dst, src);
    ASSERT_TRUE(ok);

    // 预期四段（K/V）
    EXPECT_EQ(src.size(), 2u);
    EXPECT_EQ(dst.size(), 2u);
    // D2H: src 为 GPU buffer，dst 为 CPU buffer
    for (const auto& buffer : src) {
        EXPECT_EQ(buffer->where(), MemoryType::MEMORY_GPU);
    }
    for (const auto& buffer : dst) {
        EXPECT_EQ(buffer->where(), MemoryType::MEMORY_CPU);
    }
    EXPECT_EQ(src[0]->sizeBytes(), buf0.kv_addr->sizeBytes());
    EXPECT_EQ(src[1]->sizeBytes(), buf1.kv_addr->sizeBytes());
    EXPECT_EQ(dst[0]->sizeBytes(), buf0.kv_addr->sizeBytes());
    EXPECT_EQ(dst[1]->sizeBytes(), buf1.kv_addr->sizeBytes());
}

TEST_F(KVCacheMemoryConnectorTest, checkKVCacheResource_ReturnFalse_WhenResourceNull) {
    EXPECT_FALSE(connector_->checkKVCacheResource(nullptr));
}

TEST_F(KVCacheMemoryConnectorTest, checkKVCacheResource_ReturnFalse_WhenCacheKeysEmpty) {
    auto res = makeCacheResource(/*cache_keys=*/{}, /*per_layer_block_indices=*/{{}});
    EXPECT_FALSE(connector_->checkKVCacheResource(res));
}

TEST_F(KVCacheMemoryConnectorTest, checkKVCacheResource_ReturnFalse_WhenLayerBlockIdsEmpty) {
    auto res             = std::make_shared<KVCacheResource>();
    res->cache_keys      = CacheKeysType{1};
    res->layer_block_ids = LayerBlockIds{};  // empty
    EXPECT_FALSE(connector_->checkKVCacheResource(res));
}

TEST_F(KVCacheMemoryConnectorTest, checkKVCacheResource_ReturnFalse_WhenLayerBlockIdsSizeMismatch) {
    auto res        = std::make_shared<KVCacheResource>();
    res->cache_keys = CacheKeysType{1, 2};

    // Provide non-empty but size != layer_num.
    LayerBlockIds lbs;
    auto          ptr = std::make_shared<BlockIds>();
    ptr->blocks()     = {1, 1};
    lbs.emplace_back(ptr);
    res->layer_block_ids = std::move(lbs);

    EXPECT_FALSE(connector_->checkKVCacheResource(res));
}

TEST_F(KVCacheMemoryConnectorTest, checkKVCacheResource_ReturnFalse_WhenBlocksNumLessThanCacheKeysSize) {
    auto res        = std::make_shared<KVCacheResource>();
    res->cache_keys = CacheKeysType{1, 2};

    LayerBlockIds lbs;
    lbs.reserve(cache_config_.layer_num);
    for (int i = 0; i < cache_config_.layer_num; ++i) {
        auto ptr      = std::make_shared<BlockIds>();
        ptr->blocks() = {1};  // size 1 < cache_keys.size() (2)
        lbs.emplace_back(std::move(ptr));
    }
    res->layer_block_ids = std::move(lbs);

    EXPECT_FALSE(connector_->checkKVCacheResource(res));
}

TEST_F(KVCacheMemoryConnectorTest, checkKVCacheResource_ReturnTrue_WhenResourceValid) {
    std::vector<int64_t>          cache_keys{1, 2};
    std::vector<std::vector<int>> lbs_vec{{1, 2}};  // other layers auto filled with NULL_BLOCK_IDX
    auto                          res = makeCacheResource(cache_keys, lbs_vec);
    EXPECT_TRUE(connector_->checkKVCacheResource(res));
}

TEST_F(KVCacheMemoryConnectorTest, mallocBlocks_ReturnFalse_BlockPoolNull) {
    std::vector<BlockIdxType> blocks;
    auto                      ok = connector_->mallocBlocks(nullptr, 1, blocks);
    EXPECT_FALSE(ok);
    EXPECT_TRUE(blocks.empty());
}

TEST_F(KVCacheMemoryConnectorTest, mallocBlocks_ReturnFalse_NeedBlocksZero) {
    const size_t block_size = 4096;
    auto         pool       = ensureBlockPool(block_size);
    ASSERT_NE(pool, nullptr);
    std::vector<BlockIdxType> blocks;
    auto                      ok = connector_->mallocBlocks(pool, 0, blocks);
    EXPECT_FALSE(ok);
    EXPECT_TRUE(blocks.empty());
}

TEST_F(KVCacheMemoryConnectorTest, mallocBlocks_ReturnFalse_NotEnoughFreeBlocks) {
    const size_t block_size = 4096;
    auto         pool       = ensureBlockPool(block_size);
    ASSERT_NE(pool, nullptr);
    // 占满可用块
    const size_t free_now = pool->freeBlocksNum();
    if (free_now > 0) {
        auto taken = pool->malloc(static_cast<int>(free_now));
        ASSERT_EQ(taken.size(), free_now);
    }
    std::vector<BlockIdxType> blocks;
    auto                      ok = connector_->mallocBlocks(pool, 1, blocks);
    EXPECT_FALSE(ok);
    EXPECT_TRUE(blocks.empty());
}

TEST_F(KVCacheMemoryConnectorTest, mallocBlocks_ReturnFalse_BlockPoolMallocFailed) {
    const size_t block_size = 2048;
    auto         pool       = ensureBlockPool(block_size);
    ASSERT_NE(pool, nullptr);
    const size_t free_block_num = pool->freeBlocksNum();
    // 占满可用块
    auto malloced_blocks = pool->malloc(static_cast<int>(free_block_num));
    ASSERT_EQ(malloced_blocks.size(), free_block_num);

    std::vector<BlockIdxType> blocks;
    auto                      ok = connector_->mallocBlocks(pool, 1, blocks);
    EXPECT_FALSE(ok);
    EXPECT_TRUE(blocks.empty());
    // 释放以避免影响后续测试
    pool->requestFree(malloced_blocks);
}

TEST_F(KVCacheMemoryConnectorTest, mallocBlocks_ReturnTrue_OnSufficientFree) {
    const size_t block_size = 2048;
    auto         pool       = ensureBlockPool(block_size);
    ASSERT_NE(pool, nullptr);
    const size_t              free_before = pool->freeBlocksNum();
    const size_t              need        = std::min<size_t>(3, std::max<size_t>(1, free_before / 2));
    std::vector<BlockIdxType> blocks;
    auto                      ok = connector_->mallocBlocks(pool, need, blocks);
    ASSERT_TRUE(ok);
    ASSERT_EQ(blocks.size(), need);
    // 释放以避免影响后续测试
    pool->requestFree(blocks);
}

TEST_F(KVCacheMemoryConnectorTest, freeBlocks_ReturnNoop_BlocksEmpty) {
    const size_t block_size = 4096;
    // 未创建 pool 前应返回空
    auto pool_before = connector_->getBlockPool(block_size);
    EXPECT_EQ(pool_before, nullptr);

    // 空 blocks 输入，不应创建 pool，也不应崩溃
    auto ok = connector_->freeBlocks(nullptr, {});
    EXPECT_TRUE(ok);

    auto pool_after = connector_->getBlockPool(block_size);
    EXPECT_EQ(pool_after, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, freeBlocks_ReturnFalse_PoolNull_NonEmptyBlocksAllNull) {
    const size_t block_size = 8192;
    // 未创建 pool，传入包含非法/空 block 索引，应返回 false
    auto ok = connector_->freeBlocks(nullptr, {NULL_BLOCK_IDX, -1, NULL_BLOCK_IDX});
    EXPECT_FALSE(ok);
    auto pool = connector_->getBlockPool(block_size);
    EXPECT_EQ(pool, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, freeBlocks_IncreaseFreeBlocks_WhenValidBlocksFreed) {
    const size_t block_size = 2048;
    // 显式创建 pool
    auto pool = ensureBlockPool(block_size);
    ASSERT_NE(pool, nullptr);

    const size_t free_before_alloc = pool->freeBlocksNum();
    // 分配 3 个块
    auto blocks = pool->malloc(3);
    ASSERT_EQ(blocks.size(), 3u);
    const size_t free_after_alloc = pool->freeBlocksNum();
    ASSERT_LE(free_after_alloc + 3, free_before_alloc);

    // 释放其中 2 个（包含一个 NULL_BLOCK_IDX，应该被忽略）
    auto ok1 = connector_->freeBlocks(pool, {blocks[0], NULL_BLOCK_IDX, blocks[1]});
    EXPECT_TRUE(ok1);

    const size_t free_after_free2 = pool->freeBlocksNum();
    // 应该比分配后多 2
    EXPECT_EQ(free_after_free2, free_after_alloc + 2);

    // 再释放剩余的 1 个
    auto ok2 = connector_->freeBlocks(pool, {blocks[2]});
    EXPECT_TRUE(ok2);
    const size_t free_after_free3 = pool->freeBlocksNum();
    EXPECT_EQ(free_after_free3, free_after_alloc + 3);
}

TEST_F(KVCacheMemoryConnectorTest, freeBlocks_IgnoreNullIndices_NoChange) {
    const size_t block_size = 1024;
    auto         pool       = ensureBlockPool(block_size);
    ASSERT_NE(pool, nullptr);

    const size_t free_before = pool->freeBlocksNum();
    auto         ok          = connector_->freeBlocks(pool, {NULL_BLOCK_IDX, -1, NULL_BLOCK_IDX});
    EXPECT_TRUE(ok);
    const size_t free_after = pool->freeBlocksNum();
    EXPECT_EQ(free_after, free_before);
}

TEST_F(KVCacheMemoryConnectorTest, freeBlocks_ReturnFalse_PoolNotExist_WithValidBlocks) {
    // 模拟有“有效”索引，但对应 pool 未创建，应返回 false
    auto ok = connector_->freeBlocks(nullptr, {0, 1, 2});
    EXPECT_FALSE(ok);
}

TEST_F(KVCacheMemoryConnectorTest, referenceBlocks_ReturnVoid_WhenBlocksEmpty) {
    const auto   buf        = allocator_->convertIndexToBuffer(0, 0);
    const size_t block_size = buf.kv_addr->sizeBytes();
    auto         pool       = ensureBlockPool(block_size);
    ASSERT_NE(pool, nullptr);
    auto blocks = pool->malloc(1);
    ASSERT_EQ(blocks.size(), 1u);
    const int block_idx = blocks[0];

    const int ref_before = pool->getBlockCacheRefCount(block_idx);
    connector_->referenceBlocks(pool, /*blocks=*/{});
    EXPECT_EQ(pool->getBlockCacheRefCount(block_idx), ref_before);
}

TEST_F(KVCacheMemoryConnectorTest, referenceBlocks_ReturnVoid_WhenBlockPoolNull) {
    ASSERT_NO_FATAL_FAILURE(connector_->referenceBlocks(nullptr, /*blocks=*/{1, 2}));
}

TEST_F(KVCacheMemoryConnectorTest, getBlockPool_ReturnNull_WhenNotExists) {
    // Use a size that should not be created by other tests.
    EXPECT_EQ(connector_->getBlockPool(/*block_size=*/1234567), nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, createBlockPool_ReturnNonNull_WhenValidBlockSize) {
    const auto   buf        = allocator_->convertIndexToBuffer(0, 0);
    const size_t block_size = buf.kv_addr->sizeBytes();
    auto         pool       = connector_->createBlockPool(block_size);
    ASSERT_NE(pool, nullptr);

    // `createBlockPool` should register it for later lookup.
    EXPECT_EQ(connector_->getBlockPool(block_size), pool);
}

TEST_F(KVCacheMemoryConnectorTest, putToCache_ReturnVoid_WhenCachePutFails_DuplicateKey) {
    // Prepare a pool and 2 allocated blocks with same block_size.
    const auto   buf        = allocator_->convertIndexToBuffer(0, 0);
    const size_t block_size = buf.kv_addr->sizeBytes();
    auto         pool       = ensureBlockPool(block_size);
    ASSERT_NE(pool, nullptr);

    auto blocks1 = pool->malloc(1);
    auto blocks2 = pool->malloc(1);
    ASSERT_EQ(blocks1.size(), 1u);
    ASSERT_EQ(blocks2.size(), 1u);
    const int idx1 = blocks1[0];
    const int idx2 = blocks2[0];

    MemoryBlockCache::CacheItem item1;
    item1.cache_key   = 1;
    item1.block_index = idx1;
    item1.block_size  = block_size;
    item1.is_resident = false;

    // First insert success => referenceBlocks called (all_ref increases by 1).
    connector_->putToCache(item1);
    EXPECT_EQ(pool->getBlockCacheRefCount(idx1), 2);

    // Mimic request finished: drop request ref so only cache ref remains (all_ref back to 1).
    pool->requestFree(std::vector<int>{idx1});
    EXPECT_EQ(pool->getBlockCacheRefCount(idx1), 1);

    // Duplicate cache_key should make MemoryBlockCache::put return false, and putToCache should do nothing for idx2.
    MemoryBlockCache::CacheItem item_dup = item1;
    item_dup.block_index                 = idx2;
    connector_->putToCache(item_dup);
    EXPECT_EQ(pool->getBlockCacheRefCount(idx2), 1);  // unchanged (only request ref from malloc)

    // Cleanup request refs for idx2 to avoid leaking state.
    pool->requestFree(std::vector<int>{idx2});
}

TEST_F(KVCacheMemoryConnectorTest, putToCache_ReturnVoid_WhenCacheFull_EvictsAndFreesPoppedBlock) {
    // Force cache capacity to 1 so we can test eviction.
    connector_->block_cache_->lru_cache_ = LRUCache<int64_t, MemoryBlockCache::CacheItem>(1);

    const auto   buf        = allocator_->convertIndexToBuffer(0, 0);
    const size_t block_size = buf.kv_addr->sizeBytes();
    auto         pool       = ensureBlockPool(block_size);
    ASSERT_NE(pool, nullptr);

    auto blocks1 = pool->malloc(1);
    auto blocks2 = pool->malloc(1);
    ASSERT_EQ(blocks1.size(), 1u);
    ASSERT_EQ(blocks2.size(), 1u);
    const int idx1 = blocks1[0];
    const int idx2 = blocks2[0];

    const auto free_before = pool->freeBlocksNum();

    MemoryBlockCache::CacheItem item1;
    item1.cache_key   = 100;
    item1.block_index = idx1;
    item1.block_size  = block_size;
    item1.is_resident = false;
    connector_->putToCache(item1);
    EXPECT_EQ(pool->getBlockCacheRefCount(idx1), 2);

    // Drop request ref, keep cache ref only.
    pool->requestFree(std::vector<int>{idx1});
    EXPECT_EQ(pool->getBlockCacheRefCount(idx1), 1);
    const auto free_mid = pool->freeBlocksNum();
    EXPECT_EQ(free_mid, free_before);

    MemoryBlockCache::CacheItem item2;
    item2.cache_key   = 101;
    item2.block_index = idx2;
    item2.block_size  = block_size;
    item2.is_resident = false;
    connector_->putToCache(item2);  // should evict item1 and free idx1 via cache_free path

    EXPECT_FALSE(connector_->block_cache_->contains(100));
    EXPECT_TRUE(connector_->block_cache_->contains(101));

    // idx1 should become free again after eviction (request ref already dropped).
    EXPECT_EQ(pool->getBlockCacheRefCount(idx1), 0);
    EXPECT_EQ(pool->freeBlocksNum(), free_mid + 1);

    // Cleanup request ref for idx2 (keep cache ref).
    pool->requestFree(std::vector<int>{idx2});
}

TEST_F(KVCacheMemoryConnectorTest, clearCache_ReturnVoid_WhenBlockCacheNull) {
    connector_->block_cache_.reset();
    ASSERT_NO_FATAL_FAILURE(connector_->clearCache());
}

TEST_F(KVCacheMemoryConnectorTest, clearCache_ReturnVoid_WhenBlockPoolMissingAndFreeBlocksFailed) {
    // Put one item into block_cache_ with a block_size that does not have a corresponding pool.
    const int64_t bad_block_size  = 1234567;
    const int     bad_block_index = 1;

    MemoryBlockCache::CacheItem item;
    item.cache_key   = 99001;
    item.block_index = bad_block_index;
    item.block_size  = bad_block_size;
    item.is_resident = false;
    ASSERT_TRUE(connector_->block_cache_->put(item).first);
    ASSERT_TRUE(connector_->block_cache_->contains(item.cache_key));

    connector_->clearCache();
    EXPECT_FALSE(connector_->block_cache_->contains(item.cache_key));
    EXPECT_TRUE(connector_->block_cache_->empty());
}

TEST_F(KVCacheMemoryConnectorTest, ensureEnoughFreeBlocks_ReturnTrue_WhenEnoughFree) {
    const size_t block_size = 4096;
    auto         pool       = ensureBlockPool(block_size);
    ASSERT_NE(pool, nullptr);
    const size_t free_now = pool->freeBlocksNum();
    // 需求不超过当前可用
    auto ok = connector_->ensureEnoughFreeBlocks(pool, free_now - 1);
    EXPECT_TRUE(ok);
}

TEST_F(KVCacheMemoryConnectorTest, ensureEnoughFreeBlocks_ReturnFalse_WhenInsufficient) {
    const size_t block_size = 4096;
    auto         pool       = ensureBlockPool(block_size);
    ASSERT_NE(pool, nullptr);
    // 占满可用块
    const size_t free_now = pool->freeBlocksNum();
    auto         taken    = pool->malloc(static_cast<int>(free_now));
    ASSERT_EQ(taken.size(), free_now);
    // 无可逐出的缓存项时，应返回 false
    auto ok = connector_->ensureEnoughFreeBlocks(pool, 1);
    EXPECT_FALSE(ok);
    // 释放以避免影响后续测试
    pool->requestFree(taken);
}

TEST_F(KVCacheMemoryConnectorTest, ensureEnoughFreeBlocks_ReturnTrue_FreeBlocksFromCache) {
    const size_t block_size = 4096;
    auto         pool       = ensureBlockPool(block_size);
    ASSERT_NE(pool, nullptr);
    // 占满可用块
    const size_t free_num     = pool->freeBlocksNum();
    auto         taken_blocks = pool->malloc(static_cast<int>(free_num));
    // 添加到cache中
    for (int i = 0; i < taken_blocks.size(); i++) {
        MemoryBlockCache::CacheItem item;
        item.cache_key   = i;
        item.block_index = taken_blocks.at(i);
        connector_->block_cache_->put(item);
        EXPECT_TRUE(connector_->block_cache_->contains(item.cache_key));
    }
    EXPECT_EQ(connector_->block_cache_->size(), taken_blocks.size());

    // 超过当前空闲, 会从cache中pop
    auto ok = connector_->ensureEnoughFreeBlocks(pool, free_num - 2);
    EXPECT_TRUE(ok);
    EXPECT_EQ(pool->freeBlocksNum(), free_num - 2);
    EXPECT_EQ(connector_->block_cache_->size(), 2);
}

TEST_F(KVCacheMemoryConnectorTest, waitContextDoneAsync_ReturnFalse_WhenThreadPoolNull) {
    connector_->wait_done_thread_pool_.reset();

    bool callback_called = false;
    auto ctx             = std::make_shared<rtp_llm::MemoryConnectorAsyncContext>(
        /*broadcast_result=*/nullptr, /*done_callback=*/[&](bool) { callback_called = true; });
    ASSERT_FALSE(ctx->done());

    EXPECT_FALSE(connector_->waitContextDoneAsync(ctx));
    // Should not execute callback since no thread pool.
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    EXPECT_FALSE(callback_called);
    EXPECT_FALSE(ctx->done());
}

TEST_F(KVCacheMemoryConnectorTest, waitContextDoneAsync_ReturnFalse_WhenPushTaskFails) {
    // Stop thread pool so pushTask is expected to fail.
    ASSERT_NE(connector_->wait_done_thread_pool_, nullptr);
    connector_->wait_done_thread_pool_->stop();

    bool callback_called = false;
    auto ctx             = std::make_shared<rtp_llm::MemoryConnectorAsyncContext>(
        /*broadcast_result=*/nullptr, /*done_callback=*/[&](bool) { callback_called = true; });
    ASSERT_FALSE(ctx->done());

    EXPECT_FALSE(connector_->waitContextDoneAsync(ctx));
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    EXPECT_FALSE(callback_called);
    EXPECT_FALSE(ctx->done());

    // Re-create a thread pool for remaining tests in this process.
    connector_->wait_done_thread_pool_ = std::make_shared<autil::LockFreeThreadPool>(8, 1000, nullptr, "WaitDoneTP");
    ASSERT_TRUE(connector_->wait_done_thread_pool_->start());
}

TEST_F(KVCacheMemoryConnectorTest, waitContextDoneAsync_ReturnTrue_WhenPushTaskSucceeds) {
    ASSERT_NE(connector_->wait_done_thread_pool_, nullptr);

    std::atomic<bool> callback_called{false};
    auto              ctx = std::make_shared<rtp_llm::MemoryConnectorAsyncContext>(
        /*broadcast_result=*/nullptr, /*done_callback=*/[&](bool) { callback_called.store(true); });
    ASSERT_FALSE(ctx->done());

    EXPECT_TRUE(connector_->waitContextDoneAsync(ctx));

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < deadline && !ctx->done()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(ctx->done());
    EXPECT_TRUE(callback_called.load());
}

TEST_F(KVCacheMemoryConnectorTest, isThreadPoolFull_ReturnTrue_WhenThreadPoolNull) {
    connector_->wait_done_thread_pool_.reset();
    EXPECT_TRUE(connector_->isThreadPoolFull());
}

TEST_F(KVCacheMemoryConnectorTest, isThreadPoolFull_ReturnFalse_WhenThreadPoolNotFull) {
    ASSERT_NE(connector_->wait_done_thread_pool_, nullptr);
    EXPECT_FALSE(connector_->isThreadPoolFull());
}

TEST_F(KVCacheMemoryConnectorTest, isThreadPoolFull_ReturnTrue_WhenThreadPoolFull) {
    // Use a tiny thread pool and block its single worker, then fill the queue.
    connector_->wait_done_thread_pool_.reset();
    connector_->wait_done_thread_pool_ = std::make_shared<autil::LockFreeThreadPool>(/*thread_num=*/1,
                                                                                     /*queue_size=*/1,
                                                                                     /*thread_init_func=*/nullptr,
                                                                                     /*name=*/"IsFullTP");
    ASSERT_TRUE(connector_->wait_done_thread_pool_->start());

    std::atomic<bool> block_worker{true};
    std::atomic<bool> first_task_started{false};

    // Occupy the only worker thread.
    ASSERT_EQ(connector_->wait_done_thread_pool_->pushTask([&]() {
        first_task_started.store(true);
        while (block_worker.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }),
              autil::ThreadPoolBase::ERROR_NONE);

    // Wait until the worker actually starts executing the first task.
    const auto started_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < started_deadline && !first_task_started.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(first_task_started.load());

    // Fill the queue with a second blocking task.
    (void)connector_->wait_done_thread_pool_->pushTask([&]() {
        while (block_worker.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    // Eventually the pool should be full (queue_size=1, worker busy).
    const auto full_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < full_deadline && !connector_->isThreadPoolFull()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(connector_->isThreadPoolFull());

    // Unblock and stop to avoid leaking threads.
    block_worker.store(false);
    connector_->wait_done_thread_pool_->stop();
    connector_->wait_done_thread_pool_.reset();
}

TEST_F(KVCacheMemoryConnectorTest, clearCache_ReturnKeepReferencedItems_WhenRefCountGT1) {
    const int  layer0        = 0;
    const int  gpu_block_idx = 0;
    const auto buf           = allocator_->convertIndexToBuffer(layer0, gpu_block_idx);
    ASSERT_NE(buf.kv_addr, nullptr);
    const size_t block_size = buf.kv_addr->sizeBytes();

    std::vector<int64_t> keys{90001, 90002};
    auto                 block_indices = putItemsToCache(keys, block_size);
    ASSERT_EQ(block_indices.size(), 2u);
    ASSERT_TRUE(connector_->block_cache_->contains(keys[0]));
    ASSERT_TRUE(connector_->block_cache_->contains(keys[1]));

    auto pool = requireExistingBlockPool(block_size);
    ASSERT_NE(pool, nullptr);

    // Make first block ref count > 1 so clearCache() should keep it.
    pool->blockCacheReference({block_indices[0]});
    ASSERT_GT(pool->getBlockCacheRefCount(block_indices[0]), 1);

    const auto free_before = pool->freeBlocksNum();
    connector_->clearCache();

    EXPECT_TRUE(connector_->block_cache_->contains(keys[0]));
    EXPECT_FALSE(connector_->block_cache_->contains(keys[1]));
    // One block should have been freed back.
    EXPECT_EQ(pool->freeBlocksNum(), free_before + 1);
}

}  // namespace rtp_llm::test

int main(int argc, char** argv) {
    rtp_llm::initLogger();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
