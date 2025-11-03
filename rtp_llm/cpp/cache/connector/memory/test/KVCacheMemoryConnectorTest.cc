// Copyright (c) RTP-LLM

#include <csignal>
#include <chrono>
#include <execinfo.h>
#include <thread>
#include <unistd.h>

#include <grpcpp/alarm.h>

#include "gtest/gtest.h"
#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/connector/Meta.h"
#include "rtp_llm/cpp/cache/connector/memory/KVCacheMemoryConnector.h"
#include "rtp_llm/cpp/cache/connector/memory/MemoryAsyncContext.h"
#include "rtp_llm/cpp/cache/connector/memory/MemoryBlockCache.h"
#include "rtp_llm/cpp/cache/connector/memory/test/mock/TestRpcService.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache/SingleTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
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

class TestReadMeta final: public Meta {
public:
    explicit TestReadMeta(bool enable_memory_cache, bool enable_remote_cache, std::string trace_id):
        enable_memory_cache_(enable_memory_cache), enable_remote_cache_(enable_remote_cache), trace_id_(trace_id) {}
    ~TestReadMeta() override = default;

    bool enableMemoryCache() const override {
        return enable_memory_cache_;
    }
    bool enableRemoteCache() const override {
        return enable_remote_cache_;
    }
    const std::string& trace_id() const override {
        return trace_id_;
    }
    const std::string& unique_id() const override {
        return unique_id_;
    }
    const std::vector<int64_t>& tokens() const override {
        return tokens_;
    }


private:
    bool                 enable_memory_cache_{false};
    bool                 enable_remote_cache_{false};
    std::string          trace_id_;
    std::string          unique_id_ = "";  // TODO : support lora (remote connector)
    std::vector<int64_t> tokens_;          // TODO : get tokens (remote connector)
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
        ModelSpecificConfig         model_specific_config;

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
                                   runtime_config,
                                   model_specific_config);
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

    // BlockInfo helpers: convertIndexToBuffer() now returns std::vector<BlockInfo>.
    size_t sumBlockInfosBytes(const std::vector<BlockInfo>& infos) const {
        size_t total = 0;
        for (const auto& b : infos) {
            if (b.addr && b.size_bytes > 0) {
                total += b.size_bytes;
            }
        }
        return total;
    }

    void setBlockBytes(const BlockInfo& b, size_t byte_offset, size_t byte_len, char c) const {
        ASSERT_NE(b.addr, nullptr);
        ASSERT_LE(byte_offset + byte_len, b.size_bytes);
        auto* addr = static_cast<char*>(b.addr) + byte_offset;
        if (b.is_cuda) {
            check_cuda_value(cudaMemset(addr, c, byte_len));
        } else {
            memset(addr, c, byte_len);
        }
    }

    void verifyBlockBytesEq(const BlockInfo& b, size_t byte_offset, size_t byte_len, char expected) const {
        ASSERT_NE(b.addr, nullptr);
        ASSERT_LE(byte_offset + byte_len, b.size_bytes);
        auto* addr = static_cast<const char*>(b.addr) + byte_offset;

        std::vector<unsigned char> data(byte_len, 0);
        if (b.is_cuda) {
            check_cuda_value(cudaMemcpy(data.data(), addr, byte_len, cudaMemcpyDeviceToHost));
        } else {
            memcpy(data.data(), addr, byte_len);
        }
        size_t mismatch = 0;
        for (; mismatch < byte_len; ++mismatch) {
            if (data[mismatch] != static_cast<unsigned char>(expected)) {
                break;
            }
        }
        ASSERT_EQ(mismatch, byte_len) << "mismatch at byte offset " << mismatch << " expect '" << expected << "' got 0x"
                                      << std::hex << static_cast<int>(data[mismatch]) << std::dec;
    }

    void setBlockInfosContent(const std::vector<BlockInfo>& infos, char c) const {
        for (const auto& b : infos) {
            if (!b.addr || b.size_bytes == 0) {
                continue;
            }
            setBlockBytes(b, /*byte_offset=*/0, b.size_bytes, c);
        }
    }

    void verifyBlockInfosContent(const std::vector<BlockInfo>& infos, char c) const {
        for (const auto& b : infos) {
            if (!b.addr || b.size_bytes == 0) {
                continue;
            }
            verifyBlockBytesEq(b, /*byte_offset=*/0, b.size_bytes, c);
        }
    }

    void verifyGpuBufferContent(const std::vector<KVCacheMemoryConnector::LayerBlock>& gpu_layer_blocks) const {
        for (const auto& layer_block : gpu_layer_blocks) {
            const auto gpu_bufs = allocator_->convertIndexToBuffer(layer_block.layer_id, layer_block.block_id);
            ASSERT_GT(sumBlockInfosBytes(gpu_bufs), 0u);
            verifyBlockInfosContent(gpu_bufs, static_cast<char>('k' + layer_block.layer_id));
        }
    }
    void verifyCpuBufferContent(const std::vector<KVCacheMemoryConnector::LayerBlock>& gpu_layer_blocks,
                                BlockIdxType                                           mem_block_index,
                                size_t                                                 mem_block_size) const {
        auto pool = requireExistingBlockPool(mem_block_size);
        ASSERT_NE(pool, nullptr);
        const auto mem_bufs = pool->convertIndexToBuffer(0, mem_block_index);
        ASSERT_EQ(mem_bufs.size(), 1u);
        const auto& mem_buffer = mem_bufs[0];
        ASSERT_NE(mem_buffer.addr, nullptr);
        ASSERT_GE(mem_buffer.size_bytes, mem_block_size);

        size_t offset = 0;
        for (const auto& layer_block : gpu_layer_blocks) {
            const auto gpu_bufs = allocator_->convertIndexToBuffer(layer_block.layer_id, layer_block.block_id);
            const auto bytes    = sumBlockInfosBytes(gpu_bufs);
            ASSERT_GT(bytes, 0u);

            const char expected_k = static_cast<char>('k' + layer_block.layer_id);
            verifyBlockBytesEq(mem_buffer, offset, bytes, expected_k);
            offset += bytes;
        }
    }
    void prepareBufferContent(const std::vector<KVCacheMemoryConnector::LayerBlock>& gpu_layer_blocks,
                              BlockIdxType&                                          mem_block_index,
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
            const auto gpu_bufs = allocator_->convertIndexToBuffer(layer_block.layer_id, layer_block.block_id);
            const auto bytes    = sumBlockInfosBytes(gpu_bufs);
            ASSERT_GT(bytes, 0u);
            if (fill_gpu) {
                setBlockInfosContent(gpu_bufs, static_cast<char>('k' + layer_block.layer_id));
            }
            total += bytes;
        }

        // 申请memory block
        auto pool = ensureBlockPool(total);
        ASSERT_NE(pool, nullptr);
        auto mem_blocks = pool->malloc(1);
        ASSERT_EQ(mem_blocks.size(), 1u);
        const BlockIdxType malloced_mem_block_index = static_cast<BlockIdxType>(mem_blocks[0]);
        const auto         mem_bufs                 = pool->convertIndexToBuffer(0, malloced_mem_block_index);
        ASSERT_EQ(mem_bufs.size(), 1u);
        const auto& mem_buffer = mem_bufs[0];
        ASSERT_NE(mem_buffer.addr, nullptr);
        EXPECT_GE(mem_buffer.size_bytes, total);

        // 给mem_buffer填充数据
        if (fill_cpu) {
            size_t offset = 0;
            for (const auto& layer_block : gpu_layer_blocks) {
                const auto gpu_bufs = allocator_->convertIndexToBuffer(layer_block.layer_id, layer_block.block_id);
                const auto bytes    = sumBlockInfosBytes(gpu_bufs);
                ASSERT_GT(bytes, 0u);
                setBlockBytes(mem_buffer, offset, bytes, static_cast<char>('k' + layer_block.layer_id));
                offset += bytes;
            }
        }

        mem_block_index = malloced_mem_block_index;
        mem_block_size  = total;
    }
    void addOneCopyInfoToPb(MemoryOperationRequestPB&                              req,
                            const std::vector<KVCacheMemoryConnector::LayerBlock>& gpu_layer_blocks,
                            BlockIdxType                                           mem_block_index,
                            size_t                                                 mem_block_size) const {
        auto* gb = req.add_gpu_blocks();
        for (const auto& layer_block : gpu_layer_blocks) {
            auto* lb = gb->add_layer_blocks();
            lb->set_layer_id(layer_block.layer_id);
            lb->set_block_id(layer_block.block_id);
        }
        req.add_mem_block_ids(static_cast<int32_t>(mem_block_index));
        req.add_mem_block_sizes(mem_block_size);
    }
    LayerBlockIds makeLayerBlockIds(const std::vector<std::vector<BlockIdxType>>& per_layer_block_indices,
                                    size_t                                        cache_keys_num) const {
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
    std::shared_ptr<KVCacheResource>
    makeCacheResource(const CacheKeysType&                          cache_keys,
                      const std::vector<std::vector<BlockIdxType>>& per_layer_block_indices,
                      size_t                                        reuse_len = 0) const {
        auto res             = std::make_shared<KVCacheResource>();
        res->cache_keys      = cache_keys;
        res->layer_block_ids = makeLayerBlockIds(per_layer_block_indices, cache_keys.size());
        // reuse_len in these tests means "GPU already-reused prefix length".
        // KVCacheResource::reuseBlockNum() is derived from (device + memory + remote),
        // so set device reuse here to make asyncMatch/asyncRead semantics consistent.
        res->setDeviceReuseBlockNum(reuse_len);
        // These unit tests want to include the whole cache_keys range by default.
        res->setLastBlockAligned(true);
        return res;
    }
    std::vector<BlockIdxType> putItemsToCache(const CacheKeysType& keys, size_t mem_block_size) const {
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
            const BlockIdxType block_idx = static_cast<BlockIdxType>(blocks[0]);
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
    bool waitUntilDone(const std::shared_ptr<rtp_llm::AsyncContext>& ctx, int timeout_ms = 3000) const {
        if (!ctx) {
            return false;
        }
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < deadline) {
            if (ctx->done()) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return ctx->done();
    }
};

TEST_F(KVCacheMemoryConnectorTest, init_ReturnFalse_NoWorkerAddrs) {
    // 构造空的 worker 地址，BroadcastManager::init() 会失败；业务代码使用 RTP_LLM_CHECK，
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
    EXPECT_EQ(conn->broadcast_manager_, nullptr);
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
    EXPECT_EQ(conn->broadcast_manager_, nullptr);
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
    ASSERT_NE(conn->broadcast_manager_, nullptr);
    EXPECT_EQ(conn->broadcast_manager_->workerNum(), server_addrs_.size());
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
    const size_t                           N = 3;
    CacheKeysType                          cache_keys{70001, 70002, 70003};
    std::vector<std::vector<BlockIdxType>> lbs_vec{{1, 1, 1}, {2, 2, 2}};
    auto                                   res = makeCacheResource(cache_keys, lbs_vec, /*reuse_len=*/N);

    // Even if memory has matches, asyncMatch should skip when gpu reuse covers all keys.
    const auto   bufs     = allocator_->convertIndexToBuffer(0, 0);
    const size_t mem_size = sumBlockInfosBytes(bufs);
    ASSERT_GT(mem_size, 0u);
    putItemsToCache(cache_keys, mem_size);

    auto meta      = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true,  /*enable_remote_cache=*/false, "");
    auto match_ctx = connector_->asyncMatch(res, meta);
    EXPECT_EQ(match_ctx, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, asyncMatch_ReturnNull_WhenNoPrefixMatched) {
    CacheKeysType                          cache_keys{71001, 71002};
    std::vector<std::vector<BlockIdxType>> lbs_vec{{1, 1}};
    auto                                   res = makeCacheResource(cache_keys, lbs_vec, /*reuse_len=*/0);

    // No cache prefill => matched_num == 0
    auto meta      = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true,  /*enable_remote_cache=*/false, "");
    auto match_ctx = connector_->asyncMatch(res, meta);
    EXPECT_EQ(match_ctx, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, asyncMatch_ReturnMatchedNum_WhenPrefixMatchedAndStopAtFirstMiss) {
    CacheKeysType                          cache_keys{72001, 72002, 72003};
    std::vector<std::vector<BlockIdxType>> lbs_vec{{1, 1, 1}};
    auto                                   res = makeCacheResource(cache_keys, lbs_vec, /*reuse_len=*/0);

    const auto   bufs     = allocator_->convertIndexToBuffer(0, 0);
    const size_t mem_size = sumBlockInfosBytes(bufs);
    ASSERT_GT(mem_size, 0u);

    // Only prefill first 2 keys in cache; 3rd miss => matched_num should be 2.
    putItemsToCache({cache_keys[0], cache_keys[1]}, mem_size);

    auto meta      = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true,  /*enable_remote_cache=*/false, "");
    auto match_ctx = connector_->asyncMatch(res, meta);
    ASSERT_NE(match_ctx, nullptr);
    EXPECT_TRUE(match_ctx->done());
    EXPECT_TRUE(match_ctx->success());
    EXPECT_EQ(match_ctx->matchedBlockCount(), 2u);
}

TEST_F(KVCacheMemoryConnectorTest, asyncRead_ReturnNull_OnInvalidInputs) {
    // resource is nullptr
    auto ctx_null =
        connector_->asyncRead(nullptr, nullptr, nullptr, /*start_read_block_index=*/0, /*read_block_num=*/0);
    EXPECT_EQ(ctx_null, nullptr);

    // empty cache_keys
    auto res_empty_keys = makeCacheResource({}, {{1}});
    auto ctx1 =
        connector_->asyncRead(res_empty_keys, nullptr, nullptr, /*start_read_block_index=*/0, /*read_block_num=*/0);
    EXPECT_EQ(ctx1, nullptr);

    // empty layer_block_ids
    auto res_empty_lbs        = std::make_shared<KVCacheResource>();
    res_empty_lbs->cache_keys = CacheKeysType{1};
    res_empty_lbs->layer_block_ids.clear();
    auto ctx2 =
        connector_->asyncRead(res_empty_lbs, nullptr, nullptr, /*start_read_block_index=*/0, /*read_block_num=*/0);
    EXPECT_EQ(ctx2, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, asyncRead_ReturnNull_WhenReuseLenGEKeys) {
    const size_t                           N = 3;
    CacheKeysType                          cache_keys{10001, 10002, 10003};
    std::vector<std::vector<BlockIdxType>> lbs_vec{{1, 1, 1}, {2, 2, 2}};
    auto                                   res = makeCacheResource(cache_keys, lbs_vec, N);

    // With reuse_len == keys size, asyncMatch should skip and there is nothing to read.
    auto meta      = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true,  /*enable_remote_cache=*/false, "");
    auto match_ctx = connector_->asyncMatch(res, meta);
    EXPECT_EQ(match_ctx, nullptr);
    EXPECT_EQ(res->reuseBlockNum(), N);
}

TEST_F(KVCacheMemoryConnectorTest, asyncRead_ReturnNull_WhenPlanEmpty) {
    // Simulate mismatch between match result and current cache state:
    // asyncRead does NOT call asyncMatch any more, so it relies on match_context + meta.
    // Here cache has no items, so buildCopyPlanForRead should fail and asyncRead returns nullptr.
    CacheKeysType                          cache_keys{20001, 20002};
    std::vector<std::vector<BlockIdxType>> lbs_vec{{3, 3}, {4, 4}};
    auto                                   res = makeCacheResource(cache_keys, lbs_vec);

    class TestMatchContext: public rtp_llm::AsyncMatchContext {
    public:
        explicit TestMatchContext(size_t matched): matched_(matched) {}
        void waitDone() override {
            return;
        }
        bool done() const override {
            return true;
        }
        bool success() const override {
            return true;
        }
        size_t matchedBlockCount() const override {
            return matched_;
        }

    private:
        size_t matched_{0};
    };

    auto match_ctx = std::make_shared<TestMatchContext>(/*matched=*/1);
    auto meta      = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true,  /*enable_remote_cache=*/false, "");
    auto ctx       = connector_->asyncRead(res, meta, match_ctx, /*start_read_block_index=*/0, /*read_block_num=*/1);
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, asyncRead_Success_IncrementsReuseLen_ByMatchedPrefix) {
    // 初始 reuse_len=1, 内存全部命中 => mem_match_len=3，最终 reuse_len=3
    CacheKeysType cache_keys{40001, 40002, 40003};

    const auto   bufs     = allocator_->convertIndexToBuffer(0, 0);
    const size_t mem_size = sumBlockInfosBytes(bufs);
    ASSERT_GT(mem_size, 0u);

    auto block_indices = putItemsToCache(cache_keys, mem_size);
    ASSERT_EQ(block_indices.size(), cache_keys.size());

    std::vector<std::vector<BlockIdxType>> lbs_vec{
        {101, 102, 103},  // layer0
        {201, 202, 203},  // layer1
    };
    auto res = makeCacheResource(cache_keys, lbs_vec, 1);

    auto meta      = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true,  /*enable_remote_cache=*/false, "");
    auto match_ctx = connector_->asyncMatch(res, meta);
    ASSERT_NE(match_ctx, nullptr);
    const int reuse_num = static_cast<int>(res->reuseBlockNum());
    const int read_num  = static_cast<int>(match_ctx->matchedBlockCount()) - reuse_num;
    ASSERT_GT(read_num, 0);
    auto ctx = connector_->asyncRead(res, meta, match_ctx, reuse_num, read_num);
    ASSERT_NE(ctx, nullptr);
    auto mem_ctx = std::dynamic_pointer_cast<rtp_llm::MemoryAsyncContext>(ctx);
    ASSERT_NE(mem_ctx, nullptr);
    ASSERT_TRUE(waitUntilDone(ctx));
    EXPECT_TRUE(ctx->success());
    EXPECT_EQ(res->reuseBlockNum(), 2u);  // last cache key will not be read
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
    auto broadcast_manager = std::make_shared<BroadcastManager>(addrs);
    ASSERT_TRUE(broadcast_manager->init());
    connector_->broadcast_manager_ = broadcast_manager;

    CacheKeysType cache_keys{50001, 50002};

    const auto   bufs     = allocator_->convertIndexToBuffer(0, 0);
    const size_t mem_size = sumBlockInfosBytes(bufs);
    ASSERT_GT(mem_size, 0u);

    auto block_indices = putItemsToCache(cache_keys, mem_size);
    ASSERT_EQ(block_indices.size(), cache_keys.size());

    std::vector<std::vector<BlockIdxType>> lbs_vec{{11, 12}, {21, 22}};
    auto                                   res = makeCacheResource(cache_keys, lbs_vec);

    auto meta      = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true,  /*enable_remote_cache=*/false, "");
    auto match_ctx = connector_->asyncMatch(res, meta);
    ASSERT_NE(match_ctx, nullptr);
    const int start_read_block_index = 0;
    const int read_block_num         = static_cast<int>(match_ctx->matchedBlockCount());
    auto      ctx = connector_->asyncRead(res, meta, match_ctx, start_read_block_index, read_block_num);
    ASSERT_NE(ctx, nullptr);
    auto mem_ctx = std::dynamic_pointer_cast<rtp_llm::MemoryAsyncContext>(ctx);
    ASSERT_NE(mem_ctx, nullptr);
    ASSERT_TRUE(waitUntilDone(ctx));
    EXPECT_FALSE(ctx->success());
    EXPECT_EQ(res->reuseBlockNum(), 0u);

    connector_->broadcast_manager_.reset();
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
    auto broadcast_manager = std::make_shared<BroadcastManager>(addrs);
    ASSERT_TRUE(broadcast_manager->init());
    connector_->broadcast_manager_ = broadcast_manager;

    CacheKeysType cache_keys{60001, 60002};

    const auto   bufs     = allocator_->convertIndexToBuffer(0, 0);
    const size_t mem_size = sumBlockInfosBytes(bufs);
    ASSERT_GT(mem_size, 0u);

    auto block_indices = putItemsToCache(cache_keys, mem_size);
    ASSERT_EQ(block_indices.size(), cache_keys.size());

    std::vector<std::vector<BlockIdxType>> lbs_vec{{31, 32}, {41, 42}};
    auto                                   res = makeCacheResource(cache_keys, lbs_vec);

    auto meta      = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true,  /*enable_remote_cache=*/false, "");
    auto match_ctx = connector_->asyncMatch(res, meta);
    ASSERT_NE(match_ctx, nullptr);
    const int start_read_block_index = 0;
    const int read_block_num         = static_cast<int>(match_ctx->matchedBlockCount());
    auto      ctx = connector_->asyncRead(res, meta, match_ctx, start_read_block_index, read_block_num);
    ASSERT_NE(ctx, nullptr);
    auto mem_ctx = std::dynamic_pointer_cast<rtp_llm::MemoryAsyncContext>(ctx);
    ASSERT_NE(mem_ctx, nullptr);
    ASSERT_TRUE(waitUntilDone(ctx));
    EXPECT_FALSE(ctx->success());
    EXPECT_EQ(res->reuseBlockNum(), 0u);

    connector_->broadcast_manager_.reset();
    for (auto& s : servers) {
        s->shutdown();
    }
    servers.clear();
    addrs.clear();
}

TEST_F(KVCacheMemoryConnectorTest, asyncRead_ReturnNull_WhenThreadPoolFull) {
    // 在单测里稳定模拟 startCopyAsync() 失败：把线程池替换成“未启动”的线程池，
    // 这样 pushTask 会返回非 ERROR_NONE，从而 asyncRead 返回 nullptr。
    auto old_pool = connector_->wait_done_thread_pool_;

    connector_->wait_done_thread_pool_ = std::make_shared<autil::LockFreeThreadPool>(/*thread_num=*/1,
                                                                                     /*queue_size=*/1,
                                                                                     /*thread_init_func=*/nullptr,
                                                                                     /*name=*/"AsyncReadNotStartedTP");
    // 验证线程池未启动时 pushTask 会失败（避免平台/实现差异导致用例不稳定）。
    EXPECT_NE(connector_->wait_done_thread_pool_->pushTask([]() {}), autil::ThreadPoolBase::ERROR_NONE);

    CacheKeysType cache_keys{70001, 70002};
    const auto    bufs     = allocator_->convertIndexToBuffer(0, 0);
    const size_t  mem_size = sumBlockInfosBytes(bufs);
    ASSERT_GT(mem_size, 0u);
    putItemsToCache(cache_keys, mem_size);
    auto res       = makeCacheResource(cache_keys, {{1, 2}, {3, 4}});
    auto meta      = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true,  /*enable_remote_cache=*/false, "");
    auto match_ctx = connector_->asyncMatch(res, meta);
    ASSERT_NE(match_ctx, nullptr);
    const int start_read_block_index = 0;
    const int read_block_num         = static_cast<int>(match_ctx->matchedBlockCount());

    auto ctx = connector_->asyncRead(res, meta, match_ctx, start_read_block_index, read_block_num);
    EXPECT_EQ(ctx, nullptr);

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
    const int                              layer0        = 0;
    const int                              gpu_block_idx = 1;
    CacheKeysType                          cache_keys{10, 11};
    std::vector<std::vector<BlockIdxType>> lbs_vec{
        {static_cast<BlockIdxType>(gpu_block_idx), static_cast<BlockIdxType>(gpu_block_idx)}};
    auto res = makeCacheResource(cache_keys, lbs_vec);

    const auto   bufs  = allocator_->convertIndexToBuffer(layer0, gpu_block_idx);
    const size_t total = sumBlockInfosBytes(bufs);
    ASSERT_GT(total, 0u);

    // 预置到 cache
    auto block_indices = putItemsToCache(cache_keys, total);
    ASSERT_EQ(block_indices.size(), cache_keys.size());

    const size_t cache_size_before = connector_->block_cache_->size();

    auto meta = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true, /*enable_remote_cache=*/false, "");
    auto ctx  = connector_->asyncWrite(res, meta);
    ASSERT_EQ(ctx, nullptr);
    EXPECT_EQ(connector_->block_cache_->size(), cache_size_before);
}

TEST_F(KVCacheMemoryConnectorTest, asyncWrite_ReturnSuccess_WhenPrefixInCacheOnlyWriteSuffix) {
    const int                              layer0 = 0;
    CacheKeysType                          cache_keys{60001, 60002, 60003};
    std::vector<std::vector<BlockIdxType>> lbs_vec{{1, 2, 3}};
    auto                                   res = makeCacheResource(cache_keys, lbs_vec);

    const auto   bufs  = allocator_->convertIndexToBuffer(layer0, /*block_id=*/1);
    const size_t total = sumBlockInfosBytes(bufs);
    ASSERT_GT(total, 0u);

    // Pre-insert only the first key, so cpu_matched_num should be 1 and only suffix gets written.
    auto pre_blocks = putItemsToCache({cache_keys[0]}, total);
    ASSERT_EQ(pre_blocks.size(), 1u);
    ASSERT_TRUE(connector_->block_cache_->contains(cache_keys[0]));
    const size_t cache_before = connector_->block_cache_->size();

    auto meta = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true, /*enable_remote_cache=*/false, "");
    auto ctx  = connector_->asyncWrite(res, meta);
    ASSERT_NE(ctx, nullptr);
    auto mem_ctx = std::dynamic_pointer_cast<rtp_llm::MemoryAsyncContext>(ctx);
    ASSERT_NE(mem_ctx, nullptr);
    ASSERT_TRUE(waitUntilDone(ctx));
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
    auto broadcast_manager = std::make_shared<BroadcastManager>(addrs);
    ASSERT_TRUE(broadcast_manager->init());
    connector_->broadcast_manager_ = broadcast_manager;

    CacheKeysType                          cache_keys{61001, 61002};
    std::vector<std::vector<BlockIdxType>> lbs_vec{{1, 2}};
    auto                                   res = makeCacheResource(cache_keys, lbs_vec);

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
            total += sumBlockInfosBytes(buffers);
        }
        return total;
    };
    const size_t total_key0 = totalBytesForKeyIndex(/*key_index=*/0);
    const size_t total_key1 = totalBytesForKeyIndex(/*key_index=*/1);
    ASSERT_GT(total_key0, 0u);
    ASSERT_GT(total_key1, 0u);
    ASSERT_NE(ensureBlockPool(total_key0), nullptr);
    ASSERT_NE(ensureBlockPool(total_key1), nullptr);

    auto meta = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true, /*enable_remote_cache=*/false, "");
    auto ctx  = connector_->asyncWrite(res, meta);
    ASSERT_NE(ctx, nullptr);

    // While in flight, insert the first key so write_done should skip inserting it.
    auto pre_blocks = putItemsToCache({cache_keys[0]}, total_key0);
    ASSERT_EQ(pre_blocks.size(), 1u);
    ASSERT_TRUE(connector_->block_cache_->contains(cache_keys[0]));

    auto mem_ctx = std::dynamic_pointer_cast<rtp_llm::MemoryAsyncContext>(ctx);
    ASSERT_NE(mem_ctx, nullptr);
    ASSERT_TRUE(waitUntilDone(ctx));
    EXPECT_TRUE(ctx->success());

    EXPECT_TRUE(connector_->block_cache_->contains(cache_keys[0]));
    EXPECT_TRUE(connector_->block_cache_->contains(cache_keys[1]));

    connector_->broadcast_manager_.reset();
    for (auto& s : servers) {
        s->shutdown();
    }
    servers.clear();
    addrs.clear();
}

TEST_F(KVCacheMemoryConnectorTest, asyncWrite_ReturnNull_WhenBuildPlanEmpty) {
    // 所有 layer 对于第一个未命中 key 的 blockIdx 都为 NULL，导致 plan 为空
    CacheKeysType cache_keys{100, 101};
    // 2 层，全部 NULL
    std::vector<std::vector<BlockIdxType>> lbs_vec{{NULL_BLOCK_IDX, NULL_BLOCK_IDX}, {NULL_BLOCK_IDX, NULL_BLOCK_IDX}};
    auto                                   res = makeCacheResource(cache_keys, lbs_vec);

    auto meta = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true, /*enable_remote_cache=*/false, "");
    auto ctx  = connector_->asyncWrite(res, meta);
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, asyncWrite_Success_AddsToBlockCache_AndKeepsMemBlocks) {
    // 默认 RPC 服务均返回 OK + mem success
    const int                              layer0        = 0;
    const int                              gpu_block_idx = 2;
    const size_t                           N             = 2;
    CacheKeysType                          cache_keys{200, 201};
    std::vector<std::vector<BlockIdxType>> lbs_vec{
        {static_cast<BlockIdxType>(gpu_block_idx), static_cast<BlockIdxType>(gpu_block_idx + 1)}};
    auto res = makeCacheResource(cache_keys, lbs_vec);

    const auto   bufs  = allocator_->convertIndexToBuffer(layer0, gpu_block_idx);
    const size_t total = sumBlockInfosBytes(bufs);
    ASSERT_GT(total, 0u);
    auto pool = ensureBlockPool(total);
    ASSERT_NE(pool, nullptr);
    const size_t free_before  = pool->freeBlocksNum();
    const size_t cache_before = connector_->block_cache_->size();

    auto meta = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true, /*enable_remote_cache=*/false, "");
    auto ctx  = connector_->asyncWrite(res, meta);
    ASSERT_NE(ctx, nullptr);
    auto mem_ctx = std::dynamic_pointer_cast<rtp_llm::MemoryAsyncContext>(ctx);
    ASSERT_NE(mem_ctx, nullptr);
    ASSERT_TRUE(waitUntilDone(ctx));
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
    auto broadcast_manager = std::make_shared<BroadcastManager>(addrs);
    ASSERT_TRUE(broadcast_manager->init());
    connector_->broadcast_manager_ = broadcast_manager;

    const int                              layer0        = 0;
    const int                              gpu_block_idx = 1;
    CacheKeysType                          cache_keys{301, 302};
    std::vector<std::vector<BlockIdxType>> lbs_vec{
        {static_cast<BlockIdxType>(gpu_block_idx), static_cast<BlockIdxType>(gpu_block_idx)}};
    auto res = makeCacheResource(cache_keys, lbs_vec);

    const auto   bufs  = allocator_->convertIndexToBuffer(layer0, gpu_block_idx);
    const size_t total = sumBlockInfosBytes(bufs);
    ASSERT_GT(total, 0u);
    auto pool = ensureBlockPool(total);
    ASSERT_NE(pool, nullptr);
    const size_t free_before  = pool->freeBlocksNum();
    const size_t cache_before = connector_->block_cache_->size();

    auto meta = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true, /*enable_remote_cache=*/false, "");
    auto ctx  = connector_->asyncWrite(res, meta);
    ASSERT_NE(ctx, nullptr);
    auto mem_ctx = std::dynamic_pointer_cast<rtp_llm::MemoryAsyncContext>(ctx);
    ASSERT_NE(mem_ctx, nullptr);
    ASSERT_TRUE(waitUntilDone(ctx));
    EXPECT_FALSE(ctx->success());
    // 应未插入缓存
    EXPECT_EQ(connector_->block_cache_->size(), cache_before);
    // 分配的块应被回收
    EXPECT_EQ(pool->freeBlocksNum(), free_before);

    connector_->broadcast_manager_.reset();
    for (auto& s : servers) {
        s->shutdown();
    }
    servers.clear();
    addrs.clear();
}

TEST_F(KVCacheMemoryConnectorTest, asyncWrite_ReturnNull_WhenThreadPoolFull) {
    // 在单测里稳定模拟 startCopyAsync() 失败：把线程池替换成“未启动”的线程池，
    // 这样 pushTask 会返回非 ERROR_NONE，从而 asyncWrite 返回 nullptr。
    auto old_pool = connector_->wait_done_thread_pool_;

    connector_->wait_done_thread_pool_ = std::make_shared<autil::LockFreeThreadPool>(/*thread_num=*/1,
                                                                                     /*queue_size=*/1,
                                                                                     /*thread_init_func=*/nullptr,
                                                                                     /*name=*/"AsyncWriteNotStartedTP");
    // 验证线程池未启动时 pushTask 会失败（避免平台/实现差异导致用例不稳定）。
    EXPECT_NE(connector_->wait_done_thread_pool_->pushTask([]() {}), autil::ThreadPoolBase::ERROR_NONE);

    const int                              layer0 = 0;
    CacheKeysType                          cache_keys{71001, 71002, 71003};
    std::vector<std::vector<BlockIdxType>> lbs_vec{{1, 2, 3}};
    auto                                   res = makeCacheResource(cache_keys, lbs_vec);

    // Pre-insert one key so cpu_matched_num < cache_keys.size() and it reaches the thread-pool-full check.
    const auto   bufs  = allocator_->convertIndexToBuffer(layer0, /*block_id=*/1);
    const size_t total = sumBlockInfosBytes(bufs);
    ASSERT_GT(total, 0u);
    ASSERT_NE(ensureBlockPool(total), nullptr);
    (void)putItemsToCache({cache_keys[0]}, total);

    auto meta = std::make_shared<TestReadMeta>(/*enable_memory_cache=*/true, /*enable_remote_cache=*/false, "");
    auto ctx  = connector_->asyncWrite(res, meta);
    EXPECT_EQ(ctx, nullptr);

    connector_->wait_done_thread_pool_.reset();
    connector_->wait_done_thread_pool_ = old_pool;
}

TEST_F(KVCacheMemoryConnectorTest, sendCopyPlan_ReturnContext_WhenNoWorkers_NoOp) {
    // BroadcastManager treats "0 workers" as a no-op success (requests.size()==workerNum()==0).
    connector_->broadcast_manager_->worker_addrs_.clear();

    std::vector<KVCacheMemoryConnector::CopyInfoPerKey> infos;
    auto result = connector_->sendCopyPlan(infos, KVCacheMemoryConnector::CopyDirection::H2D);
    ASSERT_NE(result, nullptr);
    result->waitDone();
    EXPECT_TRUE(result->success());
    EXPECT_TRUE(result->responses().empty());
}

TEST_F(KVCacheMemoryConnectorTest, sendCopyPlan_ReturnContext_AllRanksSuccess) {
    const int    layer_id      = 0;
    const int    gpu_block_idx = 2;
    const auto   gpu_bufs      = allocator_->convertIndexToBuffer(layer_id, gpu_block_idx);
    const size_t total         = sumBlockInfosBytes(gpu_bufs);
    ASSERT_GT(total, 0u);

    KVCacheMemoryConnector::CopyInfoPerKey info;
    info.cache_key        = 1;
    info.mem_block_index  = static_cast<BlockIdxType>(1);
    info.mem_block_size   = total;
    info.gpu_layer_blocks = {KVCacheMemoryConnector::LayerBlock{layer_id, static_cast<BlockIdxType>(gpu_block_idx)}};
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
    auto broadcast_manager = std::make_shared<BroadcastManager>(addrs);
    ASSERT_TRUE(broadcast_manager->init());
    connector_->broadcast_manager_ = broadcast_manager;

    const int    layer_id      = 0;
    const int    gpu_block_idx = 2;
    const auto   gpu_bufs      = allocator_->convertIndexToBuffer(layer_id, gpu_block_idx);
    const size_t total         = sumBlockInfosBytes(gpu_bufs);
    ASSERT_GT(total, 0u);

    KVCacheMemoryConnector::CopyInfoPerKey info;
    info.cache_key        = 2;
    info.mem_block_index  = static_cast<BlockIdxType>(1);
    info.mem_block_size   = total;
    info.gpu_layer_blocks = {KVCacheMemoryConnector::LayerBlock{layer_id, static_cast<BlockIdxType>(gpu_block_idx)}};
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
    connector_->broadcast_manager_.reset();
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
    auto broadcast_manager = std::make_shared<BroadcastManager>(addrs);
    ASSERT_TRUE(broadcast_manager->init());
    connector_->broadcast_manager_ = broadcast_manager;

    const int    layer_id      = 0;
    const int    gpu_block_idx = 2;
    const auto   gpu_bufs      = allocator_->convertIndexToBuffer(layer_id, gpu_block_idx);
    const size_t total         = sumBlockInfosBytes(gpu_bufs);
    ASSERT_GT(total, 0u);

    KVCacheMemoryConnector::CopyInfoPerKey info;
    info.cache_key        = 3;
    info.mem_block_index  = static_cast<BlockIdxType>(1);
    info.mem_block_size   = total;
    info.gpu_layer_blocks = {KVCacheMemoryConnector::LayerBlock{layer_id, static_cast<BlockIdxType>(gpu_block_idx)}};
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
    EXPECT_THROW((void)connector_->copyCache(req, resp), rtp_llm::RTPException);
}

TEST_F(KVCacheMemoryConnectorTest, copyCache_ReturnFalse_InvalidMemBlockSize) {
    const int    layer_id      = 0;
    const int    gpu_block_idx = 1;
    const auto   gpu_bufs      = allocator_->convertIndexToBuffer(layer_id, gpu_block_idx);
    const size_t total         = sumBlockInfosBytes(gpu_bufs);
    ASSERT_GT(total, 0u);

    MemoryOperationRequestPB req;
    auto*                    gb = req.add_gpu_blocks();
    auto*                    lb = gb->add_layer_blocks();
    lb->set_layer_id(layer_id);
    lb->set_block_id(gpu_block_idx);
    req.add_mem_block_ids(1);
    req.add_mem_block_sizes(0);  // invalid mem block size
    req.set_copy_direction(MemoryOperationRequestPB::H2D);

    MemoryOperationResponsePB resp;
    EXPECT_THROW(connector_->copyCache(req, resp), rtp_llm::RTPException);
}

TEST_F(KVCacheMemoryConnectorTest, copyCache_ReturnFalse_InvalidLayerId_BuildCopyPlanFailed) {
    const int    valid_layer   = 0;
    const int    gpu_block_idx = 1;
    const auto   gpu_bufs      = allocator_->convertIndexToBuffer(valid_layer, gpu_block_idx);
    const size_t total         = sumBlockInfosBytes(gpu_bufs);
    ASSERT_GT(total, 0u);

    auto pool = ensureBlockPool(total);
    ASSERT_NE(pool, nullptr);
    auto mem_blocks = pool->malloc(1);
    ASSERT_EQ(mem_blocks.size(), 1u);
    const BlockIdxType mem_block_index = static_cast<BlockIdxType>(mem_blocks[0]);

    MemoryOperationRequestPB req;
    auto*                    gb = req.add_gpu_blocks();
    auto*                    lb = gb->add_layer_blocks();
    lb->set_layer_id(cache_config_.layer_num);  // out of range
    lb->set_block_id(gpu_block_idx);
    req.add_mem_block_ids(mem_block_index);
    req.add_mem_block_sizes(static_cast<int64_t>(total));
    req.set_copy_direction(MemoryOperationRequestPB::H2D);

    MemoryOperationResponsePB resp;
    // 这里会在 allocator_->convertIndexToBuffer(...) 内部触发 RTP_LLM_CHECK 抛异常，而不是返回 false。
    EXPECT_THROW(connector_->copyCache(req, resp), rtp_llm::RTPException);
}

TEST_F(KVCacheMemoryConnectorTest, copyCache_ReturnTrue_H2D_SingleLayer) {
    const int    layer_id      = 0;
    const int    gpu_block_idx = 2;
    const auto   gpu_bufs      = allocator_->convertIndexToBuffer(layer_id, gpu_block_idx);
    const size_t total         = sumBlockInfosBytes(gpu_bufs);
    ASSERT_GT(total, 0u);

    // H2D 路径需要预先存在 mem pool 与有效 block
    auto pool = ensureBlockPool(total);
    ASSERT_NE(pool, nullptr);
    auto mem_blocks = pool->malloc(1);
    ASSERT_EQ(mem_blocks.size(), 1u);
    const BlockIdxType mem_block_index = static_cast<BlockIdxType>(mem_blocks[0]);
    const auto         mem_bufs        = pool->convertIndexToBuffer(0, mem_block_index);
    ASSERT_EQ(mem_bufs.size(), 1u);
    const auto& mem_buffer = mem_bufs[0];
    ASSERT_NE(mem_buffer.addr, nullptr);
    EXPECT_GE(mem_buffer.size_bytes, total);

    // 给mem_buffer填充数据
    setBlockBytes(mem_buffer, /*byte_offset=*/0, total, 'a');

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
    verifyBlockInfosContent(gpu_bufs, 'a');
}

TEST_F(KVCacheMemoryConnectorTest, copyCache_ReturnTrue_H2D_MultiLayer) {
    // 创建两个block_size不同的memory buffer
    std::vector<KVCacheMemoryConnector::LayerBlock> gpu_layer_blocks1{
        {/*layer_id*/ 0, /*block_id*/ 1},
        {/*layer_id*/ 1, /*block_id*/ 2},
    };
    BlockIdxType mem_block_index1 = NULL_BLOCK_IDX;
    size_t       mem_block_size1  = 0;
    prepareBufferContent(gpu_layer_blocks1, mem_block_index1, mem_block_size1, /*fill_gpu=*/false, /*fill_cpu=*/true);
    ASSERT_FALSE(isNullBlockIdx(mem_block_index1));
    ASSERT_NE(mem_block_size1, 0);

    std::vector<KVCacheMemoryConnector::LayerBlock> gpu_layer_blocks2{
        {/*layer_id*/ 0, /*block_id*/ 1},
        {/*layer_id*/ 1, /*block_id*/ 2},
        {/*layer_id*/ 2, /*block_id*/ 2},
    };
    BlockIdxType mem_block_index2 = NULL_BLOCK_IDX;
    size_t       mem_block_size2  = 0;
    prepareBufferContent(gpu_layer_blocks2, mem_block_index2, mem_block_size2, /*fill_gpu=*/false, /*fill_cpu=*/true);
    ASSERT_FALSE(isNullBlockIdx(mem_block_index2));
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
    const int    layer_id      = 0;
    const int    gpu_block_idx = 3;
    const auto   gpu_bufs      = allocator_->convertIndexToBuffer(layer_id, gpu_block_idx);
    const size_t total         = sumBlockInfosBytes(gpu_bufs);
    ASSERT_GT(total, 0u);

    // 给gpu_buf填充数据
    setBlockInfosContent(gpu_bufs, 'a');

    // 为确保索引有效，仍然预先创建并分配一个块
    auto pool = ensureBlockPool(total);
    ASSERT_NE(pool, nullptr);
    auto mem_blocks = pool->malloc(1);
    ASSERT_EQ(mem_blocks.size(), 1u);
    const BlockIdxType mem_block_index = static_cast<BlockIdxType>(mem_blocks[0]);
    const auto         mem_bufs        = pool->convertIndexToBuffer(0, mem_block_index);
    ASSERT_EQ(mem_bufs.size(), 1u);
    const auto& mem_buffer = mem_bufs[0];
    ASSERT_NE(mem_buffer.addr, nullptr);
    EXPECT_GE(mem_buffer.size_bytes, total);

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
    verifyBlockBytesEq(mem_buffer, /*byte_offset=*/0, total, 'a');
}

TEST_F(KVCacheMemoryConnectorTest, copyCache_ReturnTrue_D2H_MultiLayer) {
    std::vector<KVCacheMemoryConnector::LayerBlock> gpu_layer_blocks1{
        {/*layer_id*/ 0, /*block_id*/ 1},
        {/*layer_id*/ 1, /*block_id*/ 2},
    };
    BlockIdxType mem_block_index1 = NULL_BLOCK_IDX;
    size_t       mem_block_size1  = 0;
    prepareBufferContent(gpu_layer_blocks1, mem_block_index1, mem_block_size1, /*fill_gpu=*/true, /*fill_cpu=*/false);
    ASSERT_FALSE(isNullBlockIdx(mem_block_index1));
    ASSERT_NE(mem_block_size1, 0);

    std::vector<KVCacheMemoryConnector::LayerBlock> gpu_layer_blocks2{
        {/*layer_id*/ 0, /*block_id*/ 1},
        {/*layer_id*/ 1, /*block_id*/ 2},
        {/*layer_id*/ 2, /*block_id*/ 2},
    };
    BlockIdxType mem_block_index2 = NULL_BLOCK_IDX;
    size_t       mem_block_size2  = 0;
    prepareBufferContent(gpu_layer_blocks2, mem_block_index2, mem_block_size2, /*fill_gpu=*/true, /*fill_cpu=*/false);
    ASSERT_FALSE(isNullBlockIdx(mem_block_index2));
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
        const auto gpu_bufs = allocator_->convertIndexToBuffer(lb.layer_id, lb.block_id);
        ASSERT_GT(sumBlockInfosBytes(gpu_bufs), 0u);
        setBlockInfosContent(gpu_bufs, static_cast<char>('k' + lb.layer_id));
    }

    // Allocate one memory block sized to hold both layers' kv bytes contiguously.
    size_t total_bytes = 0;
    for (const auto& lb : gpu_layer_blocks) {
        const auto gpu_bufs = allocator_->convertIndexToBuffer(lb.layer_id, lb.block_id);
        total_bytes += sumBlockInfosBytes(gpu_bufs);
    }
    ASSERT_GT(total_bytes, 0u);

    auto pool = ensureBlockPool(total_bytes);
    ASSERT_NE(pool, nullptr);
    auto mem_blocks = pool->malloc(1);
    ASSERT_EQ(mem_blocks.size(), 1u);
    const BlockIdxType mem_block_index = static_cast<BlockIdxType>(mem_blocks[0]);

    const auto mem_bufs = pool->convertIndexToBuffer(0, mem_block_index);
    ASSERT_EQ(mem_bufs.size(), 1u);
    const auto& mem_buffer = mem_bufs[0];
    ASSERT_NE(mem_buffer.addr, nullptr);
    EXPECT_GE(mem_buffer.size_bytes, total_bytes);
    setBlockBytes(mem_buffer, /*byte_offset=*/0, total_bytes, 0);

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
        const auto gpu_bufs = allocator_->convertIndexToBuffer(lb.layer_id, lb.block_id);
        const auto bytes    = sumBlockInfosBytes(gpu_bufs);
        ASSERT_GT(bytes, 0u);
        verifyBlockBytesEq(mem_buffer, byte_off, bytes, static_cast<char>('k' + lb.layer_id));
        byte_off += bytes;
    }
}

}  // namespace rtp_llm::test

int main(int argc, char** argv) {
    rtp_llm::initLogger();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
