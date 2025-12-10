// Copyright (c) RTP-LLM

#include "gtest/gtest.h"

#include "rtp_llm/cpp/cache_new/KVCacheMemoryConnector.h"
#include "rtp_llm/cpp/cache_new/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache_new/BlockPool.h"
#include "rtp_llm/cpp/cache_new/MemoryBlockCache.h"
#include "rtp_llm/cpp/cache_new/SingleTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache_new/test/mock/TestRpcService.h"
#include "rtp_llm/cpp/config/GptInitParameter.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/model_rpc/TpBroadcastManager.h"

namespace rtp_llm::test {

class KVCacheMemoryConnectorTest: public ::testing::Test {
protected:
    void SetUp() override {
        device_ = createDevice();

        cache_config_ = createMockCacheConfig();
        allocator_    = std::make_shared<SingleTypeKVCacheAllocator>(cache_config_, device_, AllocationType::DEVICE);
        ASSERT_TRUE(allocator_->init());

        const int server_num = 4;
        startRpcServer(server_num);

        connector_ = std::make_shared<KVCacheMemoryConnector>(cache_config_, allocator_, device_, server_addrs_);
        ASSERT_TRUE(connector_->init());
    }

    DeviceBase*                                 device_{nullptr};
    CacheConfig                                 cache_config_;
    std::shared_ptr<KVCacheAllocator>           allocator_;
    std::shared_ptr<KVCacheMemoryConnector>     connector_;
    std::vector<std::unique_ptr<TestRpcServer>> servers_;
    std::vector<std::string>                    server_addrs_;

private:
    DeviceBase* createDevice() const {
        DeviceFactory::initDevices(GptInitParameter());
        return DeviceFactory::getDefaultDevice();
    }
    CacheConfig createMockCacheConfig(int layer_num = 4, int block_num = 10, int seq_size_per_block = 8) const {
        constexpr int kTestMemoryCacheSizeMb      = 64;
        constexpr int kTestMemoryCacheSyncTimeout = 1000;

        CacheConfig config;
        config.layer_num                          = layer_num;
        config.block_num                          = block_num;
        config.seq_size_per_block                 = seq_size_per_block;
        config.memory_block_cache_size_mb         = kTestMemoryCacheSizeMb;
        config.memory_block_cache_sync_timeout_ms = kTestMemoryCacheSyncTimeout;

        auto mha_spec                = std::make_shared<MHAKVCacheSpec>();
        mha_spec->layer_num          = layer_num;
        mha_spec->block_nums         = block_num;
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
            ASSERT_NE(gpu_buf.k_addr, nullptr);
            // ASSERT_NE(gpu_buf.v_addr, nullptr);
            verifyBufferContent(gpu_buf.k_addr, 'k' + layer_block.layer_id);
        }
    }
    void verifyCpuBufferContent(const std::vector<KVCacheMemoryConnector::LayerBlock>& gpu_layer_blocks,
                                int                                                    mem_block_index,
                                size_t                                                 mem_block_size) const {
        auto pool = requireExistingBlockPool(mem_block_size);
        ASSERT_NE(pool, nullptr);
        auto mem_buffer = pool->convertIndexToBuffer(0, mem_block_index);
        ASSERT_NE(mem_buffer.k_addr, nullptr);
        // ASSERT_NE(mem_buffer.v_addr, nullptr);

        size_t offset = 0;
        for (const auto& layer_block : gpu_layer_blocks) {
            const auto gpu_buf = allocator_->convertIndexToBuffer(layer_block.layer_id, layer_block.block_id);
            ASSERT_NE(gpu_buf.k_addr, nullptr);
            // ASSERT_NE(gpu_buf.v_addr, nullptr);

            char expected_k = 'k' + layer_block.layer_id;

            verifyBufferContent(mem_buffer.k_addr, offset, gpu_buf.k_addr->sizeBytes(), expected_k);
            offset += gpu_buf.k_addr->sizeBytes();
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
            ASSERT_NE(gpu_buf.k_addr, nullptr);
            // ASSERT_NE(gpu_buf.v_addr, nullptr);
            if (fill_gpu) {
                setBufferContent(gpu_buf.k_addr, 'k' + layer_block.layer_id);
            }
            total += gpu_buf.k_addr->sizeBytes();
        }

        // 申请memory block
        auto pool = ensureBlockPool(total);
        ASSERT_NE(pool, nullptr);
        auto mem_blocks = pool->malloc(1);
        ASSERT_EQ(mem_blocks.size(), 1u);
        auto malloced_mem_block_index = mem_blocks[0];
        auto mem_buffer               = pool->convertIndexToBuffer(0, malloced_mem_block_index);
        ASSERT_NE(mem_buffer.k_addr, nullptr);
        EXPECT_EQ(mem_buffer.k_addr->sizeBytes(), total);

        // 给mem_buffer填充数据
        if (fill_cpu) {
            size_t offset = 0;
            for (const auto& layer_block : gpu_layer_blocks) {
                const auto gpu_buf = allocator_->convertIndexToBuffer(layer_block.layer_id, layer_block.block_id);
                ASSERT_NE(gpu_buf.k_addr, nullptr);
                // ASSERT_NE(gpu_buf.v_addr, nullptr);
                setBufferContent(mem_buffer.k_addr, offset, gpu_buf.k_addr->sizeBytes(), 'k' + layer_block.layer_id);
                offset += gpu_buf.k_addr->sizeBytes();
            }
        }

        mem_block_index = malloced_mem_block_index;
        mem_block_size  = total;
    }
    void addOneCopyInfoToPb(MemoryCopyCacheRequestPB&                              req,
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
    std::shared_ptr<KVCacheResourceV1> makeCacheResource(const std::vector<int64_t>&          cache_keys,
                                                         const std::vector<std::vector<int>>& per_layer_block_indices,
                                                         size_t                               reuse_len = 0) const {
        auto res             = std::make_shared<KVCacheResourceV1>();
        res->cacheKeys()     = cache_keys;
        res->layerBlockIds() = makeLayerBlockIds(per_layer_block_indices, cache_keys.size());
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
            auto blocks = pool->malloc(1);
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
    // 构造空的 worker 地址，init 应返回 false，但应已创建 block_cache_ 与 manager
    std::vector<std::string> empty_addrs;
    auto conn = std::make_shared<KVCacheMemoryConnector>(cache_config_, allocator_, device_, empty_addrs);
    auto ok   = conn->init();
    EXPECT_FALSE(ok);
    ASSERT_NE(conn->block_cache_, nullptr);
    ASSERT_NE(conn->tp_broadcast_manager_, nullptr);
    EXPECT_EQ(conn->tp_broadcast_manager_->workerNum(), 0u);
}

TEST_F(KVCacheMemoryConnectorTest, init_ReturnTrue_WithWorkerAddrs) {
    // 使用有效的 worker 地址，init 应成功并正确设置 manager
    auto conn = std::make_shared<KVCacheMemoryConnector>(cache_config_, allocator_, device_, server_addrs_);
    auto ok   = conn->init();
    EXPECT_TRUE(ok);
    ASSERT_NE(conn->block_cache_, nullptr);
    ASSERT_NE(conn->tp_broadcast_manager_, nullptr);
    EXPECT_EQ(conn->tp_broadcast_manager_->workerNum(), server_addrs_.size());
}

TEST_F(KVCacheMemoryConnectorTest, init_Reinit_ClearsBlockPools_And_ResetsBlockCache) {
    // 预先创建一个 block pool，并向 block_cache_ 放入条目
    const size_t block_size                          = 4096;
    cache_config_.memory_block_cache_size_mb         = 64;
    cache_config_.memory_block_cache_sync_timeout_ms = 1000;
    auto pool                                        = ensureBlockPool(block_size);
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

TEST_F(KVCacheMemoryConnectorTest, asyncRead_ReturnNull_OnInvalidInputs) {
    // resource is nullptr
    auto ctx_null = connector_->asyncRead(nullptr, nullptr);
    EXPECT_EQ(ctx_null, nullptr);

    // empty cache_keys
    auto res_empty_keys = makeCacheResource({}, {{1}});
    auto ctx1           = connector_->asyncRead(res_empty_keys, nullptr);
    EXPECT_EQ(ctx1, nullptr);

    // empty layer_block_ids
    auto res_empty_lbs = makeCacheResource({1}, {});
    auto ctx2          = connector_->asyncRead(res_empty_lbs, nullptr);
    EXPECT_EQ(ctx2, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, asyncRead_ReturnNull_WhenReuseLenGEKeys) {
    const size_t                  N = 3;
    std::vector<int64_t>          cache_keys{10001, 10002, 10003};
    std::vector<std::vector<int>> lbs_vec{{1, 1, 1}, {2, 2, 2}};
    auto                          res = makeCacheResource(cache_keys, lbs_vec, N);

    auto ctx = connector_->asyncRead(res, nullptr);
    EXPECT_EQ(ctx, nullptr);
    EXPECT_EQ(res->reuseBlocksNum(), N);
}

TEST_F(KVCacheMemoryConnectorTest, asyncRead_ReturnNull_WhenPlanEmpty) {
    // 不命中任何 key，copy plan 为空，返回 nullptr
    std::vector<int64_t>          cache_keys{20001, 20002};
    std::vector<std::vector<int>> lbs_vec{{3, 3}, {4, 4}};
    auto                          res = makeCacheResource(cache_keys, lbs_vec);

    // 未向 cache 预置命中项
    auto ctx = connector_->asyncRead(res, nullptr);
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, asyncRead_ReturnNull_WhenSendCopyPlanFails_NoWorkers) {
    // 有命中项，但没有 worker，发送失败应返回 nullptr
    std::vector<int64_t> cache_keys{30001, 30002};

    const auto   buf      = allocator_->convertIndexToBuffer(0, 0);
    const size_t mem_size = buf.k_addr->sizeBytes();

    auto block_indices = putItemsToCache(cache_keys, mem_size);
    ASSERT_EQ(block_indices.size(), cache_keys.size());

    std::vector<std::vector<int>> lbs_vec{{10, 11}, {20, 21}};
    auto                          res = makeCacheResource(cache_keys, lbs_vec);

    connector_->tp_broadcast_manager_->worker_addrs_.clear();
    auto ctx = connector_->asyncRead(res, nullptr);
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(KVCacheMemoryConnectorTest, asyncRead_Success_IncrementsReuseLen_ByMatchedPrefix) {
    // 初始 reuse_len=1, 内存全部命中 => mem_match_len=3，最终 reuse_len=3
    std::vector<int64_t> cache_keys{40001, 40002, 40003};

    const auto   buf      = allocator_->convertIndexToBuffer(0, 0);
    const size_t mem_size = buf.k_addr->sizeBytes();

    auto block_indices = putItemsToCache(cache_keys, mem_size);
    ASSERT_EQ(block_indices.size(), cache_keys.size());

    std::vector<std::vector<int>> lbs_vec{
        {101, 102, 103},  // layer0
        {201, 202, 203},  // layer1
    };
    auto res = makeCacheResource(cache_keys, lbs_vec, 1);

    auto ctx = connector_->asyncRead(res, nullptr);
    ASSERT_NE(ctx, nullptr);
    ctx->waitDone();
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
    const size_t mem_size = buf.k_addr->sizeBytes();

    auto block_indices = putItemsToCache(cache_keys, mem_size);
    ASSERT_EQ(block_indices.size(), cache_keys.size());

    std::vector<std::vector<int>> lbs_vec{{11, 12}, {21, 22}};
    auto                          res = makeCacheResource(cache_keys, lbs_vec);

    auto ctx = connector_->asyncRead(res, nullptr);
    ASSERT_NE(ctx, nullptr);
    ctx->waitDone();
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
    const size_t mem_size = buf.k_addr->sizeBytes();

    auto block_indices = putItemsToCache(cache_keys, mem_size);
    ASSERT_EQ(block_indices.size(), cache_keys.size());

    std::vector<std::vector<int>> lbs_vec{{31, 32}, {41, 42}};
    auto                          res = makeCacheResource(cache_keys, lbs_vec);

    auto ctx = connector_->asyncRead(res, nullptr);
    ASSERT_NE(ctx, nullptr);
    ctx->waitDone();
    EXPECT_FALSE(ctx->success());
    EXPECT_EQ(res->reuseBlocksNum(), 0u);

    connector_->tp_broadcast_manager_.reset();
    for (auto& s : servers) {
        s->shutdown();
    }
    servers.clear();
    addrs.clear();
}

TEST_F(KVCacheMemoryConnectorTest, buildCopyPlanForRead_ReturnEmpty_WhenNoMatch) {
    // block_cache_ 无命中，返回空
    std::vector<int64_t>          cache_keys{1, 2};
    std::vector<std::vector<int>> lbs_vec{{10, 11}, {20, 21}};
    auto                          lbs             = makeLayerBlockIds(lbs_vec, cache_keys.size());
    size_t                        cpu_matched_num = 0;
    auto plan = connector_->buildCopyPlanForRead(cache_keys, lbs, /*gpu_reuse_len=*/0, cpu_matched_num);
    EXPECT_TRUE(plan.empty());
}

TEST_F(KVCacheMemoryConnectorTest, buildCopyPlanForRead_PrefixMatch_StopsOnFirstMiss) {
    // 仅前缀两个 key 命中，第三个未命中，应在未命中处停止
    std::vector<int64_t> cache_keys{10, 11, 12};
    // 预置命中项
    const auto   buf      = allocator_->convertIndexToBuffer(0, 0);
    const size_t mem_size = buf.k_addr->sizeBytes();

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
    auto   lbs             = makeLayerBlockIds(lbs_vec, cache_keys.size());
    size_t cpu_matched_num = 0;
    auto   plan            = connector_->buildCopyPlanForRead(cache_keys, lbs, /*gpu_reuse_len=*/0, cpu_matched_num);
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
    const size_t         mem_size = buf.k_addr->sizeBytes();

    auto block_indices = putItemsToCache(cache_keys, mem_size);
    ASSERT_EQ(block_indices.size(), cache_keys.size());
    ASSERT_TRUE(connector_->block_cache_->contains(30));
    ASSERT_TRUE(connector_->block_cache_->contains(31));
    ASSERT_TRUE(connector_->block_cache_->contains(32));

    std::vector<std::vector<int>> lbs_vec{
        {10, 11, 12},  // layer0
        {20, 21, 22},  // layer1
    };
    auto   lbs             = makeLayerBlockIds(lbs_vec, cache_keys.size());
    size_t cpu_matched_num = 0;
    auto   plan            = connector_->buildCopyPlanForRead(cache_keys, lbs, /*gpu_reuse_len=*/2, cpu_matched_num);
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
    const size_t         mem_size = buf.k_addr->sizeBytes();

    auto block_indices = putItemsToCache(cache_keys, mem_size);
    ASSERT_EQ(block_indices.size(), cache_keys.size());

    ASSERT_TRUE(connector_->block_cache_->contains(40));

    std::vector<std::vector<int>> lbs_vec{
        {7},               // layer0
        {NULL_BLOCK_IDX},  // layer1 should be skipped
    };
    auto   lbs             = makeLayerBlockIds(lbs_vec, cache_keys.size());
    size_t cpu_matched_num = 0;
    auto   plan            = connector_->buildCopyPlanForRead(cache_keys, lbs, /*gpu_reuse_len=*/0, cpu_matched_num);
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
    const size_t         mem_size = buf.k_addr->sizeBytes();

    auto block_indices = putItemsToCache(cache_keys, mem_size);
    ASSERT_EQ(block_indices.size(), cache_keys.size());

    ASSERT_TRUE(connector_->block_cache_->contains(50));

    std::vector<std::vector<int>> empty_layers;
    auto                          lbs             = makeLayerBlockIds(empty_layers, cache_keys.size());
    size_t                        cpu_matched_num = 0;
    auto plan = connector_->buildCopyPlanForRead(cache_keys, lbs, /*gpu_reuse_len=*/0, cpu_matched_num);
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
    const size_t mem_size = buf.k_addr->sizeBytes() * 2;  // custom size

    auto block_indices = putItemsToCache(cache_keys, mem_size);
    ASSERT_EQ(block_indices.size(), cache_keys.size());

    ASSERT_TRUE(connector_->block_cache_->contains(60));

    std::vector<std::vector<int>> lbs_vec{
        {1},
    };
    auto   lbs             = makeLayerBlockIds(lbs_vec, cache_keys.size());
    size_t cpu_matched_num = 0;
    auto   plan            = connector_->buildCopyPlanForRead(cache_keys, lbs, /*gpu_reuse_len=*/0, cpu_matched_num);
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
    auto res_empty_lbs = makeCacheResource({1}, {});
    auto ctx2          = connector_->asyncWrite(res_empty_lbs, nullptr);
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
    ASSERT_NE(buf.k_addr, nullptr);
    // ASSERT_NE(buf.v_addr, nullptr);
    const size_t total = buf.k_addr->sizeBytes();

    // 预置到 cache
    auto block_indices = putItemsToCache(cache_keys, total);
    ASSERT_EQ(block_indices.size(), cache_keys.size());

    const size_t cache_size_before = connector_->block_cache_->size();

    auto ctx = connector_->asyncWrite(res, nullptr);
    ASSERT_EQ(ctx, nullptr);
    EXPECT_EQ(connector_->block_cache_->size(), cache_size_before);
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
    ASSERT_NE(buf.k_addr, nullptr);
    // ASSERT_NE(buf.v_addr, nullptr);
    const size_t total = buf.k_addr->sizeBytes();
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
    ASSERT_NE(buf.k_addr, nullptr);
    // ASSERT_NE(buf.v_addr, nullptr);
    const size_t total = buf.k_addr->sizeBytes();
    auto         pool  = ensureBlockPool(total);
    ASSERT_NE(pool, nullptr);
    const size_t free_before  = pool->freeBlocksNum();
    const size_t cache_before = connector_->block_cache_->size();

    auto ctx = connector_->asyncWrite(res, nullptr);
    ASSERT_NE(ctx, nullptr);
    ctx->waitDone();
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
    ASSERT_NE(buf.k_addr, nullptr);
    // ASSERT_NE(buf.v_addr, nullptr);
    const size_t total = buf.k_addr->sizeBytes();
    auto         pool  = ensureBlockPool(total);
    ASSERT_NE(pool, nullptr);
    const size_t free_before  = pool->freeBlocksNum();
    const size_t cache_before = connector_->block_cache_->size();

    auto ctx = connector_->asyncWrite(res, nullptr);
    ASSERT_NE(ctx, nullptr);
    ctx->waitDone();
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
    ASSERT_NE(buf0.k_addr, nullptr);
    // ASSERT_NE(buf0.v_addr, nullptr);
    const size_t total = buf0.k_addr->sizeBytes();

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
    ASSERT_NE(b0.k_addr, nullptr);
    // ASSERT_NE(b0.v_addr, nullptr);
    ASSERT_NE(b1.k_addr, nullptr);
    // ASSERT_NE(b1.v_addr, nullptr);
    const size_t t0 = b0.k_addr->sizeBytes();
    const size_t t1 = b1.k_addr->sizeBytes();

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
        ASSERT_NE(buf.k_addr, nullptr);
        // ASSERT_NE(buf.v_addr, nullptr);
        single_layer_total = buf.k_addr->sizeBytes();
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
    ASSERT_NE(buf.k_addr, nullptr);
    // ASSERT_NE(buf.v_addr, nullptr);
    const size_t total = buf.k_addr->sizeBytes();
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
    ASSERT_NE(gpu_buf.k_addr, nullptr);
    // ASSERT_NE(gpu_buf.v_addr, nullptr);
    const size_t total = gpu_buf.k_addr->sizeBytes();

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
    ASSERT_NE(gpu_buf.k_addr, nullptr);
    // ASSERT_NE(gpu_buf.v_addr, nullptr);
    const size_t total = gpu_buf.k_addr->sizeBytes();

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
    ASSERT_NE(gpu_buf.k_addr, nullptr);
    // ASSERT_NE(gpu_buf.v_addr, nullptr);
    const size_t total = gpu_buf.k_addr->sizeBytes();

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
    MemoryCopyCacheRequestPB req;
    auto*                    gb = req.add_gpu_blocks();
    auto*                    lb = gb->add_layer_blocks();
    lb->set_layer_id(0);
    lb->set_block_id(1);
    // Intentionally do not add mem_block_ids or mem_block_sizes to trigger mismatch
    req.set_copy_direction(MemoryCopyCacheRequestPB::H2D);

    MemoryCopyCacheResponsePB resp;
    auto                      ok = connector_->copyCache(req, resp);
    EXPECT_FALSE(ok);
    EXPECT_FALSE(resp.success());
}

TEST_F(KVCacheMemoryConnectorTest, copyCache_ReturnFalse_InvalidMemBlockId) {
    const int  layer_id      = 0;
    const int  gpu_block_idx = 1;
    const auto gpu_buf       = allocator_->convertIndexToBuffer(layer_id, gpu_block_idx);
    ASSERT_NE(gpu_buf.k_addr, nullptr);
    // ASSERT_NE(gpu_buf.v_addr, nullptr);
    const size_t total = gpu_buf.k_addr->sizeBytes();

    MemoryCopyCacheRequestPB req;
    auto*                    gb = req.add_gpu_blocks();
    auto*                    lb = gb->add_layer_blocks();
    lb->set_layer_id(layer_id);
    lb->set_block_id(gpu_block_idx);
    req.add_mem_block_ids(-1);  // invalid mem block id
    req.add_mem_block_sizes(static_cast<int64_t>(total));
    req.set_copy_direction(MemoryCopyCacheRequestPB::H2D);

    MemoryCopyCacheResponsePB resp;
    auto                      ok = connector_->copyCache(req, resp);
    EXPECT_FALSE(ok);
    EXPECT_FALSE(resp.success());
}

TEST_F(KVCacheMemoryConnectorTest, copyCache_ReturnFalse_InvalidMemBlockSize) {
    const int  layer_id      = 0;
    const int  gpu_block_idx = 1;
    const auto gpu_buf       = allocator_->convertIndexToBuffer(layer_id, gpu_block_idx);
    ASSERT_NE(gpu_buf.k_addr, nullptr);
    // ASSERT_NE(gpu_buf.v_addr, nullptr);

    MemoryCopyCacheRequestPB req;
    auto*                    gb = req.add_gpu_blocks();
    auto*                    lb = gb->add_layer_blocks();
    lb->set_layer_id(layer_id);
    lb->set_block_id(gpu_block_idx);
    req.add_mem_block_ids(1);
    req.add_mem_block_sizes(0);  // invalid mem block size
    req.set_copy_direction(MemoryCopyCacheRequestPB::H2D);

    MemoryCopyCacheResponsePB resp;
    auto                      ok = connector_->copyCache(req, resp);
    EXPECT_FALSE(ok);
    EXPECT_FALSE(resp.success());
}

TEST_F(KVCacheMemoryConnectorTest, copyCache_ReturnFalse_InvalidLayerId_BuildCopyPlanFailed) {
    const int  valid_layer   = 0;
    const int  gpu_block_idx = 1;
    const auto gpu_buf       = allocator_->convertIndexToBuffer(valid_layer, gpu_block_idx);
    ASSERT_NE(gpu_buf.k_addr, nullptr);
    // ASSERT_NE(gpu_buf.v_addr, nullptr);
    const size_t total = gpu_buf.k_addr->sizeBytes();

    auto pool = ensureBlockPool(total);
    ASSERT_NE(pool, nullptr);
    auto mem_blocks = pool->malloc(1);
    ASSERT_EQ(mem_blocks.size(), 1u);
    const int mem_block_index = mem_blocks[0];

    MemoryCopyCacheRequestPB req;
    auto*                    gb = req.add_gpu_blocks();
    auto*                    lb = gb->add_layer_blocks();
    lb->set_layer_id(cache_config_.layer_num);  // out of range
    lb->set_block_id(gpu_block_idx);
    req.add_mem_block_ids(mem_block_index);
    req.add_mem_block_sizes(static_cast<int64_t>(total));
    req.set_copy_direction(MemoryCopyCacheRequestPB::H2D);

    MemoryCopyCacheResponsePB resp;
    auto                      ok = connector_->copyCache(req, resp);
    EXPECT_FALSE(ok);
    EXPECT_FALSE(resp.success());
}

TEST_F(KVCacheMemoryConnectorTest, copyCache_ReturnTrue_H2D_SingleLayer) {
    const int  layer_id      = 0;
    const int  gpu_block_idx = 2;
    const auto gpu_buf       = allocator_->convertIndexToBuffer(layer_id, gpu_block_idx);
    ASSERT_NE(gpu_buf.k_addr, nullptr);
    // ASSERT_NE(gpu_buf.v_addr, nullptr);
    const size_t total = gpu_buf.k_addr->sizeBytes();

    // H2D 路径需要预先存在 mem pool 与有效 block
    auto pool = ensureBlockPool(total);
    ASSERT_NE(pool, nullptr);
    auto mem_blocks = pool->malloc(1);
    ASSERT_EQ(mem_blocks.size(), 1u);
    const int mem_block_index = mem_blocks[0];
    auto      mem_buffer      = pool->convertIndexToBuffer(0, mem_block_index);
    ASSERT_NE(mem_buffer.k_addr, nullptr);
    EXPECT_EQ(mem_buffer.k_addr->sizeBytes(), total);

    // 给mem_buffer填充数据
    size_t offset = 0;
    setBufferContent(mem_buffer.k_addr, offset, gpu_buf.k_addr->sizeBytes(), 'a');
    offset += gpu_buf.k_addr->sizeBytes();
    // setBufferContent(mem_buffer.k_addr, offset, gpu_buf.v_addr->sizeBytes(), 'b');

    MemoryCopyCacheRequestPB req;
    auto*                    gb = req.add_gpu_blocks();
    auto*                    lb = gb->add_layer_blocks();
    lb->set_layer_id(layer_id);
    lb->set_block_id(gpu_block_idx);
    req.add_mem_block_ids(mem_block_index);
    req.add_mem_block_sizes(static_cast<int64_t>(total));
    req.set_copy_direction(MemoryCopyCacheRequestPB::H2D);

    MemoryCopyCacheResponsePB resp;
    auto                      ok = connector_->copyCache(req, resp);
    EXPECT_TRUE(ok);
    EXPECT_TRUE(resp.success());

    // H2D, 验证数据是否拷贝成功
    verifyBufferContent(gpu_buf.k_addr, 'a');
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

    MemoryCopyCacheRequestPB req;
    req.set_copy_direction(MemoryCopyCacheRequestPB::H2D);
    addOneCopyInfoToPb(req, gpu_layer_blocks1, mem_block_index1, mem_block_size1);
    addOneCopyInfoToPb(req, gpu_layer_blocks2, mem_block_index2, mem_block_size2);

    MemoryCopyCacheResponsePB resp;
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
    ASSERT_NE(gpu_buf.k_addr, nullptr);
    // ASSERT_NE(gpu_buf.v_addr, nullptr);
    const size_t total = gpu_buf.k_addr->sizeBytes();

    // 给gpu_buf填充数据
    setBufferContent(gpu_buf.k_addr, 'a');
    // setBufferContent(gpu_buf.v_addr, 'b');

    // 为确保索引有效，仍然预先创建并分配一个块
    auto pool = ensureBlockPool(total);
    ASSERT_NE(pool, nullptr);
    auto mem_blocks = pool->malloc(1);
    ASSERT_EQ(mem_blocks.size(), 1u);
    const int mem_block_index = mem_blocks[0];
    auto      mem_buffer      = pool->convertIndexToBuffer(0, mem_block_index);
    ASSERT_NE(mem_buffer.k_addr, nullptr);
    EXPECT_EQ(mem_buffer.k_addr->sizeBytes(), total);

    MemoryCopyCacheRequestPB req;
    auto*                    gb = req.add_gpu_blocks();
    auto*                    lb = gb->add_layer_blocks();
    lb->set_layer_id(layer_id);
    lb->set_block_id(gpu_block_idx);
    req.add_mem_block_ids(mem_block_index);
    req.add_mem_block_sizes(static_cast<int64_t>(total));
    req.set_copy_direction(MemoryCopyCacheRequestPB::D2H);

    MemoryCopyCacheResponsePB resp;
    auto                      ok = connector_->copyCache(req, resp);
    EXPECT_TRUE(ok);
    EXPECT_TRUE(resp.success());

    // D2H, 验证数据是否拷贝成功
    verifyBufferContent(mem_buffer.k_addr, 0, gpu_buf.k_addr->sizeBytes(), 'a');
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

    MemoryCopyCacheRequestPB req;
    req.set_copy_direction(MemoryCopyCacheRequestPB::D2H);
    addOneCopyInfoToPb(req, gpu_layer_blocks1, mem_block_index1, mem_block_size1);
    addOneCopyInfoToPb(req, gpu_layer_blocks2, mem_block_index2, mem_block_size2);

    MemoryCopyCacheResponsePB resp;
    auto                      ok = connector_->copyCache(req, resp);
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
    ASSERT_NE(gpu_buf.k_addr, nullptr);
    // ASSERT_NE(gpu_buf.v_addr, nullptr);
    const size_t total_bytes = gpu_buf.k_addr->sizeBytes();

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
    const int  layer_id      = 0;
    const int  gpu_block_idx = 1;
    const auto gpu_buf       = allocator_->convertIndexToBuffer(layer_id, gpu_block_idx);
    ASSERT_NE(gpu_buf.k_addr, nullptr);
    // ASSERT_NE(gpu_buf.v_addr, nullptr);
    const size_t total_bytes = gpu_buf.k_addr->sizeBytes();

    auto pool = ensureBlockPool(total_bytes);
    ASSERT_NE(pool, nullptr);
    // 将 layer_kv_tensors_ 设置为空，模拟 convertIndexToBuffer 失败
    pool->layout_strategy_->layer_kv_tensors_.clear();

    std::vector<KVCacheMemoryConnector::LayerBlock> gpu_layer_blocks{{layer_id, gpu_block_idx}};
    std::vector<rtp_llm::BufferPtr>                 dst, src;
    const int                                       mem_block_index = 1;
    auto                                            ok              = connector_->prepareCopyBuffers(
        gpu_layer_blocks, mem_block_index, total_bytes, KVCacheMemoryConnector::CopyDirection::H2D, dst, src);
    EXPECT_FALSE(ok);
    EXPECT_TRUE(dst.empty());
    EXPECT_TRUE(src.empty());
}

TEST_F(KVCacheMemoryConnectorTest, prepareCopyBuffers_ReturnFalse_InvalidGpuBlockIdx) {
    const int  layer_id      = 0;
    const int  gpu_block_idx = 1;
    const auto gpu_buf       = allocator_->convertIndexToBuffer(layer_id, gpu_block_idx);
    ASSERT_NE(gpu_buf.k_addr, nullptr);
    // ASSERT_NE(gpu_buf.v_addr, nullptr);
    const size_t total_bytes = gpu_buf.k_addr->sizeBytes();

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
    ASSERT_NE(gpu_buf.k_addr, nullptr);
    // ASSERT_NE(gpu_buf.v_addr, nullptr);
    const size_t total_bytes = gpu_buf.k_addr->sizeBytes();

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
        std::vector<KVCacheMemoryConnector::LayerBlock> gpu_layer_blocks{{cache_config_.layer_num, gpu_block_idx}};
        std::vector<rtp_llm::BufferPtr>                 dst, src;
        auto                                            ok = connector_->prepareCopyBuffers(
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
    ASSERT_NE(gpu_buf.k_addr, nullptr);
    // ASSERT_NE(gpu_buf.v_addr, nullptr);
    const size_t k_bytes = gpu_buf.k_addr->sizeBytes();
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
    ASSERT_NE(buf0.k_addr, nullptr);
    // ASSERT_NE(buf0.v_addr, nullptr);
    ASSERT_NE(buf1.k_addr, nullptr);
    // ASSERT_NE(buf1.v_addr, nullptr);
    const size_t total = buf0.k_addr->sizeBytes() + buf1.k_addr->sizeBytes();

    auto pool = ensureBlockPool(total);
    ASSERT_NE(pool, nullptr);
    auto mem_blocks = pool->malloc(1);
    ASSERT_EQ(mem_blocks.size(), 1u);
    const int mem_block_index = mem_blocks[0];
    auto      mem_buffer      = pool->convertIndexToBuffer(0, mem_block_index);
    ASSERT_NE(mem_buffer.k_addr, nullptr);
    // ASSERT_NE(mem_buffer.v_addr, nullptr);
    EXPECT_EQ(mem_buffer.k_addr->sizeBytes(), total);
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
    EXPECT_EQ(src[0]->sizeBytes(), buf0.k_addr->sizeBytes());
    EXPECT_EQ(src[1]->sizeBytes(), buf1.k_addr->sizeBytes());
    EXPECT_EQ(dst[0]->sizeBytes(), buf0.k_addr->sizeBytes());
    EXPECT_EQ(dst[1]->sizeBytes(), buf1.k_addr->sizeBytes());
}

TEST_F(KVCacheMemoryConnectorTest, prepareCopyBuffers_ReturnTrue_D2H_SingleLayer_MemPoolNotCreated) {
    const int  layer_id      = 0;
    const int  gpu_block_idx = 3;
    const auto gpu_buf       = allocator_->convertIndexToBuffer(layer_id, gpu_block_idx);
    ASSERT_NE(gpu_buf.k_addr, nullptr);
    // ASSERT_NE(gpu_buf.v_addr, nullptr);
    const size_t k_bytes = gpu_buf.k_addr->sizeBytes();
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
    ASSERT_NE(gpu_buf.k_addr, nullptr);
    // ASSERT_NE(gpu_buf.v_addr, nullptr);
    const size_t k_bytes = gpu_buf.k_addr->sizeBytes();
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
    ASSERT_NE(buf0.k_addr, nullptr);
    // ASSERT_NE(buf0.v_addr, nullptr);
    ASSERT_NE(buf1.k_addr, nullptr);
    // ASSERT_NE(buf1.v_addr, nullptr);
    const size_t total = buf0.k_addr->sizeBytes() + buf1.k_addr->sizeBytes();

    auto pool = ensureBlockPool(total);
    ASSERT_NE(pool, nullptr);
    auto mem_blocks = pool->malloc(1);
    ASSERT_EQ(mem_blocks.size(), 1u);
    const int mem_block_index = mem_blocks[0];
    auto      mem_buffer      = pool->convertIndexToBuffer(0, mem_block_index);
    ASSERT_NE(mem_buffer.k_addr, nullptr);
    // ASSERT_NE(mem_buffer.v_addr, nullptr);
    EXPECT_EQ(mem_buffer.k_addr->sizeBytes(), total);
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
    EXPECT_EQ(src[0]->sizeBytes(), buf0.k_addr->sizeBytes());
    EXPECT_EQ(src[1]->sizeBytes(), buf1.k_addr->sizeBytes());
    EXPECT_EQ(dst[0]->sizeBytes(), buf0.k_addr->sizeBytes());
    EXPECT_EQ(dst[1]->sizeBytes(), buf1.k_addr->sizeBytes());
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

// TEST_F(KVCacheMemoryConnectorTest, getOrCreateMemoryBlockPool_ReturnNull_CreateFalse_NotExist) {
//     const size_t block_size = 4096;
//     auto         pool       = connector_->getBlockPool(block_size);
//     EXPECT_EQ(pool, nullptr);
// }

// TEST_F(KVCacheMemoryConnectorTest, getOrCreateMemoryBlockPool_ReturnNull_BlockPoolInitFailed) {
//     const size_t block_size = 0;  // invalid block size
//     auto         pool       = ensureBlockPool(block_size);
//     EXPECT_EQ(pool, nullptr);
// }

// TEST_F(KVCacheMemoryConnectorTest, getOrCreateMemoryBlockPool_ReturnNotNull_CreateTrue) {
//     const size_t block_size = 4096;
//     auto         pool       = ensureBlockPool(block_size);
//     ASSERT_NE(pool, nullptr);
// }

// TEST_F(KVCacheMemoryConnectorTest, getOrCreateMemoryBlockPool_ReturnSameInstance_ForSameSize) {
//     const size_t block_size = 8192;
//     auto         pool1      = ensureBlockPool(block_size);
//     ASSERT_NE(pool1, nullptr);
//     auto pool2 = connector_->getBlockPool(block_size);
//     ASSERT_NE(pool2, nullptr);
//     EXPECT_EQ(pool1.get(), pool2.get());
// }

// TEST_F(KVCacheMemoryConnectorTest, getOrCreateMemoryBlockPool_ReturnDifferentInstances_ForDifferentSizes) {
//     const size_t block_size_a = 2048;
//     const size_t block_size_b = 4096;
//     auto         pool_a       = ensureBlockPool(block_size_a);
//     auto         pool_b       = ensureBlockPool(block_size_b);
//     ASSERT_NE(pool_a, nullptr);
//     ASSERT_NE(pool_b, nullptr);
//     EXPECT_NE(pool_a.get(), pool_b.get());
// }

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

}  // namespace rtp_llm::test

int main(int argc, char** argv) {
    rtp_llm::initLogger();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
