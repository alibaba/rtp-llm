#include "gtest/gtest.h"

#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaConnection.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaClient.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaServer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/test/CacheStoreTestBase.h"
#include "autil/NetUtil.h"

namespace rtp_llm {

class RdmaConnectionTest: public CacheStoreTestBase {
protected:
    bool                                      initRdmaClientAndServer();
    std::shared_ptr<RdmaConnectionWriteBlock> makeWriteBlock(const std::shared_ptr<BlockBuffer>& local_buffer,
                                                             const std::shared_ptr<BlockBuffer>& peer_buffer,
                                                             WriteBlockDoneCallback              callback);
    void                                      verifyMemory(const std::shared_ptr<BlockBuffer>& block, char a);

protected:
    std::shared_ptr<RdmaClient> rdma_client_;
    std::shared_ptr<RdmaServer> rdma_server_;
    std::string                 local_ip_;
    uint32_t                    rdma_port_{0};
};

bool RdmaConnectionTest::initRdmaClientAndServer() {
    rdma_port_ = autil::NetUtil::randomPort();

    rdma_server_ = std::make_shared<RdmaServer>(memory_util_);
    if (!rdma_server_->init(rdma_port_, 1, 1)) {
        return false;
    }

    rdma_client_ = std::make_shared<RdmaClient>(memory_util_);
    if (!rdma_client_->init(1, 1)) {
        return false;
    }

    // connect first time
    local_ip_ = autil::NetUtil::getBindIp();
    return true;
}

std::shared_ptr<RdmaConnectionWriteBlock>
RdmaConnectionTest::makeWriteBlock(const std::shared_ptr<BlockBuffer>& local_buffer,
                                   const std::shared_ptr<BlockBuffer>& peer_buffer,
                                   WriteBlockDoneCallback              callback) {

    auto block_buffer_info = std::make_shared<BlockBufferInfo>();
    block_buffer_info->set_key(peer_buffer->key);
    block_buffer_info->set_len(peer_buffer->len);

    ::accl::barex::memp_t mem;
    if (!memory_util_->findMemoryMr(
            (void*)&mem, peer_buffer->addr.get(), peer_buffer->len, peer_buffer->gpu_mem, peer_buffer->adopted)) {
        return nullptr;
    }

    auto rdma_info = block_buffer_info->mutable_rdma_info();
    rdma_info->set_addr(reinterpret_cast<uint64_t>(peer_buffer->addr.get()));
    for (auto& [nicid, mr] : mem.mrs) {
        auto nic_rkey = rdma_info->add_nic_rkeys();
        nic_rkey->set_nicid(nicid);
        nic_rkey->set_rkey(mr->rkey);
    }

    return std::make_shared<RdmaConnectionWriteBlock>(local_buffer, block_buffer_info, callback);
}

void RdmaConnectionTest::verifyMemory(const std::shared_ptr<BlockBuffer>& block, char a) {
    auto buf = device_util_->mallocCPU(block->len);
    EXPECT_TRUE(device_util_->memcopy(buf, false, block->addr.get(), block->gpu_mem, block->len));
    for (int i = 0; i < block->len; i++) {
        EXPECT_EQ(a, *((char*)(buf) + i));
    }
    device_util_->freeCPU(buf);
}

TEST_F(RdmaConnectionTest, testWriteBlocks_Success) {
    if (!memory_util_->isRdmaMode()) { return; }
    ASSERT_TRUE(initRdmaClientAndServer());

    auto rdma_connection = rdma_client_->getConnection(local_ip_, rdma_port_);
    ASSERT_TRUE(rdma_connection != nullptr);

    // write 4 blocks
    uint32_t block_size = 16;

    uint32_t callback_times = 0;
    auto     callback       = [&callback_times](bool success) {
        EXPECT_TRUE(success);
        callback_times++;
    };

    // 3 continous blocks
    auto request_block_buffers = block_buffer_util_->makeRequestBlockBufferVec("request-id", 1, 3, block_size);
    ASSERT_EQ(1, request_block_buffers.size());
    auto request_block_buffer = request_block_buffers[0];

    auto local_block0 = request_block_buffer->getBlock("k_token_id_str_0_layer_id_0");
    auto peer_block0  = request_block_buffer->getBlock("v_token_id_str_0_layer_id_0");
    auto write_block0 = makeWriteBlock(local_block0, peer_block0, callback);

    auto local_block1 = request_block_buffer->getBlock("k_token_id_str_1_layer_id_0");
    auto peer_block1  = request_block_buffer->getBlock("v_token_id_str_1_layer_id_0");
    auto write_block1 = makeWriteBlock(local_block1, peer_block1, callback);

    auto local_block2 = request_block_buffer->getBlock("k_token_id_str_2_layer_id_0");
    auto peer_block2  = request_block_buffer->getBlock("v_token_id_str_2_layer_id_0");
    auto write_block2 = makeWriteBlock(local_block2, peer_block2, callback);

    ASSERT_TRUE(write_block0 != nullptr && write_block1 != nullptr && write_block2 != nullptr);

    ASSERT_TRUE(device_util_->memsetGPU((void*)write_block0->getBlock()->addr.get(), '0', block_size));
    ASSERT_TRUE(device_util_->memsetGPU((void*)write_block1->getBlock()->addr.get(), '1', block_size));
    ASSERT_TRUE(device_util_->memsetGPU((void*)write_block2->getBlock()->addr.get(), '2', block_size));

    // one independent block
    auto local_block3 = block_buffer_util_->makeBlockBuffer("a", block_size, '3', true);
    auto peer_block3  = block_buffer_util_->makeBlockBuffer("b", block_size, '9', true);
    auto write_block3 = makeWriteBlock(local_block3, peer_block3, callback);

    // do write once
    rdma_connection->writeBlocks({write_block0, write_block3, write_block1, write_block2});

    // wait till all write done, max wait 1s
    for (int i = 0; i < 100; i++) {
        if (callback_times == 4) {
            break;
        } else {
            usleep(10 * 1000);
        }
    }
    ASSERT_EQ(4, callback_times);

    // verify write success
    verifyMemory(peer_block0, '0');
    verifyMemory(peer_block1, '1');
    verifyMemory(peer_block2, '2');
    verifyMemory(peer_block3, '3');
}

TEST_F(RdmaConnectionTest, testWriteBlocks_GetWriteChannelFailed) {
    if (!memory_util_->isRdmaMode()) { return; }
    ASSERT_TRUE(initRdmaClientAndServer());

    auto rdma_connection = rdma_client_->getConnection(local_ip_, rdma_port_);
    ASSERT_TRUE(rdma_connection != nullptr);

    uint32_t block_size   = 16;
    auto     local_buffer = block_buffer_util_->makeBlockBuffer("a", block_size, '0', true);
    auto     peer_buffer  = block_buffer_util_->makeBlockBuffer("b", block_size, '1', true);

    std::mutex sync_mutex;
    auto       callback = [&sync_mutex](bool success) {
        EXPECT_FALSE(success);
        sync_mutex.unlock();
    };
    auto write_block = makeWriteBlock(local_buffer, peer_buffer, callback);

    // change write block nic_rkeys to invalid value
    write_block->nic_rkeys_.clear();
    write_block->nic_rkeys_[1024] = 2047;

    // do write once
    sync_mutex.lock();
    rdma_connection->writeBlocks({write_block});

    sync_mutex.lock();
    sync_mutex.unlock();

    // verify write failed
    auto buf = device_util_->mallocCPU(block_size);
    ASSERT_TRUE(device_util_->memcopy(buf, false, peer_buffer->addr.get(), peer_buffer->gpu_mem, block_size));
    ASSERT_EQ('1', *(char*)buf);
    device_util_->freeCPU(buf);
}

TEST_F(RdmaConnectionTest, testWriteBlocks_BlockInfoInvalid) {
    if (!memory_util_->isRdmaMode()) { return; }
    ASSERT_TRUE(initRdmaClientAndServer());

    auto rdma_connection = rdma_client_->getConnection(local_ip_, rdma_port_);
    ASSERT_TRUE(rdma_connection != nullptr);

    uint32_t block_size   = 16;
    auto     local_buffer = block_buffer_util_->makeBlockBuffer("a", block_size, '0', true);
    auto     peer_buffer  = block_buffer_util_->makeBlockBuffer("b", block_size, '1', true);

    std::mutex sync_mutex;
    auto       callback = [&sync_mutex](bool success) {
        EXPECT_FALSE(success);
        sync_mutex.unlock();
    };
    auto write_block = makeWriteBlock(local_buffer, peer_buffer, callback);

    // dereg local buffer
    memory_util_->deregUserMr(local_buffer->addr.get(), local_buffer->gpu_mem);

    // do write once
    sync_mutex.lock();
    rdma_connection->writeBlocks({write_block});

    sync_mutex.lock();
    sync_mutex.unlock();

    // verify write failed
    auto buf = device_util_->mallocCPU(block_size);
    ASSERT_TRUE(device_util_->memcopy(buf, false, peer_buffer->addr.get(), peer_buffer->gpu_mem, block_size));
    ASSERT_EQ('1', *(char*)buf);
    device_util_->freeCPU(buf);
}

TEST_F(RdmaConnectionTest, testWriteBlocks_WriteFailed) {
    if (!memory_util_->isRdmaMode()) { return; }
    ASSERT_TRUE(initRdmaClientAndServer());

    auto rdma_connection = rdma_client_->getConnection(local_ip_, rdma_port_);
    ASSERT_TRUE(rdma_connection != nullptr);

    uint32_t block_size   = 16;
    auto     local_buffer = block_buffer_util_->makeBlockBuffer("a", block_size, '0', true);
    auto     peer_buffer  = block_buffer_util_->makeBlockBuffer("b", block_size, '1', true);

    std::mutex sync_mutex;
    auto       callback = [&sync_mutex](bool success) {
        EXPECT_FALSE(success);
        sync_mutex.unlock();
    };
    auto write_block = makeWriteBlock(local_buffer, peer_buffer, callback);

    // make peer addr invalid
    write_block->getPeerBlock()->mutable_rdma_info()->set_addr(0x123);

    // do write once
    sync_mutex.lock();
    rdma_connection->writeBlocks({write_block});

    sync_mutex.lock();
    sync_mutex.unlock();

    // verify write failed
    auto buf = device_util_->mallocCPU(block_size);
    ASSERT_TRUE(device_util_->memcopy(buf, false, peer_buffer->addr.get(), peer_buffer->gpu_mem, block_size));
    ASSERT_EQ('1', *(char*)buf);
    device_util_->freeCPU(buf);

    rdma_client_.reset();
    rdma_server_.reset();
}

TEST_F(RdmaConnectionTest, testWriteBlock_AppendBlock_Success) {
    if (!memory_util_->isRdmaMode()) { return; }
    ASSERT_TRUE(initRdmaClientAndServer());

    auto request_block_buffers = block_buffer_util_->makeRequestBlockBufferVec("request-id", 1, 3, 1024);
    ASSERT_EQ(1, request_block_buffers.size());

    auto request_block_buffer = request_block_buffers[0];

    int  callback_called_times = 0;
    auto callback              = [&callback_called_times](bool success) {
        callback_called_times++;
        EXPECT_TRUE(success);
    };

    auto first_block = makeWriteBlock(request_block_buffer->getBlock("k_token_id_str_0_layer_id_0"),
                                      request_block_buffer->getBlock("v_token_id_str_0_layer_id_0"),
                                      callback);
    ASSERT_TRUE(first_block != nullptr);

    auto second_block = makeWriteBlock(request_block_buffer->getBlock("k_token_id_str_1_layer_id_0"),
                                       request_block_buffer->getBlock("v_token_id_str_1_layer_id_0"),
                                       callback);
    ASSERT_TRUE(second_block != nullptr);

    ASSERT_TRUE(first_block->appendBlock(second_block));

    // verfiy append
    ASSERT_EQ(request_block_buffer->getBlock("k_token_id_str_0_layer_id_0")->addr, first_block->getBlock()->addr);
    ASSERT_EQ(2048, first_block->getBlock()->len);
    ASSERT_TRUE(first_block->getBlock()->gpu_mem);
    ASSERT_TRUE(first_block->getBlock()->adopted);
    ASSERT_EQ(first_block->getPeerBlock()->rdma_info().addr(), first_block->getPeerBlock()->rdma_info().addr());
    ASSERT_EQ(2, first_block->getSubBlockCount());

    auto third_block = makeWriteBlock(request_block_buffer->getBlock("k_token_id_str_2_layer_id_0"),
                                      request_block_buffer->getBlock("v_token_id_str_2_layer_id_0"),
                                      callback);
    ASSERT_TRUE(third_block != nullptr);

    ASSERT_TRUE(first_block->appendBlock(third_block));
    ASSERT_EQ(3, first_block->getSubBlockCount());
    ASSERT_EQ(request_block_buffer->getBlock("k_token_id_str_0_layer_id_0")->addr, first_block->getBlock()->addr);
    ASSERT_EQ(3072, first_block->getBlock()->len);
    ASSERT_TRUE(first_block->getBlock()->gpu_mem);
    ASSERT_TRUE(first_block->getBlock()->adopted);
    ASSERT_EQ(first_block->getPeerBlock()->rdma_info().addr(), first_block->getPeerBlock()->rdma_info().addr());

    first_block->getCallback()(true);
    ASSERT_EQ(callback_called_times, 3);
}

TEST_F(RdmaConnectionTest, testWriteBlock_AppendBlock_Failed) {
    if (!memory_util_->isRdmaMode()) { return; }
    ASSERT_TRUE(initRdmaClientAndServer());

    auto request_block_buffers = block_buffer_util_->makeRequestBlockBufferVec("request-id", 1, 3, 1024);
    ASSERT_EQ(1, request_block_buffers.size());

    auto request_block_buffer = request_block_buffers[0];

    int  callback_called_times = 0;
    auto callback              = [&callback_called_times](bool success) {
        callback_called_times++;
        EXPECT_TRUE(success);
    };

    auto first_block = makeWriteBlock(request_block_buffer->getBlock("k_token_id_str_0_layer_id_0"),
                                      request_block_buffer->getBlock("v_token_id_str_0_layer_id_0"),
                                      callback);
    ASSERT_TRUE(first_block != nullptr);

    auto second_block = makeWriteBlock(request_block_buffer->getBlock("k_token_id_str_1_layer_id_0"),
                                       request_block_buffer->getBlock("v_token_id_str_1_layer_id_0"),
                                       callback);
    ASSERT_TRUE(second_block != nullptr);

    auto third_block = makeWriteBlock(request_block_buffer->getBlock("k_token_id_str_2_layer_id_0"),
                                      request_block_buffer->getBlock("v_token_id_str_2_layer_id_0"),
                                      callback);
    ASSERT_TRUE(third_block != nullptr);

    // 本端地址不连续
    {
        ASSERT_FALSE(first_block->appendBlock(third_block));

        second_block->getBlock()->gpu_mem = false;
        ASSERT_FALSE(first_block->appendBlock(second_block));
        second_block->getBlock()->gpu_mem = true;

        second_block->getBlock()->adopted = false;
        ASSERT_FALSE(first_block->appendBlock(second_block));
        second_block->getBlock()->adopted = true;
    }
    {
        // 对端地址不连续
        auto test_block = makeWriteBlock(request_block_buffer->getBlock("k_token_id_str_1_layer_id_0"),
                                         request_block_buffer->getBlock("v_token_id_str_2_layer_id_0"),
                                         callback);
        ASSERT_FALSE(first_block->appendBlock(test_block));
    }
    {
        // nic_rkeys 不一致
        second_block->nic_rkeys_[1111] = 2222;
        ASSERT_FALSE(first_block->appendBlock(second_block));
    }
}

TEST_F(RdmaConnectionTest, testBugScenario_WrongRkey_ProtectionError) {
    // 复现线上 bug: Prefill 用 Decode B 的 QP 但 Decode A 的 rkey → mlx5 protection error
    //
    // 线上场景：
    //   Decode A 在 PD_A 中注册 buffer → rkey_A
    //   Decode B 在 PD_B 中注册 buffer → rkey_B
    //   旧代码 port-only 索引导致 Prefill 拿到 B 的 QP (PD_B) 但用 A 的 rkey_A
    //   PD_B 中找不到 rkey_A → mlx5 local protection error (0x3a)
    //
    // 单进程复现：所有 QP 共享同一 PD，无法自然产生跨 PD 错误，
    // 因此通过篡改 rkey 为无效值来模拟跨 PD 场景，触发相同的硬件保护错误。
    if (!memory_util_->isRdmaMode()) { return; }
    ASSERT_TRUE(initRdmaClientAndServer());

    auto rdma_connection = rdma_client_->getConnection(local_ip_, rdma_port_);
    ASSERT_TRUE(rdma_connection != nullptr);

    uint32_t block_size   = 16;
    auto     local_buffer = block_buffer_util_->makeBlockBuffer("local", block_size, 'L', true);
    auto     peer_buffer  = block_buffer_util_->makeBlockBuffer("peer",  block_size, 'P', true);

    std::mutex sync_mutex;
    bool       write_success = true;
    auto       callback      = [&sync_mutex, &write_success](bool success) {
        write_success = success;
        sync_mutex.unlock();
    };

    auto write_block = makeWriteBlock(local_buffer, peer_buffer, callback);
    ASSERT_TRUE(write_block != nullptr);
    ASSERT_FALSE(write_block->nic_rkeys_.empty());

    // 保留正确的 NIC ID（getConnection 匹配成功），篡改 rkey 为无效值
    // 等价于线上：QP 匹配到了（NIC 正确），但 rkey 属于另一个进程的 PD
    for (auto& [nicid, rkey] : write_block->nic_rkeys_) {
        rkey = 0xDEADBEEF;
    }

    sync_mutex.lock();
    rdma_connection->writeBlocks({write_block});

    sync_mutex.lock();
    sync_mutex.unlock();

    // RDMA 硬件拒绝写入：rkey 在当前 PD 中无法映射到有效 MR
    // dmesg 会记录: mlx5_bond_X/Y: QP XXXXX error: local protection error (0x3a 0x0 0x95)
    EXPECT_FALSE(write_success);

    verifyMemory(peer_buffer, 'P');

    rdma_client_.reset();
    rdma_server_.reset();
}

}  // namespace rtp_llm