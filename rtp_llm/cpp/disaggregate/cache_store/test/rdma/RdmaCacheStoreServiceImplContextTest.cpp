#include "gtest/gtest.h"
#include <gmock/gmock.h>

#include "autil/EnvUtil.h"
#include "autil/NetUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/test/CacheStoreTestBase.h"

#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaConnection.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaMessager.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaCacheStoreServiceImpl.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaCacheStoreServiceImplContext.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaServerHostConnections.h"
#include "aios/network/accl-barex/include/accl/barex/xchannel.h"

namespace rtp_llm {

class MockCacheLoadServiceClosure: public RPCClosure {
public:
    MockCacheLoadServiceClosure(const std::shared_ptr<MemoryUtil>&                           memory_util,
                                const std::shared_ptr<RequestBlockBuffer>                    request_block_buffer,
                                arpc::ANetRPCController*                                     controller,
                                CacheLoadRequest*                                            request,
                                CacheLoadResponse*                                           response,
                                CacheStoreLoadDoneCallback                                   callback,
                                const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collector):
        request_block_buffer_(request_block_buffer) {}
        
    virtual ~MockCacheLoadServiceClosure() {}

public:
    void Run() override {
        run_done_ = true;
    }
    bool runDone() const {
        return run_done_;
    }

public:
    bool                                run_done_ = false;
    std::shared_ptr<RequestBlockBuffer> request_block_buffer_;
};

class RdmaCacheStoreServiceImplContextTest: public CacheStoreTestBase {
protected:
    void TearDown() override {
        // stop all callback before delete done, or write callback may happen after done delete
        client_.reset();
        server_.reset();
        if (!done_) {
            delete done_;
        }
        memory_util_.reset();
    }

    bool initMessager();
    bool initContext();
    void loadThreadFunction(const std::shared_ptr<BlockBuffer>& block);
    bool waitDoneRun();
    void verifyContextRunDone(bool                         run_done,
                              int                          unloaded_block_cnt,
                              int                          write_cnt,
                              bool                         context_done_run,
                              KvCacheStoreServiceErrorCode error_code);

protected:
    CacheLoadRequest*                                     request_;
    CacheLoadResponse*                                    response_;
    std::shared_ptr<CacheStoreServerLoadMetricsCollector> collector_;
    MockCacheLoadServiceClosure*                          done_{nullptr};

    std::shared_ptr<RequestBlockBufferStore> server_buffer_store_;
    std::shared_ptr<RdmaConnection>          rdma_connection_;

    std::shared_ptr<RdmaCacheStoreServiceImplContext> context_;

    std::shared_ptr<RdmaMessager> client_;
    std::shared_ptr<RdmaMessager> server_;

    uint32_t port_;
    uint32_t rdma_port_;
};

bool RdmaCacheStoreServiceImplContextTest::initMessager() {
    server_buffer_store_ = std::make_shared<RequestBlockBufferStore>(memory_util_);
    port_                = autil::NetUtil::randomPort();
    rdma_port_           = autil::NetUtil::randomPort();
    client_              = std::make_shared<RdmaMessager>(memory_util_, nullptr, nullptr);

    MessagerInitParams client_init_params;
    client_init_params.server_port      = autil::NetUtil::randomPort();
    client_init_params.rdma_server_port = autil::NetUtil::randomPort();
    if (!client_->init(client_init_params)) {
        return false;
    }

    server_ = std::make_shared<RdmaMessager>(memory_util_, server_buffer_store_, nullptr);
    MessagerInitParams server_init_params;
    server_init_params.server_port      = port_;
    server_init_params.rdma_server_port = rdma_port_;
    if (!server_->init(server_init_params)) {
        return false;
    }
    return true;
}

bool RdmaCacheStoreServiceImplContextTest::initContext() {
    auto client_rdma_connection = client_->rdma_client_->getConnection(autil::NetUtil::getBindIp(), rdma_port_);
    if (client_rdma_connection == nullptr) {
        return false;
    }

    auto request_block_buffer = std::make_shared<RequestBlockBuffer>("request-1");
    for (int i = 0; i < 10; i++) {
        auto block = block_buffer_util_->makeBlockBuffer("b" + std::to_string(i), 1024, '0' + i, true);
        request_block_buffer->addBlock(block);
    }

    auto load_request =
        std::make_shared<LoadRequest>(autil::NetUtil::getBindIp(), port_, 0, request_block_buffer, nullptr, 1000, 1, 0);
    request_ = client_->makeLoadRequest(client_rdma_connection, load_request);
    if (request_ == nullptr) {
        return false;
    }
    arpc::ANetRPCController* controller = new arpc::ANetRPCController();
    controller->SetExpireTime(1000);

    response_ = new CacheLoadResponse;
    if (response_ == nullptr) {
        return false;
    }
    done_ = new MockCacheLoadServiceClosure(
        memory_util_, request_block_buffer, controller, request_, response_, nullptr, nullptr);

    if (done_ == nullptr) {
        return false;
    }

    auto client_conns = client_rdma_connection->getAllConnections();
    std::vector<AddrPortChannel> addr_port_channels;
    for (auto& conn : client_conns) {
        auto local = conn->GetLocalAddrAndPort();
        addr_port_channels.push_back({local.first, (uint32_t)local.second, conn->GetChannel()->GetPeerChannel()});
    }
    rdma_connection_ = server_->rdma_server_->getConnection(addr_port_channels);
    if (rdma_connection_ == nullptr) {
        return false;
    }

    auto collector = std::make_shared<CacheStoreServerLoadMetricsCollector>(nullptr, 1, 2, 3);
    context_       = std::make_shared<RdmaCacheStoreServiceImplContext>(
        request_, response_, collector, done_, server_buffer_store_, rdma_connection_);
    return context_ != nullptr;
}

void RdmaCacheStoreServiceImplContextTest::verifyContextRunDone(bool                         run_done,
                                                                int                          unloaded_block_cnt,
                                                                int                          write_cnt,
                                                                bool                         context_done_run,
                                                                KvCacheStoreServiceErrorCode error_code) {
    ASSERT_EQ(done_->runDone(), run_done);
    ASSERT_EQ(context_->unloaded_blocks_.size(), unloaded_block_cnt);
    ASSERT_LE(context_->write_cnt_.load(), write_cnt);
    ASSERT_EQ(context_->done_run_.load(), context_done_run);
    ASSERT_EQ(response_->error_code(), error_code);
}

bool RdmaCacheStoreServiceImplContextTest::waitDoneRun() {
    for (int i = 0; i < 5000; i++) {  // wait 5s
        if (done_->runDone()) {
            break;
        }
        usleep(1000);
    }
    return done_->runDone();
}

TEST_F(RdmaCacheStoreServiceImplContextTest, loadBlock_Success) {
    if (!memory_util_->isRdmaMode()) { return; }
    ASSERT_TRUE(initMessager());
    ASSERT_TRUE(initContext());
    ASSERT_EQ(context_->unloaded_blocks_.size(), 10);

    std::vector<std::shared_ptr<BlockBuffer>> blocks;
    for (int i = 0; i < 10; i++) {
        auto block = block_buffer_util_->makeBlockBuffer("b" + std::to_string(i), 1024, 'a' + i, true);
        blocks.push_back(block);
    }
    context_->loadBlockOnRdma(true, blocks);

    ASSERT_TRUE(waitDoneRun());
    verifyContextRunDone(true, 0, 10, true, KvCacheStoreServiceErrorCode::EC_SUCCESS);

    auto cpu_buffer = device_util_->mallocCPU(1024);
    for (int i = 0; i < 10; i++) {
        auto block = done_->request_block_buffer_->getBlock("b" + std::to_string(i));
        ASSERT_TRUE(block != nullptr);

        ASSERT_TRUE(device_util_->memcopy(cpu_buffer, false, block->addr.get(), block->gpu_mem, block->len));
        ASSERT_EQ('a' + i, *static_cast<char*>(cpu_buffer));
    }
    device_util_->freeCPU(cpu_buffer);
}

TEST_F(RdmaCacheStoreServiceImplContextTest, loadBlockTimeout) {
    if (!memory_util_->isRdmaMode()) { return; }
    ASSERT_TRUE(initMessager());
    ASSERT_TRUE(initContext());
    ASSERT_EQ(context_->unloaded_blocks_.size(), 10);

    std::vector<std::shared_ptr<BlockBuffer>> blocks;
    for (int i = 0; i < 10; i++) {
        auto block = block_buffer_util_->makeBlockBuffer("b" + std::to_string(i), 1024, 'a' + i, true);
        blocks.push_back(block);
    }

    std::mutex mutex;
    mutex.lock();
    auto timer_callback = [this, &mutex]() {
        this->context_->runFailed(KvCacheStoreServiceErrorCode::EC_FAILED_LOAD_BUFFER);
        mutex.unlock();
    };
    auto timer = server_->timer_manager_->addTimer(request_->timeout_ms(), std::move(timer_callback));
    context_->setTimer(timer);
    ASSERT_TRUE(timer != nullptr);

    std::vector<std::thread> load_threads;
    for (int i = 0; i < 7; ++i) {
        load_threads.emplace_back([this, i, block = blocks[i]]() { this->loadThreadFunction(block); });
    }
    for (auto& t : load_threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    mutex.lock();
    mutex.unlock();

    ASSERT_TRUE(waitDoneRun());
    verifyContextRunDone(true, 3, 7, true, KvCacheStoreServiceErrorCode::EC_FAILED_LOAD_BUFFER);
}

TEST_F(RdmaCacheStoreServiceImplContextTest, loadBlock_Canceled) {
    if (!memory_util_->isRdmaMode()) { return; }
    ASSERT_TRUE(initMessager());
    ASSERT_TRUE(initContext());
    ASSERT_EQ(context_->unloaded_blocks_.size(), 10);

    std::vector<std::shared_ptr<BlockBuffer>> blocks;
    for (int i = 0; i < 5; i++) {
        auto block = block_buffer_util_->makeBlockBuffer("b" + std::to_string(i), 1024, 'a' + i, true);
        blocks.push_back(block);
    }
    context_->loadBlockOnRdma(true, blocks);
    context_->loadBlockOnRdma(false, {});

    ASSERT_TRUE(waitDoneRun());
    verifyContextRunDone(true, 5, 5, true, KvCacheStoreServiceErrorCode::EC_FAILED_LOAD_BUFFER);
}

TEST_F(RdmaCacheStoreServiceImplContextTest, loadBlock_WriteFailed) {
    if (!memory_util_->isRdmaMode()) { return; }
    ASSERT_TRUE(initMessager());
    ASSERT_TRUE(initContext());
    ASSERT_EQ(context_->unloaded_blocks_.size(), 10);

    // change request block 5 nic rkey to invalid
    auto block = context_->unloaded_blocks_["b5"];
    ASSERT_TRUE(block != nullptr);
    auto rdma_info = block->mutable_rdma_info();
    for (int i = 0; i < rdma_info->nic_rkeys_size(); i++) {
        rdma_info->mutable_nic_rkeys(i)->set_rkey(0);
    }

    std::vector<std::shared_ptr<BlockBuffer>> blocks;
    for (int i = 0; i < 10; i++) {
        auto block = block_buffer_util_->makeBlockBuffer("b" + std::to_string(i), 1024, 'a' + i, true);
        blocks.push_back(block);
    }
    context_->loadBlockOnRdma(true, blocks);

    ASSERT_TRUE(waitDoneRun());
    verifyContextRunDone(true, 0, 9, true, KvCacheStoreServiceErrorCode::EC_FAILED_RDMA_WRITE);
}

TEST_F(RdmaCacheStoreServiceImplContextTest, loadBlock_AfterDoneRun) {
    if (!memory_util_->isRdmaMode()) { return; }
    ASSERT_TRUE(initMessager());
    ASSERT_TRUE(initContext());
    ASSERT_EQ(context_->unloaded_blocks_.size(), 10);

    // force run failed
    context_->runFailed(KvCacheStoreServiceErrorCode::EC_FAILED_LOAD_BUFFER);
    ASSERT_TRUE(context_->response_ == nullptr);
    ASSERT_TRUE(waitDoneRun());
    verifyContextRunDone(true, 10, 0, true, KvCacheStoreServiceErrorCode::EC_FAILED_LOAD_BUFFER);

    auto block = block_buffer_util_->makeBlockBuffer("b0", 1024, 'a', true);
    context_->loadBlockOnRdma(true, {block});
}

void RdmaCacheStoreServiceImplContextTest::loadThreadFunction(const std::shared_ptr<BlockBuffer>& block) {
    context_->loadBlockOnRdma(true, {block});
}

TEST_F(RdmaCacheStoreServiceImplContextTest, loadBlockMultiThread) {
    if (!memory_util_->isRdmaMode()) { return; }
    ASSERT_TRUE(initMessager());
    ASSERT_TRUE(initContext());
    ASSERT_EQ(context_->unloaded_blocks_.size(), 10);

    std::vector<std::shared_ptr<BlockBuffer>> blocks;
    for (int i = 0; i < 10; i++) {
        auto block = block_buffer_util_->makeBlockBuffer("b" + std::to_string(i), 1024, 'a' + i, true);
        blocks.push_back(block);
    }

    std::vector<std::thread> load_threads;
    for (int i = 0; i < 10; ++i) {
        load_threads.emplace_back([this, i, block = blocks[i]]() { this->loadThreadFunction(block); });
    }
    for (auto& t : load_threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    ASSERT_TRUE(waitDoneRun());
    verifyContextRunDone(true, 0, 10, true, KvCacheStoreServiceErrorCode::EC_SUCCESS);

    for (int i = 0; i < response_->blocks_size(); i++) {
        const auto& block = response_->blocks(i);
        ASSERT_EQ(block.key(), "b" + std::to_string(i));
        ASSERT_EQ(block.len(), 1024);
    }
}

}  // namespace rtp_llm
