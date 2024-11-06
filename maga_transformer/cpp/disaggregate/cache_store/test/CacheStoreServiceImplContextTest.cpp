#include "gtest/gtest.h"
#include <gmock/gmock.h>

#include "autil/EnvUtil.h"
#include "autil/NetUtil.h"
#include "maga_transformer/cpp/disaggregate/cache_store/NormalCacheStore.h"
#include "maga_transformer/cpp/disaggregate/cache_store/CacheStoreServiceImplContext.h"
#include "maga_transformer/cpp/disaggregate/cache_store/CacheLoadServiceClosure.h"
#include "maga_transformer/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "maga_transformer/cpp/disaggregate/cache_store/BlockBufferUtil.h"
#include "maga_transformer/cpp/disaggregate/cache_store/MessagerClient.h"
#include "maga_transformer/cpp/disaggregate/cache_store/Interface.h"

namespace rtp_llm{
class MockCacheLoadServiceClosure : public CacheLoadServiceClosure {
public:
    MockCacheLoadServiceClosure(const std::shared_ptr<MemoryUtil>&                           memory_util,
                            const std::shared_ptr<RequestBlockBuffer>                    request_block_buffer,
                            arpc::ANetRPCController*                                     controller,
                            CacheLoadRequest*                                            request,
                            CacheLoadResponse*                                           response,
                            CacheStoreLoadDoneCallback                                   callback,
                            const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collector):
    CacheLoadServiceClosure(memory_util, request_block_buffer, controller, request, response, callback, collector){}
        
    virtual ~MockCacheLoadServiceClosure(){}
public:
    MOCK_METHOD0(Run,void());
};

class CacheStoreServiceImplContextTest: public ::testing::Test{
protected:
    bool initCacheStores();
    bool initContext();
    void loadThreadFunction(int id);

    void SetUp() override {
        memory_util_       = std::make_shared<MemoryUtil>(createMemoryUtilImpl(autil::EnvUtil::getEnv(kEnvRdmaMode, false)));
        block_buffer_util_ = std::make_shared<BlockBufferUtil>(memory_util_);
    }

    void TearDown() override {
        delete done_;
        cache_store1_.reset();
        memory_util_.reset();
    }

protected:
    std::shared_ptr<NormalCacheStore> cache_store1_;
    std::shared_ptr<MemoryUtil>      memory_util_;
    std::shared_ptr<BlockBufferUtil> block_buffer_util_;
    CacheLoadRequest* request_;
    CacheLoadResponse* response_;
    std::shared_ptr<CacheStoreServerLoadMetricsCollector> collector_;
    MockCacheLoadServiceClosure* done_;
    std::shared_ptr<arpc::TimerManager> timer_manager_;

    std::shared_ptr<CacheStoreServiceImplContext> context_;
};

bool CacheStoreServiceImplContextTest::initCacheStores() {
    auto port1 = autil::NetUtil::randomPort();
    auto port2 = autil::NetUtil::randomPort();

    CacheStoreInitParams params1;
    params1.listen_port   = port1;
    params1.connect_port  = port2;
    params1.enable_metric = false;
    params1.memory_util   = memory_util_;

    cache_store1_ = NormalCacheStore::createNormalCacheStore(params1);
    if (cache_store1_ == nullptr) {
        return false;
    }
    return true;
}
bool CacheStoreServiceImplContextTest::initContext(){
    auto request_block_buffer = std::make_shared<RequestBlockBuffer>("request-1");
    for(int i = 0; i < 10; i++){
        auto block = block_buffer_util_->makeBlockBuffer("b"+std::to_string(i), 1024, '0'+i, true);
        request_block_buffer->addBlock(block);
    }
    request_ = cache_store1_->messager_client_->makeLoadRequest(request_block_buffer,1000);
    if(request_ == nullptr){
        return false;
    }
    arpc::ANetRPCController* controller = new arpc::ANetRPCController();
    controller->SetExpireTime(1000);
    
    response_ = new CacheLoadResponse;
    if(response_  == nullptr){
        return false;
    }
    done_ = new MockCacheLoadServiceClosure(memory_util_, request_block_buffer, controller, request_, response_, nullptr, nullptr);
    
    if(done_ == nullptr){
        return false;
    }
    timer_manager_=cache_store1_->messager_server_->timer_manager_;
    if(timer_manager_ == nullptr){
        return false;
    }

    context_ = std::make_shared<CacheStoreServiceImplContext>(request_, response_, nullptr, done_);
    return context_!=nullptr;
}

TEST_F(CacheStoreServiceImplContextTest, loadBlock){
    ASSERT_TRUE(initCacheStores());
    ASSERT_TRUE(initContext());
    
    context_->setUnLoadedBlocks();
    ASSERT_EQ(context_->unloaded_blocks_.size(),10);

    EXPECT_CALL(*done_, Run()).Times(1);
    for(int i=0; i<10; i++){
        auto block = block_buffer_util_->makeBlockBuffer("b"+std::to_string(i), 1024, 'a'+i, true);
        context_->loadBlockOnTcp(block);
        ASSERT_EQ(context_->unloaded_blocks_.size(),10-i-1);
    }
    
    ASSERT_EQ(context_->unloaded_blocks_.size(),0);
    ASSERT_EQ(context_->isAllLoaded(),true);
    ASSERT_EQ(context_->reentrant_flag_.load(),true);
    ASSERT_EQ(context_->is_timeout_, false);

    ASSERT_EQ(response_->blocks_size(), 10);
    for (int i = 0; i < response_->blocks_size(); i++) {
        const auto& block = response_->blocks(i);
        ASSERT_EQ(block.key(), "b"+std::to_string(i));
        ASSERT_EQ(block.content().data()[0], 'a'+i);
        ASSERT_EQ(block.len(), 1024);
    }
}

void CacheStoreServiceImplContextTest::loadThreadFunction(int i) {
    auto block = block_buffer_util_->makeBlockBuffer("b" + std::to_string(i), 1024, 'a' + i, true);
    context_->loadBlockOnTcp(block);
}

TEST_F(CacheStoreServiceImplContextTest, loadBlockMultiThread){
    ASSERT_TRUE(initCacheStores());
    ASSERT_TRUE(initContext());
    context_->setUnLoadedBlocks();
    ASSERT_EQ(context_->unloaded_blocks_.size(),10);

    EXPECT_CALL(*done_, Run()).Times(1);
    std::vector<std::thread> load_threads;
    for (int i = 0; i < 10; ++i) {
        load_threads.emplace_back([this, i](){
            this->loadThreadFunction(i);
        });
    }
    for (auto& t : load_threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    ASSERT_EQ(context_->unloaded_blocks_.size(),0);
    ASSERT_EQ(context_->isAllLoaded(),true);
    ASSERT_EQ(context_->reentrant_flag_.load(),true);
    ASSERT_EQ(context_->is_timeout_, false);

    ASSERT_EQ(response_->blocks_size(), 10);
    for (int i = 0; i < response_->blocks_size(); i++) {
        const auto& block = response_->blocks(i);
        ASSERT_EQ(block.key(), "b"+std::to_string(i));
        ASSERT_EQ(block.content().data()[0], 'a'+i);
        ASSERT_EQ(block.len(), 1024);
    }
}

TEST_F(CacheStoreServiceImplContextTest, loadBlockTimeout){
    ASSERT_TRUE(initCacheStores());
    ASSERT_TRUE(initContext());
    std::mutex mutex;
    mutex.lock();
    auto timer_callback = [this, &mutex](){
        this->context_->setTimeOut();
        if(!this->context_->isAllLoaded()){
            this->context_->runFailed(KvCacheStoreServiceErrorCode::EC_FAILED_LOAD_BUFFER);
        }
        mutex.unlock();
    };
    auto timer = timer_manager_->addTimer(request_->timeout_ms(), std::move(timer_callback));
    context_->setTimer(timer);
    ASSERT_TRUE(timer!=nullptr);

    context_->setUnLoadedBlocks();
    ASSERT_EQ(context_->unloaded_blocks_.size(),10);

    EXPECT_CALL(*done_, Run()).Times(1);
    std::vector<std::thread> load_threads;
    for (int i = 0; i < 7; ++i) {
        load_threads.emplace_back([this, i](){
            this->loadThreadFunction(i);
        });
    }
    for (auto& t : load_threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    mutex.lock();
    mutex.unlock();
    ASSERT_EQ(context_->unloaded_blocks_.size(),3);
    ASSERT_EQ(context_->isAllLoaded(),false);
    ASSERT_EQ(context_->reentrant_flag_.load(),true);
    ASSERT_EQ(context_->is_timeout_, true);

    ASSERT_EQ(response_->blocks_size(), 0);
}
}
