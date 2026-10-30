#include "gtest/gtest.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpCacheStoreLoadServiceClosure.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/test/CacheStoreTestBase.h"
#include "autil/NetUtil.h"
#include "autil/EnvUtil.h"
#include <atomic>
#include <future>
#include <thread>

namespace rtp_llm {

class TcpCacheStoreLoadServiceClosureTest: public CacheStoreTestBase {
protected:
    TcpCacheStoreLoadServiceClosure*
    makeClosure(arpc::ErrorCode arpc_ec, KvCacheStoreServiceErrorCode resp_ec, CacheStoreLoadDoneCallback callback);
};

TcpCacheStoreLoadServiceClosure* TcpCacheStoreLoadServiceClosureTest::makeClosure(arpc::ErrorCode              arpc_ec,
                                                                                  KvCacheStoreServiceErrorCode resp_ec,
                                                                                  CacheStoreLoadDoneCallback callback) {
    auto request_buffer = std::make_shared<RequestBlockBuffer>("request-id");
    auto controller     = new arpc::ANetRPCController();
    auto request        = new CacheLoadRequest;
    auto response       = new CacheLoadResponse;
    auto collector      = std::make_shared<CacheStoreClientLoadMetricsCollector>(nullptr, 1, 1);

    if (arpc_ec != arpc::ARPC_ERROR_NONE) {
        controller->SetFailed("failed");
        controller->SetErrorCode(arpc_ec);
    }

    response->set_error_code(resp_ec);

    return new TcpCacheStoreLoadServiceClosure(
        memory_util_, request_buffer, controller, request, response, callback, collector);
}

TEST_F(TcpCacheStoreLoadServiceClosureTest, testRun_Success) {
    std::mutex mutex;
    auto       callback = [&mutex](bool ok, CacheStoreErrorCode ec) {
        ASSERT_TRUE(ok);
        ASSERT_EQ(CacheStoreErrorCode::None, ec);
        mutex.unlock();
    };

    auto closure = makeClosure(arpc::ARPC_ERROR_NONE, KvCacheStoreServiceErrorCode::EC_SUCCESS, callback);
    ASSERT_TRUE(closure != nullptr);
    closure->Run();

    mutex.lock();
    mutex.unlock();
}

TEST_F(TcpCacheStoreLoadServiceClosureTest, DefersResponseCompletionOffIoThread) {
    auto block = block_buffer_util_->makeBlockBuffer("key", 32, '0', false);
    ASSERT_NE(block, nullptr);
    autil::ThreadPool pool(1, 8, false, "CacheLoadTest");
    ASSERT_TRUE(pool.start());
    std::promise<void> worker_started;
    std::promise<void> release_worker;
    auto               release = release_worker.get_future().share();
    ASSERT_EQ(pool.pushTask([&]() {
        worker_started.set_value();
        release.wait();
    }),
              autil::ThreadPoolBase::ERROR_NONE);
    worker_started.get_future().wait();

    const auto         io_thread = std::this_thread::get_id();
    std::atomic<bool>  completed{false};
    std::promise<void> done;
    auto               done_future = done.get_future();
    auto               closure     = makeClosure(
        arpc::ARPC_ERROR_NONE, KvCacheStoreServiceErrorCode::EC_SUCCESS, [&](bool ok, CacheStoreErrorCode ec) {
            EXPECT_TRUE(ok);
            EXPECT_EQ(ec, CacheStoreErrorCode::None);
            EXPECT_NE(std::this_thread::get_id(), io_thread);
            EXPECT_EQ(std::string(static_cast<char*>(block->addr.get()), block->len), std::string(32, 'x'));
            completed = true;
            done.set_value();
        });
    closure->request_block_buffer_->addBlock(block);
    auto response_block = closure->response_->add_blocks();
    response_block->set_key("key");
    response_block->set_len(32);
    response_block->set_content(std::string(32, 'x'));
    closure->response_pool_ = &pool;
    closure->Run();
    EXPECT_FALSE(completed.load());
    release_worker.set_value();
    EXPECT_EQ(done_future.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    pool.stop();
}

TEST_F(TcpCacheStoreLoadServiceClosureTest, RejectedResponseWorkCompletesWithFailure) {
    autil::ThreadPool stopped_pool(1, 1, false, "CacheLoadTest");
    stopped_pool.stop();
    int  callback_count = 0;
    auto closure        = makeClosure(
        arpc::ARPC_ERROR_NONE, KvCacheStoreServiceErrorCode::EC_SUCCESS, [&](bool ok, CacheStoreErrorCode ec) {
            ++callback_count;
            EXPECT_FALSE(ok);
            EXPECT_EQ(ec, CacheStoreErrorCode::PushWorkerItemFailed);
        });
    closure->response_pool_ = &stopped_pool;
    closure->Run();
    EXPECT_EQ(callback_count, 1);
}

TEST_F(TcpCacheStoreLoadServiceClosureTest, testRun_ControllerFailed) {
    std::mutex mutex;
    auto       callback = [&mutex](bool ok, CacheStoreErrorCode ec) {
        ASSERT_FALSE(ok);
        ASSERT_EQ(CacheStoreErrorCode::LoadSendRequestFailed, ec);
        mutex.unlock();
    };

    auto closure = makeClosure(arpc::ARPC_ERROR_CONNECTION_CLOSED, KvCacheStoreServiceErrorCode::EC_SUCCESS, callback);
    ASSERT_TRUE(closure != nullptr);
    closure->Run();

    mutex.lock();
    mutex.unlock();
}

TEST_F(TcpCacheStoreLoadServiceClosureTest, testRun_ResponseFailed) {
    std::mutex mutex;
    auto       callback = [&mutex](bool ok, CacheStoreErrorCode ec) {
        ASSERT_FALSE(ok);
        ASSERT_EQ(CacheStoreErrorCode::LoadBufferTimeout, ec);
        mutex.unlock();
    };

    auto closure = makeClosure(arpc::ARPC_ERROR_NONE, KvCacheStoreServiceErrorCode::EC_FAILED_LOAD_BUFFER, callback);
    ASSERT_TRUE(closure != nullptr);
    closure->Run();

    mutex.lock();
    mutex.unlock();
}

TEST_F(TcpCacheStoreLoadServiceClosureTest, testRun_BlockSizeError) {
    std::mutex mutex;
    auto       callback = [&mutex](bool ok, CacheStoreErrorCode ec) {
        ASSERT_FALSE(ok);
        ASSERT_EQ(CacheStoreErrorCode::LoadBufferTimeout, ec);
        mutex.unlock();
    };

    auto closure = makeClosure(arpc::ARPC_ERROR_NONE, KvCacheStoreServiceErrorCode::EC_SUCCESS, callback);
    ASSERT_TRUE(closure != nullptr);

    uint32_t block_size = 16;
    closure->request_block_buffer_->addBlock(block_buffer_util_->makeBlockBuffer("a", block_size, '0', true));
    closure->Run();

    mutex.lock();
    mutex.unlock();
}

TEST_F(TcpCacheStoreLoadServiceClosureTest, testRun_BlockContentError) {
    std::mutex mutex;
    auto       callback = [&mutex](bool ok, CacheStoreErrorCode ec) {
        ASSERT_FALSE(ok);
        ASSERT_EQ(CacheStoreErrorCode::LoadBufferTimeout, ec);
        mutex.unlock();
    };

    auto closure = makeClosure(arpc::ARPC_ERROR_NONE, KvCacheStoreServiceErrorCode::EC_SUCCESS, callback);
    ASSERT_TRUE(closure != nullptr);

    uint32_t block_size = 16;
    closure->request_block_buffer_->addBlock(block_buffer_util_->makeBlockBuffer("a", block_size, '0', true));
    closure->response_->add_blocks()->set_len(0);
    closure->Run();

    mutex.lock();
    mutex.unlock();
}

}  // namespace rtp_llm
