#include "gtest/gtest.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpCacheStoreLoadServiceClosure.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/test/CacheStoreTestBase.h"
#include "autil/NetUtil.h"
#include "autil/EnvUtil.h"

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