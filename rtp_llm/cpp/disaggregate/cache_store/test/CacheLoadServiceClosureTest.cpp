#include "gtest/gtest.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheLoadServiceClosure.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/test/CacheStoreTestBase.h"
#include "autil/NetUtil.h"
#include "autil/EnvUtil.h"

namespace rtp_llm {

class CacheLoadServiceClosureTest: public CacheStoreTestBase {
protected:
    CacheLoadServiceClosure*
    makeClosure(arpc::ErrorCode arpc_ec, KvCacheStoreServiceErrorCode resp_ec, CacheStoreLoadDoneCallback callback);
};

CacheLoadServiceClosure* CacheLoadServiceClosureTest::makeClosure(arpc::ErrorCode              arpc_ec,
                                                                  KvCacheStoreServiceErrorCode resp_ec,
                                                                  CacheStoreLoadDoneCallback   callback) {
    auto request_buffer = std::make_shared<RequestBlockBuffer>("request-id");
    auto controller     = new arpc::ANetRPCController();
    auto request        = new CacheLoadRequest;
    auto response       = new CacheLoadResponse;

    if (arpc_ec != arpc::ARPC_ERROR_NONE) {
        controller->SetFailed("failed");
        controller->SetErrorCode(arpc_ec);
    }

    response->set_error_code(resp_ec);

    return new CacheLoadServiceClosure(memory_util_, request_buffer, controller, request, response, callback, nullptr);
}

TEST_F(CacheLoadServiceClosureTest, testRun_Success) {
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

TEST_F(CacheLoadServiceClosureTest, testRun_ControllerFailed) {
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

TEST_F(CacheLoadServiceClosureTest, testRun_ResponseFailed) {
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

TEST_F(CacheLoadServiceClosureTest, testRun_BlockSizeError) {
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

TEST_F(CacheLoadServiceClosureTest, testRun_BlockContentError) {
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

TEST_F(CacheLoadServiceClosureTest, testFromArpcErrorCode) {
    auto closure = makeClosure(
        arpc::ARPC_ERROR_NONE, KvCacheStoreServiceErrorCode::EC_SUCCESS, [](bool ok, CacheStoreErrorCode ec) {});
    ASSERT_TRUE(closure != nullptr);

    ASSERT_EQ(CacheStoreErrorCode::CallPrefillTimeout, closure->fromArpcErrorCode(arpc::ARPC_ERROR_TIMEOUT));

    ASSERT_EQ(CacheStoreErrorCode::LoadSendRequestFailed,
              closure->fromArpcErrorCode(arpc::ARPC_ERROR_CONNECTION_CLOSED));
    ASSERT_EQ(CacheStoreErrorCode::LoadSendRequestFailed,
              closure->fromArpcErrorCode(arpc::ARPC_ERROR_METHOD_NOT_FOUND));
    ASSERT_EQ(CacheStoreErrorCode::LoadSendRequestFailed, closure->fromArpcErrorCode(arpc::ARPC_ERROR_POST_PACKET));

    ASSERT_EQ(CacheStoreErrorCode::PushWorkerItemFailed, closure->fromArpcErrorCode(arpc::ARPC_ERROR_PUSH_WORKITEM));
    ASSERT_EQ(CacheStoreErrorCode::PushWorkerItemFailed, closure->fromArpcErrorCode(arpc::ARPC_ERROR_QUEUE_FULL));

    ASSERT_EQ(CacheStoreErrorCode::LoadErrorUnknown, closure->fromArpcErrorCode(arpc::ARPC_ERROR_APP_MIN));

    delete closure;
}

TEST_F(CacheLoadServiceClosureTest, testFromResponseErrorCode) {
    auto closure = makeClosure(
        arpc::ARPC_ERROR_NONE, KvCacheStoreServiceErrorCode::EC_SUCCESS, [](bool ok, CacheStoreErrorCode ec) {});
    ASSERT_TRUE(closure != nullptr);

    ASSERT_EQ(CacheStoreErrorCode::None, closure->fromResponseErrorCode(KvCacheStoreServiceErrorCode::EC_SUCCESS));
    ASSERT_EQ(CacheStoreErrorCode::LoadSendRequestFailed,
              closure->fromResponseErrorCode(KvCacheStoreServiceErrorCode::EC_FAILED_INVALID_REQ));
    ASSERT_EQ(CacheStoreErrorCode::LoadRdmaConnectFailed,
              closure->fromResponseErrorCode(KvCacheStoreServiceErrorCode::EC_FAILED_RDMA_CONNECTION));
    ASSERT_EQ(CacheStoreErrorCode::LoadRdmaWriteFailed,
              closure->fromResponseErrorCode(KvCacheStoreServiceErrorCode::EC_FAILED_RDMA_WRITE));
    ASSERT_EQ(CacheStoreErrorCode::LoadBufferTimeout,
              closure->fromResponseErrorCode(KvCacheStoreServiceErrorCode::EC_FAILED_LOAD_BUFFER));
    ASSERT_EQ(CacheStoreErrorCode::LoadErrorUnknown,
              closure->fromResponseErrorCode(KvCacheStoreServiceErrorCode::EC_FAILED_INTERNAL));

    delete closure;
}

}  // namespace rtp_llm