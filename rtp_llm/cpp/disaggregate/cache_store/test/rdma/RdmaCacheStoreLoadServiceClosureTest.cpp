#include "gtest/gtest.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaCacheStoreLoadServiceClosure.h"
#include "rtp_llm/cpp/disaggregate/cache_store/test/CacheStoreTestBase.h"
#include "autil/NetUtil.h"
#include "autil/EnvUtil.h"

namespace rtp_llm {

class RdmaCacheStoreLoadServiceClosureTest: public CacheStoreTestBase {
protected:
    RdmaCacheStoreLoadServiceClosure*
    makeClosure(arpc::ErrorCode arpc_ec, KvCacheStoreServiceErrorCode resp_ec, CacheStoreLoadDoneCallback callback);
};

RdmaCacheStoreLoadServiceClosure* RdmaCacheStoreLoadServiceClosureTest::makeClosure(
    arpc::ErrorCode arpc_ec, KvCacheStoreServiceErrorCode resp_ec, CacheStoreLoadDoneCallback callback) {
    auto request_buffer = std::make_shared<RequestBlockBuffer>("request-id");
    auto controller     = new arpc::ANetRPCController();
    auto request        = new CacheLoadRequest;
    auto response       = new CacheLoadResponse;

    if (arpc_ec != arpc::ARPC_ERROR_NONE) {
        controller->SetFailed("failed");
        controller->SetErrorCode(arpc_ec);
    }

    response->set_error_code(resp_ec);

    auto collector = std::make_shared<CacheStoreClientLoadMetricsCollector>(nullptr, 1, 1);
    return new RdmaCacheStoreLoadServiceClosure(
        nullptr, nullptr, memory_util_, request_buffer, controller, request, response, callback, collector);
}

TEST_F(RdmaCacheStoreLoadServiceClosureTest, testRun_Success) {
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

TEST_F(RdmaCacheStoreLoadServiceClosureTest, testRun_ControllerFailed) {
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

TEST_F(RdmaCacheStoreLoadServiceClosureTest, testRun_ResponseFailed) {
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

}  // namespace rtp_llm