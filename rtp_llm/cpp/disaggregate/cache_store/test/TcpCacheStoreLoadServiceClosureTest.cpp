#include "gtest/gtest.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpCacheStoreLoadServiceClosure.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/test/CacheStoreTestBase.h"
#include "rtp_llm/cpp/utils/DevicePin.h"
#include "autil/NetUtil.h"
#include "autil/EnvUtil.h"

#include <limits>

namespace rtp_llm {

class TcpCacheStoreLoadServiceClosureTest: public CacheStoreTestBase {
protected:
    TcpCacheStoreLoadServiceClosure* makeClosure(arpc::ErrorCode              arpc_ec,
                                                 KvCacheStoreServiceErrorCode resp_ec,
                                                 CacheStoreLoadDoneCallback   callback,
                                                 int                          device_id = -1);
};

class TcpCacheStoreLoadServiceClosureNoDeviceTest: public TcpCacheStoreLoadServiceClosureTest {
protected:
    // Pin failure is intentionally tested without CacheStoreTestBase's device-0 runtime initialization.
    void SetUp() override {}
};

TcpCacheStoreLoadServiceClosure* TcpCacheStoreLoadServiceClosureTest::makeClosure(arpc::ErrorCode              arpc_ec,
                                                                                  KvCacheStoreServiceErrorCode resp_ec,
                                                                                  CacheStoreLoadDoneCallback   callback,
                                                                                  int device_id) {
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
        memory_util_, request_buffer, controller, request, response, callback, collector, device_id);
}

TEST_F(TcpCacheStoreLoadServiceClosureNoDeviceTest, testRun_DevicePinFailed) {
#if USING_CUDA || USING_ROCM
    bool                callback_called = false;
    bool                callback_ok     = true;
    CacheStoreErrorCode callback_ec     = CacheStoreErrorCode::None;
    auto                callback        = [&](bool ok, CacheStoreErrorCode ec) {
        callback_called = true;
        callback_ok     = ok;
        callback_ec     = ec;
    };

    auto closure = makeClosure(
        arpc::ARPC_ERROR_NONE, KvCacheStoreServiceErrorCode::EC_SUCCESS, callback, std::numeric_limits<int>::max());
    ASSERT_NE(nullptr, closure);
    closure->Run();

    EXPECT_TRUE(callback_called);
    EXPECT_FALSE(callback_ok);
    EXPECT_EQ(CacheStoreErrorCode::LoadErrorUnknown, callback_ec);
#else
    GTEST_SKIP() << "device pinning is a no-op in CPU-only builds";
#endif
}

TEST_F(TcpCacheStoreLoadServiceClosureNoDeviceTest, testRun_SwitchesToRequestedDevice) {
#if USING_CUDA
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 2) {
        GTEST_SKIP() << "device switching requires at least two visible CUDA devices";
    }

    int original_device = -1;
    ASSERT_EQ(cudaSuccess, cudaGetDevice(&original_device));
    ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
#elif USING_ROCM
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 2) {
        GTEST_SKIP() << "device switching requires at least two visible ROCm devices";
    }

    int original_device = -1;
    ASSERT_EQ(hipSuccess, hipGetDevice(&original_device));
    ASSERT_EQ(hipSuccess, hipSetDevice(0));
#else
    GTEST_SKIP() << "device pinning is a no-op in CPU-only builds";
#endif

#if USING_CUDA || USING_ROCM
    bool                callback_called = false;
    bool                callback_ok     = false;
    CacheStoreErrorCode callback_ec     = CacheStoreErrorCode::LoadErrorUnknown;
    auto                callback        = [&](bool ok, CacheStoreErrorCode ec) {
        callback_called = true;
        callback_ok     = ok;
        callback_ec     = ec;
    };

    auto closure =
        makeClosure(arpc::ARPC_ERROR_NONE, KvCacheStoreServiceErrorCode::EC_SUCCESS, callback, /*device_id=*/1);
    ASSERT_NE(nullptr, closure);
    closure->Run();

#if USING_CUDA
    int current_device = -1;
    EXPECT_EQ(cudaSuccess, cudaGetDevice(&current_device));
    EXPECT_EQ(1, current_device);
    EXPECT_EQ(cudaSuccess, cudaSetDevice(original_device));
#elif USING_ROCM
    int current_device = -1;
    EXPECT_EQ(hipSuccess, hipGetDevice(&current_device));
    EXPECT_EQ(1, current_device);
    EXPECT_EQ(hipSuccess, hipSetDevice(original_device));
#endif
    EXPECT_TRUE(callback_called);
    EXPECT_TRUE(callback_ok);
    EXPECT_EQ(CacheStoreErrorCode::None, callback_ec);
#endif
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
