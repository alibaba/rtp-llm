#include "gtest/gtest.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpCacheStoreLoadServiceClosure.h"
#include "rtp_llm/cpp/disaggregate/cache_store/LoadContext.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/test/CacheStoreTestBase.h"
#include "rtp_llm/cpp/utils/DevicePin.h"
#include "autil/NetUtil.h"
#include "autil/EnvUtil.h"

#include <condition_variable>
#include <chrono>
#include <future>
#include <limits>
#include <thread>

namespace rtp_llm {

class TcpCacheStoreLoadServiceClosureTest: public CacheStoreTestBase {
protected:
    TcpCacheStoreLoadServiceClosure* makeClosure(arpc::ErrorCode              arpc_ec,
                                                 KvCacheStoreServiceErrorCode resp_ec,
                                                 CacheStoreLoadDoneCallback   callback,
                                                 int                          device_id = -1,
                                                 const std::shared_ptr<LoadCopyFence>& copy_fence = nullptr);
};

class TcpCacheStoreLoadServiceClosureNoDeviceTest: public TcpCacheStoreLoadServiceClosureTest {
protected:
    // Pin failure is intentionally tested without CacheStoreTestBase's device-0 runtime initialization.
    void SetUp() override {}
};

TcpCacheStoreLoadServiceClosure* TcpCacheStoreLoadServiceClosureTest::makeClosure(arpc::ErrorCode              arpc_ec,
                                                                                  KvCacheStoreServiceErrorCode resp_ec,
                                                                                  CacheStoreLoadDoneCallback   callback,
                                                                                  int device_id,
                                                                                  const std::shared_ptr<LoadCopyFence>& copy_fence) {
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
        memory_util_, request_buffer, controller, request, response, callback, collector, device_id, copy_fence);
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

TEST_F(TcpCacheStoreLoadServiceClosureTest, testRun_ClosedFenceSkipsCopy) {
    constexpr uint32_t block_size = 16;
    auto               fence      = std::make_shared<LoadCopyFence>();
    auto               block      = block_buffer_util_->makeBlockBuffer("a", block_size, 'B', false);

    bool                callback_called = false;
    bool                callback_ok     = true;
    CacheStoreErrorCode callback_ec     = CacheStoreErrorCode::None;
    auto callback = [&](bool ok, CacheStoreErrorCode ec) {
        callback_called = true;
        callback_ok     = ok;
        callback_ec     = ec;
    };

    auto closure =
        makeClosure(arpc::ARPC_ERROR_NONE, KvCacheStoreServiceErrorCode::EC_SUCCESS, callback, -1, fence);
    closure->request_block_buffer_->addBlock(block);
    auto response_block = closure->response_->add_blocks();
    response_block->set_key("a");
    response_block->set_len(block_size);
    response_block->set_content(std::string(block_size, 'A'));

    fence->closeAndDrain();
    closure->Run();

    EXPECT_TRUE(callback_called);
    EXPECT_FALSE(callback_ok);
    EXPECT_EQ(CacheStoreErrorCode::LoadBufferTimeout, callback_ec);
    EXPECT_EQ(std::string(block_size, 'B'), std::string(static_cast<const char*>(block->addr.get()), block_size));
}

TEST(LoadCopyFenceTest, closeDrainsActiveCopyAndRejectsFutureCopy) {
    auto fence = std::make_shared<LoadCopyFence>();

    std::mutex              mutex;
    std::condition_variable cond;
    bool                    copy_entered   = false;
    bool                    release_copy   = false;
    bool                    close_returned = false;

    std::thread copy_thread([&]() {
        EXPECT_TRUE(fence->runIfOpen([&]() {
            std::unique_lock<std::mutex> lock(mutex);
            copy_entered = true;
            cond.notify_all();
            cond.wait(lock, [&]() { return release_copy; });
        }));
    });

    {
        std::unique_lock<std::mutex> lock(mutex);
        cond.wait(lock, [&]() { return copy_entered; });
    }

    std::thread close_thread([&]() {
        fence->closeAndDrain();
        std::lock_guard<std::mutex> lock(mutex);
        close_returned = true;
        cond.notify_all();
    });

    const auto close_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (!fence->closed() && std::chrono::steady_clock::now() < close_deadline) {
        std::this_thread::yield();
    }
    EXPECT_TRUE(fence->closed());
    {
        std::lock_guard<std::mutex> lock(mutex);
        EXPECT_FALSE(close_returned);
        release_copy = true;
    }
    cond.notify_all();

    copy_thread.join();
    close_thread.join();

    {
        std::lock_guard<std::mutex> lock(mutex);
        EXPECT_TRUE(close_returned);
    }
    bool mutated = false;
    EXPECT_FALSE(fence->runIfOpen([&]() { mutated = true; }));
    EXPECT_FALSE(mutated);
}

TEST(LoadContextTest, timeoutDrainsActiveCopyAndLateCallbackCannotOverwrite) {
    auto context = std::make_shared<LoadContext>(nullptr, false);
    context->expect_layer_cnt_ = 1;
    context->deadline_ms_      = 0;
    auto fence                 = context->copy_fence_;

    std::mutex              mutex;
    std::condition_variable cond;
    bool                    copy_entered  = false;
    bool                    release_copy  = false;
    bool                    wait_returned = false;

    std::thread copy_thread([&]() {
        EXPECT_TRUE(fence->runIfOpen([&]() {
            std::unique_lock<std::mutex> lock(mutex);
            copy_entered = true;
            cond.notify_all();
            cond.wait(lock, [&]() { return release_copy; });
        }));
    });
    {
        std::unique_lock<std::mutex> lock(mutex);
        cond.wait(lock, [&]() { return copy_entered; });
    }

    std::thread wait_thread([&]() {
        context->waitDone();
        std::lock_guard<std::mutex> lock(mutex);
        wait_returned = true;
        cond.notify_all();
    });

    const auto timeout_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (!fence->closed() && std::chrono::steady_clock::now() < timeout_deadline) {
        std::this_thread::yield();
    }
    EXPECT_TRUE(fence->closed());
    {
        std::lock_guard<std::mutex> lock(mutex);
        EXPECT_FALSE(wait_returned);
        release_copy = true;
    }
    cond.notify_all();

    copy_thread.join();
    wait_thread.join();

    EXPECT_EQ(
        ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT, context->getErrorInfo().code()
    );
    auto request_buffer = std::make_shared<RequestBlockBuffer>("late-request");
    context->updateResult(
        false, CacheStoreErrorCode::LoadErrorUnknown, request_buffer
    );
    EXPECT_EQ(
        ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT, context->getErrorInfo().code()
    );
    EXPECT_TRUE(context->failedBlockDebugInfos().empty());
}

TEST(LoadContextTest, cancelIsTerminalForLateCallback) {
    auto context = std::make_shared<LoadContext>(nullptr, false);
    context->expect_layer_cnt_  = 1;
    context->deadline_ms_       = std::numeric_limits<int64_t>::max();
    context->check_cancel_func_ = []() { return true; };

    context->waitDone();
    EXPECT_EQ(ErrorCode::CANCELLED, context->getErrorInfo().code());

    auto request_buffer = std::make_shared<RequestBlockBuffer>("late-request");
    context->updateResult(
        false, CacheStoreErrorCode::LoadErrorUnknown, request_buffer
    );
    EXPECT_EQ(ErrorCode::CANCELLED, context->getErrorInfo().code());
    EXPECT_TRUE(context->failedBlockDebugInfos().empty());
}

TEST(LoadContextTest, directWriteTimeoutKeepsLeaseUntilCallback) {
    auto context = std::make_shared<LoadContext>(nullptr, true);
    context->expect_layer_cnt_ = 1;
    context->deadline_ms_      = 0;

    auto wait_future =
        std::async(std::launch::async, [&]() { context->waitDone(); });

    bool       terminal = false;
    const auto terminal_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (!terminal && std::chrono::steady_clock::now() < terminal_deadline) {
        {
            std::lock_guard<std::mutex> lock(context->mutex_);
            terminal = context->terminal_;
        }
        std::this_thread::yield();
    }
    EXPECT_TRUE(terminal);
    EXPECT_EQ(std::future_status::timeout,
              wait_future.wait_for(std::chrono::milliseconds(10)));
    EXPECT_FALSE(context->copy_fence_->closed());

    auto request_buffer = std::make_shared<RequestBlockBuffer>("rdma-request");
    context->updateResult(true, CacheStoreErrorCode::None, request_buffer);

    EXPECT_EQ(std::future_status::ready,
              wait_future.wait_for(std::chrono::seconds(1)));
    wait_future.get();
    EXPECT_EQ(ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT,
              context->getErrorInfo().code());
}

}  // namespace rtp_llm
