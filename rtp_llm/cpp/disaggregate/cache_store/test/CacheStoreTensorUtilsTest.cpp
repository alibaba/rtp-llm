#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreTensorUtils.h"
#include "rtp_llm/cpp/disaggregate/cache_store/LockedBlockBufferManager.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBufferStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpCacheStoreServiceImpl.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpClient.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TimerManager.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {

namespace {
bool    runtime_initialized = false;
int64_t runtime_device_id   = 0;
}

bool isRuntimeInitialized() {
    return runtime_initialized;
}

int64_t getDeviceId() {
    return runtime_device_id;
}

void cudaPreRun(int) {}

void execNoBlockCopy(const CopyParams& params) {
    params.check();
    params.dst.copy_(params.src, false);
}

namespace {

class TestMemoryUtil: public MemoryUtil {
public:
    bool regUserMr(void*, uint64_t, bool, uint64_t) override {
        return true;
    }

    bool deregUserMr(void*, bool) override {
        return true;
    }

    bool isMemoryMr(void*, uint64_t, bool, bool) override {
        return true;
    }

    bool findMemoryMr(void*, void*, uint64_t, bool, bool) override {
        return true;
    }

    bool isRdmaMode() override {
        return false;
    }
};

class TestClosure: public google::protobuf::Closure {
public:
    void Run() override {
        called = true;
    }

    bool called{false};
};

class TestTcpCacheStoreServiceImpl: public TcpCacheStoreServiceImpl {
public:
    using TcpCacheStoreServiceImpl::TcpCacheStoreServiceImpl;

    void blockRead(::google::protobuf::RpcController* controller,
                   const ::BlockReadRequest*          request,
                   BlockReadResponse*                 response,
                   ::google::protobuf::Closure*       done) {
        blockReadImpl(controller, request, response, done);
    }
};

}  // namespace

TEST(CacheStoreTensorUtilsTest, BlockReadUsesCpuTensorForCpuBlocks) {
    auto                         memory_util                = std::make_shared<TestMemoryUtil>();
    auto                         request_block_buffer_store = std::make_shared<RequestBlockBufferStore>(memory_util);
    auto                         timer_manager              = std::make_shared<TimerManager>();
    auto                         locked_block_manager       = std::make_shared<LockedBlockBufferManager>();
    auto                         tcp_client                 = std::make_shared<TcpClient>();
    TestTcpCacheStoreServiceImpl service(
        memory_util, request_block_buffer_store, nullptr, timer_manager, locked_block_manager, tcp_client, 0);

    std::vector<char> cpu_data{'a', 'b', 'c', 'd', 'e', 'f'};
    auto              cpu_addr = std::shared_ptr<void>(cpu_data.data(), [](void*) {});
    auto              cpu_block =
        std::make_shared<BlockBuffer>("cpu_block", cpu_addr, static_cast<uint32_t>(cpu_data.size()), false, true);
    ASSERT_TRUE(request_block_buffer_store->regUserBuffers({cpu_block}));

    BlockReadRequest request;
    auto*            block = request.add_blocks();
    block->set_key("cpu_block");
    block->set_addr(reinterpret_cast<int64_t>(cpu_data.data()) + 1);
    block->set_len(3);

    BlockReadResponse response;
    TestClosure       done;
    service.blockRead(nullptr, &request, &response, &done);

    ASSERT_TRUE(done.called);
    ASSERT_EQ(response.error_code(), KvCacheStoreServiceErrorCode::EC_SUCCESS);
    ASSERT_EQ(response.blocks_size(), 1);
    EXPECT_EQ(response.blocks(0).content(), "bcd");
}

TEST(CacheStoreTensorUtilsTest, UsesRuntimeDeviceForGpuTensorOptions) {
    runtime_initialized = true;
    runtime_device_id   = 3;

    const auto options = cacheStoreByteTensorOptions(true);

    EXPECT_EQ(options.device().type(), torch::kCUDA);
    EXPECT_EQ(options.device().index(), 3);

    runtime_initialized = false;
    runtime_device_id   = 0;
}

TEST(CacheStoreTensorUtilsTest, DefaultsToDeviceZeroBeforeRuntimeInitialization) {
    runtime_initialized = false;
    runtime_device_id   = 7;

    const auto options = cacheStoreByteTensorOptions(true);

    EXPECT_EQ(options.device().type(), torch::kCUDA);
    EXPECT_EQ(options.device().index(), 0);

    runtime_device_id = 0;
}

}  // namespace rtp_llm
