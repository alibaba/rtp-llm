#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/embed.h>

#include "rtp_llm/cpp/disaggregate/cache_store/NoRdmaMemoryUtilImpl.h"
#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"
#include "rtp_llm/cpp/model_rpc/RemoteRpcServer.h"
#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"

namespace rtp_llm {
namespace {

struct InitTrace {
    std::vector<std::string>  events;
    std::vector<bool>         rdma_modes;
    std::weak_ptr<MemoryUtil> memory_util;
    int                       released = 0;
};

InitTrace init_trace;

class RecordingMemoryUtil: public NoRdmaMemoryUtilImpl {
public:
    ~RecordingMemoryUtil() override {
        ++init_trace.released;
    }
};

class ScopedRemoteRpcServer: public RemoteRpcServer {
public:
    ~ScopedRemoteRpcServer() override {
        if (getEngine()) {
            stop();
        }
    }

    const std::shared_ptr<MemoryUtil>& memoryUtil() const {
        return cache_store_->getMemoryUtil();
    }
};

class RemoteRpcServerInitTest: public DeviceTestBase {
public:
    static void SetUpTestSuite() {
        if (!Py_IsInitialized()) {
            interpreter_ = std::make_unique<py::scoped_interpreter>();
        }
    }

    static void TearDownTestSuite() {
        interpreter_.reset();
    }

    void SetUp() override {
        DeviceTestBase::SetUp();
        init_trace                         = {};
        NormalExecutor::test_model_factory = [](const GptModelInitParams&) {
            init_trace.events.emplace_back("model");
            return std::make_unique<MockModel>(100);
        };
    }

    void TearDown() override {
        NormalExecutor::test_model_factory = nullptr;
        DeviceTestBase::TearDown();
    }

    EngineInitParams initParams(bool rdma_mode) {
        CustomConfig  config;
        ModelConfig   model_config;
        RuntimeConfig runtime_config;
        KVCacheConfig kv_cache_config;
        config.role_type     = RoleType::PREFILL;
        auto params          = createEngineInitParams(config, model_config, runtime_config, kv_cache_config);
        params.server_config = py::none();
        params.parallelism_config.role_type                    = RoleType::PREFILL;
        params.pd_sep_config.cache_store_rdma_mode             = rdma_mode;
        params.pd_sep_config.cache_store_listen_port           = 0;
        params.cache_store_config.thread_count                 = 1;
        params.cache_store_config.messager_io_thread_count     = 1;
        params.cache_store_config.messager_worker_thread_count = 1;
        return params;
    }

private:
    static std::unique_ptr<py::scoped_interpreter> interpreter_;
};

std::unique_ptr<py::scoped_interpreter> RemoteRpcServerInitTest::interpreter_;

TEST_F(RemoteRpcServerInitTest, StopClosesBackgroundWorkWhileCacheOwnersRetainMemory) {
    for (bool rdma_mode : {false, true}) {
        ScopedRemoteRpcServer server;
        ASSERT_TRUE(server.init(initParams(rdma_mode), nullptr, py::none()).ok());
        auto retained_store = server.resource().cache_store;
        ASSERT_NE(retained_store, nullptr);
        auto memory = retained_store->getMemoryUtil();
        ASSERT_NE(retained_store->messager_, nullptr);
        ASSERT_NE(retained_store->thread_pool_, nullptr);

        // RtpLLMOp owns the service through this base type.
        LocalRpcServer& service = server;
        service.stop();
        EXPECT_EQ(retained_store->messager_, nullptr);
        EXPECT_EQ(retained_store->thread_pool_, nullptr);
        EXPECT_EQ(retained_store->getMemoryUtil(), memory);
        service.stop();
        EXPECT_EQ(retained_store->getMemoryUtil(), memory);
    }
}

}  // namespace
}  // namespace rtp_llm

// Wrap only the external factory boundary. The real server, engine and cache
// store execute, while the test needs neither a RDMA NIC nor a full model.
extern "C" std::shared_ptr<rtp_llm::MemoryUtil> __wrap__ZN7rtp_llm20createMemoryUtilImplEb(bool rdma_mode) {
    rtp_llm::init_trace.events.emplace_back("memory");
    rtp_llm::init_trace.rdma_modes.push_back(rdma_mode);
    auto memory_util                = std::make_shared<rtp_llm::RecordingMemoryUtil>();
    rtp_llm::init_trace.memory_util = memory_util;
    return memory_util;
}
