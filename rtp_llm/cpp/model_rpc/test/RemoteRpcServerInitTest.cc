#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/embed.h>

#include "rtp_llm/cpp/disaggregate/cache_store/NoRdmaMemoryUtilImpl.h"
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

TEST_F(RemoteRpcServerInitTest, RdmaInitializesBeforeEngineAndReusesPreparedMemoryUtil) {
    auto params = initParams(true);
    {
        ScopedRemoteRpcServer server;
        ASSERT_TRUE(server.init(params, nullptr, py::none()).ok());
        EXPECT_EQ(init_trace.events, (std::vector<std::string>{"memory", "model"}));
        EXPECT_EQ(init_trace.rdma_modes, (std::vector<bool>{true}));
        EXPECT_EQ(server.memoryUtil(), init_trace.memory_util.lock());
        EXPECT_EQ(init_trace.released, 0);
    }
    EXPECT_TRUE(init_trace.memory_util.expired());
    EXPECT_EQ(init_trace.released, 1);
}

TEST_F(RemoteRpcServerInitTest, LocalInitFailureReleasesPreparedMemoryUtil) {
    auto params = initParams(true);
    // This invalid local-MM configuration returns before engine construction.
    params.model_config_.mm_model_config.is_multimodal = true;
    params.vit_config.vit_separation                   = VitSeparation::VIT_SEPARATION_LOCAL;
    ScopedRemoteRpcServer server;
    EXPECT_EQ(server.init(params, nullptr, py::none()).error_code(), grpc::StatusCode::FAILED_PRECONDITION);
    EXPECT_EQ(init_trace.events, (std::vector<std::string>{"memory"}));
    EXPECT_EQ(init_trace.rdma_modes, (std::vector<bool>{true}));
    EXPECT_EQ(server.getEngine(), nullptr);
    EXPECT_TRUE(init_trace.memory_util.expired());
    EXPECT_EQ(init_trace.released, 1);
}

TEST_F(RemoteRpcServerInitTest, EngineConstructionFailureReleasesPreparedMemoryUtil) {
    auto params                        = initParams(true);
    NormalExecutor::test_model_factory = [](const GptModelInitParams&) -> std::unique_ptr<ModelBase> {
        init_trace.events.emplace_back("model");
        throw std::runtime_error("injected model construction failure");
    };
    ScopedRemoteRpcServer server;
    EXPECT_THROW(server.init(params, nullptr, py::none()), std::runtime_error);
    EXPECT_EQ(init_trace.events, (std::vector<std::string>{"memory", "model"}));
    EXPECT_EQ(init_trace.rdma_modes, (std::vector<bool>{true}));
    EXPECT_TRUE(init_trace.memory_util.expired());
    EXPECT_EQ(init_trace.released, 1);
}

TEST_F(RemoteRpcServerInitTest, TcpKeepsMemoryInitializationAfterEngine) {
    auto                  params = initParams(false);
    ScopedRemoteRpcServer server;
    ASSERT_TRUE(server.init(params, nullptr, py::none()).ok());
    EXPECT_EQ(init_trace.events, (std::vector<std::string>{"model", "memory"}));
    EXPECT_EQ(init_trace.rdma_modes, (std::vector<bool>{false}));
    EXPECT_EQ(server.memoryUtil(), init_trace.memory_util.lock());
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
