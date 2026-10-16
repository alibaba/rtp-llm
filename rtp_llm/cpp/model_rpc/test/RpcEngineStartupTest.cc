#include <arpa/inet.h>
#include <memory>
#include <netinet/in.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <unistd.h>

#include <pybind11/embed.h>

#include "autil/Scope.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/model_rpc/RemoteRpcServer.h"
#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"

namespace rtp_llm {
namespace {

class RpcEngineStartupTest: public DeviceTestBase {
public:
    static void SetUpTestSuite() {
        if (!Py_IsInitialized()) {
            interpreter_ = std::make_unique<py::scoped_interpreter>();
        }
    }

    static void TearDownTestSuite() {
        interpreter_.reset();
    }

    void TearDown() override {
        NormalExecutor::test_model_factory = nullptr;
        DeviceTestBase::TearDown();
    }

protected:
    EngineInitParams makeParams(RoleType role) {
        CustomConfig config;
        config.role_type = role;
        ModelConfig   model;
        RuntimeConfig runtime;
        KVCacheConfig cache;
        auto          params                            = createEngineInitParams(config, model, runtime, cache);
        params.py_model                                 = py::none();
        params.server_config                            = py::none();
        params.pd_sep_config.cache_store_rdma_mode      = false;
        params.pd_sep_config.cache_store_listen_port    = 0;
        params.cache_store_config.cache_store_rdma_mode = false;
        NormalExecutor::test_model_factory              = [vocab = model.vocab_size](const GptModelInitParams&) {
            return std::make_unique<MockModel>(vocab);
        };
        return params;
    }

    void stopEngine(const std::shared_ptr<NormalEngine>& engine) {
        py::gil_scoped_release release;
        EXPECT_TRUE(engine->stop().ok());
    }

private:
    inline static std::unique_ptr<py::scoped_interpreter> interpreter_;
};

class RemoteRpcEngineStartupTest: public RpcEngineStartupTest, public testing::WithParamInterface<RoleType> {};

TEST_P(RemoteRpcEngineStartupTest, CacheStoreFailureNeverStartsEngineLoop) {
    auto      params          = makeParams(GetParam());
    const int reserved_socket = socket(AF_INET, SOCK_STREAM, 0);
    ASSERT_GE(reserved_socket, 0);
    autil::ScopeGuard close_socket([&] { close(reserved_socket); });
    sockaddr_in       address{};
    address.sin_family      = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port        = 0;
    ASSERT_EQ(bind(reserved_socket, reinterpret_cast<sockaddr*>(&address), sizeof(address)), 0);
    ASSERT_EQ(listen(reserved_socket, 1), 0);
    socklen_t address_size = sizeof(address);
    ASSERT_EQ(getsockname(reserved_socket, reinterpret_cast<sockaddr*>(&address), &address_size), 0);
    params.pd_sep_config.cache_store_listen_port = ntohs(address.sin_port);

    RemoteRpcServer server;
    try {
        server.init(params, nullptr, py::none());
        ADD_FAILURE() << "Expected the occupied cache-store listener to fail";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("cache store init failed"), std::string::npos);
    }
    auto engine = std::dynamic_pointer_cast<NormalEngine>(server.getEngine());
    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine->resourceContext().cache_manager->getCacheStore(), nullptr);
    // The real cache-store listener failed after engine construction. A worker
    // must not have entered its TP loop while that dependency was unavailable.
    EXPECT_FALSE(engine->running_.load());
    EXPECT_EQ(engine->loop_thread_, nullptr);
    stopEngine(engine);
    EXPECT_EQ(engine->executor_, nullptr);
}

TEST_P(RemoteRpcEngineStartupTest, SuccessfulCacheStoreBindingStartsEngineLoop) {
    auto            params = makeParams(GetParam());
    RemoteRpcServer server;
    ASSERT_TRUE(server.init(params, nullptr, py::none()).ok());
    auto engine = std::dynamic_pointer_cast<NormalEngine>(server.getEngine());
    ASSERT_NE(engine, nullptr);
    ASSERT_NE(engine->resourceContext().cache_manager->getCacheStore(), nullptr);
    EXPECT_TRUE(engine->running_.load());
    EXPECT_NE(engine->loop_thread_, nullptr);
    stopEngine(engine);
}

TEST_F(RpcEngineStartupTest, LocalInitializationStillStartsEngineLoop) {
    auto           params = makeParams(RoleType::PDFUSION);
    LocalRpcServer server;
    ASSERT_TRUE(server.init(params, nullptr, py::none()).ok());
    auto engine = std::dynamic_pointer_cast<NormalEngine>(server.getEngine());
    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine->resourceContext().cache_manager->getCacheStore(), nullptr);
    EXPECT_TRUE(engine->running_.load());
    EXPECT_NE(engine->loop_thread_, nullptr);
    stopEngine(engine);
}

INSTANTIATE_TEST_SUITE_P(PrefillAndDecode,
                         RemoteRpcEngineStartupTest,
                         testing::Values(RoleType::PREFILL, RoleType::DECODE));

}  // namespace
}  // namespace rtp_llm
