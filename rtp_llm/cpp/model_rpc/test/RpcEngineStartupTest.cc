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

}  // namespace
}  // namespace rtp_llm
