#include "autil/EnvUtil.h"
#include "rtp_llm/cpp/normal_engine/test/MockEngine.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"

namespace rtp_llm {

class DeferredEngineStartTest: public DeviceTestBase {
protected:
    std::shared_ptr<NormalEngine> prepareEngine() {
        ModelConfig model_config;
        RuntimeConfig runtime_config;
        KVCacheConfig kv_cache_config;
        kv_cache_config.kv_cache_event_publisher_type = "kvcm";
        kv_cache_config.kv_cache_event_host_ip_port    = "192.0.2.10:8080";
        auto params = createEngineInitParams(CustomConfig{}, model_config, runtime_config, kv_cache_config);
        const auto vocab = model_config.vocab_size;
        NormalExecutor::test_model_factory = [vocab](const GptModelInitParams&) {
            return std::make_unique<MockModel>(vocab);
        };
        auto engine = std::make_shared<NormalEngine>(params, nullptr, true);
        NormalExecutor::test_model_factory = nullptr;
        return engine;
    }

    void TearDown() override {
        NormalExecutor::test_model_factory = nullptr;
        DeviceTestBase::TearDown();
    }
};

TEST_F(DeferredEngineStartTest, OrdinaryStartupStillStartsImmediately) {
    auto engine = createMockEngine(CustomConfig{});
    EXPECT_TRUE(engine->running_.load());
    EXPECT_NE(engine->loop_thread_, nullptr);
    EXPECT_NE(engine->resourceContext().cache_manager->blockTreeCache(), nullptr);
}

TEST_F(DeferredEngineStartTest, RpcReleaseStartsEachRankAndOnlyLeaderCreatesBroadcaster) {
    autil::EnvGuard restored_ip("RequestedIP", "192.0.2.20");
    for (int tp_rank : {0, 1}) {
        SCOPED_TRACE(tp_rank);
        auto engine = prepareEngine();
        LocalRpcServer server;
        server.engine_ = engine;
        // Only RPC role selection varies here; the mock engine uses one GPU.
        server.maga_init_params_.parallelism_config.tp_rank = tp_rank;
        const auto& cache_manager = engine->resourceContext().cache_manager;
        const auto checkpoint_tensors = cache_manager->gpuCacheTensors();
        ASSERT_FALSE(checkpoint_tensors.empty());
        ASSERT_EQ(cache_manager->blockTreeCache(), nullptr);
        ASSERT_EQ(cache_manager->cache_event_publisher_, nullptr);
        RuntimeConfig restored_runtime;
        restored_runtime.worker_addrs = {"192.0.2.20:2"};
        restored_runtime.worker_grpc_addrs = {"127.0.0.1:1"};
        ASSERT_NO_THROW(server.updateRuntimeEndpoints(restored_runtime));
        EXPECT_EQ(cache_manager->runtime_config_.worker_addrs, restored_runtime.worker_addrs);
        EXPECT_EQ(cache_manager->runtime_config_.worker_grpc_addrs, restored_runtime.worker_grpc_addrs);
        EXPECT_EQ(cache_manager->kv_cache_config_.kv_cache_event_host_ip_port, "192.0.2.20:8080");
        ASSERT_EQ(engine->loop_thread_, nullptr);
        ASSERT_EQ(server.tp_broadcaster_, nullptr);
        ASSERT_NO_THROW(server.startServices());
        EXPECT_TRUE(engine->running_.load());
        auto first_thread = engine->loop_thread_;
        ASSERT_NE(first_thread, nullptr);
        auto first_broadcaster = server.tp_broadcaster_;
        auto block_tree = cache_manager->blockTreeCache();
        ASSERT_NE(block_tree, nullptr);
        const auto released_tensors = cache_manager->gpuCacheTensors();
        ASSERT_EQ(released_tensors.size(), checkpoint_tensors.size());
        for (size_t i = 0; i < checkpoint_tensors.size(); ++i) {
            EXPECT_EQ(released_tensors[i].data_ptr(), checkpoint_tensors[i].data_ptr());
        }
        EXPECT_EQ(first_broadcaster != nullptr, tp_rank == 0);
        ASSERT_NO_THROW(server.startServices());
        EXPECT_EQ(engine->loop_thread_, first_thread);
        EXPECT_EQ(server.tp_broadcaster_, first_broadcaster);
        EXPECT_EQ(cache_manager->blockTreeCache(), block_tree);
        EXPECT_ANY_THROW(server.updateRuntimeEndpoints(restored_runtime));
    }
}

TEST_F(DeferredEngineStartTest, AbortedTemplateCanStopBeforeLoopStarts) {
    auto engine = prepareEngine();
    ASSERT_TRUE(engine->resourceContext().cache_manager);
    EXPECT_FALSE(engine->running_.load());
    EXPECT_EQ(engine->loop_thread_, nullptr);
    EXPECT_TRUE(engine->stop().ok());
    // Destruction must also tolerate an initialized engine with no loop.
    engine.reset();
}

TEST_F(DeferredEngineStartTest, ReleaseStartsOneLoopAndGenerates) {
    auto engine = prepareEngine();
    ASSERT_FALSE(engine->running_.load());
    ASSERT_EQ(engine->loop_thread_, nullptr);
    std::shared_ptr<EngineBase> base = engine;
    ASSERT_TRUE(base->startLoop().ok());
    auto first_thread = engine->loop_thread_;
    ASSERT_NE(first_thread, nullptr);
    ASSERT_TRUE(base->startLoop().ok());
    EXPECT_EQ(engine->loop_thread_, first_thread);

    auto input = std::make_shared<GenerateInput>();
    input->input_ids = torch::tensor({1, 2, 3}, torch::kInt32);
    input->generate_config = std::make_shared<GenerateConfig>();
    input->generate_config->max_new_tokens = 3;
    input->generate_config->is_streaming = false;
    auto stream = engine->enqueue(input);
    auto output = stream->nextOutput();
    ASSERT_TRUE(output.ok()) << output.status().ToString();
    EXPECT_EQ(output.value().generate_outputs[0].aux_info.output_len, 3);
    EXPECT_TRUE(stream->hasEvent(StreamEvents::GenerateDone));
}

}  // namespace rtp_llm
