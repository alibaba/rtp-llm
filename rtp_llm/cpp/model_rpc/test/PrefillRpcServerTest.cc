#include "gtest/gtest.h"
#include <memory>

#include "torch/all.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestBase.h"

namespace rtp_llm {

class TestablePrefillRpcServer : public PrefillRpcServer {
public:
    EngineInitParams& initParams() { return maga_init_params_; }

    ErrorInfo collectStreamOutputPublic(std::shared_ptr<GenerateStream>       stream,
                                        const std::shared_ptr<GenerateInput>& input) {
        grpc::ServerContext ctx;
        GenerateOutputs     outputs;
        return collectStreamOutput(&ctx, stream, input, outputs);
    }
};

class PrefillRpcServerTest: public DeviceTestBase {};

TEST_F(PrefillRpcServerTest, waitStreamBeforeRunReturnsSchedulerEnqueueErrorImmediately) {
    // block_num=1, tokens_per_block=2 → maxAvailableTokensNum=2 < inputLength=3,
    // so checkInputLength fails synchronously with EXCEEDS_KV_CACHE_MAX_LEN.
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 1, 1, 4, 2, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;

    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;

    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    std::shared_ptr<GenerateInput> query = std::make_shared<GenerateInput>();
    query->input_ids                     = torch::tensor({1, 2, 3}, torch::kInt32);
    query->generate_config               = std::make_shared<GenerateConfig>();

    auto stream =
        std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);

    ASSERT_FALSE(scheduler.enqueue(stream).ok());
    ASSERT_TRUE(stream->hasError());
    ASSERT_EQ(stream->getStatus(), StreamState::FINISHED);
    ASSERT_EQ(stream->statusInfo().code(), ErrorCode::EXCEEDS_KV_CACHE_MAX_LEN);

    TestablePrefillRpcServer server;
    server.initParams().pd_sep_config.prefill_max_wait_timeout_ms = 1;

    auto error_info = server.waitStreamBeforeRun(stream);

    ASSERT_TRUE(error_info.hasError());
    ASSERT_EQ(error_info.code(), ErrorCode::EXCEEDS_KV_CACHE_MAX_LEN);
    ASSERT_NE(error_info.ToString().find("kv cache max available tokens num"), std::string::npos);
}

TEST_F(PrefillRpcServerTest, collectStreamOutputReturnsErrorForFailedBatchEnqueue) {
    // block_num=1, tokens_per_block=2 → maxAvailableTokensNum=2 < inputLength=3,
    // so checkInputLength fails synchronously with EXCEEDS_KV_CACHE_MAX_LEN.
    CacheConfig                     cache_config  = makeMhaCacheConfig(1, 1, 1, 4, 2, rtp_llm::DataType::TYPE_FP16);
    std::shared_ptr<KVCacheManager> cache_manager = std::make_shared<KVCacheManager>(cache_config);
    ASSERT_TRUE(cache_manager->init());

    ResourceContext resource_context;
    resource_context.cache_manager = cache_manager;

    ModelConfig model_config;
    model_config.max_seq_len = 8192;

    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                     = 100;
    runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;

    PDSepConfig         pd_sep_config;
    ParallelismConfig   parallelism_config;
    ModelSpecificConfig model_specific_config;
    FIFOScheduler       scheduler(
        runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager);

    std::shared_ptr<GenerateInput> query = std::make_shared<GenerateInput>();
    query->input_ids                     = torch::tensor({1, 2, 3}, torch::kInt32);
    query->generate_config               = std::make_shared<GenerateConfig>();

    auto stream =
        std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);

    auto streams = scheduler.batchEnqueue({stream});
    ASSERT_EQ(streams.size(), 1u);
    ASSERT_TRUE(streams[0]->hasError());
    ASSERT_EQ(streams[0]->getStatus(), StreamState::FINISHED);

    TestablePrefillRpcServer server;
    auto                     err = server.collectStreamOutputPublic(streams[0], query);

    ASSERT_TRUE(err.hasError());
    ASSERT_EQ(err.code(), streams[0]->statusInfo().code());
}

}  // namespace rtp_llm
