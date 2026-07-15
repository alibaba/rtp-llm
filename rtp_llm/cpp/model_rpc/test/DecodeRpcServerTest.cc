#include <gtest/gtest.h>

#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"
#include "rtp_llm/cpp/testing/TestLogCapture.h"

namespace rtp_llm {

TEST(DecodeRpcServerTest, MtpCacheKeyUsesSharedBaseModelIdForEverySlot) {
    constexpr size_t mtp_base_model_id = 17;

    for (size_t mtp_model_id = 0; mtp_model_id < 2; ++mtp_model_id) {
        EXPECT_EQ(DecodeRpcServer::makeMTPModuleCacheKey(mtp_base_model_id, "101", /*layer_id=*/0),
                  "model_id_17_token_id_str_101_layer_id_0")
            << "mtp_model_id=" << mtp_model_id;
    }
}

TEST(DecodeRpcServerTest, MtpLoadPlanContainsOnlyModule0) {
    auto module0          = std::make_unique<EngineInitParams>();
    module0->model_id     = 17;
    auto module1          = std::make_unique<EngineInitParams>();
    module1->model_id     = 23;
    auto mtp_model_params = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
    mtp_model_params->push_back(std::move(module0));
    mtp_model_params->push_back(std::move(module1));
    ProposeModelEngineInitParams propose_params(SP_TYPE_MTP, /*gen_num_per_cycle=*/2, std::move(mtp_model_params));

    const auto plan = DecodeRpcServer::makeMTPModuleLoadPlan(&propose_params);

    ASSERT_EQ(plan.size(), 1);
    EXPECT_EQ(plan[0].module_index, 0);
    EXPECT_EQ(plan[0].engine_init_params, propose_params.mtp_model_params_->at(0).get());
    EXPECT_EQ(plan[0].cache_model_id, 17);
}

TEST(DecodeRpcServerTest, MtpLoadPlanRejectsMissingModule0) {
    EXPECT_TRUE(DecodeRpcServer::makeMTPModuleLoadPlan(nullptr).empty());

    ProposeModelEngineInitParams missing_params;
    EXPECT_TRUE(DecodeRpcServer::makeMTPModuleLoadPlan(&missing_params).empty());

    auto                         empty_params = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
    ProposeModelEngineInitParams no_modules(SP_TYPE_MTP, /*gen_num_per_cycle=*/2, std::move(empty_params));
    EXPECT_TRUE(DecodeRpcServer::makeMTPModuleLoadPlan(&no_modules).empty());

    auto mtp_model_params = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
    mtp_model_params->push_back(nullptr);
    mtp_model_params->push_back(std::make_unique<EngineInitParams>());
    ProposeModelEngineInitParams null_module0(SP_TYPE_MTP, /*gen_num_per_cycle=*/2, std::move(mtp_model_params));
    EXPECT_TRUE(DecodeRpcServer::makeMTPModuleLoadPlan(&null_module0).empty());
}

TEST(DecodeRpcServerTest, MtpLoadPlanIgnoresInactiveModules) {
    auto mtp_model_params = std::make_unique<std::vector<std::unique_ptr<EngineInitParams>>>();
    mtp_model_params->push_back(std::make_unique<EngineInitParams>());
    mtp_model_params->push_back(nullptr);
    ProposeModelEngineInitParams propose_params(SP_TYPE_MTP, /*gen_num_per_cycle=*/2, std::move(mtp_model_params));

    const auto plan = DecodeRpcServer::makeMTPModuleLoadPlan(&propose_params);

    ASSERT_EQ(plan.size(), 1);
    EXPECT_EQ(plan[0].engine_init_params, propose_params.mtp_model_params_->at(0).get());
}

TEST(DecodeRpcServerTest, ReadFailureLogContainsPeerErrorAndEveryBlockKey) {
    test::TestLogCapture log_capture("read_cache_failure");
    DecodeRpcServer::logReadFailures(/*request_id=*/42,
                                     "127.0.0.1:1:2",
                                     ErrorCode::CACHE_STORE_LOAD_CONNECT_FAILED,
                                     "connect failed",
                                     {"blocks={kv_key_0,kv_key_1}"});

    const auto log_content = log_capture.content();
    EXPECT_NE(log_content.find("PD_CACHE_KEY_READ_FAILED"), std::string::npos);
    EXPECT_NE(log_content.find("127.0.0.1:1:2"), std::string::npos);
    EXPECT_NE(log_content.find("kv_key_0"), std::string::npos);
    EXPECT_NE(log_content.find("kv_key_1"), std::string::npos);
}

TEST(DecodeRpcServerTest, ReadTimeoutLogsKeysAndCancellationIsSilent) {
    test::TestLogCapture log_capture("read_cache_timeout_cancel");
    DecodeRpcServer::logReadFailures(
        /*request_id=*/43, "peer", ErrorCode::LOAD_CACHE_TIMEOUT, "timeout", {"blocks={timeout_key}"});
    DecodeRpcServer::logReadFailures(
        /*request_id=*/44, "peer", ErrorCode::CANCELLED, "cancelled", {"blocks={cancelled_key}"});

    const auto log_content = log_capture.content();
    EXPECT_NE(log_content.find("timeout_key"), std::string::npos);
    EXPECT_EQ(log_content.find("cancelled_key"), std::string::npos);
}

}  // namespace rtp_llm
