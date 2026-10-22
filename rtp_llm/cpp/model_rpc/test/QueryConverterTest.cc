#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/StaticConfig.h"
#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>

#include "rtp_llm/cpp/engine_base/stream/FrontendSpTpotSamples.h"

#define private public
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServer.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

using namespace std;
namespace rtp_llm {

class QueryConverterTest: public DeviceTestBase {};

TEST(QueryConverterCaptureTest, CaptureIntentDoesNotChangeGenerationOrRejectReuseConflict) {
    EXPECT_FALSE(GenerateConfig().capture_hidden_states);
    EXPECT_EQ(GenerateConfigPB::kCaptureHiddenStatesFieldNumber, 77);
    GenerateConfigPB omitted;
    omitted.set_max_new_tokens(4);
    EXPECT_FALSE(QueryConverter::transGenerateConfig(&omitted)->capture_hidden_states);
    for (int tokens : {0, 4}) {
        for (bool reuse : {false, true}) {
            for (bool capture : {false, true}) {
                GenerateConfigPB wire;
                wire.set_max_new_tokens(tokens);
                wire.set_reuse_cache(reuse);
                wire.set_capture_hidden_states(capture);
                const auto config = QueryConverter::transGenerateConfig(&wire);
                EXPECT_EQ(config->capture_hidden_states, capture);
                EXPECT_EQ(config->reuse_cache, reuse);
                EXPECT_EQ(config->max_new_tokens, tokens);
            }
        }
    }
}

TEST(PrefillRpcServerTest, ThinkModeBudgetUsesPdSeparationWhenMaxNewTokensIsOne) {
    GenerateConfigPB generate_config;
    generate_config.set_max_new_tokens(1);
    generate_config.set_num_beams(1);
    generate_config.set_num_return_sequences(1);
    generate_config.set_can_use_pd_separation(true);
    generate_config.set_in_think_mode(true);
    generate_config.set_max_thinking_tokens(131072);
    generate_config.add_end_think_token_ids(8);
    generate_config.add_end_think_token_ids(9);

    EXPECT_TRUE(PrefillRpcServer::shouldUsePdSeparation(generate_config));
}

TEST(PrefillRpcServerTest, SingleTokenRequestPreservesBasePdRouting) {
    GenerateConfigPB generate_config;
    generate_config.set_max_new_tokens(1);
    generate_config.set_num_beams(1);
    generate_config.set_num_return_sequences(1);
    generate_config.set_can_use_pd_separation(true);
    generate_config.set_in_think_mode(true);
    generate_config.set_max_thinking_tokens(-1);

    EXPECT_TRUE(PrefillRpcServer::shouldUsePdSeparation(generate_config));
}

TEST(QueryConverterPrefillTest, ZeroAndPositiveBudgetsPreserveWireSemantics) {
    for (const int tokens : {0, 1, 7, std::numeric_limits<int>::max()}) {
        for (const bool explicit_prefill : {false, true}) {
            SCOPED_TRACE(tokens);
            SCOPED_TRACE(explicit_prefill);
            GenerateConfigPB wire;
            wire.set_max_new_tokens(tokens);
            wire.set_prefill_only(explicit_prefill);
            wire.set_reuse_cache(true);
            wire.set_can_use_pd_separation(true);
            if (explicit_prefill && tokens != 0) {
                EXPECT_THROW(QueryConverter::resolveMaxNewTokens(wire), std::invalid_argument);
                EXPECT_THROW(QueryConverter::transGenerateConfig(&wire), std::invalid_argument);
                EXPECT_THROW(PrefillRpcServer::shouldUsePdSeparation(wire), std::invalid_argument);
                continue;
            }
            const auto config = QueryConverter::transGenerateConfig(&wire);
            EXPECT_EQ(QueryConverter::resolveMaxNewTokens(wire), tokens);
            EXPECT_EQ(config->max_new_tokens, tokens);
            EXPECT_EQ(config->isPrefillOnly(), tokens == 0);
            EXPECT_TRUE(config->reuse_cache);
            EXPECT_EQ(PrefillRpcServer::shouldUsePdSeparation(wire), tokens > 0);
        }
    }
    GenerateConfigPB omitted;
    EXPECT_EQ(QueryConverter::resolveMaxNewTokens(omitted), 0);
    EXPECT_TRUE(QueryConverter::transGenerateConfig(&omitted)->isPrefillOnly());
}

TEST(QueryConverterPrefillTest, NegativeBudgetsCannotBypassValidation) {
    for (const int tokens : {-1, std::numeric_limits<int>::min()}) {
        for (const bool explicit_prefill : {false, true}) {
            GenerateConfigPB wire;
            wire.set_max_new_tokens(tokens);
            wire.set_prefill_only(explicit_prefill);
            wire.set_can_use_pd_separation(false);
            EXPECT_THROW(QueryConverter::resolveMaxNewTokens(wire), std::invalid_argument);
            EXPECT_THROW(QueryConverter::transGenerateConfig(&wire), std::invalid_argument);
            EXPECT_THROW(PrefillRpcServer::shouldUsePdSeparation(wire), std::invalid_argument);
        }
    }
}

TEST(QueryConverterPrefillTest, UnsupportedPrefillOutputsAndSamplingAreRejected) {
    struct InvalidConfigCase {
        const char* name;
        void (*configure)(GenerateConfigPB*);
    };
    const std::array<InvalidConfigCase, 12> cases{{
        {"min_new_tokens", [](GenerateConfigPB* c) { c->set_min_new_tokens(1); }},
        {"num_beams", [](GenerateConfigPB* c) { c->set_num_beams(2); }},
        {"variable_num_beams", [](GenerateConfigPB* c) { c->add_variable_num_beams(1); }},
        {"num_return_sequences", [](GenerateConfigPB* c) { c->set_num_return_sequences(2); }},
        {"return_logits", [](GenerateConfigPB* c) { c->set_return_logits(true); }},
        {"calculate_loss", [](GenerateConfigPB* c) { c->set_calculate_loss(1); }},
        {"return_softmax_probs", [](GenerateConfigPB* c) { c->set_return_softmax_probs(true); }},
        {"return_all_probs", [](GenerateConfigPB* c) { c->set_return_all_probs(true); }},
        {"return_cum_log_probs", [](GenerateConfigPB* c) { c->set_return_cum_log_probs(true); }},
        {"return_logprobs", [](GenerateConfigPB* c) { c->set_return_logprobs(true); }},
        {"return_hidden_states", [](GenerateConfigPB* c) { c->set_return_hidden_states(true); }},
        {"return_all_hidden_states", [](GenerateConfigPB* c) { c->set_return_all_hidden_states(true); }},
    }};
    for (const auto& test_case : cases) {
        for (const bool explicit_prefill : {false, true}) {
            SCOPED_TRACE(test_case.name);
            GenerateConfigPB wire;
            wire.set_prefill_only(explicit_prefill);
            test_case.configure(&wire);
            try {
                QueryConverter::transGenerateConfig(&wire);
                FAIL() << "unsupported prefill-only configuration accepted";
            } catch (const std::invalid_argument& e) {
                EXPECT_NE(std::string(e.what()).find(test_case.name), std::string::npos);
            }
        }
    }
}

TEST(QueryConverterPrefillTest, ZeroBudgetBypassesPdEvenWithThinkingBudget) {
    GenerateConfigPB wire;
    wire.set_can_use_pd_separation(true);
    wire.set_in_think_mode(true);
    wire.set_max_thinking_tokens(131072);
    for (const bool explicit_prefill : {false, true}) {
        wire.set_prefill_only(explicit_prefill);
        EXPECT_FALSE(PrefillRpcServer::shouldUsePdSeparation(wire));
    }
}

TEST(QueryConverterPrefillTest, MainAuxField75DoesNotAliasPrefillField76) {
    EXPECT_EQ(GenerateConfigPB::kPrefillOnlyFieldNumber, 76);
    // Main's optional BoolValue aux_info is unknown on this branch. Both wrapper
    // values must be skipped without changing field 76 or the base aux default.
    const std::array<std::string, 3> payloads{{
        std::string("\xda\x04\x00\xe0\x04\x01", 6),
        std::string("\xda\x04\x02\x08\x01\xe0\x04\x01", 8),
        std::string("\xe0\x04\x01", 3),
    }};
    for (const auto& payload : payloads) {
        GenerateConfigPB wire;
        ASSERT_TRUE(wire.ParseFromString(payload));
        EXPECT_EQ(wire.max_new_tokens(), 0);
        EXPECT_TRUE(wire.prefill_only());
        const auto config = QueryConverter::transGenerateConfig(&wire);
        EXPECT_TRUE(config->isPrefillOnly());
        EXPECT_TRUE(config->aux_info);
        GenerateConfigPB round_trip;
        ASSERT_TRUE(round_trip.ParseFromString(wire.SerializeAsString()));
        EXPECT_TRUE(round_trip.prefill_only());
    }
}

TEST(QueryConverterPrefillTest, ErrorCodesUseExplicitWireMappings) {
    EXPECT_EQ(static_cast<int>(ErrorCodePB::INVALID_PARAMS), 26);
    EXPECT_EQ(static_cast<int>(ErrorCodePB::EXECUTION_EXCEPTION), 27);
    EXPECT_EQ(transErrorCodeToRPC(ErrorCode::INVALID_PARAMS), ErrorCodePB::INVALID_PARAMS);
    EXPECT_EQ(transRPCErrorCode(ErrorCodePB::INVALID_PARAMS), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(transErrorCodeToRPC(ErrorCode::EXECUTION_EXCEPTION), ErrorCodePB::EXECUTION_EXCEPTION);
    EXPECT_EQ(transRPCErrorCode(ErrorCodePB::EXECUTION_EXCEPTION), ErrorCode::EXECUTION_EXCEPTION);
    EXPECT_EQ(transErrorCodeToGrpc(ErrorCode::INVALID_PARAMS), grpc::StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(transErrorCodeToGrpc(ErrorCode::EXECUTION_EXCEPTION), grpc::StatusCode::INTERNAL);
}

TEST_F(QueryConverterTest, testTransInput) {
    GenerateInputPB input;
    input.mutable_request_info()->set_frontend_ip("10.0.0.1");
    input.mutable_request_info()->set_dash_ip("10.0.0.2");
    input.mutable_request_info()->set_trace_id("trace-123");
    input.mutable_request_info()->set_request_id("source-request-123");
    input.mutable_request_info()->set_source_role("frontend");
    input.set_priority(70);
    input.add_token_ids(0);
    input.add_token_ids(1);

    auto generate_config_pb = input.mutable_generate_config();
    generate_config_pb->set_min_new_tokens(4);
    generate_config_pb->set_max_new_tokens(5);
    generate_config_pb->set_num_beams(1);
    generate_config_pb->set_num_return_sequences(1);
    generate_config_pb->set_top_k(6);
    generate_config_pb->set_top_p(0.6);
    generate_config_pb->set_temperature(0.1);
    generate_config_pb->set_repetition_penalty(0.2);
    generate_config_pb->mutable_top_p_decay()->set_value(0.7);
    generate_config_pb->mutable_top_p_min()->set_value(0.3);
    generate_config_pb->mutable_top_p_reset_ids()->set_value(7);
    generate_config_pb->mutable_task_id()->set_value("8");
    generate_config_pb->set_calculate_loss(1);
    generate_config_pb->set_return_hidden_states(true);
    generate_config_pb->set_frontend_metric_streaming(true);
    for (int i = 0; i < 2; ++i) {
        auto* stop_words = generate_config_pb->mutable_stop_words_list()->add_rows();
        for (int j = 0; j < 3; ++j) {
            stop_words->add_values(i * 3 + j);
        }
    }
    auto  generate_input = QueryConverter::transQuery(&input);
    auto& input_ids      = generate_input->input_ids;
    ASSERT_EQ(input_ids.numel(), 2);
    ASSERT_EQ(input_ids.data_ptr<int32_t>()[0], 0);
    ASSERT_EQ(generate_input->request_info.frontend_ip, "10.0.0.1");
    ASSERT_EQ(generate_input->request_info.dash_ip, "10.0.0.2");
    ASSERT_EQ(generate_input->request_info.trace_id, "trace-123");
    ASSERT_EQ(generate_input->request_info.request_id, "source-request-123");
    ASSERT_EQ(generate_input->request_info.source_role, "frontend");
    ASSERT_EQ(generate_input->priority, 70);
    auto generate_config = generate_input->generate_config;
    ASSERT_EQ(generate_config->min_new_tokens, 4);
    ASSERT_EQ(generate_config->max_new_tokens, 5);
    ASSERT_EQ(generate_config->num_beams, 1);
    ASSERT_EQ(generate_config->num_return_sequences, 1);
    ASSERT_EQ(generate_config->top_k, 6);
    ASSERT_FLOAT_EQ(generate_config->top_p, 0.6);
    ASSERT_FLOAT_EQ(generate_config->temperature, 0.1);
    ASSERT_FLOAT_EQ(generate_config->repetition_penalty, 0.2);
    ASSERT_FLOAT_EQ(generate_config->top_p_decay.value(), 0.7);
    ASSERT_FLOAT_EQ(generate_config->top_p_min.value(), 0.3);
    ASSERT_EQ(generate_config->top_p_reset_ids.value(), 7);
    ASSERT_EQ(generate_config->task_id.value(), "8");
    ASSERT_EQ(generate_config->calculate_loss, 1);
    ASSERT_TRUE(generate_config->return_hidden_states);
    ASSERT_TRUE(generate_config->frontend_metric_streaming);
    ASSERT_FALSE(generate_config->return_logits);
    ASSERT_EQ(generate_config->stop_words_list.size(), 2);
    vector<int> stop_words_1{0, 1, 2};
    vector<int> stop_words_2{3, 4, 5};
    ASSERT_EQ(generate_config->stop_words_list[0], stop_words_1);
    ASSERT_EQ(generate_config->stop_words_list[1], stop_words_2);
}

TEST_F(QueryConverterTest, RoleAddrReadsLegacyTypedAndDualWritePayloads) {
    GenerateConfigPB config;

    auto* legacy = config.add_role_addrs();
    legacy->set_role(RoleAddrPB::PREFILL);
    legacy->set_ip("legacy-prefill");

    auto* string_only = config.add_role_addrs();
    string_only->set_role_str("DECODE");
    string_only->set_ip("string-decode");

    auto* dual = config.add_role_addrs();
    dual->set_role(RoleAddrPB::VIT);
    dual->set_role_str("VIT");
    dual->set_ip("dual-vit");

    const auto role_addrs = QueryConverter::getRoleAddrs(&config);
    ASSERT_EQ(role_addrs.size(), 3);
    EXPECT_EQ(role_addrs[0].role, RoleType::PREFILL);
    EXPECT_EQ(role_addrs[1].role, RoleType::DECODE);
    EXPECT_EQ(role_addrs[2].role, RoleType::VIT);
}

TEST_F(QueryConverterTest, RoleAddrPreservesPdfusionDefaultAndRejectsConflicts) {
    GenerateConfigPB legacy_pdfusion;
    legacy_pdfusion.add_role_addrs()->set_role(RoleAddrPB::PDFUSION);
    EXPECT_EQ(QueryConverter::getRoleAddrs(&legacy_pdfusion)[0].role, RoleType::PDFUSION);

    GenerateConfigPB conflict;
    auto*            conflicting = conflict.add_role_addrs();
    conflicting->set_role(RoleAddrPB::PREFILL);
    conflicting->set_role_str("DECODE");
    const bool core_dump_on_exception            = StaticConfig::user_ft_core_dump_on_exception;
    StaticConfig::user_ft_core_dump_on_exception = false;
    EXPECT_THROW(QueryConverter::getRoleAddrs(&conflict), std::runtime_error);
    StaticConfig::user_ft_core_dump_on_exception = core_dump_on_exception;

    GenerateConfigPB omitted_legacy_default;
    omitted_legacy_default.add_role_addrs();
    EXPECT_EQ(QueryConverter::getRoleAddrs(&omitted_legacy_default)[0].role, RoleType::PDFUSION);
}

TEST_F(QueryConverterTest, testTransOutput) {
    auto output_token_ids = torch::empty({1, 3}, torch::kInt32);
    auto data             = output_token_ids.data_ptr<int>();
    for (int i = 0; i < 3; ++i) {
        data[i] = i;
    }
    GenerateOutputs outputs;
    GenerateOutput  res;
    res.output_ids                                      = output_token_ids;
    res.finished                                        = true;
    res.aux_info.cost_time_us                           = 1000;
    res.aux_info.iter_count                             = 9;
    res.aux_info.input_len                              = 8;
    res.aux_info.output_len                             = 7;
    res.aux_info.speculative_verify_rounds              = 3;
    res.aux_info.speculative_accepted_token_num         = 9;
    res.aux_info.speculative_proposed_draft_tokens      = 12;
    res.aux_info.context_execute_time_us                = 100;
    res.aux_info.generate_execute_time_us               = 200;
    res.aux_info.context_execute_time_with_cache_us     = 80;
    outputs.frontend_metric_only                        = true;
    outputs.frontend_context_token_num                  = 11;
    outputs.frontend_context_token_num_with_cache       = 13;
    outputs.frontend_context_execute_time_us            = 101;
    outputs.frontend_context_execute_time_with_cache_us = 81;
    outputs.frontend_generate_token_num                 = 17;
    outputs.frontend_generate_execute_time_us           = 201;
    outputs.frontend_sp_tpot_samples                    = std::make_shared<FrontendSpTpotSamples>();
    auto tpot_sample                                    = outputs.frontend_sp_tpot_samples->begin();
    tpot_sample->complete(12345.5);
    auto hidden_states_tensor = torch::empty({3, 2}, torch::kFloat32);
    auto hidden_states_data   = hidden_states_tensor.data_ptr<float>();
    for (int i = 0; i < 6; ++i) {
        hidden_states_data[i] = i;
    }
    res.hidden_states.emplace(hidden_states_tensor);
    outputs.generate_outputs.push_back(res);

    GenerateOutputsPB outputs_pb;
    QueryConverter::transResponse(&outputs_pb, &outputs, true, "", 10000);

    auto& output_pb   = outputs_pb.flatten_output();
    auto  aux_info_pb = output_pb.aux_info(0);
    EXPECT_EQ(aux_info_pb.cost_time_us(), 1000);
    EXPECT_EQ(aux_info_pb.iter_count(), 9);
    EXPECT_EQ(aux_info_pb.input_len(), 8);
    EXPECT_EQ(aux_info_pb.output_len(), 7);
    EXPECT_EQ(aux_info_pb.speculative_verify_rounds(), 3);
    EXPECT_EQ(aux_info_pb.speculative_accepted_token_num(), 9);
    EXPECT_EQ(aux_info_pb.speculative_proposed_draft_tokens(), 12);
    EXPECT_EQ(aux_info_pb.context_execute_time_us(), 100);
    EXPECT_EQ(aux_info_pb.generate_execute_time_us(), 200);
    EXPECT_EQ(aux_info_pb.context_execute_time_with_cache_us(), 80);
    EXPECT_TRUE(outputs_pb.frontend_metric_only());
    ASSERT_TRUE(outputs_pb.has_frontend_context_token_num());
    EXPECT_EQ(outputs_pb.frontend_context_token_num().value(), 11);
    ASSERT_TRUE(outputs_pb.has_frontend_context_token_num_with_cache());
    EXPECT_EQ(outputs_pb.frontend_context_token_num_with_cache().value(), 13);
    ASSERT_TRUE(outputs_pb.has_frontend_context_execute_time_us());
    EXPECT_EQ(outputs_pb.frontend_context_execute_time_us().value(), 101);
    ASSERT_TRUE(outputs_pb.has_frontend_context_execute_time_with_cache_us());
    EXPECT_EQ(outputs_pb.frontend_context_execute_time_with_cache_us().value(), 81);
    ASSERT_TRUE(outputs_pb.has_frontend_generate_token_num());
    EXPECT_EQ(outputs_pb.frontend_generate_token_num().value(), 17);
    ASSERT_TRUE(outputs_pb.has_frontend_generate_execute_time_us());
    EXPECT_EQ(outputs_pb.frontend_generate_execute_time_us().value(), 201);
    ASSERT_EQ(outputs_pb.frontend_sp_tpot_samples_size(), 1);
    EXPECT_EQ(outputs_pb.frontend_sp_tpot_samples(0).sequence_id(), 1);
    EXPECT_DOUBLE_EQ(outputs_pb.frontend_sp_tpot_samples(0).tpot_us(), 12345.5);
    EXPECT_TRUE(outputs.frontend_sp_tpot_samples->take().empty());
    auto output_ids_pb = output_pb.output_ids();
    ASSERT_EQ(output_ids_pb.data_type(), TensorPB_DataType::TensorPB_DataType_INT32);
    ASSERT_EQ(output_ids_pb.shape_size(), 3);
    ASSERT_EQ(output_ids_pb.shape(0), 1);
    ASSERT_EQ(output_ids_pb.shape(1), 1);
    ASSERT_EQ(output_ids_pb.shape(2), 3);
    auto            output_ids_string = output_ids_pb.int32_data();
    vector<int32_t> output_ids_vector;
    output_ids_vector.resize(output_ids_string.size() / sizeof(int32_t));
    std::memcpy(output_ids_vector.data(), output_ids_string.data(), output_ids_string.size());
    for (int i = 0; i < 3; ++i) {
        ASSERT_EQ(output_ids_vector[i], i);
    }
    ASSERT_TRUE(output_pb.has_hidden_states());
    auto hidden_states_pb = output_pb.hidden_states();
    ASSERT_EQ(hidden_states_pb.data_type(), TensorPB_DataType::TensorPB_DataType_FP32);
    ASSERT_EQ(hidden_states_pb.shape_size(), 3);
    ASSERT_EQ(hidden_states_pb.shape(0), 1);
    ASSERT_EQ(hidden_states_pb.shape(1), 3);
    ASSERT_EQ(hidden_states_pb.shape(2), 2);
    auto          hidden_states_string = hidden_states_pb.fp32_data();
    vector<float> hidden_states_vector;
    hidden_states_vector.resize(hidden_states_string.size() / sizeof(float));
    std::memcpy(hidden_states_vector.data(), hidden_states_string.data(), hidden_states_string.size());
    for (int i = 0; i < 6; ++i) {
        ASSERT_FLOAT_EQ(hidden_states_vector[i], i);
    }
}

TEST_F(QueryConverterTest, TransTensorPB_FP32) {

    torch::Tensor tensor = torch::rand({2, 3}, torch::kFloat32);
    TensorPB      tensor_pb;
    QueryConverter::transTensorPB(&tensor_pb, tensor);
    EXPECT_EQ(tensor_pb.data_type(), TensorPB::FP32);
    ASSERT_EQ(tensor_pb.shape_size(), 2);
    EXPECT_EQ(tensor_pb.shape(0), 2);
    EXPECT_EQ(tensor_pb.shape(1), 3);

    // 验证数据一致性
    const std::string& proto_data        = tensor_pb.fp32_data();
    const float*       proto_ptr         = reinterpret_cast<const float*>(proto_data.data());
    torch::Tensor      contiguous_tensor = tensor.contiguous();
    const float*       tensor_ptr        = contiguous_tensor.data_ptr<float>();

    ASSERT_EQ(proto_data.size(), contiguous_tensor.numel() * sizeof(float));
    for (int i = 0; i < contiguous_tensor.numel(); ++i) {
        EXPECT_FLOAT_EQ(proto_ptr[i], tensor_ptr[i]);
    }
}

TEST_F(QueryConverterTest, TransTensorPB_BF16) {
    torch::Tensor tensor = torch::rand({3}, torch::kBFloat16);
    TensorPB      tensor_pb;
    QueryConverter::transTensorPB(&tensor_pb, tensor);

    EXPECT_EQ(tensor_pb.data_type(), TensorPB::BF16);

    const std::string& proto_data    = tensor_pb.bf16_data();
    size_t             expected_size = tensor.numel() * sizeof(c10::BFloat16);
    ASSERT_EQ(proto_data.size(), expected_size);

    const char* tensor_data = static_cast<const char*>(tensor.contiguous().data_ptr());
    EXPECT_EQ(std::memcmp(proto_data.data(), tensor_data, expected_size), 0);
}

TEST_F(QueryConverterTest, TransTensorPB_ScalarShape) {
    torch::Tensor tensor = torch::tensor(42, torch::kInt32);
    TensorPB      tensor_pb;
    QueryConverter::transTensorPB(&tensor_pb, tensor);
    EXPECT_EQ(tensor_pb.shape_size(), 0);
}

TEST_F(QueryConverterTest, TransTensorPB_NonContiguous) {
    torch::Tensor tensor = torch::rand({3, 4}, torch::kFloat32).transpose(0, 1);
    TensorPB      tensor_pb;
    QueryConverter::transTensorPB(&tensor_pb, tensor);

    torch::Tensor      contiguous_tensor = tensor.contiguous();
    const std::string& proto_data        = tensor_pb.fp32_data();
    const float*       proto_ptr         = reinterpret_cast<const float*>(proto_data.data());
    const float*       tensor_ptr        = contiguous_tensor.data_ptr<float>();

    for (int i = 0; i < contiguous_tensor.numel(); ++i) {
        EXPECT_FLOAT_EQ(proto_ptr[i], tensor_ptr[i]);
    }
}

TEST_F(QueryConverterTest, TransTensorPB_UnsupportedType) {
    torch::Tensor tensor = torch::ones({1}, torch::kInt64);
    TensorPB      tensor_pb;
    EXPECT_THROW(QueryConverter::transTensorPB(&tensor_pb, tensor), std::runtime_error);
}

TEST_F(QueryConverterTest, TimeoutErrorCodeMapsToGrpcDeadline) {
    EXPECT_EQ(transErrorCodeToGrpc(ErrorCode::GENERATE_TIMEOUT), grpc::StatusCode::DEADLINE_EXCEEDED);
    EXPECT_EQ(transErrorCodeToGrpc(ErrorCode::DEADLINE_EXCEEDED), grpc::StatusCode::DEADLINE_EXCEEDED);
    EXPECT_EQ(transErrorCodeToGrpc(ErrorCode::WAIT_TO_RUN_TIMEOUT), grpc::StatusCode::DEADLINE_EXCEEDED);
    EXPECT_EQ(transErrorCodeToGrpc(ErrorCode::KEEP_ALIVE_TIMEOUT), grpc::StatusCode::DEADLINE_EXCEEDED);
}

}  // namespace rtp_llm
