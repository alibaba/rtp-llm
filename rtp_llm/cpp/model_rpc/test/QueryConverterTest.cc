#include "rtp_llm/cpp/testing/TestBase.h"
#include <memory>
#include <optional>

#define private public
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

using namespace std;
namespace rtp_llm {

class QueryConverterTest: public DeviceTestBase {};

TEST_F(QueryConverterTest, testTransInput) {
    GenerateInputPB input;
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
    generate_config_pb->mutable_json_schema()->set_value("{\"type\":\"object\"}");
    generate_config_pb->mutable_regex()->set_value("[a-z]+");
    generate_config_pb->mutable_ebnf()->set_value("root ::= \"a\"");
    generate_config_pb->mutable_structural_tag()->set_value("{\"format\":{\"type\":\"json_schema\"}}");
    generate_config_pb->mutable_task_id()->set_value("8");
    generate_config_pb->set_calculate_loss(1);
    generate_config_pb->set_return_hidden_states(true);
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
    ASSERT_EQ(generate_config->json_schema.value(), "{\"type\":\"object\"}");
    ASSERT_EQ(generate_config->regex.value(), "[a-z]+");
    ASSERT_EQ(generate_config->ebnf.value(), "root ::= \"a\"");
    ASSERT_EQ(generate_config->structural_tag.value(), "{\"format\":{\"type\":\"json_schema\"}}");
    ASSERT_EQ(generate_config->task_id.value(), "8");
    ASSERT_EQ(generate_config->calculate_loss, 1);
    ASSERT_TRUE(generate_config->return_hidden_states);
    ASSERT_FALSE(generate_config->return_logits);
    ASSERT_EQ(generate_config->stop_words_list.size(), 2);
    vector<int> stop_words_1{0, 1, 2};
    vector<int> stop_words_2{3, 4, 5};
    ASSERT_EQ(generate_config->stop_words_list[0], stop_words_1);
    ASSERT_EQ(generate_config->stop_words_list[1], stop_words_2);
}

TEST_F(QueryConverterTest, testTransOutput) {
    auto output_token_ids = torch::empty({1, 3}, torch::kInt32);
    auto data             = output_token_ids.data_ptr<int>();
    for (int i = 0; i < 3; ++i) {
        data[i] = i;
    }
    GenerateOutputs outputs;
    GenerateOutput  res;
    res.output_ids            = output_token_ids;
    res.finished              = true;
    res.aux_info.cost_time_us = 1000;
    res.aux_info.iter_count   = 9;
    res.aux_info.input_len    = 8;
    res.aux_info.output_len   = 7;
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

// -- response_format normalization -------------------------------------------
// These tests cover the server-side fallback in QueryConverter::transGenerateConfig
// that mirrors rtp_llm/cpp/model_rpc/model_rpc_client.py::_normalize_grammar_fields.
// A non-Python client (or a Python path that skips the normalizer) should get
// the same grammar-field population as the OpenAI endpoint does client-side.

namespace {
GenerateInputPB makeInputWithResponseFormat(const std::string& response_format_json) {
    GenerateInputPB input;
    input.add_token_ids(0);
    auto* config = input.mutable_generate_config();
    config->mutable_response_format()->set_value(response_format_json);
    return input;
}
}  // namespace

TEST_F(QueryConverterTest, ResponseFormat_JsonSchemaNested) {
    // OpenAI-style: {"type":"json_schema","json_schema":{"name":"x","schema":{...}}}
    auto input = makeInputWithResponseFormat(
        R"({"type":"json_schema","json_schema":{"name":"person","schema":{"type":"object","properties":{"age":{"type":"integer"}}}}})");
    auto cfg = QueryConverter::transQuery(&input)->generate_config;
    ASSERT_TRUE(cfg->json_schema.has_value());
    // Inner schema is surfaced — NOT the wrapping {name, schema} envelope.
    EXPECT_NE(cfg->json_schema->find("\"properties\""), std::string::npos);
    EXPECT_EQ(cfg->json_schema->find("\"name\""), std::string::npos);
    EXPECT_FALSE(cfg->regex.has_value());
    EXPECT_FALSE(cfg->ebnf.has_value());
    EXPECT_FALSE(cfg->structural_tag.has_value());
}

TEST_F(QueryConverterTest, ResponseFormat_JsonObjectShorthand) {
    // OpenAI "any JSON object" — should populate json_schema with the
    // literal "object" schema so xgrammar can compile it.
    auto input = makeInputWithResponseFormat(R"({"type":"json_object"})");
    auto cfg   = QueryConverter::transQuery(&input)->generate_config;
    ASSERT_TRUE(cfg->json_schema.has_value());
    EXPECT_EQ(cfg->json_schema.value(), R"({"type":"object"})");
}

TEST_F(QueryConverterTest, ResponseFormat_Regex) {
    auto input = makeInputWithResponseFormat(
        R"({"type":"regex","pattern":"#[0-9A-Fa-f]{6}"})");
    auto cfg = QueryConverter::transQuery(&input)->generate_config;
    ASSERT_TRUE(cfg->regex.has_value());
    EXPECT_EQ(cfg->regex.value(), "#[0-9A-Fa-f]{6}");
    EXPECT_FALSE(cfg->json_schema.has_value());
    EXPECT_FALSE(cfg->ebnf.has_value());
    EXPECT_FALSE(cfg->structural_tag.has_value());
}

TEST_F(QueryConverterTest, ResponseFormat_Ebnf) {
    auto input = makeInputWithResponseFormat(
        R"({"type":"ebnf","grammar":"root ::= \"hi\""})");
    auto cfg = QueryConverter::transQuery(&input)->generate_config;
    ASSERT_TRUE(cfg->ebnf.has_value());
    EXPECT_EQ(cfg->ebnf.value(), "root ::= \"hi\"");
    EXPECT_FALSE(cfg->json_schema.has_value());
    EXPECT_FALSE(cfg->regex.has_value());
    EXPECT_FALSE(cfg->structural_tag.has_value());
}

TEST_F(QueryConverterTest, ResponseFormat_StructuralTag) {
    // structural_tag payload is typically a JSON object; we serialize it back
    // to a string and stash it in generate_config.structural_tag.
    auto input = makeInputWithResponseFormat(
        R"({"type":"structural_tag","structural_tag":{"format":{"type":"sequence","elements":[]}}})");
    auto cfg = QueryConverter::transQuery(&input)->generate_config;
    ASSERT_TRUE(cfg->structural_tag.has_value());
    EXPECT_NE(cfg->structural_tag->find("\"sequence\""), std::string::npos);
    EXPECT_FALSE(cfg->json_schema.has_value());
}

TEST_F(QueryConverterTest, ResponseFormat_ExplicitFieldWinsOverResponseFormat) {
    // If the PB carries BOTH a concrete grammar field and a response_format,
    // the explicit field is authoritative — response_format is not re-parsed.
    // This matches the Python normalizer's early-return guard.
    GenerateInputPB input;
    input.add_token_ids(0);
    auto* config = input.mutable_generate_config();
    config->mutable_regex()->set_value("^[0-9]+$");
    config->mutable_response_format()->set_value(R"({"type":"json_object"})");

    auto cfg = QueryConverter::transQuery(&input)->generate_config;
    ASSERT_TRUE(cfg->regex.has_value());
    EXPECT_EQ(cfg->regex.value(), "^[0-9]+$");
    // json_object fallback must NOT fire because regex was already set.
    EXPECT_FALSE(cfg->json_schema.has_value());
}

TEST_F(QueryConverterTest, ResponseFormat_TextTypeIsNoOp) {
    // OpenAI explicit "no constraint" — must succeed and leave grammar fields empty.
    auto input = makeInputWithResponseFormat(R"({"type":"text"})");
    auto cfg   = QueryConverter::transQuery(&input)->generate_config;
    EXPECT_FALSE(cfg->json_schema.has_value());
    EXPECT_FALSE(cfg->regex.has_value());
    EXPECT_FALSE(cfg->ebnf.has_value());
    EXPECT_FALSE(cfg->structural_tag.has_value());
}

TEST_F(QueryConverterTest, ResponseFormat_EmptyStringIsNoOp) {
    // Empty string is the "absent response_format" sentinel — no envelope was
    // sent. Distinct from {} which is an explicit empty envelope and must be
    // rejected (covered by ResponseFormat_InvalidEnvelopesAreRejected).
    GenerateInputPB input;
    input.add_token_ids(0);
    input.mutable_generate_config()->mutable_response_format()->set_value("");
    auto cfg = QueryConverter::transQuery(&input)->generate_config;
    EXPECT_FALSE(cfg->json_schema.has_value());
    EXPECT_FALSE(cfg->regex.has_value());
    EXPECT_FALSE(cfg->ebnf.has_value());
    EXPECT_FALSE(cfg->structural_tag.has_value());
}

TEST_F(QueryConverterTest, ResponseFormat_InvalidEnvelopesAreRejected) {
    // Every case below routes to either the JSON-parse throw or the empty-
    // result backstop in normalizeResponseFormat, ending up as INVALID_PARAMS
    // at the gRPC layer instead of a silent unconstrained generation.
    const std::vector<std::pair<const char*, std::string>> bad = {
        {"malformed_json",          "this is not json {{{"},
        {"empty_object",            "{}"},
        {"json_schema_empty_body",  R"({"type":"json_schema","json_schema":{}})"},
        {"unknown_type",            R"({"type":"not_a_real_type"})"},
        {"missing_type",            R"({"json_schema":{"schema":{"type":"object"}}})"},
        {"json_schema_no_payload",  R"({"type":"json_schema"})"},
        {"regex_no_pattern",        R"({"type":"regex"})"},
        {"ebnf_no_grammar",         R"({"type":"ebnf"})"},
        {"structural_tag_no_body",  R"({"type":"structural_tag"})"},
    };
    for (const auto& [label, env] : bad) {
        auto input = makeInputWithResponseFormat(env);
        EXPECT_THROW(QueryConverter::transQuery(&input), std::invalid_argument)
            << "case: " << label;
    }
}

}  // namespace rtp_llm
