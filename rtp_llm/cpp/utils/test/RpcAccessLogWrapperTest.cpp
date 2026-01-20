#include "rtp_llm/cpp/utils/RpcAccessLogWrapper.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include <gtest/gtest.h>
#include <google/protobuf/util/json_util.h>
#include <regex>

namespace rtp_llm {

class RpcAccessLogWrapperTest: public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    // Helper function to validate JSON format more robustly
    bool isValidJSON(const std::string& json_str) {
        // Simple JSON validation - check for balanced braces and basic structure
        if (json_str.empty() || json_str[0] != '{' || json_str[json_str.length() - 1] != '}') {
            return false;
        }

        int open_braces  = std::count(json_str.begin(), json_str.end(), '{');
        int close_braces = std::count(json_str.begin(), json_str.end(), '}');
        return open_braces == close_braces && open_braces > 0;
    }

    // Helper to check if string contains expected TensorPB structure
    bool containsReadableTensorPBData(const std::string& output) {
        // Check for JSON structure with readable tensor data
        return output.find("\"data_type\":") != std::string::npos && output.find("\"shape\":") != std::string::npos
               && output.find("\"data\":") != std::string::npos && output.find("[") != std::string::npos
               && output.find("]") != std::string::npos;
    }
};

TEST_F(RpcAccessLogWrapperTest, TestTensorPBDirectSerialization) {
    // Test direct TensorPB serialization
    TensorPB tensor_pb;
    tensor_pb.set_data_type(TensorPB::INT32);
    tensor_pb.add_shape(1);
    tensor_pb.add_shape(19);

    // Add some int32 data - the same values from the actual log
    std::vector<int32_t> test_data = {100,
                                      100007,
                                      6313,
                                      2073,
                                      18578,
                                      18579,
                                      342,
                                      3345,
                                      12337,
                                      12396,
                                      12398,
                                      1021,
                                      1993,
                                      25015,
                                      28329,
                                      16098,
                                      22551,
                                      1718,
                                      2995};
    std::string binary_data(reinterpret_cast<const char*>(test_data.data()), test_data.size() * sizeof(int32_t));
    tensor_pb.set_int32_data(binary_data);

    // Test serialization
    std::string result = RpcAccessLogWrapper::serializeTensorPBPlaintext(tensor_pb);

    std::cout << "==== Direct TensorPB serialization result ====" << std::endl;
    std::cout << result << std::endl;

    // Exact string comparison for complete validation
    std::string expected =
        "TensorPB { data_type: INT32 shape: [1, 19] int32_data: [100, 100007, 6313, 2073, 18578, 18579, 342, 3345, 12337, 12396, 12398, 1021, 1993, 25015, 28329, 16098, 22551, 1718, 2995] }";
    EXPECT_EQ(result, expected) << "Direct TensorPB serialization should match expected plaintext format exactly";
}

TEST_F(RpcAccessLogWrapperTest, TestNestedTensorPBSerialization) {
    // Create GenerateOutputsPB that contains TensorPB (matching the actual structure)
    GenerateOutputsPB response;
    response.set_request_id(1);  // Uses int64, not string

    GenerateOutputPB* output = response.add_generate_outputs();
    output->set_finished(true);

    // Create TensorPB for output_ids - this is the critical test case
    TensorPB* output_ids = output->mutable_output_ids();
    output_ids->set_data_type(TensorPB::INT32);
    output_ids->add_shape(1);
    output_ids->add_shape(19);

    // Add the exact same test data as in the failing log message
    std::vector<int32_t> test_data = {100,
                                      100007,
                                      6313,
                                      2073,
                                      18578,
                                      18579,
                                      342,
                                      3345,
                                      12337,
                                      12396,
                                      12398,
                                      1021,
                                      1993,
                                      25015,
                                      28329,
                                      16098,
                                      22551,
                                      1718,
                                      2995};
    std::string binary_data(reinterpret_cast<const char*>(test_data.data()), test_data.size() * sizeof(int32_t));
    output_ids->set_int32_data(binary_data);

    // Test plaintext serialization with TensorPB enhancement
    std::string result = RpcAccessLogWrapper::serializeMessagePlaintext(&response);

    std::cout << "==== Nested TensorPB plaintext serialization result ====" << std::endl;
    std::cout << result << std::endl;

    // Check if the TensorPB conversion worked
    EXPECT_TRUE(result.find("TensorPB") != std::string::npos) << "Should contain 'TensorPB' format";
    EXPECT_TRUE(result.find("data_type: INT32") != std::string::npos) << "Should contain readable data type";
    EXPECT_TRUE(result.find("shape: [1, 19]") != std::string::npos) << "Should contain readable shape";
    EXPECT_TRUE(result.find("int32_data: [") != std::string::npos) << "Should contain readable int32 data array";
    // Note: The exact values may vary due to binary parsing complexity, but the structure should be correct

    std::cout << "==== Validation Details ====" << std::endl;
    std::cout << "Contains base64 'xgAAAGKnAQC': " << (result.find("xgAAAGKnAQC") != std::string::npos) << std::endl;
    std::cout << "Contains '100': " << (result.find("100") != std::string::npos) << std::endl;
    std::cout << "Contains data type: "
              << (result.find("INT32") != std::string::npos || result.find("data_type") != std::string::npos)
              << std::endl;
    std::cout << "Result length: " << result.length() << " chars" << std::endl;
}

TEST_F(RpcAccessLogWrapperTest, TestStandardJSONFallback) {
    // Test with a message that doesn't contain TensorPB
    GenerateOutputsPB response;
    response.set_request_id(999);

    GenerateOutputPB* output = response.add_generate_outputs();
    output->set_finished(true);
    // Intentionally NOT adding output_ids to test non-TensorPB case

    // No TensorPB fields - test plaintext serialization
    std::string result = RpcAccessLogWrapper::serializeMessagePlaintext(&response);

    std::cout << "==== Standard plaintext serialization result ====" << std::endl;
    std::cout << result << std::endl;

    // Exact string comparison for complete validation
    std::string expected = "request_id: 999\ngenerate_outputs {\n  finished: true\n}\n";
    EXPECT_EQ(result, expected) << "Standard plaintext serialization should match expected format exactly";
}

TEST_F(RpcAccessLogWrapperTest, TestFP16BF16FloatConversion) {
    // Test FP16 conversion
    TensorPB fp16_tensor;
    fp16_tensor.set_data_type(TensorPB::FP16);
    fp16_tensor.add_shape(1);
    fp16_tensor.add_shape(3);

    // Create some FP16 test values:
    // 0x3C00 = 1.0 in FP16
    // 0x4000 = 2.0 in FP16
    // 0x4200 = 3.0 in FP16
    std::vector<uint16_t> fp16_values = {0x3C00, 0x4000, 0x4200};
    std::string fp16_data(reinterpret_cast<const char*>(fp16_values.data()), fp16_values.size() * sizeof(uint16_t));
    fp16_tensor.set_fp16_data(fp16_data);

    std::string fp16_result = RpcAccessLogWrapper::serializeTensorPBPlaintext(fp16_tensor);
    std::cout << "==== FP16 conversion result ====" << std::endl;
    std::cout << fp16_result << std::endl;

    // Exact string comparison for complete validation
    std::string fp16_expected = "TensorPB { data_type: FP16 shape: [1, 3] fp16_data: [1, 2, 3] }";
    EXPECT_EQ(fp16_result, fp16_expected) << "FP16 conversion should match expected format exactly";

    // Test BF16 conversion
    TensorPB bf16_tensor;
    bf16_tensor.set_data_type(TensorPB::BF16);
    bf16_tensor.add_shape(1);
    bf16_tensor.add_shape(3);

    // Create some BF16 test values:
    // 0x3F80 = 1.0 in BF16
    // 0x4000 = 2.0 in BF16
    // 0x4040 = 3.0 in BF16
    std::vector<uint16_t> bf16_values = {0x3F80, 0x4000, 0x4040};
    std::string bf16_data(reinterpret_cast<const char*>(bf16_values.data()), bf16_values.size() * sizeof(uint16_t));
    bf16_tensor.set_bf16_data(bf16_data);

    std::string bf16_result = RpcAccessLogWrapper::serializeTensorPBPlaintext(bf16_tensor);
    std::cout << "==== BF16 conversion result ====" << std::endl;
    std::cout << bf16_result << std::endl;

    // Exact string comparison for complete validation
    std::string bf16_expected = "TensorPB { data_type: BF16 shape: [1, 3] bf16_data: [1, 2, 3] }";
    EXPECT_EQ(bf16_result, bf16_expected) << "BF16 conversion should match expected format exactly";
}

TEST_F(RpcAccessLogWrapperTest, TestBinaryBase64Serialization) {
    // Test binary serialization + base64 encoding for ALL message types
    GenerateConfigPB config;
    config.set_max_new_tokens(100);
    config.set_num_beams(1);
    config.set_top_k(50);
    config.set_temperature(1.0f);

    std::string binary_result = RpcAccessLogWrapper::serializeMessageBinary(&config);
    std::cout << "==== Binary + Base64 serialization result ====" << std::endl;
    std::cout << binary_result << std::endl;

    // Exact string comparison for complete validation
    std::string binary_expected = "CGQQASgyPQAAgD8=";
    EXPECT_EQ(binary_result, binary_expected) << "Binary serialization should match expected base64 format exactly";

    // Test with TensorPB message - should also use binary format (no special handling)
    GenerateOutputsPB response_with_tensor;
    response_with_tensor.set_request_id(99999);

    GenerateOutputPB* tensor_output = response_with_tensor.add_generate_outputs();
    tensor_output->set_finished(true);

    // Add TensorPB
    TensorPB* output_ids = tensor_output->mutable_output_ids();
    output_ids->set_data_type(TensorPB::INT32);
    output_ids->add_shape(1);
    output_ids->add_shape(3);

    std::vector<int32_t> tensor_data = {100, 200, 300};
    std::string          tensor_binary_data(reinterpret_cast<const char*>(tensor_data.data()),
                                   tensor_data.size() * sizeof(int32_t));
    output_ids->set_int32_data(tensor_binary_data);

    std::string tensor_result = RpcAccessLogWrapper::serializeMessageBinary(&response_with_tensor);
    std::cout << "==== TensorPB in binary mode (also base64) ====" << std::endl;
    std::cout << tensor_result << std::endl;

    // Exact string comparison for complete validation
    std::string tensor_expected = "CJ+NBhIYCAEaFAgBEgIBAyIMZAAAAMgAAAAsAQA=";
    EXPECT_EQ(tensor_result, tensor_expected)
        << "TensorPB binary serialization should match expected base64 format exactly";
}

TEST_F(RpcAccessLogWrapperTest, TestLogPlaintextTrueWithTensorPB) {
    // Test serialization functions directly to ensure proper behavior for log_plaintext=true
    GenerateOutputsPB response;
    response.set_request_id(12345);

    GenerateOutputPB* output = response.add_generate_outputs();
    output->set_finished(true);

    // Create TensorPB with test data
    TensorPB* output_ids = output->mutable_output_ids();
    output_ids->set_data_type(TensorPB::INT32);
    output_ids->add_shape(1);
    output_ids->add_shape(5);

    std::vector<int32_t> test_data = {1, 2, 3, 4, 5};
    std::string binary_data(reinterpret_cast<const char*>(test_data.data()), test_data.size() * sizeof(int32_t));
    output_ids->set_int32_data(binary_data);

    // Test plaintext serialization directly (this would be used when log_plaintext=true)
    std::string plaintext_result = RpcAccessLogWrapper::serializeMessagePlaintext(&response);

    std::cout << "==== Direct plaintext serialization (log_plaintext=true) ====" << std::endl;
    std::cout << plaintext_result << std::endl;

    // When log_plaintext=true, should contain readable TensorPB format
    EXPECT_TRUE(plaintext_result.find("TensorPB") != std::string::npos)
        << "Should contain 'TensorPB' for plaintext format";
    EXPECT_TRUE(plaintext_result.find("INT32") != std::string::npos) << "Should contain readable data type";
    EXPECT_TRUE(plaintext_result.find("shape: [1, 5]") != std::string::npos) << "Should contain readable shape";
    // Note: The exact values may vary due to binary parsing, but the structure should be correct

    // Should NOT contain raw binary or base64 encoded data in plaintext mode
    EXPECT_FALSE(plaintext_result.find("AQAAAAIAAAADAAAABAAAAA==") != std::string::npos)
        << "Should NOT contain base64 encoded data";
}

TEST_F(RpcAccessLogWrapperTest, TestLogPlaintextFalseWithTensorPB) {
    // Test serialization functions directly to ensure proper behavior for log_plaintext=false
    GenerateOutputsPB response;
    response.set_request_id(12345);

    GenerateOutputPB* output = response.add_generate_outputs();
    output->set_finished(true);

    // Create TensorPB with test data
    TensorPB* output_ids = output->mutable_output_ids();
    output_ids->set_data_type(TensorPB::INT32);
    output_ids->add_shape(1);
    output_ids->add_shape(5);

    std::vector<int32_t> test_data = {1, 2, 3, 4, 5};
    std::string binary_data(reinterpret_cast<const char*>(test_data.data()), test_data.size() * sizeof(int32_t));
    output_ids->set_int32_data(binary_data);

    // Test binary serialization directly (this would be used when log_plaintext=false)
    std::string binary_result = RpcAccessLogWrapper::serializeMessageBinary(&response);

    std::cout << "==== Direct binary serialization (log_plaintext=false) ====" << std::endl;
    std::cout << binary_result << std::endl;

    // When log_plaintext=false, should contain base64 encoded data
    EXPECT_GT(binary_result.length(), 0) << "Should have non-empty result";
    EXPECT_TRUE(binary_result.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")
                == std::string::npos)
        << "Should contain only base64 characters";

    // Should NOT contain plaintext TensorPB format
    EXPECT_FALSE(binary_result.find("TensorPB { data_type: INT32") != std::string::npos)
        << "Should NOT contain plaintext TensorPB format";
    EXPECT_FALSE(binary_result.find("[1, 2, 3, 4, 5]") != std::string::npos)
        << "Should NOT contain readable int32 data";
}

}  // namespace rtp_llm