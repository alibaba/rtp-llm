#include "rtp_llm/cpp/utils/RpcAccessLogWrapper.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include <google/protobuf/text_format.h>
#include <google/protobuf/util/json_util.h>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <regex>

namespace rtp_llm {

namespace {
// Base64 encoding function
std::string base64_encode(const std::string& input) {
    static const char* chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string        result;
    result.reserve(((input.length() + 2) / 3) * 4);

    int           i = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];

    const char* bytes_to_encode = input.c_str();
    int         in_len          = input.length();

    while (i < in_len) {
        char_array_3[0] = bytes_to_encode[i++];
        char_array_3[1] = (i < in_len) ? bytes_to_encode[i++] : 0;
        char_array_3[2] = (i < in_len) ? bytes_to_encode[i++] : 0;

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;

        for (int j = 0; j < 4 && (i - 3 + j) < in_len; j++) {
            result += chars[char_array_4[j]];
        }
    }

    // Add padding
    while (result.length() % 4) {
        result += '=';
    }

    return result;
}

// Helper function to convert FP16 to float using safe type conversion
float fp16_to_float(uint16_t fp16_value) {
    uint32_t sign = (fp16_value & 0x8000) << 16;
    uint32_t exp  = (fp16_value & 0x7C00) >> 10;
    uint32_t frac = (fp16_value & 0x03FF);

    if (exp == 0) {
        if (frac == 0) {
            // Zero
            uint32_t zero_bits = sign;
            float    zero_float;
            std::memcpy(&zero_float, &zero_bits, sizeof(float));
            return zero_float;
        } else {
            // Denormalized number
            exp = 127 - 14;
            while ((frac & 0x400) == 0) {
                frac <<= 1;
                exp--;
            }
            frac &= 0x3FF;
        }
    } else if (exp == 0x1F) {
        // Infinity or NaN
        exp = 0xFF;
    } else {
        // Normalized number
        exp += 127 - 15;
    }

    uint32_t result_bits = sign | (exp << 23) | (frac << 13);
    float    result_float;
    std::memcpy(&result_float, &result_bits, sizeof(float));
    return result_float;
}

// Helper function to convert BF16 to float using safe type conversion
float bf16_to_float(uint16_t bf16_value) {
    // BF16 has the same exponent width as FP32, just truncated mantissa
    uint32_t result_bits = static_cast<uint32_t>(bf16_value) << 16;
    float    result_float;
    std::memcpy(&result_float, &result_bits, sizeof(float));
    return result_float;
}
}  // anonymous namespace

std::string RpcAccessLogWrapper::serializeTensorPBPlaintext(const TensorPB& tensor_pb) {
    std::ostringstream oss;
    oss << "TensorPB {";

    // Add data type
    oss << " data_type: ";
    switch (tensor_pb.data_type()) {
        case TensorPB::FP32:
            oss << "FP32";
            break;
        case TensorPB::INT32:
            oss << "INT32";
            break;
        case TensorPB::FP16:
            oss << "FP16";
            break;
        case TensorPB::BF16:
            oss << "BF16";
            break;
        default:
            oss << "UNKNOWN";
            break;
    }

    // Add shape
    oss << " shape: [";
    for (int i = 0; i < tensor_pb.shape_size(); ++i) {
        if (i > 0)
            oss << ", ";
        oss << tensor_pb.shape(i);
    }
    oss << "]";

    // Add data based on type
    switch (tensor_pb.data_type()) {
        case TensorPB::FP32: {
            if (tensor_pb.fp32_data().empty()) {
                oss << " fp32_data: []";
            } else {
                oss << " fp32_data: [";
                const float* data          = reinterpret_cast<const float*>(tensor_pb.fp32_data().data());
                size_t       element_count = tensor_pb.fp32_data().size() / sizeof(float);
                for (size_t i = 0; i < element_count; ++i) {
                    if (i > 0)
                        oss << ", ";
                    oss << data[i];
                }
                oss << "]";
            }
            break;
        }
        case TensorPB::INT32: {
            if (tensor_pb.int32_data().empty()) {
                oss << " int32_data: []";
            } else {
                oss << " int32_data: [";
                const int32_t* data          = reinterpret_cast<const int32_t*>(tensor_pb.int32_data().data());
                size_t         element_count = tensor_pb.int32_data().size() / sizeof(int32_t);
                for (size_t i = 0; i < element_count; ++i) {
                    if (i > 0)
                        oss << ", ";
                    oss << data[i];
                }
                oss << "]";
            }
            break;
        }
        case TensorPB::FP16: {
            if (tensor_pb.fp16_data().empty()) {
                oss << " fp16_data: []";
            } else {
                oss << " fp16_data: [";
                const uint16_t* data          = reinterpret_cast<const uint16_t*>(tensor_pb.fp16_data().data());
                size_t          element_count = tensor_pb.fp16_data().size() / sizeof(uint16_t);
                for (size_t i = 0; i < element_count; ++i) {
                    if (i > 0)
                        oss << ", ";
                    oss << fp16_to_float(data[i]);
                }
                oss << "]";
            }
            break;
        }
        case TensorPB::BF16: {
            if (tensor_pb.bf16_data().empty()) {
                oss << " bf16_data: []";
            } else {
                oss << " bf16_data: [";
                const uint16_t* data          = reinterpret_cast<const uint16_t*>(tensor_pb.bf16_data().data());
                size_t          element_count = tensor_pb.bf16_data().size() / sizeof(uint16_t);
                for (size_t i = 0; i < element_count; ++i) {
                    if (i > 0)
                        oss << ", ";
                    oss << bf16_to_float(data[i]);
                }
                oss << "]";
            }
            break;
        }
        default:
            oss << " data: <binary data>";
            break;
    }

    oss << " }";
    return oss.str();
}

// Helper function to recursively find and replace TensorPB content in protobuf text format
std::string replaceTensorPBInText(const std::string& text) {
    std::string result = text;

    // Look for TensorPB field patterns - search for blocks that contain both "data_type:" and "shape:" and binary data
    size_t pos = 0;
    while (true) {
        size_t data_type_pos = result.find("data_type:", pos);
        if (data_type_pos == std::string::npos) {
            break;
        }

        // Find the opening brace that contains this data_type
        size_t brace_start = result.rfind('{', data_type_pos);
        if (brace_start == std::string::npos) {
            pos = data_type_pos + 10;
            continue;
        }

        // Find the matching closing brace
        size_t brace_end   = std::string::npos;
        int    brace_count = 1;
        size_t current     = brace_start + 1;

        while (current < result.length() && brace_count > 0) {
            if (result[current] == '{') {
                brace_count++;
            } else if (result[current] == '}') {
                brace_count--;
                if (brace_count == 0) {
                    brace_end = current;
                    break;
                }
            }
            current++;
        }

        if (brace_end == std::string::npos) {
            pos = data_type_pos + 10;
            continue;
        }

        // Extract the content between braces
        std::string tensor_content = result.substr(brace_start + 1, brace_end - brace_start - 1);

        // Check if this looks like a TensorPB (contains data_type, shape, and some _data field)
        if (tensor_content.find("data_type:") != std::string::npos && tensor_content.find("shape:") != std::string::npos
            && (tensor_content.find("int32_data:") != std::string::npos
                || tensor_content.find("fp32_data:") != std::string::npos
                || tensor_content.find("fp16_data:") != std::string::npos
                || tensor_content.find("bf16_data:") != std::string::npos)) {

            // Find the field name before the opening brace
            size_t field_start = brace_start;
            while (field_start > 0 && (result[field_start - 1] == ' ' || result[field_start - 1] == '\t')) {
                field_start--;
            }
            while (field_start > 0 && result[field_start - 1] != ' ' && result[field_start - 1] != '\t'
                   && result[field_start - 1] != '\n') {
                field_start--;
            }

            std::string field_name = result.substr(field_start, brace_start - field_start);
            field_name.erase(field_name.find_last_not_of(" \t") + 1);  // trim trailing spaces

            // Parse the TensorPB content manually
            TensorPB tensor_pb;

            // Extract data_type using simple string search
            size_t dt_pos = tensor_content.find("data_type:");
            if (dt_pos != std::string::npos) {
                dt_pos += 10;  // skip "data_type:"
                while (dt_pos < tensor_content.length() && std::isspace(tensor_content[dt_pos]))
                    dt_pos++;

                size_t dt_end = dt_pos;
                while (dt_end < tensor_content.length() && !std::isspace(tensor_content[dt_end])
                       && tensor_content[dt_end] != '\n')
                    dt_end++;

                std::string data_type = tensor_content.substr(dt_pos, dt_end - dt_pos);
                if (data_type == "INT32") {
                    tensor_pb.set_data_type(TensorPB::INT32);
                } else if (data_type == "FP32") {
                    tensor_pb.set_data_type(TensorPB::FP32);
                } else if (data_type == "FP16") {
                    tensor_pb.set_data_type(TensorPB::FP16);
                } else if (data_type == "BF16") {
                    tensor_pb.set_data_type(TensorPB::BF16);
                }
            }

            // Extract shape values
            size_t shape_search_pos = 0;
            while ((shape_search_pos = tensor_content.find("shape:", shape_search_pos)) != std::string::npos) {
                shape_search_pos += 6;  // skip "shape:"
                while (shape_search_pos < tensor_content.length() && std::isspace(tensor_content[shape_search_pos])) {
                    shape_search_pos++;
                }

                size_t shape_end = shape_search_pos;
                while (shape_end < tensor_content.length() && std::isdigit(tensor_content[shape_end])) {
                    shape_end++;
                }

                if (shape_end > shape_search_pos) {
                    std::string shape_str = tensor_content.substr(shape_search_pos, shape_end - shape_search_pos);
                    try {
                        int shape_val = std::stoi(shape_str);
                        tensor_pb.add_shape(shape_val);
                    } catch (const std::exception&) {
                        // Skip invalid shape values
                    }
                }
                shape_search_pos = shape_end;
            }

            // Extract binary data from int32_data field
            size_t data_pos = tensor_content.find("int32_data:");
            if (data_pos != std::string::npos) {
                size_t quote_start = tensor_content.find('"', data_pos);
                if (quote_start != std::string::npos) {
                    size_t quote_end = quote_start + 1;
                    // Find the closing quote, handling escaped quotes
                    while (quote_end < tensor_content.length()) {
                        if (tensor_content[quote_end] == '"'
                            && (quote_end == quote_start + 1 || tensor_content[quote_end - 1] != '\\')) {
                            break;
                        }
                        quote_end++;
                    }

                    if (quote_end < tensor_content.length()) {
                        std::string binary_data = tensor_content.substr(quote_start + 1, quote_end - quote_start - 1);
                        tensor_pb.set_int32_data(binary_data);
                    }
                }
            }

            // Use our custom serialization
            std::string tensor_output = RpcAccessLogWrapper::serializeTensorPBPlaintext(tensor_pb);

            // Replace the entire TensorPB block with field name
            std::string replacement = field_name + " " + tensor_output;

            result.replace(field_start, brace_end - field_start + 1, replacement);
            pos = field_start + replacement.length();
        } else {
            pos = data_type_pos + 10;
        }
    }

    return result;
}

std::string RpcAccessLogWrapper::serializeMessagePlaintext(const google::protobuf::Message* message) {
    if (!message) {
        return "null";
    }

    // Special handling for TensorPB to make it more readable
    if (message->GetDescriptor()->name() == "TensorPB") {
        TensorPB tensor_pb;
        tensor_pb.CopyFrom(*message);
        return serializeTensorPBPlaintext(tensor_pb);
    }

    // For other messages that may contain TensorPB fields, first get standard text format
    std::string output;
    google::protobuf::TextFormat::PrintToString(*message, &output);

    // Then replace any TensorPB structures with readable format
    return replaceTensorPBInText(output);
}

// Removed complex TensorPB JSON replacement functions - using simple binary serialization for log_plaintext=false

std::string RpcAccessLogWrapper::serializeMessageBinary(const google::protobuf::Message* message) {
    if (!message) {
        return "null";
    }

    // Binary serialization + base64 encoding for all messages
    std::string binary_data;
    if (!message->SerializeToString(&binary_data)) {
        RTP_LLM_LOG_ERROR("Failed to serialize message to binary");
        return message->ShortDebugString();
    }

    return base64_encode(binary_data);
}

// 带日志级别的查询日志记录函数
void RpcAccessLogWrapper::logQuery(const RpcAccessLogConfig&        config,
                                   const std::string&               requestType,
                                   const google::protobuf::Message* request,
                                   uint32_t                         logLevel) {
    // 检查日志级别是否启用
    if (!Logger::getAccessLogger().isLevelEnabled(logLevel)) {
        return;
    }

    // 只检查enable开关，移除access_log_interval控制
    if (!config.enable_rpc_access_log) {
        return;
    }

    // 序列化请求
    std::string request_str;

    if (config.log_plaintext) {
        request_str = serializeMessagePlaintext(request);
    } else {
        request_str = serializeMessageBinary(request);
    }

    // 根据日志级别记录日志
    LOG_QUERY_BY_LEVEL(logLevel, "%s: {\"request\": \"%s\"}", requestType.c_str(), request_str.c_str());
}

// 带日志级别的访问日志记录函数
void RpcAccessLogWrapper::logAccess(const RpcAccessLogConfig&        config,
                                    const std::string&               requestType,
                                    const google::protobuf::Message* request,
                                    const google::protobuf::Message* output,
                                    const std::string&               errorMsg,
                                    uint32_t                         logLevel) {
    // 检查日志级别是否启用
    if (!Logger::getAccessLogger().isLevelEnabled(logLevel)) {
        return;
    }

    // 只检查enable开关，移除access_log_interval控制
    if (!config.enable_rpc_access_log) {
        return;
    }

    // 序列化请求和响应
    std::string request_str, output_str;

    if (config.log_plaintext) {
        request_str = serializeMessagePlaintext(request);
        output_str  = serializeMessagePlaintext(output);
    } else {
        request_str = serializeMessageBinary(request);
        output_str  = serializeMessageBinary(output);
    }

    // 根据日志级别记录日志
    LOG_ACCESS_BY_LEVEL(logLevel,
                        "%s: {\"request\": \"%s\", \"response\": \"%s\", \"error_msg\": \"%s\"}",
                        requestType.c_str(),
                        request_str.c_str(),
                        output_str.c_str(),
                        errorMsg.c_str());
}

}  // namespace rtp_llm