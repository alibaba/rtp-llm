#include "rtp_llm/cpp/utils/RpcAccessLogWrapper.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include <google/protobuf/text_format.h>
#include <google/protobuf/util/json_util.h>
#include <sstream>
#include <iomanip>

namespace rtp_llm {

namespace {
std::string serializeTensorPBPlaintext(const TensorPB& tensor_pb) {
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
                for (size_t i = 0; i < element_count && i < 10; ++i) {  // Limit to first 10 elements
                    if (i > 0)
                        oss << ", ";
                    oss << data[i];
                }
                if (element_count > 10) {
                    oss << ", ... (" << element_count << " elements total)";
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
                for (size_t i = 0; i < element_count && i < 10; ++i) {  // Limit to first 10 elements
                    if (i > 0)
                        oss << ", ";
                    oss << data[i];
                }
                if (element_count > 10) {
                    oss << ", ... (" << element_count << " elements total)";
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
                for (size_t i = 0; i < element_count && i < 10; ++i) {  // Limit to first 10 elements
                    if (i > 0)
                        oss << ", ";
                    oss << "0x" << std::hex << data[i] << std::dec;
                }
                if (element_count > 10) {
                    oss << ", ... (" << element_count << " elements total)";
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
                for (size_t i = 0; i < element_count && i < 10; ++i) {  // Limit to first 10 elements
                    if (i > 0)
                        oss << ", ";
                    oss << "0x" << std::hex << data[i] << std::dec;
                }
                if (element_count > 10) {
                    oss << ", ... (" << element_count << " elements total)";
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

class TensorPBFieldValuePrinter: public google::protobuf::TextFormat::FastFieldValuePrinter {
public:
    void PrintMessageStart(const google::protobuf::Message&                 message,
                           int                                              field_index,
                           int                                              field_count,
                           bool                                             single_line_mode,
                           google::protobuf::TextFormat::BaseTextGenerator* generator) const override {
        if (message.GetDescriptor()->name() == "TensorPB") {
            // 重新使用已有的serializeTensorPBPlaintext函数
            TensorPB tensor_pb;
            tensor_pb.CopyFrom(message);
            std::string output = serializeTensorPBPlaintext(tensor_pb);
            generator->PrintString(output);
            return;
        }
        // 对于其他消息类型，使用默认打印
        google::protobuf::TextFormat::FastFieldValuePrinter::PrintMessageStart(
            message, field_index, field_count, single_line_mode, generator);
    }
};
}  // namespace

std::string RpcAccessLogWrapper::serializeMessagePlaintext(const google::protobuf::Message* message) {
    if (!message) {
        return "null";
    }

    // Special handling for TensorPB to make it more readable
    if (message->GetDescriptor()->name() == "TensorPB") {
        TensorPB tensor_pb;
        tensor_pb.CopyFrom(*message);
        // 使用匿名命名空间中的函数
        return [&tensor_pb]() {
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
                        for (size_t i = 0; i < element_count && i < 10; ++i) {  // Limit to first 10 elements
                            if (i > 0)
                                oss << ", ";
                            oss << data[i];
                        }
                        if (element_count > 10) {
                            oss << ", ... (" << element_count << " elements total)";
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
                        for (size_t i = 0; i < element_count && i < 10; ++i) {  // Limit to first 10 elements
                            if (i > 0)
                                oss << ", ";
                            oss << data[i];
                        }
                        if (element_count > 10) {
                            oss << ", ... (" << element_count << " elements total)";
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
                        for (size_t i = 0; i < element_count && i < 10; ++i) {  // Limit to first 10 elements
                            if (i > 0)
                                oss << ", ";
                            oss << "0x" << std::hex << data[i] << std::dec;
                        }
                        if (element_count > 10) {
                            oss << ", ... (" << element_count << " elements total)";
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
                        for (size_t i = 0; i < element_count && i < 10; ++i) {  // Limit to first 10 elements
                            if (i > 0)
                                oss << ", ";
                            oss << "0x" << std::hex << data[i] << std::dec;
                        }
                        if (element_count > 10) {
                            oss << ", ... (" << element_count << " elements total)";
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
        }();
    }

    // 使用自定义TextFormat打印机处理包含TensorPB字段的消息
    google::protobuf::TextFormat::Printer printer;
    printer.SetDefaultFieldValuePrinter(new TensorPBFieldValuePrinter());
    std::string output;
    printer.PrintToString(*message, &output);
    return output;
}

std::string RpcAccessLogWrapper::serializeMessageWithCompress(const google::protobuf::Message* message) {
    if (!message) {
        return "null";
    }
    std::string json_string;
    auto        status = google::protobuf::util::MessageToJsonString(*message, &json_string);
    if (!status.ok()) {
        RTP_LLM_LOG_ERROR("Failed to serialize message to JSON: %s", status.ToString().c_str());
        return message->ShortDebugString();
    }
    return json_string;
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
        request_str = serializeMessageWithCompress(request);
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
        request_str = serializeMessageWithCompress(request);
        output_str  = serializeMessageWithCompress(output);
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