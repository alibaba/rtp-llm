#pragma once
#include <string>
#include "autil/legacy/jsonizable.h"

namespace rtp_llm {
namespace rtp_llm_master {
class MasterErrorResponse: public autil::legacy::Jsonizable {
public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("error_code", error_code, error_code);
        json.Jsonize("message", message, message);
    }

    static std::string CreateJsonString(int error_code, const std::string& error_msg) {
        MasterErrorResponse response;
        response.error_code = error_code;
        response.message    = error_msg;
        return autil::legacy::ToJsonString(response, /*isCompact=*/true);
    }

public:
    int         error_code;
    std::string message;
};

class MasterInfo: public autil::legacy::Jsonizable {
public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("request_id", request_id, request_id);
        json.Jsonize("prefix_length", prefix_length, prefix_length);
        json.Jsonize("input_length", input_length, input_length);
        json.Jsonize("expect_execute_time_ms", expect_execute_time_ms, expect_execute_time_ms);
        json.Jsonize("expect_wait_time_ms", expect_wait_time_ms, expect_wait_time_ms);
        json.Jsonize("machine_info", machine_info, machine_info);
        json.Jsonize("tokenize_cost_time_ms", tokenize_cost_time_ms, tokenize_cost_time_ms);
        json.Jsonize("estimate_cost_time_ms", estimate_cost_time_ms, estimate_cost_time_ms);
    }

public:
    int         prefix_length;
    int         input_length;
    int64_t     request_id;
    int64_t     expect_execute_time_ms;
    int64_t expect_wait_time_ms;
    int64_t     tokenize_cost_time_ms;
    int64_t     estimate_cost_time_ms;
    std::string machine_info;
};

class MasterSuccessResponse: public autil::legacy::Jsonizable {
public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("ip", ip, ip);
        json.Jsonize("port", port, port);
        json.Jsonize("master_info", master_info, master_info);
    }
    static std::string CreateJsonString(const std::string& ip, int port, const MasterInfo& master_info) {
        MasterSuccessResponse response;
        response.ip          = ip;
        response.port        = port;
        response.master_info = master_info;
        return autil::legacy::ToJsonString(response, /*isCompact=*/true);
    }

public:
    std::string ip;
    int         port;
    MasterInfo  master_info;
};

}  // namespace rtp_llm_master
}  // namespace rtp_llm
