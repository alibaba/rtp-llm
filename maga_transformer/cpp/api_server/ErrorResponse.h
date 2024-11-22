#pragma once

#include "autil/legacy/jsonizable.h"

namespace rtp_llm {

class ErrorResponse: public autil::legacy::Jsonizable {
public:
    ~ErrorResponse() override = default;

public:
    void               Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override;
    static std::string CreateErrorResponseJsonString(int error_code, const std::string& error_msg);

public:
    int         error_code;
    std::string error_msg;
};

}  // namespace rtp_llm