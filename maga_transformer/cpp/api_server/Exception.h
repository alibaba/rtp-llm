#pragma once

#include <exception>
#include <string>

namespace rtp_llm {

class HttpApiServerException: public std::exception {
public:
    enum Type {
        CONCURRENCY_LIMIT_ERROR = 409,
        CANCELLED_ERROR = 499,
        ERROR_INPUT_FORMAT_ERROR = 507,
        GPT_NOT_FOUND_ERROR = 508,
        NO_PROMPT_ERROR = 509,
        EMPTY_PROMPT_ERROR = 510,
        LONG_PROMPT_ERROR = 511,
        ERROR_STOP_LIST_FORMAT = 512,
        UNKNOWN_ERROR = 514,
        UNSUPPORTED_OPERATION = 515,
        ERROR_GENERATE_CONFIG_FORMAT = 516,
        TOKENIZER_ERROR = 517,
        MULTIMODAL_ERROR = 518,
        UPDATE_ERROR = 601,
        MALLOC_ERROR = 602,
        GENERATE_TIMEOUT_ERROR = 603,
        GET_HOST_ERROR = 604,
        GET_CONNECTION_ERROR = 605,
        CONNECT_ERROR = 606,
        CONNECTION_RESET_BY_PEER_ERROR = 607,
        REMOTE_ALLOCATE_RESOURCE_ERROR = 608,
        REMOTE_LOAD_KV_CACHE_ERROR = 609,
        REMOTE_GENERATE_ERROR = 610,
    };

    HttpApiServerException(Type type, const std::string& message) : type_(type), message_(message) {}

    virtual const char* what() const noexcept override {
        return message_.c_str();
    }

    Type getType() const {
        return type_;
    }

    std::string getMessage() const {
        return message_;
    }

    static std::string formatException(const std::exception& e);

private:
    Type type_;
    std::string message_;
};

}  // namespace rtp_llm
