#pragma once

#include "rtp_llm/cpp/api_server/http_server/http_server/HttpResponseWriter.h"
#include "rtp_llm/cpp/api_server/http_server/http_server/HttpRequest.h"
#include "rtp_llm/cpp/api_server/TokenProcessor.h"

#include "autil/legacy/jsonizable.h"

namespace rtp_llm {

class TokenizerEncodeRequest: public autil::legacy::Jsonizable {
public:
    TokenizerEncodeRequest()           = default;
    ~TokenizerEncodeRequest() override = default;

public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override;

public:
    std::optional<bool>        return_offsets_mapping;
    std::optional<std::string> prompt;
};

class TokenizerService {
public:
    TokenizerService(const std::shared_ptr<TokenProcessor>& token_processor);
    ~TokenizerService() = default;

public:
    void tokenizerEncode(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                         const http_server::HttpRequest&                         request);

private:
    std::shared_ptr<TokenProcessor> token_processor_;
};

}  // namespace rtp_llm
