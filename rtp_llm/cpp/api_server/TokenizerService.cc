#include "rtp_llm/cpp/api_server/TokenizerService.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/api_server/ErrorResponse.h"
#include "rtp_llm/cpp/api_server/TokenizerEncodeResponse.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <optional>

namespace rtp_llm {

#define JSONIZE_OPTIONAL(field)                                                                                        \
    try {                                                                                                              \
        using Type = decltype(field)::value_type;                                                                      \
        Type field##Tmp;                                                                                               \
        json.Jsonize(#field, field##Tmp);                                                                              \
        field = field##Tmp;                                                                                            \
    } catch (autil::legacy::ExceptionBase & e) {                                                                       \
        if (field.has_value() == false) {                                                                              \
            field = std::nullopt;                                                                                      \
        }                                                                                                              \
    }

void TokenizerEncodeRequest::Jsonize(Jsonizable::JsonWrapper& json) {
    JSONIZE_OPTIONAL(prompt);
    JSONIZE_OPTIONAL(return_offsets_mapping);
}

TokenizerService::TokenizerService(const std::shared_ptr<TokenProcessor>& token_processor):
    token_processor_(token_processor) {}

void TokenizerService::tokenizerEncode(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                                       const http_server::HttpRequest&                         request) {
    writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
    writer->AddHeader("Content-Type", "application/json");
    if (!ParallelInfo::globalParallelInfo().isMaster()) {
        RTP_LLM_LOG_WARNING("gang worker should not access /tokenizer/encode api directly");
        auto msg = ErrorResponse::CreateErrorResponseJsonString(
            515, "gang worker should not access /tokenizer/encode api directly");
        writer->Write(msg);
        return;
    }

    const auto body = request.GetBody();
    try {
        TokenizerEncodeRequest req;
        autil::legacy::FromJsonString(req, body);
        if (req.prompt.has_value() == false) {
            RTP_LLM_LOG_WARNING("tokenizer encode failed, request has no prompt, request body: %s", body.c_str());
            writer->SetStatus(500, "Internal Server Error");
            auto msg =
                ErrorResponse::CreateErrorResponseJsonString(514, "tokenizer encode failed, request has no prompt");
            writer->Write(msg);
            return;
        }
        auto prompt         = req.prompt.value();
        bool offset_mapping = false;
        if (req.return_offsets_mapping.has_value()) {
            offset_mapping = req.return_offsets_mapping.value();
        }
        std::shared_ptr<TokenizerEncodeResponse> tokenizer_response;
        if (offset_mapping) {
            tokenizer_response = token_processor_->tokenizer(prompt);
        } else {
            auto                     token_ids = token_processor_->encode(prompt);
            std::vector<std::string> tokens;
            for (auto id : token_ids) {
                tokens.push_back(token_processor_->decode(std::vector<int>{id}));
            }
            tokenizer_response            = std::make_shared<TokenizerEncodeResponse>();
            tokenizer_response->token_ids = token_ids;
            tokenizer_response->tokens    = tokens;
        }

        if (!tokenizer_response) {
            RTP_LLM_LOG_WARNING("tokenizer encode failed, response is null, request body: %s", body.c_str());
            writer->SetStatus(500, "Internal Server Error");
            auto msg =
                ErrorResponse::CreateErrorResponseJsonString(514, "tokenizer encode failed, maybe tokenizer failed");
            writer->Write(msg);
            return;
        }
        tokenizer_response->return_offset_mapping = offset_mapping;

        auto response_json_str = ToJsonString(*tokenizer_response, /*isCompact=*/true);
        writer->Write(response_json_str);
        return;
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING(
            "tokenizer encode failed, found exception. request body: %s, exception: [%s]", body.c_str(), e.what());
        writer->SetStatus(500, "Internal Server Error");
        auto msg = ErrorResponse::CreateErrorResponseJsonString(514, "tokenizer encode failed, exception occurred");
        writer->Write(msg);
        return;
    }
}

}  // namespace rtp_llm
