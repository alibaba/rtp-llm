#pragma once

#include "autil/legacy/jsonizable.h"

namespace rtp_llm {

class TokenizerEncodeResponse: public autil::legacy::Jsonizable {
public:
    TokenizerEncodeResponse()           = default;
    ~TokenizerEncodeResponse() override = default;

public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        if (json.GetMode() == autil::legacy::FastJsonizableBase::Mode::TO_JSON) {
            if (return_offset_mapping) {
                json.Jsonize("offset_mapping", offset_mapping);
            }
        } else {
            json.Jsonize("offset_mapping", offset_mapping, offset_mapping);
        }
        json.Jsonize("token_ids", token_ids, token_ids);
        json.Jsonize("tokens", tokens, tokens);
        json.Jsonize("error", error, error);
    }

public:
    std::vector<std::vector<int>> offset_mapping;
    std::vector<std::string>      tokens;
    std::vector<int>              token_ids;
    std::string                   error;
    bool                          return_offset_mapping{false};
};

}  // namespace rtp_llm