#pragma once
#include "autil/legacy/jsonizable.h"

namespace rtp_llm {
namespace rtp_llm_master {

class TokenizeResponse: public autil::legacy::Jsonizable {
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("token_ids", token_ids);
    }

public:
    std::vector<int> token_ids;
};

}  // namespace rtp_llm_master
}  // namespace rtp_llm