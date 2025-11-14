#pragma once
#include <vector>
#include <string>

namespace rtp_llm {

struct RoleSpecialTokens {
    std::vector<int64_t> token_ids;
    std::vector<int64_t> eos_token_ids;
};

struct SpecialTokens {
    int64_t                           bos_token_id           = -1;
    int64_t                           eos_token_id           = 0;
    int64_t                           pad_token_id           = 0;
    int64_t                           decoder_start_token_id = -1;
    RoleSpecialTokens                 user;
    RoleSpecialTokens                 assistant;
    RoleSpecialTokens                 system;
    std::vector<std::vector<int64_t>> stop_words_id_list;
    std::vector<std::string>          stop_words_str_list;
};

}