#pragma once

#include <cstring>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "autil/legacy/jsonizable.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

class TreeDecodeConfig: public autil::legacy::Jsonizable {
public:
    int32_t                                     start_token_id;
    int32_t                                     end_token_id;
    std::string                                 sep;
    std::map<std::string, std::vector<int32_t>> prefix_dict;

    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("start_token_id", start_token_id, 225);
        json.Jsonize("end_token_id", end_token_id, 2);
        json.Jsonize("sep", sep, "_");
        json.Jsonize("prefix_dict", prefix_dict, prefix_dict);
    }
};

// Legacy startup-only Prefix Map support is intentionally unchanged. Runtime
// hot updates use ConstraintTreeCsrManager exclusively.
class PrefixToCandidateTokens {
public:
    const std::unordered_set<int32_t>& getCandidateTokens(const std::string& key) {
        std::lock_guard<std::mutex>        lock(mutex_);
        static std::unordered_set<int32_t> EMPTY;
        if (!init_success_) {
            RTP_LLM_LOG_WARNING("PrefixToCandidateTokens is not initialized yet");
            return EMPTY;
        }
        auto iter = prefix_to_cadicates_.find(key);
        if (prefix_to_cadicates_.end() == iter) {
            return EMPTY;
        }
        return iter->second;
    }

    bool isValidStatus(const std::string& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        return prefix_to_cadicates_.find(key) != prefix_to_cadicates_.end();
    }

    bool initSuccess() {
        return init_success_;
    }

    int32_t startTokenId() {
        return config.start_token_id;
    }

    int32_t endTokenId() {
        return config.end_token_id;
    }

    std::string generateNextKey(std::string old_key, int next) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!old_key.empty()) {
            old_key = old_key + config.sep;
        }
        return old_key + std::to_string(next);
    }

    void reloadPrefixDictWithPrefix(std::string dir_path, std::string tree_decode_config) {
        RTP_LLM_LOG_INFO("PrefixToCandidateTokens load filepath : %s", tree_decode_config.c_str());
        if (!tree_decode_config.empty()) {
            reloadPrefixDict(dir_path + "/" + tree_decode_config);
        }
    }

    void reloadPrefixDict(std::string file_path) {
        loadPrefixDict(std::move(file_path));
    }

public:
    static std::shared_ptr<PrefixToCandidateTokens> instance() {
        static std::shared_ptr<PrefixToCandidateTokens> singleton(new PrefixToCandidateTokens());
        return singleton;
    }

private:
    PrefixToCandidateTokens()                               = default;
    PrefixToCandidateTokens(const PrefixToCandidateTokens&) = delete;
    PrefixToCandidateTokens(PrefixToCandidateTokens&&)      = delete;
    PrefixToCandidateTokens& operator=(const PrefixToCandidateTokens&) = delete;

    void loadPrefixDict(const std::string& file_path) {
        std::lock_guard<std::mutex> lock(mutex_);
        init_success_ = false;
        prefix_to_cadicates_.clear();
        std::ifstream file(file_path);
        if (!file) {
            RTP_LLM_LOG_INFO("PrefixToCandidateTokens load failed: unable to open file[%s]", file_path.c_str());
            return;
        }

        try {
            std::ostringstream content;
            content << file.rdbuf();
            autil::legacy::FromJsonString(config, content.str());
        } catch (autil::legacy::ExceptionBase& e) {
            RTP_LLM_LOG_INFO("PrefixToCandidateTokens load failed: file[%s] is not valid json", file_path.c_str());
            return;
        }
        for (const auto& [prefix, candidates] : config.prefix_dict) {
            prefix_to_cadicates_[prefix] = std::unordered_set<int32_t>(candidates.begin(), candidates.end());
        }
        init_success_ = true;
        RTP_LLM_LOG_INFO("PrefixToCandidateTokens load [%s] successfully", file_path.c_str());
    }

private:
    std::mutex                                                   mutex_;
    TreeDecodeConfig                                             config;
    std::unordered_map<std::string, std::unordered_set<int32_t>> prefix_to_cadicates_;
    bool                                                         init_success_ = false;
};

using PrefixToCandidateTokensPtr = std::shared_ptr<PrefixToCandidateTokens>;

}  // namespace rtp_llm
