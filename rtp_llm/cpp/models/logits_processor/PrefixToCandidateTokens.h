#pragma once

#include <vector>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <mutex>
#include "autil/legacy/jsonizable.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

class TreeDecodeConfig: public autil::legacy::Jsonizable {
public:
    int32_t                                     start_token_id;
    int32_t                                     end_token_id;
    std::string                                 sep;
    std::map<std::string, std::vector<int32_t>> prefix_dict;
    std::map<std::string, float>                weight_dict;

    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("start_token_id", start_token_id, 225);
        json.Jsonize("end_token_id", end_token_id, 2);
        json.Jsonize("sep", sep, "_");
        json.Jsonize("prefix_dict", prefix_dict, prefix_dict);
        json.Jsonize("weight_dict", weight_dict, {});
    }
};

class PrefixToCandidateTokens {
public:
    const std::unordered_set<int32_t>& getCandidateTokens(const std::string& key) {
        std::lock_guard<std::mutex>        lock(mutex_);
        static std::unordered_set<int32_t> EMPTY;
        if (!init_success_) {
            static std::unordered_set<int32_t> EMPTY;
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
        auto                        iter = prefix_to_cadicates_.find(key);
        if (prefix_to_cadicates_.end() == iter) {
            return false;
        } else {
            return true;
        }
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
    std::string getSep() {
        return config.sep;
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
        if (tree_decode_config.size() > 0) {
            std::string prefix_dict_path = dir_path + "/" + tree_decode_config;
            reloadPrefixDict(prefix_dict_path);
        }
    }
    void reloadPrefixDict(std::string file_path) {
        loadPrefixDict(file_path);
    }

    const std::unordered_map<std::string, float>& getWeightDict() {
        return weight_dict_;
    }

public:
    static std::shared_ptr<PrefixToCandidateTokens> instance() {
        static std::shared_ptr<PrefixToCandidateTokens> t(new PrefixToCandidateTokens());
        return t;
    }

private:
    PrefixToCandidateTokens() {}
    PrefixToCandidateTokens(PrefixToCandidateTokens&)                  = delete;
    PrefixToCandidateTokens(PrefixToCandidateTokens&&)                 = delete;
    PrefixToCandidateTokens& operator=(const PrefixToCandidateTokens&) = delete;
    void                     loadPrefixDict(std::string file_path) {
        std::lock_guard<std::mutex> lock(mutex_);
        init_success_ = false;
        prefix_to_cadicates_.clear();
        weight_dict_.clear();
        std::ifstream file(file_path);
        if (!file) {
            std::stringstream ss;
            ss << "Unable to open file[" << file_path << "]" << std::endl;
            RTP_LLM_LOG_INFO("PrefixToCandidateTokens load failed: %s", ss.str().c_str());
            return;
        }

        try {
            std::ostringstream ss;
            ss << file.rdbuf();
            autil::legacy::FromJsonString(config, ss.str());
        } catch (autil::legacy::ExceptionBase& e) {
            std::stringstream ss;
            ss << "file[" << file_path << "]'s format is not json" << std::endl;
            RTP_LLM_LOG_INFO("PrefixToCandidateTokens load failed: %s", ss.str().c_str());
            return;
        }
        for (auto kv : config.prefix_dict) {
            std::unordered_set<int32_t> tmp_set;
            for (auto token_id : kv.second) {
                tmp_set.insert(token_id);
            }
            prefix_to_cadicates_[kv.first] = tmp_set;
        }
        weight_dict_.insert(config.weight_dict.begin(), config.weight_dict.end());
        file.close();
        init_success_ = true;
        RTP_LLM_LOG_INFO("PrefixToCandidateTokens load [%s] successfully", file_path.c_str());
    }

private:
    std::mutex                                                   mutex_;
    TreeDecodeConfig                                             config;
    std::unordered_map<std::string, std::unordered_set<int32_t>> prefix_to_cadicates_;
    std::unordered_map<std::string, float>                       weight_dict_;
    bool                                                         init_success_ = false;
};

typedef std::shared_ptr<PrefixToCandidateTokens> PrefixToCandidateTokensPtr;
}  // namespace rtp_llm
