#pragma once

#include <vector>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "autil/legacy/jsonizable.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rapidjson/reader.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/error/en.h"

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

// SAX handler for streaming JSON parsing
class PrefixDictHandler: public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, PrefixDictHandler> {
public:
    PrefixDictHandler(TreeDecodeConfig& cfg, std::unordered_map<std::string, std::unordered_set<int32_t>>& prefix_map):
        config_(cfg),
        prefix_map_(prefix_map),
        in_prefix_dict_(false),
        in_array_(false),
        current_key_(),
        dict_key_(),
        current_array_(),
        depth_(0) {
        config_.start_token_id = 225;  // default
        config_.end_token_id   = 2;    // default
        config_.sep            = "_";  // default
    }

    bool Null() {
        return true;
    }
    bool Bool(bool) {
        return true;
    }

    bool Int(int i) {
        return handleNumber(static_cast<int32_t>(i));
    }

    bool Uint(unsigned u) {
        return handleNumber(static_cast<int32_t>(u));
    }

    bool Int64(int64_t i) {
        return handleNumber(static_cast<int32_t>(i));
    }

    bool Uint64(uint64_t u) {
        return handleNumber(static_cast<int32_t>(u));
    }

    bool Double(double) {
        return true;
    }

    bool String(const char* str, rapidjson::SizeType length, bool) {
        if (in_array_) {
            // String in array - shouldn't happen for token IDs, but handle gracefully
            return true;
        } else if (current_key_ == "sep") {
            config_.sep = std::string(str, length);
            current_key_.clear();
        } else if (in_prefix_dict_ && !dict_key_.empty()) {
            // String value in prefix_dict - shouldn't happen, but handle gracefully
            return true;
        }
        return true;
    }

    bool StartObject() {
        depth_++;
        if (current_key_ == "prefix_dict") {
            in_prefix_dict_ = true;
        }
        return true;
    }

    bool Key(const char* str, rapidjson::SizeType length, bool) {
        current_key_ = std::string(str, length);
        if (in_prefix_dict_) {
            // This is a key in prefix_dict (e.g., "key1", "key2")
            dict_key_ = current_key_;
        }
        return true;
    }

    bool EndObject(rapidjson::SizeType) {
        depth_--;
        if (in_prefix_dict_ && depth_ == 1) {
            // Finished prefix_dict object (depth 1 is root object)
            in_prefix_dict_ = false;
        }
        if (!in_prefix_dict_) {
            current_key_.clear();
        }
        return true;
    }

    bool StartArray() {
        if (in_prefix_dict_ && !dict_key_.empty()) {
            in_array_ = true;
            current_array_.clear();
        }
        return true;
    }

    bool EndArray(rapidjson::SizeType) {
        if (in_array_ && !dict_key_.empty()) {
            // Finished one prefix_dict entry array
            if (!current_array_.empty()) {
                std::unordered_set<int32_t> tmp_set;
                for (auto token_id : current_array_) {
                    tmp_set.insert(token_id);
                }
                prefix_map_[dict_key_] = std::move(tmp_set);
            }
            current_array_.clear();
            dict_key_.clear();
        }
        in_array_ = false;
        return true;
    }

private:
    bool handleNumber(int32_t value) {
        if (in_array_) {
            current_array_.push_back(value);
        } else if (current_key_ == "start_token_id") {
            config_.start_token_id = value;
            current_key_.clear();
        } else if (current_key_ == "end_token_id") {
            config_.end_token_id = value;
            current_key_.clear();
        }
        return true;
    }

    TreeDecodeConfig&                                             config_;
    std::unordered_map<std::string, std::unordered_set<int32_t>>& prefix_map_;
    bool                                                          in_prefix_dict_;
    bool                                                          in_array_;
    std::string                                                   current_key_;
    std::string                                                   dict_key_;
    std::vector<int32_t>                                          current_array_;
    int                                                           depth_;
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
    std::string generateNextKey(std::string old_key, int next) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!old_key.empty()) {
            old_key = old_key + config.sep;
        }
        return old_key + std::to_string(next);
    }
    void reloadPrefixDictWithPrefix(const std::string& dir_path, const std::string& tree_decode_config) {
        RTP_LLM_LOG_INFO("PrefixToCandidateTokens load filepath : %s", tree_decode_config.c_str());
        if (tree_decode_config.size() > 0) {
            std::string prefix_dict_path = dir_path + "/" + tree_decode_config;
            reloadPrefixDict(prefix_dict_path);
        }
    }
    void reloadPrefixDict(const std::string& file_path) {
        loadPrefixDict(file_path);
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
    void                     loadPrefixDict(const std::string& file_path) {
        std::lock_guard<std::mutex> lock(mutex_);
        init_success_ = false;
        prefix_to_cadicates_.clear();

        // Try optimized streaming parser first for large files
        if (loadPrefixDictStreaming(file_path)) {
            init_success_ = true;
            RTP_LLM_LOG_INFO("PrefixToCandidateTokens load [%s] successfully (streaming)", file_path.c_str());
            return;
        }

        // Fallback to original method for compatibility
        RTP_LLM_LOG_INFO("PrefixToCandidateTokens falling back to legacy parser");
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
        file.close();
        init_success_ = true;
        RTP_LLM_LOG_INFO("PrefixToCandidateTokens load [%s] successfully", file_path.c_str());
    }

    bool loadPrefixDictStreaming(const std::string& file_path) {
        FILE* fp = fopen(file_path.c_str(), "rb");
        if (!fp) {
            return false;
        }

        // Use a reasonable buffer size (1MB) for streaming
        const size_t buffer_size = 1024 * 1024;
        char*        buffer      = new char[buffer_size];

        rapidjson::FileReadStream is(fp, buffer, buffer_size);
        PrefixDictHandler         handler(config, prefix_to_cadicates_);
        rapidjson::Reader         reader;

        bool success = false;
        if (reader.Parse(is, handler)) {
            success = true;
        } else {
            rapidjson::ParseErrorCode e = reader.GetParseErrorCode();
            size_t                    o = reader.GetErrorOffset();
            RTP_LLM_LOG_WARNING(
                "PrefixToCandidateTokens streaming parse error at offset %zu: %s", o, rapidjson::GetParseError_En(e));
        }

        delete[] buffer;
        fclose(fp);
        return success;
    }

private:
    std::mutex                                                   mutex_;
    TreeDecodeConfig                                             config;
    std::unordered_map<std::string, std::unordered_set<int32_t>> prefix_to_cadicates_;
    bool                                                         init_success_ = false;
};

typedef std::shared_ptr<PrefixToCandidateTokens> PrefixToCandidateTokensPtr;
}  // namespace rtp_llm
