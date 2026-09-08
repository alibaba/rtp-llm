#pragma once

#include <vector>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <map>
#include <unordered_map>
#include <unordered_set>
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
        if (skip_depth_ > 0) {
            return true;
        }
        if (depth_ == 0) {
            return typeError("root must be a JSON object");
        }
        return rejectScalar("null");
    }
    bool Bool(bool) {
        if (skip_depth_ > 0) {
            return true;
        }
        if (depth_ == 0) {
            return typeError("root must be a JSON object");
        }
        return rejectScalar("bool");
    }

    bool Int(int i) {
        if (skip_depth_ > 0) {
            return true;
        }
        if (depth_ == 0) {
            return typeError("root must be a JSON object");
        }
        return handleNumber(static_cast<int64_t>(i));
    }

    bool Uint(unsigned u) {
        if (skip_depth_ > 0) {
            return true;
        }
        if (depth_ == 0) {
            return typeError("root must be a JSON object");
        }
        return handleNumber(static_cast<int64_t>(u));
    }

    bool Int64(int64_t i) {
        if (skip_depth_ > 0) {
            return true;
        }
        if (depth_ == 0) {
            return typeError("root must be a JSON object");
        }
        return handleNumber(i);
    }

    bool Uint64(uint64_t u) {
        if (skip_depth_ > 0) {
            return true;
        }
        if (depth_ == 0) {
            return typeError("root must be a JSON object");
        }
        // Clamp values above int64_t max to a sentinel that handleNumber will treat as out-of-range.
        int64_t v = (u > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) ?
                        std::numeric_limits<int64_t>::max() :
                        static_cast<int64_t>(u);
        return handleNumber(v);
    }

    bool Double(double) {
        if (skip_depth_ > 0) {
            return true;
        }
        if (depth_ == 0) {
            return typeError("root must be a JSON object");
        }
        return rejectScalar("double");
    }

    bool String(const char* str, rapidjson::SizeType length, bool) {
        if (skip_depth_ > 0) {
            return true;
        }
        if (depth_ == 0) {
            return typeError("root must be a JSON object");
        }
        if (in_array_) {
            return typeError("prefix_dict array element must be an integer, got string");
        }
        if (depth_ == 1) {
            // value of a root-level key
            if (current_key_ == "sep") {
                config_.sep = std::string(str, length);
                current_key_.clear();
                return true;
            }
            if (current_key_ == "start_token_id" || current_key_ == "end_token_id") {
                return typeError("start_token_id/end_token_id must be an integer, got string");
            }
            if (current_key_ == "prefix_dict") {
                return typeError("prefix_dict must be an object, got string");
            }
            return true;  // unknown root key, ignore (matches legacy)
        }
        if (in_prefix_dict_ && !dict_key_.empty()) {
            return typeError("prefix_dict value must be an array, got string");
        }
        return true;
    }

    bool StartObject() {
        if (skip_depth_ > 0) {
            skip_depth_++;
            return true;
        }
        if (in_array_) {
            return typeError("prefix_dict array element must be an integer, got object");
        }
        if (depth_ == 1) {
            // object as the value of a root-level key: only prefix_dict may be an object
            if (current_key_ == "prefix_dict") {
                in_prefix_dict_ = true;
            } else if (current_key_ == "start_token_id" || current_key_ == "end_token_id") {
                return typeError("start_token_id/end_token_id must be an integer, got object");
            } else if (current_key_ == "sep") {
                return typeError("sep must be a string, got object");
            } else if (!current_key_.empty()) {
                // Unknown root-level key whose value is an object. The legacy parser silently
                // ignores unknown fields, so for backward compatibility skip the whole subtree
                // instead of failing the entire config. Note: depth_ is intentionally NOT
                // incremented here; the subtree (including its matching EndObject) is tracked by
                // skip_depth_ alone.
                RTP_LLM_LOG_INFO("PrefixToCandidateTokens SAX: skipping unknown root field '%s' (object value)",
                                 current_key_.c_str());
                skip_depth_ = 1;
                return true;
            }
        } else if (depth_ >= 2) {
            // an object where a prefix_dict value (array) is expected
            return typeError("prefix_dict value must be an array, got object");
        }
        depth_++;
        return true;
    }

    bool Key(const char* str, rapidjson::SizeType length, bool) {
        if (skip_depth_ > 0) {
            return true;
        }
        current_key_ = std::string(str, length);
        if (in_prefix_dict_) {
            // This is a key in prefix_dict (e.g., "key1", "key2")
            dict_key_ = current_key_;
        }
        return true;
    }

    bool EndObject(rapidjson::SizeType) {
        if (skip_depth_ > 0) {
            skip_depth_--;
            if (skip_depth_ == 0) {
                current_key_.clear();
            }
            return true;
        }
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
        if (skip_depth_ > 0) {
            skip_depth_++;
            return true;
        }
        // Nested arrays are not a valid candidate-token list (only a flat array of ids is allowed).
        if (in_array_) {
            return typeError("nested arrays are not allowed in a prefix_dict value");
        }
        // Arrays are only valid as prefix_dict values (the candidate token id list).
        if (in_prefix_dict_ && !dict_key_.empty()) {
            in_array_ = true;
            current_array_.clear();
            return true;
        }
        // Unknown root-level key whose value is an array: skip the whole subtree (legacy tolerates
        // unknown fields). A known root key with an array value still falls through to typeError.
        if (depth_ == 1 && !current_key_.empty() && !isKnownRootKey(current_key_)) {
            RTP_LLM_LOG_INFO("PrefixToCandidateTokens SAX: skipping unknown root field '%s' (array value)",
                             current_key_.c_str());
            skip_depth_ = 1;
            return true;
        }
        return typeError("unexpected array (only prefix_dict values may be arrays)");
    }

    bool EndArray(rapidjson::SizeType) {
        if (skip_depth_ > 0) {
            skip_depth_--;
            if (skip_depth_ == 0) {
                current_key_.clear();
            }
            return true;
        }
        if (in_array_ && !dict_key_.empty()) {
            std::unordered_set<int32_t> tmp_set;
            for (auto token_id : current_array_) {
                tmp_set.insert(token_id);
            }
            prefix_map_[dict_key_] = std::move(tmp_set);
            current_array_.clear();
            dict_key_.clear();
        }
        in_array_ = false;
        return true;
    }

public:
    bool schemaError() const {
        return schema_error_;
    }

private:
    static bool isKnownRootKey(const std::string& key) {
        return key == "start_token_id" || key == "end_token_id" || key == "sep" || key == "prefix_dict";
    }

    bool typeError(const std::string& msg) {
        // Returning false aborts the rapidjson parse. schema_error_ records that the abort was
        // caused by a schema/type violation (not a syntax error), so loadPrefixDict can fail
        // deterministically instead of papering over it with the legacy parser.
        schema_error_ = true;
        RTP_LLM_LOG_WARNING("PrefixToCandidateTokens SAX schema/type error: %s", msg.c_str());
        return false;
    }

    // bool/null/double are never valid values anywhere in tree_decode_config. They are tolerated
    // (and ignored, matching the legacy parser) only as the value of an unknown root-level key.
    bool rejectScalar(const char* json_type) {
        if (in_array_) {
            return typeError(std::string("prefix_dict array element must be an integer, got ") + json_type);
        }
        if (depth_ == 1
            && (current_key_ == "start_token_id" || current_key_ == "end_token_id" || current_key_ == "sep"
                || current_key_ == "prefix_dict")) {
            return typeError("key '" + current_key_ + "' has an unexpected " + json_type + " value");
        }
        if (in_prefix_dict_ && !dict_key_.empty()) {
            return typeError(std::string("prefix_dict value must be an array, got ") + json_type);
        }
        return true;  // unknown root key value, ignore
    }

    bool handleNumber(int64_t v) {
        // Determine whether an integer is allowed at the current position and where it belongs.
        bool to_start = false;
        bool to_end   = false;
        bool to_array = false;
        if (in_array_) {
            to_array = true;
        } else if (depth_ == 1) {
            if (current_key_ == "start_token_id") {
                to_start = true;
            } else if (current_key_ == "end_token_id") {
                to_end = true;
            } else if (current_key_ == "sep") {
                return typeError("sep must be a string, got number");
            } else if (current_key_ == "prefix_dict") {
                return typeError("prefix_dict must be an object, got number");
            } else {
                return true;  // unknown root key, ignore
            }
        } else if (in_prefix_dict_ && !dict_key_.empty()) {
            return typeError("prefix_dict value must be an array, got number");
        } else {
            return true;  // ignore
        }

        // The position accepts an integer; range-check before storing.
        if (v < std::numeric_limits<int32_t>::min() || v > std::numeric_limits<int32_t>::max()) {
            if (to_start || to_end) {
                // start/end token ids are scalar config values: an out-of-range value would silently
                // fall back to the default token, which is misleading. Treat it as a schema error.
                return typeError("start_token_id/end_token_id value out of int32_t range");
            }
            // Candidate token arrays stay lenient: skip out-of-range entries rather than failing.
            RTP_LLM_LOG_WARNING("PrefixToCandidateTokens SAX: integer value %lld overflows int32_t, skipping",
                                (long long)v);
            return true;
        }
        int32_t value = static_cast<int32_t>(v);
        if (to_start) {
            config_.start_token_id = value;
            current_key_.clear();
        } else if (to_end) {
            config_.end_token_id = value;
            current_key_.clear();
        } else if (to_array) {
            // Candidate token ids are used directly as logits/vocab indices, so a negative value is
            // invalid and would throw at logits time (DFAUtil::getCandidateTokenIds). Skip it here so
            // a bad entry never reaches the runtime, while keeping the rest of the list usable.
            if (value < 0) {
                RTP_LLM_LOG_WARNING("PrefixToCandidateTokens SAX: negative candidate token id %d skipped", value);
            } else {
                current_array_.push_back(value);
            }
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
    // While > 0, the parser is inside an unknown root-level container being skipped. It counts the
    // nesting level of that subtree (incremented on Start*/decremented on End*) so the whole subtree
    // is ignored, matching the legacy parser's tolerance of unknown fields.
    int  skip_depth_   = 0;
    bool schema_error_ = false;
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
        bool schema_error = false;
        if (loadPrefixDictStreaming(file_path, schema_error)) {
            init_success_ = true;
            RTP_LLM_LOG_INFO("PrefixToCandidateTokens load [%s] successfully (streaming)", file_path.c_str());
            return;
        }
        if (schema_error) {
            // The file is syntactically valid JSON but violates the tree_decode_config schema/types.
            // This is a definitive error: do NOT fall back to the legacy parser, which could
            // silently coerce mismatched types and report a misleading success.
            RTP_LLM_LOG_WARNING("PrefixToCandidateTokens load [%s] failed schema validation; not falling back",
                                file_path.c_str());
            prefix_to_cadicates_.clear();
            return;
        }

        // Soft failure (file open error or JSON syntax error): fall back to the legacy parser.
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

    // Returns true on success. On failure, sets schema_error=true when the file is valid JSON but
    // violates the tree_decode_config schema/types (a definitive error); leaves it false for soft
    // failures (cannot open file, JSON syntax error) where the legacy parser may still be tried.
    bool loadPrefixDictStreaming(const std::string& file_path, bool& schema_error) {
        schema_error = false;
        auto fp      = std::unique_ptr<FILE, decltype(&fclose)>(fopen(file_path.c_str(), "rb"), &fclose);
        if (!fp) {
            return false;
        }

        // Use a reasonable buffer size (1MB) for streaming
        constexpr size_t buffer_size = 1024 * 1024;
        auto             buffer      = std::make_unique<char[]>(buffer_size);

        rapidjson::FileReadStream                                    is(fp.get(), buffer.get(), buffer_size);
        TreeDecodeConfig                                             local_config;
        std::unordered_map<std::string, std::unordered_set<int32_t>> local_prefix_to_candidates;
        PrefixDictHandler                                            handler(local_config, local_prefix_to_candidates);
        rapidjson::Reader                                            reader;

        bool success = false;
        if (reader.Parse(is, handler)) {
            success = true;
            // NOTE: the streaming path only fills start/end/sep on config and writes the candidate
            // sets directly into prefix_to_cadicates_; it intentionally does NOT populate
            // config.prefix_dict (unlike the legacy parser). Runtime lookups only ever use
            // prefix_to_cadicates_, so this is equivalent. Do not rely on config.prefix_dict after a
            // successful streaming load.
            config               = std::move(local_config);
            prefix_to_cadicates_ = std::move(local_prefix_to_candidates);
        } else {
            // Distinguish a handler-triggered schema/type abort from a genuine JSON syntax error.
            schema_error                = handler.schemaError();
            rapidjson::ParseErrorCode e = reader.GetParseErrorCode();
            size_t                    o = reader.GetErrorOffset();
            RTP_LLM_LOG_WARNING(
                "PrefixToCandidateTokens streaming parse error at offset %zu: %s", o, rapidjson::GetParseError_En(e));
        }

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