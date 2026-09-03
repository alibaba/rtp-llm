#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "autil/legacy/jsonizable.h"

namespace rtp_llm {

class TreeDecodeConfig: public autil::legacy::Jsonizable {
public:
    int32_t                                     start_token_id = 225;
    int32_t                                     end_token_id   = 2;
    std::string                                 sep            = "_";
    std::map<std::string, std::vector<int32_t>> prefix_dict;

    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("start_token_id", start_token_id, 225);
        json.Jsonize("end_token_id", end_token_id, 2);
        json.Jsonize("sep", sep, std::string("_"));
        json.Jsonize("prefix_dict", prefix_dict, prefix_dict);
    }
};

// A decode request keeps one immutable snapshot for its whole lifetime. Updating
// the process-wide tree therefore cannot change an in-flight request midway.
class PrefixTreeSnapshot {
public:
    const std::vector<int32_t>& getCandidateTokens(const std::string& key) const;
    bool                        isValidStatus(const std::string& key) const;
    std::string                 generateNextKey(const std::string& old_key, int32_t next) const;

    uint64_t version() const {
        return version_;
    }
    int32_t startTokenId() const {
        return start_token_id_;
    }
    int32_t endTokenId() const {
        return end_token_id_;
    }
    size_t prefixCount() const {
        return prefix_to_candidates_.size();
    }

private:
    friend class PrefixToCandidateTokens;

    PrefixTreeSnapshot(uint64_t                                    version,
                       int32_t                                     start_token_id,
                       int32_t                                     end_token_id,
                       std::string                                 sep,
                       std::map<std::string, std::vector<int32_t>> prefix_to_candidates);

private:
    uint64_t                                    version_;
    int32_t                                     start_token_id_;
    int32_t                                     end_token_id_;
    std::string                                 sep_;
    std::map<std::string, std::vector<int32_t>> prefix_to_candidates_;
};

using PrefixTreeSnapshotPtr = std::shared_ptr<const PrefixTreeSnapshot>;

enum class PrefixTreeUpdateCode {
    UPDATED,
    ALREADY_CURRENT,
    STALE_VERSION,
    INVALID_CONFIG,
    IO_ERROR,
};

struct PrefixTreeUpdateResult {
    PrefixTreeUpdateCode code;
    uint64_t             current_version;
    std::string          message;

    bool ok() const {
        return code == PrefixTreeUpdateCode::UPDATED || code == PrefixTreeUpdateCode::ALREADY_CURRENT;
    }
};

const char* prefixTreeUpdateCodeName(PrefixTreeUpdateCode code);

// Process-wide owner of the current immutable prefix-tree snapshot.
class PrefixToCandidateTokens {
public:
    static std::shared_ptr<PrefixToCandidateTokens> instance();

    PrefixTreeSnapshotPtr snapshot() const;
    bool                  initSuccess() const;
    uint64_t              currentVersion() const;
    int32_t               startTokenId() const;
    int32_t               endTokenId() const;

    PrefixTreeUpdateResult updatePrefixDict(uint64_t version, TreeDecodeConfig config);
    PrefixTreeUpdateResult updatePrefixDictFromJson(const std::string& json);
    PrefixTreeUpdateResult reloadPrefixDict(const std::string& file_path);
    PrefixTreeUpdateResult reloadPrefixDictWithPrefix(const std::string& dir_path,
                                                      const std::string& tree_decode_config);

private:
    PrefixToCandidateTokens()                                          = default;
    PrefixToCandidateTokens(const PrefixToCandidateTokens&)            = delete;
    PrefixToCandidateTokens(PrefixToCandidateTokens&&)                 = delete;
    PrefixToCandidateTokens& operator=(const PrefixToCandidateTokens&) = delete;

private:
    mutable std::mutex    update_mutex_;
    PrefixTreeSnapshotPtr snapshot_;
};

using PrefixToCandidateTokensPtr = std::shared_ptr<PrefixToCandidateTokens>;

}  // namespace rtp_llm
