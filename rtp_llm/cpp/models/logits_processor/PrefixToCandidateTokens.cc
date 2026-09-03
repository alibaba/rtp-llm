#include "rtp_llm/cpp/models/logits_processor/PrefixToCandidateTokens.h"

#include <algorithm>
#include <atomic>
#include <fstream>
#include <limits>
#include <sstream>
#include <utility>

#include "autil/legacy/jsonizable.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

class PrefixTreeWireUpdate: public autil::legacy::Jsonizable {
public:
    uint64_t         version = 0;
    TreeDecodeConfig config;

    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("version", version, version);
        json.Jsonize("start_token_id", config.start_token_id, 225);
        json.Jsonize("end_token_id", config.end_token_id, 2);
        json.Jsonize("sep", config.sep, std::string("_"));
        json.Jsonize("prefix_dict", config.prefix_dict, config.prefix_dict);
    }
};

}  // namespace

PrefixTreeSnapshot::PrefixTreeSnapshot(uint64_t                                    version,
                                       int32_t                                     start_token_id,
                                       int32_t                                     end_token_id,
                                       std::string                                 sep,
                                       std::map<std::string, std::vector<int32_t>> prefix_to_candidates):
    version_(version),
    start_token_id_(start_token_id),
    end_token_id_(end_token_id),
    sep_(std::move(sep)),
    prefix_to_candidates_(std::move(prefix_to_candidates)) {}

const std::vector<int32_t>& PrefixTreeSnapshot::getCandidateTokens(const std::string& key) const {
    static const std::vector<int32_t> kEmpty;
    const auto                        iter = prefix_to_candidates_.find(key);
    return iter == prefix_to_candidates_.end() ? kEmpty : iter->second;
}

bool PrefixTreeSnapshot::isValidStatus(const std::string& key) const {
    return prefix_to_candidates_.find(key) != prefix_to_candidates_.end();
}

std::string PrefixTreeSnapshot::generateNextKey(const std::string& old_key, int32_t next) const {
    if (old_key.empty()) {
        return std::to_string(next);
    }
    return old_key + sep_ + std::to_string(next);
}

const char* prefixTreeUpdateCodeName(PrefixTreeUpdateCode code) {
    switch (code) {
        case PrefixTreeUpdateCode::UPDATED:
            return "updated";
        case PrefixTreeUpdateCode::ALREADY_CURRENT:
            return "already_current";
        case PrefixTreeUpdateCode::STALE_VERSION:
            return "stale_version";
        case PrefixTreeUpdateCode::INVALID_CONFIG:
            return "invalid_config";
        case PrefixTreeUpdateCode::IO_ERROR:
            return "io_error";
    }
    return "unknown";
}

std::shared_ptr<PrefixToCandidateTokens> PrefixToCandidateTokens::instance() {
    static std::shared_ptr<PrefixToCandidateTokens> singleton(new PrefixToCandidateTokens());
    return singleton;
}

PrefixTreeSnapshotPtr PrefixToCandidateTokens::snapshot() const {
    return std::atomic_load_explicit(&snapshot_, std::memory_order_acquire);
}

bool PrefixToCandidateTokens::initSuccess() const {
    return snapshot() != nullptr;
}

uint64_t PrefixToCandidateTokens::currentVersion() const {
    const auto current = snapshot();
    return current ? current->version() : 0;
}

int32_t PrefixToCandidateTokens::startTokenId() const {
    const auto current = snapshot();
    return current ? current->startTokenId() : TreeDecodeConfig().start_token_id;
}

int32_t PrefixToCandidateTokens::endTokenId() const {
    const auto current = snapshot();
    return current ? current->endTokenId() : TreeDecodeConfig().end_token_id;
}

PrefixTreeUpdateResult PrefixToCandidateTokens::updatePrefixDict(uint64_t version, TreeDecodeConfig config) {
    if (version == 0) {
        return {PrefixTreeUpdateCode::INVALID_CONFIG, currentVersion(), "version must be greater than zero"};
    }
    const auto active = snapshot();
    if (active && version < active->version()) {
        return {PrefixTreeUpdateCode::STALE_VERSION, active->version(), "a newer tree is already active"};
    }
    if (active && version == active->version()) {
        return {PrefixTreeUpdateCode::ALREADY_CURRENT, active->version(), "tree version is already active"};
    }
    if (config.start_token_id < 0 || config.end_token_id < 0) {
        return {PrefixTreeUpdateCode::INVALID_CONFIG, currentVersion(), "token ids must not be negative"};
    }
    if (config.start_token_id == config.end_token_id) {
        return {PrefixTreeUpdateCode::INVALID_CONFIG,
                currentVersion(),
                "start_token_id and end_token_id must be different"};
    }
    if (config.sep.empty()) {
        return {PrefixTreeUpdateCode::INVALID_CONFIG, currentVersion(), "separator must not be empty"};
    }
    if (config.prefix_dict.empty()) {
        return {PrefixTreeUpdateCode::INVALID_CONFIG, currentVersion(), "prefix_dict must not be empty"};
    }

    for (auto& [prefix, token_ids] : config.prefix_dict) {
        if (prefix.empty() || token_ids.empty()) {
            return {PrefixTreeUpdateCode::INVALID_CONFIG,
                    currentVersion(),
                    "prefixes and candidate lists must not be empty"};
        }
        for (const auto token_id : token_ids) {
            if (token_id < 0) {
                return {
                    PrefixTreeUpdateCode::INVALID_CONFIG, currentVersion(), "candidate token ids must not be negative"};
            }
        }
        std::sort(token_ids.begin(), token_ids.end());
        token_ids.erase(std::unique(token_ids.begin(), token_ids.end()), token_ids.end());
    }

    // The expensive allocation happens before taking update_mutex_. Readers never
    // take this lock; the critical section only checks the version and swaps a pointer.
    const size_t          prefix_count = config.prefix_dict.size();
    PrefixTreeSnapshotPtr next(new PrefixTreeSnapshot(
        version, config.start_token_id, config.end_token_id, config.sep, std::move(config.prefix_dict)));

    std::lock_guard<std::mutex> lock(update_mutex_);
    const auto                  current = snapshot();
    if (current && version < current->version()) {
        return {PrefixTreeUpdateCode::STALE_VERSION, current->version(), "a newer tree is already active"};
    }
    if (current && version == current->version()) {
        return {PrefixTreeUpdateCode::ALREADY_CURRENT, current->version(), "tree version is already active"};
    }

    std::atomic_store_explicit(&snapshot_, std::move(next), std::memory_order_release);
    RTP_LLM_LOG_INFO("PrefixToCandidateTokens activated version [%llu], prefix count [%zu]",
                     static_cast<unsigned long long>(version),
                     prefix_count);
    return {PrefixTreeUpdateCode::UPDATED, version, "tree activated"};
}

PrefixTreeUpdateResult PrefixToCandidateTokens::updatePrefixDictFromJson(const std::string& json) {
    PrefixTreeWireUpdate update;
    try {
        autil::legacy::FromJsonString(update, json);
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("PrefixToCandidateTokens cannot parse update: %s", e.what());
        return {PrefixTreeUpdateCode::INVALID_CONFIG,
                currentVersion(),
                "request body is not a valid constraint-tree update"};
    }
    return updatePrefixDict(update.version, std::move(update.config));
}

PrefixTreeUpdateResult PrefixToCandidateTokens::reloadPrefixDict(const std::string& file_path) {
    RTP_LLM_LOG_INFO("PrefixToCandidateTokens load filepath: %s", file_path.c_str());
    std::ifstream file(file_path);
    if (!file) {
        RTP_LLM_LOG_WARNING("PrefixToCandidateTokens cannot open file: %s", file_path.c_str());
        return {PrefixTreeUpdateCode::IO_ERROR, currentVersion(), "unable to open tree config file"};
    }

    TreeDecodeConfig config;
    try {
        std::ostringstream content;
        content << file.rdbuf();
        autil::legacy::FromJsonString(config, content.str());
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("PrefixToCandidateTokens cannot parse file [%s]: %s", file_path.c_str(), e.what());
        return {PrefixTreeUpdateCode::INVALID_CONFIG, currentVersion(), "tree config file is not valid JSON"};
    }

    const uint64_t active_version = currentVersion();
    if (active_version == std::numeric_limits<uint64_t>::max()) {
        return {PrefixTreeUpdateCode::INVALID_CONFIG, active_version, "tree version is exhausted"};
    }
    return updatePrefixDict(active_version + 1, std::move(config));
}

PrefixTreeUpdateResult PrefixToCandidateTokens::reloadPrefixDictWithPrefix(const std::string& dir_path,
                                                                           const std::string& tree_decode_config) {
    if (tree_decode_config.empty()) {
        return {PrefixTreeUpdateCode::ALREADY_CURRENT, currentVersion(), "tree decode is not configured"};
    }
    return reloadPrefixDict(dir_path + "/" + tree_decode_config);
}

}  // namespace rtp_llm
