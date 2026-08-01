#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <numeric>
#include <vector>

#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/HashUtil.h"

namespace rtp_llm {

using RequestPrefixKey = int64_t;

// Immutable, tagless view passed to connector match implementations.  It is
// intentionally independent of native group cache keys and block tables.
class RequestPrefixMatchView {
public:
    RequestPrefixMatchView(const std::vector<RequestPrefixKey>& keys,
                           size_t                               match_span_tokens,
                           size_t                               token_extent,
                           size_t                               match_limit_tokens,
                           size_t                               write_limit_tokens,
                           size_t                               reuse_tokens):
        keys_(&keys),
        match_span_tokens_(match_span_tokens),
        token_extent_(token_extent),
        match_limit_tokens_(match_limit_tokens),
        write_limit_tokens_(write_limit_tokens),
        reuse_tokens_(reuse_tokens) {}

    const std::vector<RequestPrefixKey>& keys() const {
        return *keys_;
    }
    size_t matchSpanTokens() const {
        return match_span_tokens_;
    }
    size_t tokenExtent() const {
        return token_extent_;
    }
    size_t matchLimitTokens() const {
        return match_limit_tokens_;
    }
    size_t writeLimitTokens() const {
        return write_limit_tokens_;
    }
    size_t reuseTokens() const {
        return reuse_tokens_;
    }

private:
    const std::vector<RequestPrefixKey>* keys_;
    size_t                               match_span_tokens_;
    size_t                               token_extent_;
    size_t                               match_limit_tokens_;
    size_t                               write_limit_tokens_;
    size_t                               reuse_tokens_;
};

// Request-owned connector control plane. Native tag-local cache keys and
// physical block resources remain in CacheGroupResource.
class RequestPrefixResource {
public:
    RequestPrefixResource() = default;

    RequestPrefixResource(const RequestPrefixResource& other) {
        std::lock_guard<std::mutex> lock(other.reuse_mutex_);
        copyFromLocked(other);
    }

    RequestPrefixResource& operator=(const RequestPrefixResource& other) {
        if (this == &other) {
            return *this;
        }
        std::scoped_lock lock(reuse_mutex_, other.reuse_mutex_);
        copyFromLocked(other);
        return *this;
    }

    void configure(const CacheTopology& topology) {
        size_t span     = 1;
        bool   reusable = false;
        for (const auto& group : topology.groups()) {
            if (!group.policy.enable_prefix_reuse) {
                continue;
            }
            reusable         = true;
            const size_t gcd = std::gcd(span, group.seq_size_per_block);
            RTP_LLM_CHECK_WITH_INFO(span / gcd <= std::numeric_limits<size_t>::max() / group.seq_size_per_block,
                                    "request prefix match-span LCM overflow");
            span = span / gcd * group.seq_size_per_block;
        }
        match_span_tokens_ = reusable ? span : 1;
        reset();
    }

    void reset() {
        std::lock_guard<std::mutex> lock(reuse_mutex_);
        keys_.clear();
        token_extent_        = 0;
        match_limit_tokens_  = 0;
        write_limit_tokens_  = 0;
        device_reuse_tokens_ = 0;
        memory_reuse_tokens_ = 0;
        remote_reuse_tokens_ = 0;
    }

    void rebuild(int32_t* tokens, size_t token_count) {
        RTP_LLM_CHECK_WITH_INFO(tokens != nullptr || token_count == 0,
                                "RequestPrefixResource::rebuild received null tokens");
        keys_.clear();
        token_extent_       = token_count;
        match_limit_tokens_ = token_count == 0 ? 0 : ((token_count - 1) / match_span_tokens_) * match_span_tokens_;
        write_limit_tokens_ = (token_count / match_span_tokens_) * match_span_tokens_;
        keys_.reserve((token_count + match_span_tokens_ - 1) / match_span_tokens_);
        RequestPrefixKey hash = 0;
        for (size_t begin = 0; begin < token_count; begin += match_span_tokens_) {
            const size_t end = std::min(begin + match_span_tokens_, token_count);
            hash             = hashInt64Array(hash, tokens + begin, tokens + end);
            keys_.push_back(hash);
        }
    }

    RequestPrefixMatchView matchView() const {
        return RequestPrefixMatchView(
            keys_, match_span_tokens_, token_extent_, match_limit_tokens_, write_limit_tokens_, reuseTokens());
    }

    const std::vector<RequestPrefixKey>& keys() const {
        return keys_;
    }
    size_t matchSpanTokens() const {
        return match_span_tokens_;
    }
    size_t tokenExtent() const {
        return token_extent_;
    }
    size_t completePrefixEndpoint() const {
        return write_limit_tokens_;
    }
    size_t matchLimitTokens() const {
        return match_limit_tokens_;
    }
    size_t writeLimitTokens() const {
        return write_limit_tokens_;
    }

    size_t deviceReuseTokens() const {
        std::lock_guard<std::mutex> lock(reuse_mutex_);
        return device_reuse_tokens_;
    }
    size_t memoryReuseTokens() const {
        std::lock_guard<std::mutex> lock(reuse_mutex_);
        return memory_reuse_tokens_;
    }
    size_t remoteReuseTokens() const {
        std::lock_guard<std::mutex> lock(reuse_mutex_);
        return remote_reuse_tokens_;
    }
    size_t reuseTokens() const {
        std::lock_guard<std::mutex> lock(reuse_mutex_);
        return device_reuse_tokens_ + memory_reuse_tokens_ + remote_reuse_tokens_;
    }

    void setDeviceReuseTokens(size_t tokens) {
        std::lock_guard<std::mutex> lock(reuse_mutex_);
        validateTierReuseTokens(tokens, memory_reuse_tokens_, remote_reuse_tokens_);
        device_reuse_tokens_ = tokens;
    }
    void setMemoryReuseTokens(size_t tokens) {
        std::lock_guard<std::mutex> lock(reuse_mutex_);
        validateTierReuseTokens(tokens, device_reuse_tokens_, remote_reuse_tokens_);
        memory_reuse_tokens_ = tokens;
    }
    void setRemoteReuseTokens(size_t tokens) {
        std::lock_guard<std::mutex> lock(reuse_mutex_);
        validateTierReuseTokens(tokens, device_reuse_tokens_, memory_reuse_tokens_);
        remote_reuse_tokens_ = tokens;
    }

private:
    void copyFromLocked(const RequestPrefixResource& other) {
        match_span_tokens_   = other.match_span_tokens_;
        keys_                = other.keys_;
        token_extent_        = other.token_extent_;
        match_limit_tokens_  = other.match_limit_tokens_;
        write_limit_tokens_  = other.write_limit_tokens_;
        device_reuse_tokens_ = other.device_reuse_tokens_;
        memory_reuse_tokens_ = other.memory_reuse_tokens_;
        remote_reuse_tokens_ = other.remote_reuse_tokens_;
    }

    void validateTierReuseTokens(size_t tokens, size_t other_a, size_t other_b) const {
        RTP_LLM_CHECK_WITH_INFO(tokens % match_span_tokens_ == 0,
                                "request prefix tier reuse tokens=%zu are not aligned to match span=%zu",
                                tokens,
                                match_span_tokens_);
        RTP_LLM_CHECK_WITH_INFO(tokens <= match_limit_tokens_ && other_a <= match_limit_tokens_ - tokens
                                    && other_b <= match_limit_tokens_ - tokens - other_a,
                                "request prefix cumulative reuse tokens exceed match limit: tier=%zu other=%zu/%zu "
                                "limit=%zu",
                                tokens,
                                other_a,
                                other_b,
                                match_limit_tokens_);
    }

    size_t                        match_span_tokens_{1};
    std::vector<RequestPrefixKey> keys_;
    size_t                        token_extent_{0};
    size_t                        match_limit_tokens_{0};
    size_t                        write_limit_tokens_{0};
    size_t                        device_reuse_tokens_{0};
    size_t                        memory_reuse_tokens_{0};
    size_t                        remote_reuse_tokens_{0};
    mutable std::mutex            reuse_mutex_;
};

}  // namespace rtp_llm
