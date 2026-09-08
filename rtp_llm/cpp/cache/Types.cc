#include "rtp_llm/cpp/cache/Types.h"

#include <algorithm>

#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

namespace rtp_llm {

int MallocInfo::incrSeqLen() const {
    return incr_seq_len_override >= 0 ? incr_seq_len_override : complete_token_ids->seqLength();
}

bool MallocInfo::hasInitSeqLenForTag(const std::string& tag) const {
    const auto it = init_seq_len_by_tag.find(tag);
    return it != init_seq_len_by_tag.end() && it->second >= 0;
}

int MallocInfo::initSeqLenForTag(const std::string& tag, int fallback) const {
    const auto it = init_seq_len_by_tag.find(tag);
    return it == init_seq_len_by_tag.end() || it->second < 0 ? fallback : std::min(fallback, it->second);
}

int MallocInfo::reserveStepForTag(const std::string& tag, int fallback) const {
    return hasInitSeqLenForTag(tag) ? 0 : fallback;
}

}  // namespace rtp_llm
