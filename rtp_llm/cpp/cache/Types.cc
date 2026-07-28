#include "rtp_llm/cpp/cache/Types.h"

#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

namespace rtp_llm {

int MallocInfo::incrSeqLen() const {
    return incr_seq_len_override >= 0 ? incr_seq_len_override : complete_token_ids->seqLength();
}

}  // namespace rtp_llm
