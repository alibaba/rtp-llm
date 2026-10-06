#pragma once

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

// A forced Prefill batch is local to its scheduler. Decode DP ranks receive
// subsets of that group, so inheriting the same barrier can prevent admission.
// Call only on the D-bound ALLOCATE clone; never change the original P request.
inline void normalizePdAllocateScheduling(GenerateInputPB& input) {
    if (!input.has_generate_config() || !input.generate_config().has_force_batch()
        || input.generate_config().force_batch().value() == 0) {
        return;
    }
    input.set_batch_group_size(1);
    input.clear_batch_group_id();
    auto* config = input.mutable_generate_config();
    config->clear_force_batch();
    config->clear_batch_group_timeout();
}

}  // namespace rtp_llm
