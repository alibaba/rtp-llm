#pragma once

#include <string>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

inline void encodeCompletionKey(const TransferCompletionKey& key, TransferCompletionKeyPB& wire) {
    wire.set_trace_id(key.trace_id);
    switch (key.legacy_wire_field) {
        case TransferCompletionKey::LegacyWireField::TRANSFER_TRACE_ID:
            wire.set_transfer_trace_id(std::stoull(key.trace_id));
            break;
        case TransferCompletionKey::LegacyWireField::KVCM_TRACE_ID:
            wire.set_kvcm_trace_id(key.trace_id);
            break;
        case TransferCompletionKey::LegacyWireField::NONE:
            wire.clear_trace();
            break;
    }
}

inline std::string decodeCompletionTraceId(const TransferCompletionKeyPB& wire) {
    std::string legacy_id;
    switch (wire.trace_case()) {
        case TransferCompletionKeyPB::kTransferTraceId:
            if (wire.transfer_trace_id() == 0)
                return {};
            legacy_id = std::to_string(wire.transfer_trace_id());
            break;
        case TransferCompletionKeyPB::kKvcmTraceId:
            legacy_id = wire.kvcm_trace_id();
            break;
        default:
            return wire.trace_id();
    }
    if (!wire.trace_id().empty() && wire.trace_id() != legacy_id)
        return {};
    return legacy_id;
}

}
