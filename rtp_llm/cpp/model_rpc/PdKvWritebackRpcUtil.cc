#include "rtp_llm/cpp/model_rpc/PdKvWritebackRpcUtil.h"

namespace rtp_llm {
namespace {

PdKvWritebackCompatibility pdKvWritebackCompatibilityFromPB(const PdKvWritebackCompatibilityPB& pb) {
    PdKvWritebackCompatibility compatibility;
    compatibility.seq_size_per_block = pb.seq_size_per_block();
    compatibility.layer_count        = pb.layer_count();
    compatibility.group_count        = pb.group_count();
    compatibility.partition_count    = pb.partition_count();
    compatibility.layer_to_group_id.assign(pb.layer_to_group_id().begin(), pb.layer_to_group_id().end());
    compatibility.group_types.assign(pb.group_types().begin(), pb.group_types().end());
    return compatibility;
}

}  // namespace

PdKvWritebackLaunchRequest pdKvWritebackLaunchRequestFromPB(const PdKvWritebackRequestPB& pb) {
    PdKvWritebackLaunchRequest request;
    request.manifest.request_id           = pb.request_id();
    request.manifest.request_key          = pb.request_key();
    request.manifest.final_token_count    = pb.final_token_count();
    request.manifest.reusable_block_count = pb.reusable_block_count();
    request.manifest.cache_keys.assign(pb.cache_keys().begin(), pb.cache_keys().end());
    request.manifest.group_block_ids.reserve(pb.group_block_ids_size());
    for (const auto& group_pb : pb.group_block_ids()) {
        request.manifest.group_block_ids.emplace_back(group_pb.block_ids().begin(), group_pb.block_ids().end());
    }
    request.source      = pdKvWritebackCompatibilityFromPB(pb.source());
    request.destination = pdKvWritebackCompatibilityFromPB(pb.destination());
    request.decode_worker_addrs.assign(pb.decode_worker_addrs().begin(), pb.decode_worker_addrs().end());
    request.prefill_worker_addrs.assign(pb.prefill_worker_addrs().begin(), pb.prefill_worker_addrs().end());
    request.deadline_ms = pb.deadline_us() > 0 ? pb.deadline_us() / 1000 : 0;
    return request;
}

}  // namespace rtp_llm
