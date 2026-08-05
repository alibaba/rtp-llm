#pragma once

#include <cstdint>
#include <map>
#include <vector>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "torch/all.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/models/position_ids/PositionIdsGenerator.h"

namespace rtp_llm {

struct TensorHolder;

struct NormalModelInputGathererConfig {
    size_t                                num_layers{};
    size_t                                vocab_size{};
    size_t                                input_vocab_size{};
    bool                                  has_positional_encoding{};
    bool                                  is_multimodal{};
    PositionIdsStyle                      mm_position_ids_style{};
    size_t                                position_id_len_factor{};
    RoleType                              role_type{};
    bool                                  decode_entrance{};
    size_t                                block_stride_bytes{};
    size_t                                scale_stride_bytes{};
    size_t                                seq_size_per_block{};
    size_t                                kernel_seq_size_per_block{};
    bool                                  use_opaque_kv_cache_store = false;
    std::map<std::string, CacheGroupType> kv_cache_group_types;
    std::map<std::string, size_t>         group_kernel_blocks_per_kv_block;
    bool                                  warm_up{};
    bool                                  enable_detail_log{};
};

class NormalModelInputGatherer {
public:
    explicit NormalModelInputGatherer(const NormalModelInputGathererConfig& config);

    absl::StatusOr<GptModelInputs> gather(const StreamGroups& stream_groups, TensorHolder& host_holder) const;

    // Refresh only the per-group CUDA kernel block tables. Read-only over
    // streams: no step(), no physical block tables, and no other gather step.
    absl::Status gatherKvCacheKernelBlockIds(GptModelInputs&     model_input,
                                             const StreamGroups& stream_groups,
                                             TensorHolder&       host_holder) const;

private:
    GptModelInputs allocateModelInputBuffers(const StreamGroups& stream_groups) const;
    absl::Status   processDecodeStreams(GptModelInputs& model_input, const StreamGroups& stream_groups) const;
    absl::Status   processContextStreams(GptModelInputs&     model_input,
                                         const StreamGroups& stream_groups,
                                         TensorHolder&       host_holder) const;

    NormalModelInputGathererConfig config_;
};

}  // namespace rtp_llm
