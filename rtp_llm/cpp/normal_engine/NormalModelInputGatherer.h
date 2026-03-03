#pragma once

#include <cstdint>
#include <vector>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/core/OpData.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/models/position_ids/PositionIdsGenerator.h"

namespace rtp_llm {

struct NormalModelInputGathererConfig {
    size_t                      num_layers{};
    size_t                      vocab_size{};
    size_t                      input_vocab_size{};
    bool                        has_positional_encoding{};
    bool                        is_multimodal{};
    PositionIdsStyle            mm_position_ids_style{};
    size_t                      position_id_len_factor{};
    RoleType                    role_type{};
    bool                        decode_entrance{};
    size_t                      block_stride_bytes{};
    size_t                      scale_stride_bytes{};
    size_t                      seq_size_per_block{};
    size_t                      kernel_seq_size_per_block{};
    size_t                      kernel_blocks_per_kv_block = 1;
    size_t                      kv_cache_group_nums        = 1;
    std::vector<int32_t>        layer_to_kv_cache_group_id;
    std::vector<CacheGroupType> kv_cache_group_types;
    bool                        warm_up{};
    bool                        enable_detail_log{};
};

class NormalModelInputGatherer {
public:
    explicit NormalModelInputGatherer(const NormalModelInputGathererConfig& config);

    absl::StatusOr<GptModelInputs> gather(const StreamGroups& stream_groups) const;

private:
    GptModelInputs allocateModelInputBuffers(const StreamGroups& stream_groups) const;
    void           initializeKvCacheMetadata(GptModelInputs& model_input) const;
    absl::Status   processDecodeStreams(GptModelInputs& model_input, const StreamGroups& stream_groups) const;
    absl::Status   processContextStreams(GptModelInputs& model_input, const StreamGroups& stream_groups) const;

    NormalModelInputGathererConfig config_;
};

}  // namespace rtp_llm
