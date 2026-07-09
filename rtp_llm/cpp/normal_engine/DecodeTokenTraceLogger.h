#pragma once

#include <string>
#include <vector>

#include "torch/all.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"

namespace rtp_llm {

struct DecodeTokenTraceConfig {
    bool                     enabled              = false;
    std::vector<std::string> filters;
    std::string              output_path;
    bool                     capture_peers        = true;
    int                      max_blocks_per_group = 16;
    bool                     bad_watch_enabled    = false;
    std::string              bad_watch_output_path;
    int                      bad_watch_tail_size  = 128;
    int                      bad_watch_min_cf     = 4;

    static DecodeTokenTraceConfig fromEnv();
    static DecodeTokenTraceConfig fromValues(bool               enabled,
                                             const std::string& filter_csv,
                                             const std::string& output_path,
                                             bool               capture_peers,
                                             int                max_blocks_per_group,
                                             bool               bad_watch_enabled,
                                             const std::string& bad_watch_output_path,
                                             int                bad_watch_tail_size,
                                             int                bad_watch_min_cf);

    bool matches(const std::string& trace_id) const;
};

class DecodeTokenTraceLogger {
public:
    static bool enabled();
    static void logDispatchBatch(const StreamGroups&    stream_groups,
                                 const torch::Tensor&  token_ids_cpu,
                                 const torch::Tensor&  success_cpu);

    static std::string jsonEscape(const std::string& value);
};

}  // namespace rtp_llm
