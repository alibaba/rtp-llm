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

    static DecodeTokenTraceConfig fromEnv();
    static DecodeTokenTraceConfig fromValues(bool               enabled,
                                             const std::string& filter_csv,
                                             const std::string& output_path,
                                             bool               capture_peers,
                                             int                max_blocks_per_group);

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
