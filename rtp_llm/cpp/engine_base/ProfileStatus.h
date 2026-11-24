#pragma once

#include <string>
#include <cstdint>

namespace rtp_llm {

struct ProfileStatus {
    bool        supported = false;
    std::string state     = "finished";
    std::string trace_name;
    bool        stop_requested      = false;
    bool        collection_complete = false;
    bool        export_complete     = false;
    std::string collection_error;
    std::string export_error;
    std::string file_name;
    int64_t     capture_begin_ns = 0;
    int64_t     capture_end_ns   = 0;
};

}  // namespace rtp_llm
