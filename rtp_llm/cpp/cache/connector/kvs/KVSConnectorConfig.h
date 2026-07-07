#pragma once

#include <cstddef>
#include <string>

namespace rtp_llm {

struct KVSConnectorConfig {
    std::string object_namespace  = "rtp_llm";
    std::string cache_key_version = "1";
    int         timeout_ms        = 12000;
    int         lease_term_sec    = 60;
    int         worker_thread_num = 8;
    int         worker_queue_size = 1024;
    bool        inline_execute    = false;

    // KVS client endpoint. RTP only passes these to the backend adapter.
    std::string endpoint_url;
    std::string socket_path;
    std::string read_peer = "local";
};

}  // namespace rtp_llm
