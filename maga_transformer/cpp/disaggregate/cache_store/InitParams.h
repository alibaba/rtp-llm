#pragma once

#include <stdint.h>
#include "maga_transformer/cpp/disaggregate/cache_store/MemoryUtil.h"

namespace rtp_llm {

class CacheStoreInitParams {
public:
    uint32_t connect_port{15002};
    uint32_t listen_port{15002};

    uint32_t thread_count{2};
    uint32_t queue_size{100};

    // memory util
    void*    stream{nullptr};
    bool     rdma_mode{true};
    uint32_t rdma_connect_port{0};
    uint32_t rdma_listen_port{0};

    bool enable_metric{true};

    // for test
    std::shared_ptr<MemoryUtil> memory_util;
};

}  // namespace rtp_llm