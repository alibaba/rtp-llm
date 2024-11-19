#pragma once

#include <vector>
#include <string>
#include "maga_transformer/cpp/model_rpc/RPCPool.h"
#include "maga_transformer/cpp/disaggregate/cache_store/NormalCacheStore.h"

namespace rtp_llm {

struct RemoteServerResource {
    int isParallel() const {
        return workers.size() != 0;
    }

    std::vector<std::string> workers;
    std::shared_ptr<NormalCacheStore> cache_store;
    RPCPool rpc_pool;
};

}  // namespace rtp_llm
