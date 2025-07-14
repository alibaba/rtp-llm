#pragma once

#include <vector>
#include <string>
#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"

namespace rtp_llm {

struct RemoteServerResource {
    int isTensorParallel() const {
        return workers.size() != 1;
    }

    std::vector<std::string>          workers;
    std::vector<std::string>          grpc_workers;
    std::shared_ptr<NormalCacheStore> cache_store;
    RPCPool                           rpc_pool;
};

}  // namespace rtp_llm
