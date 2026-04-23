#include "rtp_llm/cpp/pybind/common/blockUtil.h"
#include "rtp_llm/cpp/utils/HashUtil.h"

std::vector<rtp_llm::CacheKeyType> getBlockCacheKey(const std::vector<std::vector<int64_t>>& token_ids_list) {
    std::vector<rtp_llm::CacheKeyType> cache_keys;
    int64_t                            hash = 0;
    for (const auto& token_ids : token_ids_list) {
        hash = rtp_llm::hashInt64Vector(hash, token_ids);
        cache_keys.push_back(static_cast<rtp_llm::CacheKeyType>(hash));
    }
    return cache_keys;
}

void registerCommon(py::module& m) {
    m.def("get_block_cache_keys", &getBlockCacheKey, py::arg("token_ids_list"));
}