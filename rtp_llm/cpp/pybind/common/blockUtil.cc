#include "rtp_llm/cpp/pybind/common/blockUtil.h"
#include "rtp_llm/cpp/cache/CacheKeyConfig.h"
#include "rtp_llm/cpp/utils/HashUtil.h"

std::vector<int64_t> getBlockCacheKey(const std::vector<std::vector<int64_t>>& token_ids_list, int64_t initial_hash) {
    std::vector<int64_t> block_ids;
    int64_t              hash = initial_hash;
    for (const auto& token_ids : token_ids_list) {
        hash = rtp_llm::hashInt64Vector(hash, token_ids);
        block_ids.push_back(hash);
    }
    return block_ids;
}

void registerCommon(py::module& m) {
    m.def("get_block_cache_keys", &getBlockCacheKey, py::arg("token_ids_list"), py::arg("initial_hash") = 0);
    m.attr("DSV41_SWA_BOUNDED_REPLAY_CACHE_KEY_SEED") = py::int_(rtp_llm::DSV41_SWA_BOUNDED_REPLAY_CACHE_KEY_SEED);
}
