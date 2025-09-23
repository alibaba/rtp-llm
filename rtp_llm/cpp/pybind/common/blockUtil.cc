#include "rtp_llm/cpp/pybind/common/blockUtil.h"
#include "rtp_llm/cpp/utils/HashUtil.h"

std::vector<int64_t> getBlockCacheKey(const std::vector<std::vector<int64_t>>& token_ids_list) {
    std::vector<int64_t> block_ids;
    int64_t              hash = 0;
    for (const auto& token_ids : token_ids_list) {
        hash = rtp_llm::hashInt64Vector(hash, token_ids);
        block_ids.push_back(hash);
    }
    return block_ids;
}

void registerCommon(py::module& m) {
    m.def("get_block_cache_keys", &getBlockCacheKey, py::arg("token_ids_list"));
}