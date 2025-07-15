#include "rtp_llm/cpp/dataclass/KvCacheInfo.h"
#include "rtp_llm/cpp/utils/Logger.h"
namespace rtp_llm {

void registerKvCacheInfo(const py::module& m) {
    pybind11::class_<KVCacheInfo>(m, "KVCacheInfo")
        .def(pybind11::init<>())
        .def_readwrite("cached_keys", &KVCacheInfo::cached_keys)
        .def_readwrite("available_kv_cache", &KVCacheInfo::available_kv_cache)
        .def_readwrite("total_kv_cache", &KVCacheInfo::total_kv_cache)
        .def_readwrite("block_size", &KVCacheInfo::block_size)
        .def_readwrite("version", &KVCacheInfo::version);
}

};  // namespace rtp_llm