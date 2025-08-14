#include "rtp_llm/cpp/dataclass/CacheStatusInfo.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/th_op/ConfigModules.h"

namespace rtp_llm {

void registerCacheStatusInfo(const py::module& m) {
    pybind11::class_<CacheStatusInfo>(m, "CacheStatusInfo")
        .def(pybind11::init<>())
        .def_readwrite("cached_keys", &CacheStatusInfo::cached_keys)
        .def_readwrite("available_kv_cache", &CacheStatusInfo::available_kv_cache)
        .def_readwrite("total_kv_cache", &CacheStatusInfo::total_kv_cache)
        .def_readwrite("block_size", &CacheStatusInfo::block_size)
        .def_readwrite("version", &CacheStatusInfo::version);
}

};  // namespace rtp_llm

