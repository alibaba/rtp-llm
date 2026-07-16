#include <pybind11/pybind11.h>

#include "rtp_llm/cpp/cache/CacheConfigCreator.h"

namespace py = pybind11;

namespace rtp_llm::test {

void validateBasicConfig(const ModelConfig& model_config) {
    ParallelismConfig parallelism_config;
    parallelism_config.tp_size = 1;
    (void)CacheConfigCreator::createBasicConfig(
        model_config, parallelism_config, /*is_mtp=*/false, /*gen_num_per_cycle=*/0);
}

PYBIND11_MODULE(libcache_config_creator_py_test, m) {
    m.def("validate_basic_config", &validateBasicConfig);
}

}  // namespace rtp_llm::test
