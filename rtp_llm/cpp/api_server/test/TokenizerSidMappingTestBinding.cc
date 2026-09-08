#include "rtp_llm/cpp/api_server/tokenizer/Tokenizer.h"

// Bind the real implementation, not a duplicate Python manifest generator.
PYBIND11_MODULE(tokenizer_sid_mapping_test_lib, m) {
    m.def("sid_mapping_json", [](py::object tokenizer) { return rtp_llm::Tokenizer(tokenizer).sidMappingJson(); });
}
