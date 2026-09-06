#include <pybind11/pybind11.h>

#include "rtp_llm/cpp/multimodal_processor/transport/kvcm/MMKvcmWriter.h"

PYBIND11_MODULE(libmm_kvcm_writer, module) {
    rtp_llm::registerMMKvcmWriter(module);
}
