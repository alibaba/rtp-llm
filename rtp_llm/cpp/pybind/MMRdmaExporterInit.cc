#include <pybind11/pybind11.h>

#include "rtp_llm/cpp/multimodal_processor/transport/rdma/MMRdmaExporter.h"

PYBIND11_MODULE(libmm_rdma_exporter, module) {
    rtp_llm::registerMMRdmaExporter(module);
}
