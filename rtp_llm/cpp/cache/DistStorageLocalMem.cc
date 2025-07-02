#include "rtp_llm/cpp/cache/DistStorageLocalMem.h"

namespace rtp_llm {

DistStorageLocalMem::DistStorageLocalMem(const kmonitor::MetricsReporterPtr& metrics_reporter):
    metrics_reporter_(metrics_reporter) {}

bool DistStorageLocalMem::init(const DistStorageLocalMemInitParams& init_params) {
    // TODO:
    return false;
}

bool DistStorageLocalMem::lookup(const DistStorage::Item& item) {
    // TODO:
    return false;
}

bool DistStorageLocalMem::get(const DistStorage::Item& item) {
    // TODO:
    return false;
}

bool DistStorageLocalMem::put(const DistStorage::Item& item) {
    // TODO:
    return false;
}

bool DistStorageLocalMem::del(const DistStorage::Item& item) {
    // TODO:
    return false;
}

}  // namespace rtp_llm