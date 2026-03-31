#include "rtp_llm/cpp/core/ExecCtxExport.h"
#include "rtp_llm/cpp/core/ExecOps.h"
using namespace rtp_llm;

namespace torch_ext {

DeviceType ExecCtxExporter::getDeviceType() {
    return exec_params_.device_type;
}

int64_t ExecCtxExporter::getDeviceId() {
    return exec_params_.device_id;
}

}  // namespace torch_ext
