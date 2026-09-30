#include "rtp_llm/models_py/bindings/CrcBlockCopy.h"

namespace rtp_llm {
struct CrcBlockCopyBatch::Impl {};
bool CrcBlockCopyBatch::available() {
    return false;
}
CrcBlockCopyBatch::CrcBlockCopyBatch(int, size_t, size_t, size_t) {
    throw std::runtime_error("CRC block copy requires CUDA 13 x86 or ARM");
}
CrcBlockCopyBatch::~CrcBlockCopyBatch() = default;
CrcCopyStatus CrcBlockCopyBatch::store(const std::vector<CrcCopyItem>&) {
    return CrcCopyStatus::DEVICE_ERROR;
}
CrcCopyStatus CrcBlockCopyBatch::load(const std::vector<CrcCopyItem>&) {
    return CrcCopyStatus::DEVICE_ERROR;
}
CrcCopyStatus CrcBlockCopyBatch::validate(const std::vector<CrcCopyItem>&) {
    return CrcCopyStatus::DEVICE_ERROR;
}
}  // namespace rtp_llm
