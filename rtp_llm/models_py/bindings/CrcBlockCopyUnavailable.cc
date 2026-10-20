#include "rtp_llm/models_py/bindings/CrcBlockCopy.h"

namespace rtp_llm {

struct CrcBlockCopyBatch::Impl {};

bool CrcBlockCopyBatch::available() {
    return false;
}

size_t CrcBlockCopyBatch::deviceMemoryBytes() const {
    return 0;
}

CrcBlockCopyBatch::CrcBlockCopyBatch(int, size_t, size_t, size_t, std::shared_ptr<CrcDumpWriter>) {
    throw std::runtime_error("CRC block copy requires CUDA 13 x86 or ARM");
}

CrcBlockCopyBatch::~CrcBlockCopyBatch() = default;

CrcCopyStatus
CrcBlockCopyBatch::store(const std::vector<CrcCopyItem>&, std::vector<CrcCopyFailure>*) {
    return CrcCopyStatus::DEVICE_ERROR;
}

CrcCopyStatus
CrcBlockCopyBatch::load(const std::vector<CrcCopyItem>&, std::vector<CrcCopyFailure>*) {
    return CrcCopyStatus::DEVICE_ERROR;
}

}  // namespace rtp_llm
