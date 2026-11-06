#include "rtp_llm/models_py/bindings/CrcBlockCopy.h"

namespace rtp_llm {

struct CrcBlockCopyBatch::Impl {};

bool CrcBlockCopyBatch::available() {
    return false;
}

size_t CrcBlockCopyBatch::deviceMemoryBytesFor(size_t, size_t, size_t) {
    return 0;
}

size_t CrcBlockCopyBatch::deviceMemoryBytes() const {
    return 0;
}

size_t CrcBlockCopyBatch::hostMemoryBytes() const {
    return 0;
}

CrcBlockCopyBatch::CrcBlockCopyBatch(int, size_t, size_t, size_t, std::shared_ptr<CrcDumpWriter>) {
    throw std::runtime_error("CRC block copy requires CUDA 13 x86 or ARM");
}

CrcBlockCopyBatch::~CrcBlockCopyBatch() = default;

CrcCopyState CrcBlockCopyBatch::store(const std::vector<CrcCopyItem>&, std::vector<CrcCopyFailure>&) {
    return CrcCopyState::DEVICE_ERROR;
}

CrcCopyState CrcBlockCopyBatch::load(const std::vector<CrcCopyItem>&, std::vector<CrcCopyFailure>&) {
    return CrcCopyState::DEVICE_ERROR;
}

CrcCopyState CrcBlockCopyBatch::trySubmit(const std::vector<CrcCopyItem>&, CrcCopyDirection) {
    return CrcCopyState::DEVICE_ERROR;
}

CrcCopyState CrcBlockCopyBatch::poll() {
    throw std::runtime_error("CRC async copy requires CUDA 13 x86 or ARM");
}

const std::vector<CrcCopyFailure>& CrcBlockCopyBatch::failures() const {
    throw std::runtime_error("CRC async copy requires CUDA 13 x86 or ARM");
}

CrcCopyState CrcBlockCopyBatch::finish() {
    return CrcCopyState::DEVICE_ERROR;
}

void CrcBlockCopyBatch::setTestHook(std::function<int(const char*, void*, void*)>) {}

}  // namespace rtp_llm
