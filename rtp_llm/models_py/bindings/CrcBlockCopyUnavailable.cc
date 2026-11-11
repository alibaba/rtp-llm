#include "rtp_llm/models_py/bindings/CrcBlockCopy.h"

namespace rtp_llm {

struct CrcCopyWorkspace::Impl {};

bool CrcCopyWorkspace::available() {
    return false;
}

CrcCopyMemoryUsage CrcCopyWorkspace::memoryUsage(size_t, size_t, size_t) {
    return {};
}

size_t CrcCopyWorkspace::deviceMemoryBytes() const {
    return 0;
}

size_t CrcCopyWorkspace::hostMemoryBytes() const {
    return 0;
}

CrcCopyWorkspace::CrcCopyWorkspace(int, size_t, size_t, size_t, std::shared_ptr<CrcDumpWriter>) {
    throw std::runtime_error("CRC block copy requires CUDA 13 x86 or ARM");
}

CrcCopyWorkspace::~CrcCopyWorkspace() = default;

CrcCopyState CrcCopyWorkspace::evict(const std::vector<CrcCopyItem>&, std::vector<CrcCopyFailure>&) {
    return CrcCopyState::DEVICE_ERROR;
}

CrcCopyState CrcCopyWorkspace::load(const std::vector<CrcCopyItem>&, std::vector<CrcCopyFailure>&) {
    return CrcCopyState::DEVICE_ERROR;
}

CrcCopyState CrcCopyWorkspace::trySubmit(const std::vector<CrcCopyItem>&, CrcCopyDirection) {
    return CrcCopyState::DEVICE_ERROR;
}

CrcCopyState CrcCopyWorkspace::poll() {
    throw std::runtime_error("CRC async copy requires CUDA 13 x86 or ARM");
}

const std::vector<CrcCopyFailure>& CrcCopyWorkspace::failures() const {
    throw std::runtime_error("CRC async copy requires CUDA 13 x86 or ARM");
}

CrcCopyState CrcCopyWorkspace::finish() {
    return CrcCopyState::DEVICE_ERROR;
}

void CrcCopyWorkspace::setTestHook(std::function<int(const char*, void*, void*)>) {}

}  // namespace rtp_llm
