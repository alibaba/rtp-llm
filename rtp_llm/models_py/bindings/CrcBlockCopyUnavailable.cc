#include "rtp_llm/models_py/bindings/CrcBlockCopy.h"
#include <stdexcept>

namespace rtp_llm {
struct CrcBlockCopy::Impl {};
bool CrcBlockCopy::supported() {
    return false;
}
CrcBlockCopy::CrcBlockCopy(const std::vector<size_t>&, size_t) {
    throw std::runtime_error("memory cache block CRC requires a CUDA 13 x86 or ARM backend");
}
CrcBlockCopy::~CrcBlockCopy() = default;
void CrcBlockCopy::gather(size_t, const std::vector<CrcBlockCopyTile>&, bool) {
    throw std::runtime_error("memory cache block CRC backend unavailable");
}
CrcBlockCopyResult CrcBlockCopy::store(void*, size_t, bool, const std::function<bool()>&) {
    throw std::runtime_error("memory cache block CRC backend unavailable");
}
CrcBlockCopyResult CrcBlockCopy::loadAndValidate(const void*, size_t, bool, const std::function<bool()>&) {
    throw std::runtime_error("memory cache block CRC backend unavailable");
}
void CrcBlockCopy::scatter(size_t, const std::vector<CrcBlockCopyTile>&) {
    throw std::runtime_error("memory cache block CRC backend unavailable");
}
}  // namespace rtp_llm
