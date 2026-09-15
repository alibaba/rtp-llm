#include "rtp_llm/models_py/bindings/CrcBlockCopy.h"
#include <stdexcept>

namespace rtp_llm {
struct CrcBlockCopy::Impl {};
size_t CrcBlockCopy::footerOffset(size_t bytes) {
    return (bytes + 15) & ~size_t(15);
}
size_t CrcBlockCopy::transferBytes(size_t bytes) {
    return footerOffset(bytes) + sizeof(CrcBlockFooter);
}
size_t CrcBlockCopy::storageBytes(size_t bytes) {
    return (transferBytes(bytes) + sizeof(CrcBlockHostMetadata) + 15) & ~size_t(15);
}
bool CrcBlockCopy::supported() {
    return false;
}
CrcBlockCopy::CrcBlockCopy(const std::vector<size_t>&, size_t) {
    throw std::runtime_error("memory cache block CRC requires the CUDA 13 x86 backend");
}
CrcBlockCopy::~CrcBlockCopy() = default;
void CrcBlockCopy::gather(size_t, const std::vector<CrcBlockCopyTile>&, bool) {
    throw std::runtime_error("memory cache block CRC backend unavailable");
}
CrcBlockCopyResult CrcBlockCopy::store(void*, size_t, int64_t, bool, const std::function<bool()>&) {
    throw std::runtime_error("memory cache block CRC backend unavailable");
}
CrcBlockCopyResult CrcBlockCopy::loadAndValidate(const void*, size_t, bool, const std::function<bool()>&) {
    throw std::runtime_error("memory cache block CRC backend unavailable");
}
void CrcBlockCopy::scatter(size_t, const std::vector<CrcBlockCopyTile>&) {
    throw std::runtime_error("memory cache block CRC backend unavailable");
}
}  // namespace rtp_llm
