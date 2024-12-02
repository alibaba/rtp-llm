#include "maga_transformer/cpp/disaggregate/cache_store/MemoryUtil.h"

#include <cstring>
#include <cstdlib>

namespace rtp_llm {

MemoryUtil::MemoryUtil(std::unique_ptr<MemoryUtil> impl): instance_(std::move(impl)) {}

bool MemoryUtil::isRdmaMode() {
    throw std::runtime_error("not implemeted");
}

MemoryUtil& MemoryUtil::getInstance() {
    throw std::runtime_error("not implemeted");
}

void* MemoryUtil::mallocCPU(size_t size) {
    throw std::runtime_error("not implemeted");
}

void MemoryUtil::freeCPU(void* ptr) {
    throw std::runtime_error("not implemeted");
}

void* MemoryUtil::mallocGPU(size_t size) {
    throw std::runtime_error("not implemeted");
}

void MemoryUtil::freeGPU(void* ptr) {
    throw std::runtime_error("not implemeted");
}

bool MemoryUtil::regUserMr(void* buf, uint64_t size, bool gpu) {
    throw std::runtime_error("not implemeted");
}

bool MemoryUtil::deregUserMr(void* buf, bool gpu) {
    throw std::runtime_error("not implemeted");
}

bool MemoryUtil::isMemoryMr(void* ptr, uint64_t size, bool gpu, bool adopted) {
    throw std::runtime_error("not implemeted");
}

bool MemoryUtil::findMemoryMr(void* mem_info, void* buf, uint64_t size, bool gpu, bool adopted) {
    throw std::runtime_error("not implemeted");
}

void MemoryUtil::memsetCPU(void* ptr, int value, size_t len) {
    throw std::runtime_error("not implemeted");
}

bool MemoryUtil::memsetGPU(void* ptr, int value, size_t len) {
    throw std::runtime_error("not implemeted");
}

bool MemoryUtil::memcopy(void* dst, bool dst_gpu, const void* src, bool src_gpu, size_t size) {
    throw std::runtime_error("not implemeted");
}

bool MemoryUtil::gpuEventBarrier(void* event) {
    throw std::runtime_error("not implemeted");
}

}  // namespace rtp_llm
