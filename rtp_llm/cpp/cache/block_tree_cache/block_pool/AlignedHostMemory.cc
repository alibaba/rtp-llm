#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/AlignedHostMemory.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>

#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <linux/memfd.h>

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostMemoryRegistration.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {
namespace {

enum class HostPinMode {
    ALLOCATOR,
    REGISTER,
    MEMFD_REGISTER
};

HostPinMode hostPinMode() {
    const char* value = std::getenv("RTP_LLM_HOST_BLOCK_POOL_PIN_MODE");
    if (value == nullptr || std::string(value) == "allocator") {
        return HostPinMode::ALLOCATOR;
    }
    if (std::string(value) == "register") {
        return HostPinMode::REGISTER;
    }
    if (std::string(value) == "memfd_register") {
        return HostPinMode::MEMFD_REGISTER;
    }
    throw std::invalid_argument(std::string("invalid RTP_LLM_HOST_BLOCK_POOL_PIN_MODE='") + value
                                + "', expected 'register', 'memfd_register', or 'allocator'");
}

void releaseRegisteredCpuMapping(void* ptr, size_t size_bytes) noexcept {
    if (ptr == nullptr) {
        return;
    }
    std::string error_message;
    if (!unregisterHostMemory(ptr, error_message)) {
        return;
    }
    (void)munmap(ptr, size_bytes);
}

torch::Tensor allocateRegisteredCpuTensor(size_t size_bytes, bool use_memfd, int* shared_memory_fd) {
    RTP_LLM_CHECK(shared_memory_fd != nullptr);
    *shared_memory_fd = -1;
    int fd            = -1;
    if (use_memfd) {
        fd = static_cast<int>(::syscall(SYS_memfd_create, "rtp_llm_block_tree_host_pool", MFD_CLOEXEC));
        if (fd < 0) {
            throw std::runtime_error(std::string("memfd_create failed: ") + std::strerror(errno));
        }
        if (::ftruncate(fd, static_cast<off_t>(size_bytes)) != 0) {
            const int saved_errno = errno;
            (void)::close(fd);
            throw std::runtime_error(std::string("memfd ftruncate failed: ") + std::strerror(saved_errno));
        }
    }
    void* ptr = mmap(
        nullptr, size_bytes, PROT_READ | PROT_WRITE, use_memfd ? MAP_SHARED : (MAP_PRIVATE | MAP_ANONYMOUS), fd, 0);
    if (ptr == MAP_FAILED) {
        if (fd >= 0) {
            (void)::close(fd);
        }
        throw std::runtime_error(std::string("host pool mmap failed: ") + std::strerror(errno));
    }
    std::string error_message;
    if (!registerHostMemory(ptr, size_bytes, error_message)) {
        (void)munmap(ptr, size_bytes);
        if (fd >= 0) {
            (void)::close(fd);
        }
        throw std::runtime_error("host memory registration failed: " + error_message);
    }
    try {
        auto tensor = torch::from_blob(
            ptr,
            {static_cast<int64_t>(size_bytes)},
            [size_bytes](void* registered_ptr) { releaseRegisteredCpuMapping(registered_ptr, size_bytes); },
            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
        *shared_memory_fd = fd;
        return tensor;
    } catch (...) {
        releaseRegisteredCpuMapping(ptr, size_bytes);
        if (fd >= 0) {
            (void)::close(fd);
        }
        throw;
    }
}

}  // namespace

AlignedHostMemory::AlignedHostMemory(size_t usable_bytes, size_t alignment, const std::string& allocation_name) {
    backing_bytes_ = usable_bytes + alignment;
    try {
        const auto mode = hostPinMode();
        if (mode == HostPinMode::ALLOCATOR) {
            backing_ =
                torch::empty({static_cast<int64_t>(backing_bytes_)},
                             torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true));
        } else {
            if (!hostMemoryRegistrationSupported()) {
                throw std::runtime_error("registered host block pool is not supported by this device backend");
            }
            backing_ =
                allocateRegisteredCpuTensor(backing_bytes_, mode == HostPinMode::MEMFD_REGISTER, &shared_memory_fd_);
            registered_ = true;
        }
    } catch (const std::exception& e) {
        RTP_LLM_FAIL("allocate pinned host memory failed, allocation=%s usable_bytes=%zu error=%s",
                     allocation_name.c_str(),
                     usable_bytes,
                     e.what());
    }
    RTP_LLM_CHECK_WITH_INFO(registered_ || backing_.is_pinned(),
                            "host allocation [%s] must use pinned CPU memory",
                            allocation_name.c_str());

    const auto raw_base = reinterpret_cast<uintptr_t>(backing_.data_ptr<uint8_t>());
    data_               = reinterpret_cast<uint8_t*>((raw_base + alignment - 1) / alignment * alignment);
}

AlignedHostMemory::~AlignedHostMemory() {
    backing_ = torch::Tensor();
    if (shared_memory_fd_ >= 0) {
        (void)::close(shared_memory_fd_);
    }
}

uint8_t* AlignedHostMemory::data() const {
    return data_;
}

bool AlignedHostMemory::isRegistered() const {
    return registered_;
}

size_t AlignedHostMemory::backingBytes() const {
    return backing_bytes_;
}

}  // namespace rtp_llm
