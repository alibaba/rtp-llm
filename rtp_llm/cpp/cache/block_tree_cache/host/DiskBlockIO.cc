#include "rtp_llm/cpp/cache/block_tree_cache/host/DiskBlockIO.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sstream>
#include <unistd.h>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm::block_tree_cache {
namespace {

constexpr size_t kDirectIOAlignment = 4096;

bool isAligned(uint64_t value, size_t alignment) {
    return value % alignment == 0;
}

}  // namespace

PosixDiskBlockIO::~PosixDiskBlockIO() {
    close();
}

DiskBlockIOStatus PosixDiskBlockIO::openAndPreallocate(const std::string& file_path, size_t bytes, bool buffered_io) {
    close();

    if (bytes == 0) {
        RTP_LLM_LOG_ERROR("openAndPreallocate rejected zero-byte file, file=%s", file_path.c_str());
        return DiskBlockIOStatus::INVALID_SIZE;
    }

    file_path_   = file_path;
    bytes_       = bytes;
    buffered_io_ = buffered_io;

    int flags = O_CREAT | O_RDWR | O_CLOEXEC;
    if (!buffered_io_) {
#ifdef O_DIRECT
        flags |= O_DIRECT;
#else
        RTP_LLM_LOG_ERROR("O_DIRECT is not supported on this platform, file=%s", file_path.c_str());
        return DiskBlockIOStatus::IO_ERROR;
#endif
    }

    fd_ = ::open(file_path.c_str(), flags, 0644);
    if (fd_ < 0) {
        RTP_LLM_LOG_ERROR("open disk block file failed, file=%s, error=%s", file_path.c_str(), std::strerror(errno));
        return DiskBlockIOStatus::IO_ERROR;
    }

    int rc = ::posix_fallocate(fd_, 0, static_cast<off_t>(bytes));
    if (rc == EOPNOTSUPP || rc == ENOSYS) {
        RTP_LLM_LOG_WARNING(
            "posix_fallocate unsupported, falling back to ftruncate, file=%s, bytes=%zu, error=%s",
            file_path.c_str(),
            bytes,
            std::strerror(rc));
        if (::ftruncate(fd_, static_cast<off_t>(bytes)) != 0) {
            RTP_LLM_LOG_ERROR(
                "ftruncate disk block file failed, file=%s, bytes=%zu, error=%s",
                file_path.c_str(),
                bytes,
                std::strerror(errno));
            close();
            return DiskBlockIOStatus::IO_ERROR;
        }
        return DiskBlockIOStatus::OK;
    }
    if (rc != 0) {
        RTP_LLM_LOG_ERROR("posix_fallocate disk block file failed, file=%s, bytes=%zu, error=%s",
                          file_path.c_str(),
                          bytes,
                          std::strerror(rc));
        close();
        return DiskBlockIOStatus::IO_ERROR;
    }
    return DiskBlockIOStatus::OK;
}

DiskBlockIOStatus PosixDiskBlockIO::validate(uint64_t offset, const void* buffer, size_t bytes) const {
    if (buffer == nullptr || bytes == 0) {
        RTP_LLM_LOG_ERROR("disk block io rejected invalid size/buffer, file=%s, offset=%lu, bytes=%zu",
                          file_path_.c_str(),
                          offset,
                          bytes);
        return DiskBlockIOStatus::INVALID_SIZE;
    }

    // Alignment is derived purely from the requested I/O mode and the call
    // parameters, never from the fd or from whether open() actually managed to get
    // O_DIRECT semantics from the underlying filesystem. This keeps the check
    // deterministic across filesystems that reject O_DIRECT outright.
    if (!buffered_io_) {
        const auto addr = reinterpret_cast<uintptr_t>(buffer);
        if (!isAligned(offset, kDirectIOAlignment) || !isAligned(addr, kDirectIOAlignment)
            || !isAligned(bytes, kDirectIOAlignment)) {
            RTP_LLM_LOG_ERROR("direct disk io alignment failed, file=%s, offset=%lu, addr=%p, bytes=%zu",
                              file_path_.c_str(),
                              offset,
                              buffer,
                              bytes);
            return DiskBlockIOStatus::ALIGNMENT_ERROR;
        }
    }

    if (offset > bytes_ || bytes > bytes_ - offset) {
        RTP_LLM_LOG_ERROR("disk block io out of preallocated range, file=%s, offset=%lu, bytes=%zu, capacity=%zu",
                          file_path_.c_str(),
                          offset,
                          bytes,
                          bytes_);
        return DiskBlockIOStatus::INVALID_SIZE;
    }

    return DiskBlockIOStatus::OK;
}

DiskBlockIOStatus PosixDiskBlockIO::read(uint64_t offset, void* dst, size_t bytes) {
    const auto validation = validate(offset, dst, bytes);
    if (validation != DiskBlockIOStatus::OK) {
        return validation;
    }
    if (fd_ < 0) {
        RTP_LLM_LOG_ERROR("disk block io read on closed file, file=%s", file_path_.c_str());
        return DiskBlockIOStatus::IO_ERROR;
    }

    size_t done = 0;
    while (done < bytes) {
        const auto rc = ::pread(fd_, static_cast<char*>(dst) + done, bytes - done, static_cast<off_t>(offset + done));
        if (rc < 0) {
            if (errno == EINTR) {
                continue;
            }
            RTP_LLM_LOG_ERROR("pread disk block file failed, file=%s, offset=%lu, bytes=%zu, done=%zu, error=%s",
                              file_path_.c_str(),
                              offset,
                              bytes,
                              done,
                              std::strerror(errno));
            return DiskBlockIOStatus::IO_ERROR;
        }
        if (rc == 0) {
            // EOF before satisfying the full request.
            RTP_LLM_LOG_ERROR("pread disk block file got EOF, file=%s, offset=%lu, bytes=%zu, done=%zu",
                              file_path_.c_str(),
                              offset,
                              bytes,
                              done);
            return DiskBlockIOStatus::PARTIAL_FAILURE;
        }
        done += static_cast<size_t>(rc);
    }
    return DiskBlockIOStatus::OK;
}

DiskBlockIOStatus PosixDiskBlockIO::write(uint64_t offset, const void* src, size_t bytes) {
    const auto validation = validate(offset, src, bytes);
    if (validation != DiskBlockIOStatus::OK) {
        return validation;
    }
    if (fd_ < 0) {
        RTP_LLM_LOG_ERROR("disk block io write on closed file, file=%s", file_path_.c_str());
        return DiskBlockIOStatus::IO_ERROR;
    }

    size_t done = 0;
    while (done < bytes) {
        const auto rc =
            ::pwrite(fd_, static_cast<const char*>(src) + done, bytes - done, static_cast<off_t>(offset + done));
        if (rc < 0) {
            if (errno == EINTR) {
                continue;
            }
            RTP_LLM_LOG_ERROR("pwrite disk block file failed, file=%s, offset=%lu, bytes=%zu, done=%zu, error=%s",
                              file_path_.c_str(),
                              offset,
                              bytes,
                              done,
                              std::strerror(errno));
            return DiskBlockIOStatus::IO_ERROR;
        }
        if (rc == 0) {
            RTP_LLM_LOG_ERROR("pwrite disk block file made no progress, file=%s, offset=%lu, bytes=%zu, done=%zu",
                              file_path_.c_str(),
                              offset,
                              bytes,
                              done);
            return DiskBlockIOStatus::PARTIAL_FAILURE;
        }
        done += static_cast<size_t>(rc);
    }
    return DiskBlockIOStatus::OK;
}

DiskBlockIOStatus PosixDiskBlockIO::read(const std::vector<DiskRead>& reads) {
    for (const auto& item : reads) {
        const auto status = read(item.offset, item.buffer, item.bytes);
        if (status != DiskBlockIOStatus::OK) {
            return status;
        }
    }
    return DiskBlockIOStatus::OK;
}

DiskBlockIOStatus PosixDiskBlockIO::write(const std::vector<DiskWrite>& writes) {
    for (const auto& item : writes) {
        const auto status = write(item.offset, item.buffer, item.bytes);
        if (status != DiskBlockIOStatus::OK) {
            return status;
        }
    }
    return DiskBlockIOStatus::OK;
}

void PosixDiskBlockIO::close() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

std::string PosixDiskBlockIO::debugString() const {
    std::ostringstream oss;
    oss << "PosixDiskBlockIO{file=" << file_path_ << ", bytes=" << bytes_
        << ", io=" << (buffered_io_ ? "buffered" : "direct") << ", open=" << (fd_ >= 0) << "}";
    return oss.str();
}

}  // namespace rtp_llm::block_tree_cache
