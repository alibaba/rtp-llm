#include "rtp_llm/cpp/cache/connector/memory/DiskBlockIO.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sstream>
#include <unistd.h>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

constexpr size_t kDirectIOAlignment = 4096;

}  // namespace

PosixDiskBlockIO::~PosixDiskBlockIO() {
    close();
}

bool PosixDiskBlockIO::openAndPreallocate(const std::string& file_path, size_t bytes, bool buffered_io) {
    close();
    file_path_   = file_path;
    bytes_       = bytes;
    buffered_io_ = buffered_io;

    int flags = O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC;
    if (!buffered_io_) {
#ifdef O_DIRECT
        flags |= O_DIRECT;
#else
        RTP_LLM_LOG_ERROR("O_DIRECT is not supported on this platform");
        return false;
#endif
    }

    fd_ = ::open(file_path.c_str(), flags, 0600);
    if (fd_ < 0) {
        RTP_LLM_LOG_ERROR("open disk kv file failed, file=%s, error=%s", file_path.c_str(), std::strerror(errno));
        return false;
    }

    const int rc = ::posix_fallocate(fd_, 0, static_cast<off_t>(bytes));
    if (rc != 0) {
        RTP_LLM_LOG_ERROR("posix_fallocate disk kv file failed, file=%s, bytes=%zu, error=%s",
                          file_path.c_str(),
                          bytes,
                          std::strerror(rc));
        close();
        return false;
    }
    return true;
}

bool PosixDiskBlockIO::checkDirectIOAlignment(uint64_t offset, const void* buffer, size_t bytes) const {
    if (buffered_io_) {
        return true;
    }
    const auto addr = reinterpret_cast<uintptr_t>(buffer);
    if (offset % kDirectIOAlignment != 0 || addr % kDirectIOAlignment != 0 || bytes % kDirectIOAlignment != 0) {
        RTP_LLM_LOG_ERROR("direct disk io alignment failed, file=%s, offset=%lu, addr=%p, bytes=%zu",
                          file_path_.c_str(),
                          offset,
                          buffer,
                          bytes);
        return false;
    }
    return true;
}

bool PosixDiskBlockIO::read(uint64_t offset, void* dst, size_t bytes) {
    if (fd_ < 0 || dst == nullptr || offset + bytes > bytes_ || !checkDirectIOAlignment(offset, dst, bytes)) {
        return false;
    }
    size_t done = 0;
    while (done < bytes) {
        const auto rc = ::pread(fd_, static_cast<char*>(dst) + done, bytes - done, static_cast<off_t>(offset + done));
        if (rc < 0) {
            if (errno == EINTR) {
                continue;
            }
            RTP_LLM_LOG_ERROR("pread disk kv file failed, file=%s, offset=%lu, bytes=%zu, done=%zu, error=%s",
                              file_path_.c_str(),
                              offset,
                              bytes,
                              done,
                              std::strerror(errno));
            return false;
        }
        if (rc == 0) {
            RTP_LLM_LOG_ERROR("pread disk kv file got EOF, file=%s, offset=%lu, bytes=%zu, done=%zu",
                              file_path_.c_str(),
                              offset,
                              bytes,
                              done);
            return false;
        }
        done += static_cast<size_t>(rc);
    }
    return true;
}

bool PosixDiskBlockIO::write(uint64_t offset, const void* src, size_t bytes) {
    if (fd_ < 0 || src == nullptr || offset + bytes > bytes_ || !checkDirectIOAlignment(offset, src, bytes)) {
        return false;
    }
    size_t done = 0;
    while (done < bytes) {
        const auto rc =
            ::pwrite(fd_, static_cast<const char*>(src) + done, bytes - done, static_cast<off_t>(offset + done));
        if (rc < 0) {
            if (errno == EINTR) {
                continue;
            }
            RTP_LLM_LOG_ERROR("pwrite disk kv file failed, file=%s, offset=%lu, bytes=%zu, done=%zu, error=%s",
                              file_path_.c_str(),
                              offset,
                              bytes,
                              done,
                              std::strerror(errno));
            return false;
        }
        if (rc == 0) {
            RTP_LLM_LOG_ERROR("pwrite disk kv file made no progress, file=%s, offset=%lu, bytes=%zu, done=%zu",
                              file_path_.c_str(),
                              offset,
                              bytes,
                              done);
            return false;
        }
        done += static_cast<size_t>(rc);
    }
    return true;
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
        << ", io=" << (buffered_io_ ? "buffered" : "direct") << "}";
    return oss.str();
}

}  // namespace rtp_llm
