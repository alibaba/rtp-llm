#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rtp_llm::block_tree_cache {

enum class DiskBlockIOStatus {
    OK,
    INVALID_SIZE,
    ALIGNMENT_ERROR,
    IO_ERROR,
    PARTIAL_FAILURE,
};

struct DiskRead {
    uint64_t offset;
    void*    buffer;
    size_t   bytes;
};

struct DiskWrite {
    uint64_t    offset;
    const void* buffer;
    size_t      bytes;
};

// DiskBlockIO is a thin, block-pool-agnostic abstraction over POSIX file I/O for a
// single backing disk file. It owns no scheduling or block-allocation policy -
// DiskBlockPool (task 5) is responsible for mapping block indices to file offsets and
// driving this interface.
class DiskBlockIO {
public:
    virtual ~DiskBlockIO() = default;

    // Opens (creating if necessary) the file at file_path and preallocates it to
    // exactly `bytes`. When buffered_io is false, the file is opened for direct I/O
    // and all subsequent read/write offsets, buffer addresses, and sizes must be
    // 4096-byte aligned.
    virtual DiskBlockIOStatus openAndPreallocate(const std::string& file_path, size_t bytes, bool buffered_io) = 0;

    virtual DiskBlockIOStatus read(uint64_t offset, void* dst, size_t bytes)        = 0;
    virtual DiskBlockIOStatus write(uint64_t offset, const void* src, size_t bytes) = 0;

    // Batch helpers. The first version simply loops the single-item call and returns
    // immediately on the first non-OK status, leaving any remaining requests unissued.
    virtual DiskBlockIOStatus read(const std::vector<DiskRead>& reads)    = 0;
    virtual DiskBlockIOStatus write(const std::vector<DiskWrite>& writes) = 0;

    virtual void close() = 0;

    virtual std::string debugString() const = 0;
};

// PosixDiskBlockIO implements DiskBlockIO on top of open/posix_fallocate/pread/pwrite.
class PosixDiskBlockIO: public DiskBlockIO {
public:
    PosixDiskBlockIO() = default;
    ~PosixDiskBlockIO() override;

    DiskBlockIOStatus openAndPreallocate(const std::string& file_path, size_t bytes, bool buffered_io) override;

    DiskBlockIOStatus read(uint64_t offset, void* dst, size_t bytes) override;
    DiskBlockIOStatus write(uint64_t offset, const void* src, size_t bytes) override;

    DiskBlockIOStatus read(const std::vector<DiskRead>& reads) override;
    DiskBlockIOStatus write(const std::vector<DiskWrite>& writes) override;

    void close() override;

    std::string debugString() const override;

private:
    // Validates offset/buffer/bytes against direct-I/O alignment and against the
    // preallocated file size. This is purely parameter- and state-based (it never
    // touches the fd), so it behaves identically whether or not the underlying
    // open() call actually succeeded in getting O_DIRECT semantics from the
    // filesystem.
    DiskBlockIOStatus validate(uint64_t offset, const void* buffer, size_t bytes) const;

    int         fd_{-1};
    std::string file_path_;
    size_t      bytes_{0};
    bool        buffered_io_{true};
};

}  // namespace rtp_llm::block_tree_cache
