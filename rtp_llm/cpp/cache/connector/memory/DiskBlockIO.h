#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace rtp_llm {

class IDiskBlockIO {
public:
    virtual ~IDiskBlockIO() = default;

    virtual bool        openAndPreallocate(const std::string& file_path, size_t bytes, bool buffered_io) = 0;
    virtual bool        read(uint64_t offset, void* dst, size_t bytes)                                   = 0;
    virtual bool        write(uint64_t offset, const void* src, size_t bytes)                            = 0;
    virtual void        close()                                                                          = 0;
    virtual std::string debugString() const                                                              = 0;
};

class PosixDiskBlockIO: public IDiskBlockIO {
public:
    PosixDiskBlockIO() = default;
    ~PosixDiskBlockIO() override;

    bool        openAndPreallocate(const std::string& file_path, size_t bytes, bool buffered_io) override;
    bool        read(uint64_t offset, void* dst, size_t bytes) override;
    bool        write(uint64_t offset, const void* src, size_t bytes) override;
    void        close() override;
    std::string debugString() const override;

private:
    bool checkDirectIOAlignment(uint64_t offset, const void* buffer, size_t bytes) const;

private:
    int         fd_{-1};
    std::string file_path_;
    size_t      bytes_{0};
    bool        buffered_io_{true};
};

}  // namespace rtp_llm
