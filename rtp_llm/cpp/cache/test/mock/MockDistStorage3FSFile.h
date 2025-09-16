#pragma once

#include "gmock/gmock.h"

#include "rtp_llm/cpp/cache/DistStorage3FSFile.h"

namespace rtp_llm {
namespace threefs {

class MockDistStorage3FSFile: public DistStorage3FSFile {
public:
    MockDistStorage3FSFile(const ThreeFSFileConfig& config,
                           const ThreeFSIovHandle&  read_iov_handle,
                           const ThreeFSIovHandle&  write_iov_handle,
                           size_t                   read_timeout_ms  = 1000,
                           size_t                   write_timeout_ms = 2000):
        DistStorage3FSFile(config, read_iov_handle, write_iov_handle, read_timeout_ms, write_timeout_ms) {}
    ~MockDistStorage3FSFile() override = default;

public:
    MOCK_METHOD(bool, exists, (), (const, override));
    MOCK_METHOD(bool, open, (bool write), (override));
    MOCK_METHOD(bool, write, (const std::vector<DistStorage::Iov>& iovs), (override));
    MOCK_METHOD(bool, read, (const std::vector<DistStorage::Iov>& iovs), (override));
    MOCK_METHOD(bool, del, (), (override));
    MOCK_METHOD(bool, close, (), (override));
};

}  // namespace threefs
}  // namespace rtp_llm
