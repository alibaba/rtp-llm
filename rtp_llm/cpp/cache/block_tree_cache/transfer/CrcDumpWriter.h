#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "rtp_llm/models_py/bindings/CrcBlockCopy.h"

namespace rtp_llm {

class GroupSet;
struct TransferDescriptor;

// One writer per rank serializes snapshot capture and file writes across slots.
// After a successful write it keeps the newest backup_count complete dumps.
class CrcDumpWriter {
public:
    using Metadata                       = CrcCopyMetadata;
    static constexpr size_t kBackupCount = 20;

    static std::shared_ptr<CrcDumpWriter> forRank(int64_t world_rank);
    static Metadata                       dumpMetadata(const TransferDescriptor& descriptor, const GroupSet& group);
    explicit CrcDumpWriter(int64_t     world_rank,
                           std::string root        = "logs/kv_cache_crc",
                           size_t      backup_count = kBackupCount);
    // Captures and writes under the same lock; diagnostic failures preserve the CRC verdict.
    std::string
    write(const CrcCopyItem& item, const CrcCopyFailure& failure, const CrcSnapshotCapture& capture_snapshot) noexcept;

private:
    void rotate();

    int64_t           world_rank_;
    std::string       rank_dir_;
    size_t            backup_count_;
    std::mutex        mutex_;
};

}  // namespace rtp_llm
