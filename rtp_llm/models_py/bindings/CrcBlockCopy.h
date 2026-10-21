#pragma once

#include <cstddef>
#include <functional>
#include <map>
#include <limits>
#include <memory>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <vector>

namespace rtp_llm {

class CrcDumpWriter;
using CrcCopyMetadata = std::map<std::string, std::string>;

struct CrcCopyTile {
    void*  device{nullptr};
    size_t offset{0};
    size_t bytes{0};
};

struct CrcCopyItem {
    void*                    host{nullptr};
    size_t                   payload_bytes{0};
    size_t                   capacity_bytes{0};
    std::vector<CrcCopyTile> tiles;
    // Evaluated only when dumping a failure; referenced inputs live until copy returns.
    std::function<CrcCopyMetadata()> dump_metadata{};
};

enum class CrcCopyStatus {
    OK,
    INVALID_ARGS,
    CRC_MISMATCH,
    CRC_COMPUTE_ERROR,
    DEVICE_ERROR
};

struct CrcCopyFailure {
    size_t        item_index;
    uint32_t      expected;
    uint32_t      actual;
    CrcCopyStatus status{CrcCopyStatus::CRC_MISMATCH};
    int           nvcomp_status{0};
    // Populated only after capture is admitted. CPU is observed after the
    // verdict; GPU contains the exact staging record that was checked.
    std::vector<uint8_t> cpu_snapshot;
    std::vector<uint8_t> staging_snapshot;
    std::string          capture_error;
    bool                 checked_footer_available{true};
};

// The writer calls this synchronously while the failed workspace is still held.
using CrcSnapshotCapture = std::function<CrcCopyFailure()>;

// Bounded synchronous CRC32C workspace. max_tiles limits the entire batch.
// store/load require ordered, gap-free tiles covering exactly payload_bytes.
// Host storage must have encodedBytes(payload) capacity.
// Every pointer must remain live until the method returns. Calls are serialized.
// A failed store may have written host bytes: callers must not publish them.
// load verifies ALL items before scattering any item into its device tiles.
class CrcBlockCopyBatch {
public:
    CrcBlockCopyBatch(int                            device_index,
                      size_t                         max_items,
                      size_t                         max_payload_bytes,
                      size_t                         max_tiles,
                      std::shared_ptr<CrcDumpWriter> dump_writer = nullptr);
    ~CrcBlockCopyBatch();
    CrcBlockCopyBatch(const CrcBlockCopyBatch&)            = delete;
    CrcBlockCopyBatch& operator=(const CrcBlockCopyBatch&) = delete;

    static bool   available();
    static size_t encodedBytes(size_t payload_bytes) {
        if (payload_bytes > std::numeric_limits<size_t>::max() - 19) {
            throw std::overflow_error("CRC encoded block size overflow");
        }
        return (payload_bytes + 4 + 15) & ~size_t(15);
    }

    // Explicit GPU allocations owned by this workspace; excludes pinned host and CUDA/nvCOMP internal memory.
    size_t deviceMemoryBytes() const;

    CrcCopyStatus store(const std::vector<CrcCopyItem>& items,
                        std::vector<CrcCopyFailure>*    failures = nullptr);
    CrcCopyStatus load(const std::vector<CrcCopyItem>& items,
                       std::vector<CrcCopyFailure>*    failures = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rtp_llm
