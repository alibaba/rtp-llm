#pragma once

#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <vector>

namespace rtp_llm {

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
    std::string   dump_path;
    CrcCopyStatus status{CrcCopyStatus::CRC_MISMATCH};
    int           nvcomp_status{0};
    // Populated only after capture is admitted. CPU is observed after the
    // verdict; GPU contains the exact staging record that was checked.
    std::vector<uint8_t> cpu_snapshot;
    std::vector<uint8_t> staging_snapshot;
    std::string          capture_error;
    bool                 checked_footer_available{true};
};

// Invoked only for failed items, before allocating/copying diagnostic snapshots.
// A service can reserve its dump budget here. An empty callback disables capture.
using CrcCopyCapture = std::function<bool(size_t item_index)>;

// Bounded synchronous CRC32C workspace. max_tiles limits the entire batch.
// store/load require ordered, gap-free tiles covering exactly payload_bytes.
// validate ignores tiles. Host storage must have encodedBytes(payload) capacity.
// Every pointer must remain live until the method returns. Calls are serialized.
// A failed store may have written host bytes: callers must not publish them.
// load verifies ALL items before scattering any item into its device tiles.
class CrcBlockCopyBatch {
public:
    CrcBlockCopyBatch(int device_index, size_t max_items, size_t max_payload_bytes, size_t max_tiles);
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

    CrcCopyStatus store(const std::vector<CrcCopyItem>& items,
                        std::vector<CrcCopyFailure>*    failures = nullptr,
                        const CrcCopyCapture&           capture  = {});
    CrcCopyStatus load(const std::vector<CrcCopyItem>& items,
                       std::vector<CrcCopyFailure>*    failures = nullptr,
                       const CrcCopyCapture&           capture  = {});
    CrcCopyStatus validate(const std::vector<CrcCopyItem>& items,
                           std::vector<CrcCopyFailure>*    failures = nullptr,
                           const CrcCopyCapture&           capture  = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rtp_llm
