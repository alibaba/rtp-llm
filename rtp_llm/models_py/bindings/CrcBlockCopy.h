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
    // Retained for status/dump compatibility; the current CUDA kernel has no
    // separate compute-error channel and reports mismatches as CRC_MISMATCH.
    CRC_COMPUTE_ERROR,
    DEVICE_ERROR
};

struct CrcCopyFailure {
    size_t        item_index;
    uint32_t      expected;
    uint32_t      actual;
    CrcCopyStatus status{CrcCopyStatus::CRC_MISMATCH};
    // Historical nvCOMP status field, kept for dump-format compatibility; the
    // current kernel always records nvcompSuccess (0).
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
// load additionally requires device target spans across items to be disjoint:
// two items aliasing the same target would race in the scatter instead of
// failing. store only reads device sources, so aliasing source spans are safe.
// max_payload_bytes must be <= UINT32_MAX: the kernel's byte counters are
// 32-bit, so larger capacities are rejected at construction.
// Host storage must have encodedBytes(payload) capacity.
// Every pointer must remain live until the method returns. Calls are serialized.
// The caller must order the workspace after any producer of the device source
// data (the workspace streams are non-blocking).
// If CUDA completion cannot be established, the implementation aborts rather
// than returning while submitted work may still own caller pointers.
// The workspace must not be destroyed while a call is in progress; teardown
// only drains best-effort and reports (does not abort on) a CUDA failure.
// A post-scatter DEVICE_ERROR can lose the per-item verdict; treat the whole
// batch as uncommitted in that case.
// A failed store may have written host bytes: callers must not publish them.
// A failed load may have scattered corrupted bytes into device tiles: callers
// discard failed items through the reported status instead of adopting them.
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
    // Explicit pinned-host allocations owned by this workspace (pack buffers, fold tables, aliases,
    // results and per-item failure flags); excludes CUDA/allocator internal memory.
    size_t hostMemoryBytes() const;

    // Clears failures on entry, then records per-item CRC mismatches.
    CrcCopyStatus store(const std::vector<CrcCopyItem>& items, std::vector<CrcCopyFailure>& failures);
    CrcCopyStatus load(const std::vector<CrcCopyItem>& items, std::vector<CrcCopyFailure>& failures);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rtp_llm
