#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace rtp_llm {

// Stored in the last four bytes of the aligned block. Protects only the payload.
struct CrcBlockFooter {
    uint32_t crc32c{0};
};
static_assert(sizeof(CrcBlockFooter) == 4);

struct CrcBlockCopyTile {
    void*  address{nullptr};
    size_t offset{0};
    size_t bytes{0};
    bool   is_cuda{true};  // CPU-backed fixed pools use DMA, never kernel host accesses.
};

struct CrcBlockCopyResult {
    enum class FailureStage {
        INPUT,
        SOURCE_CRC,
        CRC_COMPUTE
    };
    bool     success{false};
    uint32_t expected_crc{0};
    uint32_t actual_crc{0};
    // Failure-only observations; these do not add a stored checksum or a GPU buffer.
    FailureStage                  failure_stage{FailureStage::INPUT};
    std::string                   error_message;
    std::optional<CrcBlockFooter> checked_footer;
    std::optional<CrcBlockFooter> staging_footer;
    bool                          gpu_crc_observed{false};
    uint32_t                      gpu_crc_status{0};
    bool                          output_written{false};
    // Full staging payload, captured before a failed load can reach gather/scatter.
    std::vector<uint8_t> staging_snapshot;
};

// One instance is exclusively leased by one copy task. Allocation and CRC tuning
// happen in the constructor, never in the successful copy path.
class CrcBlockCopy {
public:
    // [packed payload][padding][CRC32C]. Allocation and transfer use the same size.
    static size_t storageBytes(size_t payload_bytes) {
        return (payload_bytes + sizeof(CrcBlockFooter) + 15) & ~size_t(15);
    }
    static size_t footerOffset(size_t payload_bytes) {
        return storageBytes(payload_bytes) - sizeof(CrcBlockFooter);
    }
    static bool supported();

    CrcBlockCopy(const std::vector<size_t>& payload_sizes, size_t max_tiles);
    ~CrcBlockCopy();
    CrcBlockCopy(const CrcBlockCopy&)            = delete;
    CrcBlockCopy& operator=(const CrcBlockCopy&) = delete;

    // gather queues all pool copies on this workspace's stream. store seals the
    // CRC, performs D2H and waits for completion. Inheritance first uses load.
    // Execution methods return INPUT on invalid arguments instead of throwing.
    CrcBlockCopyResult
    gather(size_t payload_bytes, const std::vector<CrcBlockCopyTile>& tiles, bool preserve_payload = false);
    CrcBlockCopyResult store(void*                        host_block,
                             size_t                       payload_bytes,
                             bool                         host_is_pinned  = true,
                             const std::function<bool()>& capture_failure = {});

    // load validates the copied GPU payload against its CRC and returns that decision to
    // the CPU. The caller invokes scatter only after a successful result.
    CrcBlockCopyResult loadAndValidate(const void*                  host_block,
                                       size_t                       payload_bytes,
                                       bool                         host_is_pinned  = true,
                                       const std::function<bool()>& capture_failure = {});
    CrcBlockCopyResult scatter(size_t payload_bytes, const std::vector<CrcBlockCopyTile>& tiles);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rtp_llm
