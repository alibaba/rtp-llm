#pragma once

// Implementation details shared with the CUDA backend's tests. These helpers
// do not extend CrcBlockCopyBatch's public interface or record format.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>
#include <nvcomp/crc32.h>

namespace rtp_llm {
namespace crc_block_copy_internal {

constexpr size_t kSegmentBytes      = 16 * 1024;
constexpr size_t kMinSegmentedItems = 16;

constexpr size_t segmentCount(size_t payload) noexcept {
    return payload / kSegmentBytes + (payload % kSegmentBytes != 0);
}

constexpr bool shouldSegment(size_t count, size_t largest) noexcept {
    return count >= kMinSegmentedItems && largest > kSegmentBytes;
}

struct BackingInfo {
    size_t payload_bytes;
    size_t first_segment;
    size_t segment_count;
    size_t matrix_offset;  // Offset in uint32_t elements, not bytes.
};

struct Result {
    uint32_t       expected;
    uint32_t       actual;
    nvcompStatus_t status;
};
static_assert(sizeof(Result) == 12, "CRC result transfer layout must remain unchanged");

// The caller provides 32 * segmentCount(payload) uint32_t elements. Columns
// are stored at out[bit * segmentCount(payload) + segment].
void makeSuffixMatrices(size_t payload, uint32_t* out);

// All array arguments point to device memory. The caller owns their capacities
// and lifetime, and must order access to the stream. This launches the actual
// production combine/seal kernel; tests use it to inject nvCOMP status errors.
cudaError_t launchCombineAndFinish(unsigned char*        staging,
                                   size_t                stride,
                                   const BackingInfo*    backings,
                                   const uint32_t*       matrices,
                                   const uint32_t*       checksums,
                                   const nvcompStatus_t* statuses,
                                   Result*               results,
                                   size_t                count,
                                   bool                  store,
                                   cudaStream_t          stream);

}  // namespace crc_block_copy_internal
}  // namespace rtp_llm
