#pragma once

// Implementation details shared with the CUDA backend's tests. These helpers
// do not extend CrcBlockCopyBatch's public interface or record format.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>
#include <nvcomp/crc32.h>

namespace rtp_llm {
namespace crc_block_copy_internal {

struct Result {
    uint32_t       expected;
    uint32_t       actual;
    nvcompStatus_t status;
};
static_assert(sizeof(Result) == 12, "CRC result transfer layout must remain unchanged");

// failed points to mapped pinned host memory through its device alias; all other
// arrays reside on the device. Read flags only after stream completion, and read
// results only for failed items. The caller owns all capacities and lifetimes.
// Tests use this production kernel to inject nvCOMP status errors.
cudaError_t launchFinishCrc(unsigned char*        staging,
                            size_t                stride,
                            const size_t*         lengths,
                            const uint32_t*       checksums,
                            const nvcompStatus_t* statuses,
                            Result*               results,
                            bool*                 failed,
                            size_t                count,
                            bool                  store,
                            cudaStream_t          stream);

}  // namespace crc_block_copy_internal
}  // namespace rtp_llm
