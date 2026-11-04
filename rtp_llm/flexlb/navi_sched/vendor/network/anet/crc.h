#pragma once
/** @file crc.h
 * Define the interface of crc checksum
 * This file derives from apsara/common/apsaraheader/crc.h
 */


#include <stdint.h>

/// define namespace
namespace anet {

typedef uint32_t (*DoCrc32cImpl)(uint32_t, const uint8_t *, uint64_t);

bool IsSSE42Supported();
DoCrc32cImpl GetDoCrc32cImpl();

const char *GetCrcFuncStr(void);

// OPERATIONS
/** Calculate checksum for an input byte stream,
 * result is stored in checksum pool
 *
 * @param initCrc Initial checksum value
 * @param data Input data to calculate checksum
 * @param length Length of input data
 *
 * @return Calculated checksum
 */
inline uint32_t DoCrc32c(uint32_t initCrc, const uint8_t *data, uint64_t length) {
    static DoCrc32cImpl doCrc32c = GetDoCrc32cImpl();
    return doCrc32c(initCrc, data, length);
}

uint32_t DoCrc32c_Lookup(uint32_t initCrc, const uint8_t *data, uint64_t length);

uint32_t DoCrc32c_Intel(uint32_t initCrc, const uint8_t *data, uint64_t length);

} // namespace anet

