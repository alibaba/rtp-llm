/*
 * Stub nvshmem.h for ROCm + deep_ep builds.
 * The system /usr/include/nvshmem.h device headers redefine __syncwarp (already
 * provided by HIP) and use device APIs that don't match on ROCm. This stub
 * is used when compiling with enable_deep_ep on ROCm so that deep_ep's
 * configs_hip.cuh gets this file instead of the system one (by putting this
 * target first in the include path).
 */
#ifndef NVSHMEM_ROCm_STUB_NVSHMEM_H_
#define NVSHMEM_ROCm_STUB_NVSHMEM_H_

#ifdef __HIP_PLATFORM_AMD__

#include <stddef.h>
#include <stdint.h>
#include "nvshmemx.h"

/* Prevent deep_ep configs_hip.cuh from pulling in device headers that
 * redefine __syncwarp and cause nvshmemi_threadgroup_sync errors. */
#define NVSHMEM_DISABLE_DEVICE_FOR_ROCm 1

#else

#include_next <nvshmem.h>

#endif /* __HIP_PLATFORM_AMD__ */

#endif /* NVSHMEM_ROCm_STUB_NVSHMEM_H_ */
