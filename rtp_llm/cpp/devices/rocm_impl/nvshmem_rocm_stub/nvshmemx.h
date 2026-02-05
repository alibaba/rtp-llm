/*
 * Stub nvshmemx.h for ROCm + deep_ep builds.
 * configs_hip.cuh includes <nvshmemx.h>, which would pull in system NVSHMEM
 * device headers (__syncwarp redefinition, nvshmemi_threadgroup_sync errors).
 * This stub is used instead so no system nvshmemx/nvshmem device code is included.
 */
#ifndef NVSHMEM_ROCm_STUB_NVSHMEMX_H_
#define NVSHMEM_ROCm_STUB_NVSHMEMX_H_

#ifdef __HIP_PLATFORM_AMD__

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  char data[128];
} nvshmemx_uniqueid_v1;
typedef nvshmemx_uniqueid_v1 nvshmemx_uniqueid_t;

#ifdef __cplusplus
}
#endif

#else

#include_next <nvshmemx.h>

#endif /* __HIP_PLATFORM_AMD__ */

#endif /* NVSHMEM_ROCm_STUB_NVSHMEMX_H_ */
