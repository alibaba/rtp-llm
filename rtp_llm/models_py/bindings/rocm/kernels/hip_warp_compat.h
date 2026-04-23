// hip_warp_compat.h — Host-side warpSize compatibility for ROCm 7.x
//
// In ROCm 7.x, `warpSize` is a device-only builtin and cannot be used in host
// code. AMD GCN/CDNA architectures always have a warp (wavefront) size of 64.
// This header provides a host-side constant so that Triton-generated host
// launchers can reference warpSize without compiler errors.

#pragma once

#if !defined(__HIP_DEVICE_COMPILE__)
#define warpSize 64
#endif
