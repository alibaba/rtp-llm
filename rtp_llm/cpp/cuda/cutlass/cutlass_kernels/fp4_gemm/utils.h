#pragma once

#include <cuda_runtime.h>

#define CHECK_CUDA_SUCCESS(cmd)                                         \
  do {                                                                  \
    cudaError_t e = cmd;                                                \
    if (e != cudaSuccess) {                                             \
      std::stringstream _message;                                       \
      auto s = cudaGetErrorString(e);                                   \
      _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__; \
      throw std::runtime_error(_message.str());                         \
    }                                                                   \
  } while (0)

inline int getSMVersion() {
  int device{-1};
  cudaGetDevice(&device);
  int sm_major = 0;
  int sm_minor = 0;
  cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device);
  return sm_major * 10 + sm_minor;
}

