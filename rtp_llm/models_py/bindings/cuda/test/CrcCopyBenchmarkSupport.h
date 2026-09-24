#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_runtime_api.h>

#include "rtp_llm/models_py/bindings/CrcBlockCopy.h"

namespace rtp_llm::crc_copy_benchmark {

// Owns the actual production DeviceHostCopyPlan values without exposing their
// transitive framework/Torch dependencies to the CUDA translation unit.
class FrameworkCopyPlan {
public:
    explicit FrameworkCopyPlan(const std::vector<CrcCopyItem>& items);
    ~FrameworkCopyPlan();
    FrameworkCopyPlan(const FrameworkCopyPlan&)            = delete;
    FrameworkCopyPlan& operator=(const FrameworkCopyPlan&) = delete;

    uintptr_t touchMetadata() const;

private:
    friend class FrameworkCopies;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class FrameworkCopies {
public:
    explicit FrameworkCopies(int device);
    ~FrameworkCopies();
    FrameworkCopies(const FrameworkCopies&)            = delete;
    FrameworkCopies& operator=(const FrameworkCopies&) = delete;

    void copyBatch(const FrameworkCopyPlan& plan, bool store);
    void copyStaged(const FrameworkCopyPlan& plan, bool store);

    // Production's thread-local stream is private. The experimental 3D path
    // uses a separate stream from the same nondefault Torch stream pool.
    cudaStream_t copy3dStream() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rtp_llm::crc_copy_benchmark
