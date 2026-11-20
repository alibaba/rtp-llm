#pragma once

namespace deep_jit {

// The pointer itself is the kernel argument storage; do not take its address.
struct NoRefPtr {
    void* ptr = nullptr;
};

}  // namespace deep_jit
