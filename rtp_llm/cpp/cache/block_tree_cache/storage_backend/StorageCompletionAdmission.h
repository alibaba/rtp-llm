#pragma once

#include <cstddef>

namespace rtp_llm {

// Task-owned admission lifetime only; the original task teardown releases it.
// Physical UNKNOWN custody belongs to the native FROZEN mechanism separately.
class StorageCompletionAdmission {
public:
    virtual ~StorageCompletionAdmission() = default;
    virtual void onQueued() noexcept = 0;
    virtual void onRunning() noexcept = 0;
    // A preparation task may have reserved space for all request pages.
    // Drop the excess only after returning the original request references.
    virtual void reduceTo(std::size_t) noexcept {}
};

}  // namespace rtp_llm
