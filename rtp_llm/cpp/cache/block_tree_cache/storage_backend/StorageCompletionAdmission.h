#pragma once

namespace rtp_llm {

// Task-owned admission lifetime only; the original task teardown releases it.
// Physical UNKNOWN custody belongs to the native FROZEN mechanism separately.
class StorageCompletionAdmission {
public:
    virtual ~StorageCompletionAdmission() = default;
};

}  // namespace rtp_llm
