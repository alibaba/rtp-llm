#pragma once

namespace rtp_llm {

// Internal ownership annotation, not a transport or business API. A lease can
// outlive the executor when uncertain DMA has frozen its source pages.
class StorageCompletionAdmission {
public:
    virtual ~StorageCompletionAdmission() = default;
    virtual void onQueued() noexcept      = 0;
    virtual void onRunning() noexcept     = 0;
    virtual void onQuarantined() noexcept = 0;
};

}  // namespace rtp_llm
