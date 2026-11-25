#pragma once

#include <cstdint>

namespace rtp_llm {

enum class PrefillPhase : uint8_t {
    PREPARE  = 1,
    LAYER    = 2,
    GRAPH    = 3,
    TAIL     = 4,
    GPU_WAIT = 5,
    GPU_DONE = 6,
};

struct PrefillProgressSlot;

// One writer per rank. Only calls on the active executor thread publish progress.
class PrefillProgressScope {
public:
    explicit PrefillProgressScope(bool eligible) noexcept;
    ~PrefillProgressScope();

    PrefillProgressScope(const PrefillProgressScope&)            = delete;
    PrefillProgressScope& operator=(const PrefillProgressScope&) = delete;

    static void initialize(const char* slot_path) noexcept;
    static bool enabled() noexcept;
    static bool active() noexcept;
    static void mark(PrefillPhase phase) noexcept;
    static void markLayer(int layer_index) noexcept;

private:
    void publish(PrefillPhase phase, uint64_t layer = 0) noexcept;

    PrefillProgressSlot* slot_           = nullptr;
    uint64_t             thread_id_bits_ = 0;
};

}  // namespace rtp_llm
