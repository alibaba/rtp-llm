#pragma once

namespace rtp_llm {

typedef enum memorytype_enum {
    MEMORY_CPU,
    MEMORY_CPU_PINNED,
    MEMORY_GPU
} MemoryType;

}  // namespace rtp_llm
