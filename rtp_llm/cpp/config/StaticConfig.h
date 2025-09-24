#pragma once
#include <string>

namespace rtp_llm {

// these configs are used in static or global method.
struct StaticConfig {
    static bool user_ft_core_dump_on_exception;
    static bool user_disable_pdl;
};

}  // namespace rtp_llm
