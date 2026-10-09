#pragma once

#include <stdexcept>

namespace rtp_llm {

// Only translated after Python's collective preflight, before model cache writes.
class V41InputRejected: public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

}  // namespace rtp_llm
