#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/TransferValidation.h"

#include <iostream>
#include <limits>

// This test deliberately has no Torch, RPC or device dependency so the wire
// bounds can also be checked before a target runtime is provisioned.
int main() {
    using rtp_llm::transfer::tcp::isRepresentableBlockSize;
    using rtp_llm::transfer::tcp::matchesBlockPayload;
    const size_t maximum = std::numeric_limits<uint32_t>::max();
    struct Case {
        const char* name;
        size_t allocation;
        uint32_t declared;
        size_t actual;
        bool valid;
    };
    const Case cases[] = {
        {"empty", 0, 0, 0, true},
        {"complete", 64, 64, 64, true},
        {"truncated", 64, 64, 63, false},
        {"missing payload", 64, 64, 0, false},
        {"oversized payload", 64, 64, 65, false},
        {"short declared length", 64, 63, 64, false},
        {"long declared length", 64, 65, 64, false},
        {"unexpected nonempty payload", 0, 0, 1, false},
        {"uint32 boundary", maximum, static_cast<uint32_t>(maximum), maximum, true},
    };
    for (const auto& test : cases) {
        if (matchesBlockPayload(test.allocation, test.declared, test.actual) != test.valid) {
            std::cerr << "FAILED: " << test.name << '\n';
            return 1;
        }
    }
    if (!isRepresentableBlockSize(maximum)) {
        return 1;
    }
    if (std::numeric_limits<size_t>::max() > maximum) {
        // These comparisons must not narrow a 4 GiB+ allocation to uint32.
        if (isRepresentableBlockSize(maximum + 1)
            || matchesBlockPayload(maximum + 1, 0, 0)
            || matchesBlockPayload(maximum + 65, 64, 64)) {
            std::cerr << "FAILED: uint32 wraparound\n";
            return 1;
        }
    }
    std::cout << "Transfer payload bounds and uint32 wraparound checks passed\n";
    return 0;
}
