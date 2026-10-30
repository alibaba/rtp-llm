#include "rtp_llm/cpp/utils/math_utils.h"

#include <limits>
#include <stdexcept>

namespace {

void check(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template<typename Exception, typename Callable>
void checkThrows(Callable call, const char* message) {
    try {
        call();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(message);
}

}  // namespace

int main() {
    using rtp_llm::mtpDraftPrefillRequestCapacity;

    // Check minimal SP padding against an independent enumeration, including
    // the normalized token-bucket -> capture-request-count contract.
    for (int requests = 1; requests <= 64; ++requests) {
        for (int query_width = 1; query_width <= 16; ++query_width) {
            for (int sp = 1; sp <= 16; ++sp) {
                int expected = requests;
                while (expected * query_width % sp != 0) {
                    ++expected;
                }
                check(rtp_llm::padSequenceParallelRequests(requests, query_width, sp) == static_cast<size_t>(expected),
                      "request padding must be the smallest SP-compatible count");
                const int capacity = mtpDraftPrefillRequestCapacity(requests * query_width, 80, query_width, sp);
                check(capacity == expected, "token bucket must use the same SP request padding");
                const int tokens = capacity * query_width;
                check(mtpDraftPrefillRequestCapacity(tokens, 80, query_width, sp) == tokens / query_width,
                      "normalized token bucket must directly determine capture request count");
            }
        }
    }

    // Every scheduling mode retains Q-wide rows. Small graphs must not use
    // the shared allocation's maximum request count.
    check(mtpDraftPrefillRequestCapacity(5, 16, 5, 1) == 1, "one Q-wide row must capture one request");
    check(mtpDraftPrefillRequestCapacity(20, 16, 5, 1) == 4, "request capacity must follow its token bucket");

    // Rectangular backends retain complete Q-wide rows whose total token
    // count is divisible by SP, including non-coprime Q / SP configurations.
    check(mtpDraftPrefillRequestCapacity(5, 16, 3, 1) == 2, "rectangular capacity must round up Q");
    check(mtpDraftPrefillRequestCapacity(5, 16, 3, 8) == 8, "rectangular capacity must align Q rows to SP");
    check(mtpDraftPrefillRequestCapacity(7, 16, 6, 8) == 4, "rectangular alignment must use gcd(SP, Q)");
    check(mtpDraftPrefillRequestCapacity(24, 16, 3, 8) == 8, "aligned rectangular capacity must stay exact");
    checkThrows<std::invalid_argument>([] { mtpDraftPrefillRequestCapacity(7, 3, 6, 8); },
                                       "SP rounding beyond global B must be rejected");

    // Perform ceil, alignment and rectangular token products in wide
    // arithmetic, then reject capacities the runner's int keys cannot hold.
    constexpr int max_int = std::numeric_limits<int>::max();
    check(mtpDraftPrefillRequestCapacity(max_int, max_int, max_int, 1) == 1,
          "rectangular int boundary must remain representable");
    check(mtpDraftPrefillRequestCapacity(max_int - 1, max_int, 2, 1) == (max_int - 1) / 2,
          "largest even rectangular capacity must remain representable");
    checkThrows<std::overflow_error>([] { mtpDraftPrefillRequestCapacity(max_int, max_int, 2, 1); },
                                     "rectangular token product overflow must be rejected");
    checkThrows<std::overflow_error>([] { mtpDraftPrefillRequestCapacity(max_int, max_int, 1, max_int - 1); },
                                     "rectangular request rounding overflow must be rejected");

    checkThrows<std::invalid_argument>([] { mtpDraftPrefillRequestCapacity(0, 16, 3, 1); },
                                       "zero token capacity must be rejected");
    checkThrows<std::invalid_argument>([] { mtpDraftPrefillRequestCapacity(5, -1, 3, 1); },
                                       "negative global request capacity must be rejected");
    checkThrows<std::invalid_argument>([] { mtpDraftPrefillRequestCapacity(5, 16, 0, 1); },
                                       "zero query width must be rejected");
    checkThrows<std::invalid_argument>([] { mtpDraftPrefillRequestCapacity(5, 16, 3, -1); },
                                       "negative SP size must be rejected");
    return 0;
}
