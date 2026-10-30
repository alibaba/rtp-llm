#include "rtp_llm/cpp/cuda_graph/mtp_draft_prefill_capacity.h"

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
    using rtp_llm::selectMtpDraftPrefillTokenCapacity;

    // Every scheduling mode retains Q-wide rows. Small graphs must not use
    // the shared allocation's maximum request count.
    check(mtpDraftPrefillRequestCapacity(5, 16, 5, 1) == 1, "one Q-wide row must capture one request");
    check(mtpDraftPrefillRequestCapacity(20, 16, 5, 1) == 4, "request capacity must follow its token bucket");
    const std::vector<int> buckets = {5, 10, 15, 20};
    check(selectMtpDraftPrefillTokenCapacity(buckets, 5, 1, 16, 5, 1) == 5,
          "selection must use the smallest matching Q-wide bucket");
    check(selectMtpDraftPrefillTokenCapacity(buckets, 20, 4, 16, 5, 1) == 20,
          "selection must accept exact token and request boundaries");
    check(selectMtpDraftPrefillTokenCapacity(buckets, 5, 2, 16, 5, 1) == 10,
          "request capacity must also be sufficient");
    check(selectMtpDraftPrefillTokenCapacity(buckets, 85, 17, 16, 5, 1) == 0,
          "requests exceeding global capacity cannot select a graph");
    check(selectMtpDraftPrefillTokenCapacity(buckets, 25, 5, 16, 5, 1) == 0,
          "tokens exceeding all buckets cannot select a graph");
    check(selectMtpDraftPrefillTokenCapacity({5, 10}, 5, 3, 16, 5, 1) == 0,
          "requests exceeding every bucket cannot select a graph");
    check(selectMtpDraftPrefillTokenCapacity({}, 5, 1, 16, 5, 1) == 0,
          "empty capture range cannot select a graph");

    // Rectangular backends retain complete Q-wide rows whose total token
    // count is divisible by SP, including non-coprime Q / SP configurations.
    check(mtpDraftPrefillRequestCapacity(5, 16, 3, 1) == 2, "rectangular capacity must round up Q");
    check(mtpDraftPrefillRequestCapacity(5, 16, 3, 8) == 8, "rectangular capacity must align Q rows to SP");
    check(mtpDraftPrefillRequestCapacity(7, 16, 6, 8) == 4, "rectangular alignment must use gcd(SP, Q)");
    check(mtpDraftPrefillRequestCapacity(24, 16, 3, 8) == 8, "aligned rectangular capacity must stay exact");
    const std::vector<int> rectangular_buckets = {24, 48};
    check(selectMtpDraftPrefillTokenCapacity(rectangular_buckets, 24, 8, 16, 3, 8) == 24,
          "rectangular selection must accept exact boundaries");
    check(selectMtpDraftPrefillTokenCapacity(rectangular_buckets, 24, 9, 16, 3, 8) == 48,
          "rectangular selection must satisfy request capacity");
    check(selectMtpDraftPrefillTokenCapacity(rectangular_buckets, 25, 8, 16, 3, 8) == 48,
          "rectangular selection must satisfy token capacity");
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
    checkThrows<std::invalid_argument>([] { selectMtpDraftPrefillTokenCapacity({5}, 0, 1, 16, 3, 1); },
                                       "zero live tokens must be rejected");
    checkThrows<std::invalid_argument>([] { selectMtpDraftPrefillTokenCapacity({5}, 1, -1, 16, 3, 1); },
                                       "negative live B must be rejected");
    checkThrows<std::invalid_argument>([] { selectMtpDraftPrefillTokenCapacity({5}, 1, 1, 0, 3, 1); },
                                       "zero global B must be rejected");
    checkThrows<std::invalid_argument>([] { selectMtpDraftPrefillTokenCapacity({5}, 1, 1, 16, 0, 1); },
                                       "zero selector Q must be rejected");
    checkThrows<std::invalid_argument>([] { selectMtpDraftPrefillTokenCapacity({5}, 1, 1, 16, 3, 0); },
                                       "zero selector SP must be rejected");
    checkThrows<std::invalid_argument>([] { selectMtpDraftPrefillTokenCapacity({0}, 1, 1, 16, 3, 1); },
                                       "nonpositive buckets must be rejected");
    return 0;
}
