#include "rtp_llm/cpp/normal_engine/speculative/V4SmokeWitness.h"
#include <array>

// Test the exact on-wire prefix, grammar collision and verify rejection boundaries.
int main() {
    namespace w = rtp_llm::v4_smoke_witness;
    static_assert(sizeof(w::kLong) / sizeof(int32_t) == 55);
    static_assert(sizeof(w::kShort) / sizeof(int32_t) == 21);
    if (w::slot(21, 21, w::kShort, w::Grammar::Object, false) != 18
        || w::slot(21, 21, w::kShort, w::Grammar::Weather, false) != 21
        || w::slot(21, 21, w::kShort, w::Grammar::Other, false) != -1
        || w::slot(21, 21, w::kShort, w::Grammar::Weather, true) != 21)
        return 1;
    if (w::slot(18, 55, w::kLong, w::Grammar::Object, false) != -1
        || w::slot(18, 55, w::kLong, w::Grammar::Weather, true) != -1
        || w::slot(18, 54, w::kLong, w::Grammar::Object, true) != -1
        || w::slot(18, 55, w::kLong, w::Grammar::Object, true) != 17)
        return 2;
    std::array<int32_t, 55> prefix;
    std::copy(w::kLong, w::kLong + 55, prefix.begin());
    prefix.back()++;
    if (w::slot(18, 55, prefix.data(), w::Grammar::Object, true) != 17)
        return 3;
    prefix.front()++;
    if (w::slot(18, 55, prefix.data(), w::Grammar::Object, true) != -1)
        return 6;
    for (int col = 0; col < 5; ++col) {
        for (int n = 0; n <= 5; ++n)
            if (w::emitted(col, n, 5) != (col < n))
                return 4;
    }
    if (w::emitted(-1, 1, 5) || w::emitted(0, 6, 5) || w::emitted(5, 5, 5))
        return 5;
}
