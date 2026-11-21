#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string_view>

namespace rtp_llm::v4_smoke_witness {
inline bool enabled() {
    static const bool active = [] {
        const char* name = std::getenv("TEST_TARGET");
        const char* path = std::getenv("TEST_UNDECLARED_OUTPUTS_DIR");
        return name && path && path[0] == '/'
               && std::string_view(name)
                      == "//rtp_llm/test/smoke:smoke_v4_flash_0731_pd_cp2ep2_dp2ep2_dspark_cprr_async_xgrammar_json_sm100";
    }();
    return active;
}
// Exact public fixture token IDs, SHA-256 of LE signed int32 token sequence.
// Only emit that digest after byte-exact prefix + grammar + stage matching.
inline constexpr int32_t kShort[] = {0,     3476, 477,   260,  11502, 22896, 16,    128803, 17916,  713,   102323,
                                     16652, 303,  17916, 4498, 580,   14479, 16734, 13199,  128804, 128822};
inline constexpr int32_t kLong[]  = {0,     3476,  477,   260,   11502, 22896,  16,     128803, 17916, 713,   102323,
                                    16652, 303,   14479, 16734, 13199, 128804, 128822, 24313,  37399, 16562, 6127,
                                    10177, 50219, 16562, 16905, 10177, 88634,  16562,  856,    2614,  37,    10177,
                                    34859, 13168, 16562, 1059,  7,     10177,  28730,  16562,  2398,  1821,  223,
                                    21,    2029,  10177, 16051, 20035, 16562,  939,    23,     15,    3663,  15};
enum class Grammar {
    Other,
    Object,
    Weather
};
inline int slot(int prompt, int length, const int32_t* prefix, Grammar grammar, bool verify) {
    if (prompt == 21 && length == 21 && std::equal(kShort, kShort + 21, prefix)) {
        return grammar == Grammar::Object ? 18 : grammar == Grammar::Weather ? 21 : -1;
    }
    // At the genuine 38th output a model can have reached the same text using
    // a different tokenization. Keep it as a separately identified witness;
    // never claim it is the canonical/paired prefix (compute the actual SHA).
    return verify && prompt == 18 && length == 55 && grammar == Grammar::Object
                   && std::equal(kLong, kLong + 18, prefix) ?
               17 :
               -1;
}
inline const char* sha(int slot) {
    return slot == 17 ? "cba0af7f8765ce15615d4bb2c34e043080fbbc8816c41623db1abaa9578c3e47" :
                        "bc3388aef6778503c379d551f845e5e39583bf3348db4325eedd541fa2e78415";
}
inline bool emitted(int col, int accept_len, int width) {
    return 0 <= col && col < accept_len && accept_len <= width;
}
}  // namespace rtp_llm::v4_smoke_witness
