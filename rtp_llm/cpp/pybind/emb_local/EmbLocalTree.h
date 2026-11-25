#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rtp_llm::emb_local {

// The cache keeps only native data for a published tree. A tensor leaf
// addresses one of the compact resident slices, including repeated aliases.
struct Tree {
    enum class Kind {
        None,
        Tensor,
        Tuple,
        List,
        Dict,
        Bool,
        Integer,
        Float,
        String,
        Bytes
    };

    Kind              kind    = Kind::None;
    int64_t           integer = 0;
    double            real    = 0;
    std::string       text;
    std::vector<Tree> items;
};

}  // namespace rtp_llm::emb_local
