#pragma once

#include <vector>
#include <utility>
#include <algorithm>

namespace rtp_llm {

template<typename T, typename Pred>
void vectorRemoveIf(std::vector<T>& vec, Pred&& pred) {
    vec.erase(std::remove_if(vec.begin(), vec.end(), std::forward<Pred>(pred)), vec.end());
}

}  // namespace rtp_llm