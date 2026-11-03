#pragma once
namespace rtp_llm::engram {
struct FrontArguments {
    const void*  x;
    const void*  residual;
    const float* post;
    const float* comb;
    void*        hpost;
};
}  // namespace rtp_llm::engram
