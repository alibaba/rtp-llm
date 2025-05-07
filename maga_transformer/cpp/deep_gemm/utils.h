#pragma once
namespace rtp_llm {

enum class DeepGemmType {
    Normal,
    GroupedContiguous,
    GroupedMasked
};
}
