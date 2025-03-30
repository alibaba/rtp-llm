#pragma once
namespace fastertransformer {

enum class DeepGemmType {
    Normal,
    GroupedContiguous,
    GroupedMasked
};
}
