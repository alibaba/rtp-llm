#include "enabled.h"

namespace fastertransformer {
namespace kernels {
bool isWeightOnlyBatchedGemvEnabled(WeightOnlyQuantType qtype)
{
    const int arch = getSMVersion();
    if (qtype == WeightOnlyQuantType::Int4b)
    {
        return isEnabledForArch<cutlass::uint4b_t>(arch);
    }
    else if (qtype == WeightOnlyQuantType::Int8b)
    {
        return isEnabledForArch<uint8_t>(arch);
    }
    else
    {
        FT_CHECK_WITH_INFO(false, "Unsupported WeightOnlyQuantType");
        return false;
    }
}

}
}