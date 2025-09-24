#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/CommonDefines.h"
#include "rtp_llm/cpp/devices/ShapeCheck.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "autil/StringUtil.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include <numeric>
#include <utility>

using namespace std;

namespace rtp_llm {

GroupedGemmOutput CudaDevice::groupedGemm(const GroupedGemmParams& params) {
    // ensure A B C has the same size and dtype and shape is invalide.
    params.check();
    if (!useGroupGemm()) {
        return DeviceBase::groupedGemm(params);
    }
    size_t num   = params.A.size();
    auto   dtype = params.A[0]->type();
    RTP_LLM_CHECK_WITH_INFO(
        (dtype == DataType::TYPE_FP16 || dtype == DataType::TYPE_BF16 || dtype == DataType::TYPE_FP32),
        "cuda group gemm only support half/bf16/fp32.");
    cuggemm_runner_->setup(dtype);
    std::vector<BufferPtr> output(num);
    std::vector<void*>     a_pointers(num);
    std::vector<void*>     b_pointers(num);
    std::vector<void*>     c_pointers(num);
    std::vector<int>       m_array(num);
    std::vector<int>       n_array(num);
    std::vector<int>       k_array(num);
    float                  alpha = params.alpha;
    float                  beta  = params.beta;

    for (int i = 0; i < num; i++) {
        size_t    m = params.A[i]->shape()[0];
        size_t    n = params.B[i]->shape()[1];
        size_t    k = params.A[i]->shape()[1];
        BufferPtr c = nullptr;
        if (params.C.has_value()) {
            c = params.C.value()[i];
        } else {
            c    = allocateBuffer({dtype, {m, n}, AllocationType::DEVICE});
            beta = 0.f;
        }
        output[i]     = (std::move(c));
        a_pointers[i] = params.A[i]->data();
        b_pointers[i] = params.B[i]->data();
        c_pointers[i] = output[i]->data();
        m_array[i]    = m;
        n_array[i]    = n;
        k_array[i]    = k;
    }
    cuggemm_runner_->groupGemm(a_pointers.data(),
                               b_pointers.data(),
                               c_pointers.data(),
                               m_array.data(),
                               n_array.data(),
                               k_array.data(),
                               alpha,
                               beta,
                               num);
    check_cuda_error();
    return GroupedGemmOutput({std::move(output)});
}

}  // namespace rtp_llm
