#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <vector>

#if !USING_ROCM
#error "cuda_graph_prepare_fill_rocm_test must be built with --config=rocm"
#endif

#include "rtp_llm/models_py/bindings/cuda/kernels/cuda_graph_prepare.h"

namespace rtp_llm {
namespace {

TEST(CudaGraphPrepareFillTest, FillsLiteralAndDeviceBasedIotaRegions) {
    std::vector<int32_t> initial(10, -1);
    int32_t              source_value = 100;
    int32_t*             output       = nullptr;
    int32_t*             source       = nullptr;
    ASSERT_EQ(hipSuccess, hipMalloc(&output, initial.size() * sizeof(int32_t)));
    ASSERT_EQ(hipSuccess, hipMalloc(&source, sizeof(int32_t)));
    ASSERT_EQ(hipSuccess,
              hipMemcpy(output, initial.data(), initial.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    ASSERT_EQ(hipSuccess, hipMemcpy(source, &source_value, sizeof(int32_t), hipMemcpyHostToDevice));

    CudaGraphPrepareFillParams params;
    auto&                      literal = params.regions[params.region_count++];
    literal.ptr                        = output + 2;
    literal.count                      = 4;
    literal.value                      = 10;
    literal.step                       = 3;

    auto& from_device        = params.regions[params.region_count++];
    from_device.ptr          = output + 6;
    from_device.value_ptr    = source;
    from_device.count        = 3;
    from_device.value_offset = 5;
    from_device.step         = 2;

    invokeCudaGraphPrepareFill(params, nullptr);

    std::vector<int32_t> actual(initial.size());
    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
    ASSERT_EQ(hipSuccess,
              hipMemcpy(actual.data(), output, actual.size() * sizeof(int32_t), hipMemcpyDeviceToHost));
    EXPECT_EQ(actual, (std::vector<int32_t>{-1, -1, 10, 13, 16, 19, 105, 107, 109, -1}));

    EXPECT_EQ(hipSuccess, hipFree(source));
    EXPECT_EQ(hipSuccess, hipFree(output));
}

}  // namespace
}  // namespace rtp_llm
