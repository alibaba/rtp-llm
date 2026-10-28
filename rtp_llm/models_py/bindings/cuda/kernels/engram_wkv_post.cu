#include "rtp_llm/models_py/bindings/cuda/kernels/engram/launch.cuh"
#include "rtp_llm/models_py/bindings/cuda/kernels/engram/impls/wkv_post.cuh"
namespace rtp_llm {
namespace {
template<int BM, int BN, int Stages, int ClusterM, int ClusterN, bool Swap>
void wkv(const at::Tensor&             a,
         const at::Tensor&             sf,
         const at::Tensor&             weight,
         const at::Tensor&             weight_sf,
         const at::Tensor&             kv,
         const engram::FrontArguments& post) {
    using namespace engram;
    constexpr int LoadM = BM / ClusterN, LoadN = BN / ClusterM;
    constexpr int StoreM = Swap ? 16 : BM;
    constexpr int SFM = (BM + 127) / 128 * 128, SFN = (BN + 127) / 128 * 128;
    // Preserve DeepGEMM's selected launch footprint as well as its tile/stages.
    constexpr int Extra  = StoreM * (Swap ? BN * 2 : 128) * 2 + 32 * 8 * 3 + 2 * 8 * 3 + 8 + 4;
    constexpr int Smem   = Extra + Stages * ((LoadM + LoadN) * 128 + (SFM + SFN) * 4);
    const int     m      = a.size(0);
    auto          ma     = tensorMap(a, CU_TENSOR_MAP_DATA_TYPE_UINT8, 6144, m, 128, LoadM, 6144, 128);
    auto          mb     = tensorMap(weight, CU_TENSOR_MAP_DATA_TYPE_UINT8, 6144, 25600, 128, LoadN, 6144, 128);
    auto          msa    = tensorMap(sf, CU_TENSOR_MAP_DATA_TYPE_INT32, (m + 3) / 4 * 4, 48, SFM, 1, sf.stride(1));
    auto          msb    = tensorMap(weight_sf, CU_TENSOR_MAP_DATA_TYPE_INT32, 25600, 48, SFN, 1, weight_sf.stride(1));
    auto          md     = tensorMap(kv, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 25600, m, BN, StoreM, 25600, 128);
    auto          kernel = engram_wkv_post_kernel<cute::UMMA::Major::K,
                                                  cute::UMMA::Major::K,
                                                  32,
                                                  32,
                                                  32,
                                                  0,
                                                  25600,
                                                  6144,
                                                  BM,
                                                  BN,
                                                  128,
                                                  1,
                                                  128,
                                                  128,
                                                  128,
                                                  Stages,
                                                  2,
                                                  128,
                                                  128,
                                                  ClusterM * ClusterN,
                                                  (ClusterN > 1),
                                                  148,
                                                  Swap,
                                                  true,
                                                  GemmType::Normal,
                                                  false,
                                                  cutlass::float_e4m3_t,
                                                  cutlass::float_e4m3_t,
                                                  cutlass::bfloat16_t,
                                                  epilogue::operators::Identity>;
    launch(kernel,
           288,
           Smem,
           ClusterM * ClusterN,
           dim3(148, 1, 1),
           static_cast<int*>(nullptr),
           uint32_t(m),
           uint32_t(25600),
           uint32_t(6144),
           EpilogueOperatorArgs{},
           ma,
           mb,
           msa,
           msb,
           md,
           post);
}
}  // namespace
void engramWkvPost(const at::Tensor& a,
                   const at::Tensor& sf,
                   const at::Tensor& weight,
                   const at::Tensor& weight_sf,
                   const at::Tensor& x,
                   const at::Tensor& residual,
                   const at::Tensor& post,
                   const at::Tensor& comb,
                   const at::Tensor& kv,
                   const at::Tensor& hp) {
    const c10::cuda::CUDAGuard guard(a.device());
    const int                  m = engram::rows(a);
    using namespace engram;
    check(a, a, at::kFloat8_e4m3fn, {m, 6144});
    checkSf(sf, a, m, 48);
    check(weight, a, at::kFloat8_e4m3fn, {25600, 6144});
    checkSf(weight_sf, a, 25600, 48);
    check(x, a, at::kBFloat16, {m, 5120});
    check(residual, a, at::kBFloat16, {m, 4, 5120});
    check(post, a, at::kFloat, {m, 4, 1});
    check(comb, a, at::kFloat, {m, 4, 4});
    check(kv, a, at::kBFloat16, {m, 25600});
    check(hp, a, at::kBFloat16, {m, 4, 5120});
    const FrontArguments args{
        x.data_ptr(), residual.data_ptr(), post.data_ptr<float>(), comb.data_ptr<float>(), hp.data_ptr()};
    if (m <= 32)
        return wkv<32, 192, 7, 1, 1, false>(a, sf, weight, weight_sf, kv, args);
    if (m <= 64)
        return wkv<64, 192, 6, 1, 1, false>(a, sf, weight, weight_sf, kv, args);
    if (m <= 128)
        return wkv<128, 192, 4, 1, 1, false>(a, sf, weight, weight_sf, kv, args);
    if (m <= 256)
        return wkv<128, 192, 6, 2, 1, false>(a, sf, weight, weight_sf, kv, args);
    if (m <= 288)
        return wkv<144, 128, 8, 1, 2, true>(a, sf, weight, weight_sf, kv, args);
    if (m <= 320)
        return wkv<160, 128, 7, 1, 2, true>(a, sf, weight, weight_sf, kv, args);
    if (m <= 352)
        return wkv<176, 128, 7, 1, 2, true>(a, sf, weight, weight_sf, kv, args);
    if (m <= 384)
        return wkv<192, 128, 7, 1, 2, true>(a, sf, weight, weight_sf, kv, args);
    if (m <= 416)
        return wkv<208, 128, 7, 1, 2, true>(a, sf, weight, weight_sf, kv, args);
    if (m <= 448)
        return wkv<224, 128, 6, 1, 2, true>(a, sf, weight, weight_sf, kv, args);
    if (m <= 480)
        return wkv<240, 128, 6, 1, 2, true>(a, sf, weight, weight_sf, kv, args);
    return wkv<128, 256, 5, 2, 1, false>(a, sf, weight, weight_sf, kv, args);
}
}  // namespace rtp_llm
