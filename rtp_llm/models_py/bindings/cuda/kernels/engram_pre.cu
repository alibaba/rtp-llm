#include "rtp_llm/models_py/bindings/cuda/kernels/engram/launch.cuh"
#include "rtp_llm/models_py/bindings/cuda/kernels/engram/impls/pre.cuh"
namespace rtp_llm {
namespace {
template<int Splits>
void preLaunch(int                                m,
               const CUtensorMap&                 ma,
               const CUtensorMap&                 mb,
               const CUtensorMap&                 md,
               const at::Tensor&                  h,
               const at::Tensor&                  previous,
               const at::Tensor&                  partial,
               const at::Tensor&                  sqr,
               const at::Tensor&                  protocol,
               engram::layout::mega_mhc::MixArgs  mix,
               engram::layout::mega_mhc::NormArgs norm) {
    engram::launch(engram::engram_pre_kernel<Splits>,
                   512,
                   205440,
                   1,
                   dim3(148, 1, 1),
                   uint32_t(m),
                   ma,
                   mb,
                   md,
                   reinterpret_cast<const nv_bfloat16*>(h.data_ptr()),
                   previous.data_ptr<float>(),
                   partial.data_ptr<float>(),
                   sqr.data_ptr<float>(),
                   reinterpret_cast<unsigned long long*>(protocol.data_ptr<int64_t>()),
                   mix,
                   norm);
}
}  // namespace
void engramPre(const at::Tensor& h,
               const at::Tensor& previous,
               const at::Tensor& fn,
               const at::Tensor& scales,
               const at::Tensor& bases,
               const at::Tensor& weight,
               const at::Tensor& partial,
               const at::Tensor& sqr,
               const at::Tensor& protocol,
               const at::Tensor& pre,
               const at::Tensor& post,
               const at::Tensor& comb,
               const at::Tensor& y,
               const at::Tensor& q,
               const at::Tensor& sf,
               double            hc_eps,
               double            mix_eps,
               int64_t           iters,
               double            norm_eps) {
    const c10::cuda::CUDAGuard guard(h.device());
    const int                  m = engram::rows(h), splits = std::min(148 / ((m + 63) / 64), 80);
    using namespace engram;
    check(h, h, at::kBFloat16, {m, 4, 5120});
    check(previous, h, at::kFloat, {m, 4, 1});
    check(fn, h, at::kFloat, {24, 20480});
    check(scales, h, at::kFloat, {3});
    check(bases, h, at::kFloat, {24});
    check(weight, h, at::kBFloat16, {5120});
    check(partial, h, at::kFloat, {splits, m, 24});
    check(sqr, h, at::kFloat, {splits, m});
    check(protocol, h, at::kLong, {156});
    check(pre, h, at::kFloat, {m, 4, 1});
    check(post, h, at::kFloat, {m, 4, 1});
    check(comb, h, at::kFloat, {m, 4, 4});
    check(y, h, at::kBFloat16, {m, 5120});
    check(q, h, at::kFloat8_e4m3fn, {m, 5120});
    checkSf(sf, h, m, 40);
    TORCH_CHECK(iters >= 1 && iters <= 128 && hc_eps > 0 && mix_eps > 0 && norm_eps > 0,
                "Invalid Engram normalization configuration");
    auto ma = tensorMap(h, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 20480, m, 64, 64, 20480, 128);
    auto mb = tensorMap(fn, CU_TENSOR_MAP_DATA_TYPE_TFLOAT32, 20480, 24, 64, 32, 20480, 128);
    auto md = tensorMap(partial, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 24, m, 32, 64, 24, 128, splits, m * 24);
    const layout::mega_mhc::MixArgs  mix{scales.data_ptr<float>(),
                                         bases.data_ptr<float>(),
                                         pre.data_ptr<float>(),
                                         post.data_ptr<float>(),
                                         comb.data_ptr<float>(),
                                         float(hc_eps),
                                         float(mix_eps),
                                         2.f,
                                         float(mix_eps),
                                         uint32_t(iters)};
    const layout::mega_mhc::NormArgs norm{uint32_t(m),
                                          reinterpret_cast<const nv_bfloat16*>(weight.data_ptr()),
                                          reinterpret_cast<const nv_bfloat16*>(h.data_ptr()),
                                          float(norm_eps),
                                          1.f,
                                          reinterpret_cast<nv_bfloat16*>(y.data_ptr()),
                                          reinterpret_cast<__nv_fp8_e4m3*>(q.data_ptr()),
                                          reinterpret_cast<uint32_t*>(sf.data_ptr<int32_t>()),
                                          sf.stride(0),
                                          sf.stride(1),
                                          nullptr,
                                          0};
    switch (splits) {
#define RTP_ENGRAM_SPLIT(S)                                                                                            \
    case S:                                                                                                            \
        return preLaunch<S>(m, ma, mb, md, h, previous, partial, sqr, protocol, mix, norm)
        RTP_ENGRAM_SPLIT(80);
        RTP_ENGRAM_SPLIT(74);
        RTP_ENGRAM_SPLIT(49);
        RTP_ENGRAM_SPLIT(37);
        RTP_ENGRAM_SPLIT(29);
        RTP_ENGRAM_SPLIT(24);
        RTP_ENGRAM_SPLIT(21);
        RTP_ENGRAM_SPLIT(18);
#undef RTP_ENGRAM_SPLIT
    }
    TORCH_CHECK(false, "Unsupported Engram split configuration");
}
}  // namespace rtp_llm
