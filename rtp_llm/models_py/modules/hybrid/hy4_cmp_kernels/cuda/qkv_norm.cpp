#include <torch/extension.h>
#include <cmath>
#include <vector>
#include <string>
#include "qkv_norm.cuh"

void qkv_norm_quant(const at::Tensor& x, const at::Tensor& qw, const at::Tensor& kw,
                    const at::Tensor& q8, const at::Tensor& scale,
                    const std::optional<at::Tensor>& qb, const at::Tensor& kb,
                    int qr, int kr, double qe, double ke, bool pdl, std::uint64_t event,
                        const std::optional<at::Tensor>& cache, const std::optional<at::Tensor>& slots,
                        const std::optional<at::Tensor>& positions, const std::optional<at::Tensor>& cos_sin,
                        const std::string& release) {
    TORCH_CHECK(release=="late" || release=="early" || release=="after_wait", "release must be late, early or after_wait");
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2,
                "qkv must be CUDA BF16 matrix");
    const auto m = x.size(0);
    TORCH_CHECK(m > 0 && m <= 256 && qr >= 128 && qr <= 4096 && (qr & (qr-1)) == 0
                && kr >= 128 && kr <= 1024 && (kr & (kr-1)) == 0, "unsupported dimensions");
    TORCH_CHECK(x.size(1) >= qr+kr && x.stride(1) == 1 && x.stride(0) >= x.size(1), "invalid input strides");
    TORCH_CHECK(std::isfinite(qe) && qe > 0 && std::isfinite(ke) && ke > 0
                && std::isfinite(float(qe)) && float(qe) > 0
                && std::isfinite(float(ke)) && float(ke) > 0, "invalid epsilon");
    auto check = [&](const at::Tensor& t, at::ScalarType dtype) {
        TORCH_CHECK(t.device() == x.device() && t.scalar_type() == dtype, "device/dtype mismatch");
    };
    check(qw, at::kBFloat16); check(kw, at::kBFloat16);
    TORCH_CHECK(qw.dim() == 1 && qw.numel() == qr && qw.is_contiguous()
                && kw.dim() == 1 && kw.numel() == kr && kw.is_contiguous(), "gamma shape/layout mismatch");
    auto matrix = [&](const at::Tensor& t, at::ScalarType dtype, int width) {
        check(t, dtype);
        TORCH_CHECK(t.dim() == 2 && t.size(0) == m && t.size(1) == width && t.is_contiguous(),
                    "output shape/layout mismatch");
    };
    matrix(q8, at::ScalarType::Float8_e4m3fn, qr); matrix(kb, at::kBFloat16, kr);
    if (qb) matrix(*qb, at::kBFloat16, qr);
    check(scale, at::kInt);
    TORCH_CHECK(scale.dim() == 2 && scale.size(0) == m && scale.size(1) == qr/128
                && scale.stride(0) == 1 && scale.stride(1) >= m && scale.stride(1)%4 == 0,
                "scale must be column-major, TMA-aligned packed int32");
    std::vector<at::Tensor> writes = {q8, scale, kb};
    if (qb) writes.push_back(*qb);
    std::vector<at::Tensor> reads = {x, qw, kw};
    TORCH_CHECK(bool(cache)==bool(slots) && bool(cache)==bool(positions) && bool(cache)==bool(cos_sin),
                "all cache metadata must be supplied together");
    if(cache) {
        TORCH_CHECK(qr==2048 && kr==512 && x.size(1)>=qr+kr+64,"cache fusion requires HY4 2048/512/64");
        check(*cache,at::kBFloat16); check(*slots,at::kLong);
        check(*positions,at::kInt); check(*cos_sin,at::kFloat);
        TORCH_CHECK((cache->dim()==2 || cache->dim()==3) && cache->size(-1)==576 && cache->stride(-1)==1
                    && cache->size(0)>0 && cache->stride(-2)>=576,"invalid BF16 MLA cache layout");
        if(cache->dim()==3) TORCH_CHECK(cache->size(1)>0 && cache->stride(0)>=cache->size(1)*cache->stride(1),
                                        "overlapping cache pages");
        TORCH_CHECK(slots->dim()==1 && slots->size(0)==m && slots->stride(0)>0 &&
                    positions->dim()==1 && positions->size(0)==m && positions->stride(0)>0,"invalid cache metadata layout");
        TORCH_CHECK(cos_sin->dim()==2 && cos_sin->size(0)>0 && cos_sin->size(1)==64 &&
                    cos_sin->stride(1)==1 && cos_sin->stride(0)>=64,"invalid cos_sin layout");
        writes.push_back(*cache); reads.push_back(*slots); reads.push_back(*positions); reads.push_back(*cos_sin);
    }
    auto overlaps = [](const at::Tensor& a, const at::Tensor& b) {
        auto end = [](const at::Tensor& t) {
            std::uintptr_t n = 1;
            for (int i=0; i<t.dim(); ++i) n += (t.size(i)-1)*t.stride(i);
            return reinterpret_cast<std::uintptr_t>(t.data_ptr()) + n*t.element_size();
        };
        return reinterpret_cast<std::uintptr_t>(a.data_ptr()) < end(b)
            && reinterpret_cast<std::uintptr_t>(b.data_ptr()) < end(a);
    };
    for (size_t i=0; i<writes.size(); ++i) {
        for (const auto& r: reads) TORCH_CHECK(!overlaps(writes[i], r), "input/output overlap");
        for (size_t j=i+1; j<writes.size(); ++j)
            TORCH_CHECK(!overlaps(writes[i], writes[j]), "output/output overlap");
    }
    hy4_qkv_norm_launch(x,qw,kw,q8,scale,qb,kb,qr,kr,float(qe),float(ke),pdl,event,cache,slots,positions,cos_sin,release=="early" ? 1 : (release=="after_wait" ? 2 : 0));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("qkv_norm_quant", &qkv_norm_quant); m.def("_test_copy", &qkv_test_copy); }
