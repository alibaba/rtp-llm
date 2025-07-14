#pragma once

#include "rocm/include/hipblaslt/hipblaslt.h"
#include "absl/container/node_hash_map.h"

#include <string>
#include <utility>

namespace rtp_llm {
namespace rocm {

template<typename T, hipblasStatus_t (*Destroy)(T)>
struct Deleter {
    void operator()(T ptr) const {
        (void)Destroy(ptr);
    }
};

template<typename T, hipblasStatus_t (*Destroy)(T)>
using Ptr = std::unique_ptr<std::remove_pointer_t<T>, Deleter<T, Destroy>>;

struct hipblasLtMatmulInfo {
    hipblasLtMatmulAlgo_t                                      algo;
    Ptr<hipblasLtMatmulDesc_t, hipblasLtMatmulDescDestroy>     opDesc;
    Ptr<hipblasLtMatrixLayout_t, hipblasLtMatrixLayoutDestroy> ADesc, BDesc, CDesc;
};

struct hipblasLtAlgoConfig {
    hipblasOperation_t   trans_a;
    hipblasOperation_t   trans_b;
    int32_t              m;
    int32_t              n;
    int32_t              k;
    hipDataType          A_data_type;
    int32_t              lda;
    int64_t              stride_a;
    hipDataType          B_data_type;
    int32_t              ldb;
    int64_t              stride_b;
    hipDataType          C_data_type;
    int32_t              ldc;
    int64_t              stride_c;
    hipblasComputeType_t compute_type;
    int32_t              batch_count;
    hipblasLtEpilogue_t  epilogue;

    friend bool operator==(const hipblasLtAlgoConfig& a, const hipblasLtAlgoConfig& b) {
        return a.trans_a == b.trans_a && a.trans_b == b.trans_b && a.m == b.m && a.n == b.n && a.k == b.k
               && a.A_data_type == b.A_data_type && a.lda == b.lda && a.stride_a == b.stride_a
               && a.B_data_type == b.B_data_type && a.ldb == b.ldb && a.stride_b == b.stride_b
               && a.C_data_type == b.C_data_type && a.ldc == b.ldc && a.stride_c == b.stride_c
               && a.compute_type == b.compute_type && a.batch_count == b.batch_count && a.epilogue == b.epilogue;
    }

    template<typename H>
    friend H AbslHashValue(H h, const hipblasLtAlgoConfig& c) {
        return H::combine(std::move(h),
                          c.trans_a,
                          c.trans_b,
                          c.m,
                          c.n,
                          c.k,
                          c.A_data_type,
                          c.lda,
                          c.stride_a,
                          c.B_data_type,
                          c.ldb,
                          c.stride_b,
                          c.C_data_type,
                          c.ldc,
                          c.stride_c,
                          c.compute_type,
                          c.batch_count);
    }
};

class hipblasAlgoMap {
private:
    absl::node_hash_map<hipblasLtAlgoConfig, hipblasLtMatmulInfo> algo_map_;

public:
    void loadGemmConfig(const std::string& filename, hipblasLtHandle_t handle);

    const hipblasLtMatmulInfo* getAlgo(const hipblasOperation_t   trans_a,
                                       const hipblasOperation_t   trans_b,
                                       const int32_t              m,
                                       const int32_t              n,
                                       const int32_t              k,
                                       const hipDataType          A_data_type,
                                       const int32_t              lda,
                                       const int64_t              stride_a,
                                       const hipDataType          B_data_type,
                                       const int32_t              ldb,
                                       const int64_t              stride_b,
                                       const hipDataType          C_data_type,
                                       const int32_t              ldc,
                                       const int64_t              stride_c,
                                       const hipblasComputeType_t compute_type,
                                       const int32_t              batch_count,
                                       const hipblasLtEpilogue_t  epilogue);
};

}  // namespace rocm
}  // namespace rtp_llm
