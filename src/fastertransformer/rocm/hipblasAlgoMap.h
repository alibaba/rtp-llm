#pragma once

#include "rocm/include/hipblaslt/hipblaslt.h"
#include "absl/container/node_hash_map.h"

#include <string>
#include <utility>

namespace fastertransformer {
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

    friend bool operator==(const hipblasLtAlgoConfig& a, const hipblasLtAlgoConfig& b) {
        return std::memcmp(&a, &b, sizeof a) == 0;
    }

    template<typename H>
    friend H AbslHashValue(H h, const hipblasLtAlgoConfig& c) {
        return H::combine_contiguous(std::move(h), reinterpret_cast<const char*>(&c), sizeof c);
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
                                       const int32_t              batch_count);
};

}  // namespace rocm
}  // namespace fastertransformer
