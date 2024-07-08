#pragma once

#include "hip_utils.h"

#include <map>
#include <string>
#include <unordered_map>
#include <utility>

namespace fastertransformer {
namespace rocm {

#define GEMM_NUM 6
#define GEMM_CONFIG "gemm_config.in"
#define IGEMM_CONFIG "igemm_config.in"
#define SPGEMM_CONFIG "spgemm_config.in"
#define SPIGEMM_CONFIG "spigemm_config.in"

enum HipblasDataType {
    FLOAT_DATATYPE    = 0,
    HALF_DATATYPE     = 1,
    BFLOAT16_DATATYPE = 2,
    INT8_DATATYPE     = 3,
    FP8_DATATYPE      = 4
};

typedef struct {
    int   algoId, customOption, tile, splitK_val;
    int   swizzle, reductionScheme, workspaceSize;
    float exec_time;
} hipblasLtMatmulAlgo_info;

/* Structure to store information about different run trials */
typedef struct {
    hipblasLtMatmulAlgo_t algo;
    hipblasStatus_t       status;
    float                 time;
    size_t                workspaceSize;  // actual memory workspace needed
    int                   customOption;
    float                 wavesCount;
} customMatmulPerf_t;

struct hipblasAlgoConfig_t {
    int               batch_count;
    int               m;
    int               n;
    int               k;
    hipblasDatatype_t data_type;
    bool              operator==(hipblasAlgoConfig_t const& config) const {
        return (batch_count == config.batch_count) && (m == config.m) && (n == config.n) && (k == config.k)
               && (data_type == config.data_type);
    }
};

class hipblasAlgoConfig_hasher {
public:
    std::size_t operator()(hipblasAlgoConfig_t const& config) const {
        return config.batch_count * 98317ull ^ config.m * 49157ull ^ config.n * 24593ull ^ config.k * 196613ull
               ^ static_cast<int>(config.data_type) * 6151ull;
    }
};

class hipblasAlgoMap {
private:
    std::unordered_map<hipblasAlgoConfig_t, hipblasLtMatmulAlgo_info, hipblasAlgoConfig_hasher> algo_map_;
    std::string                                                                                 config_filename_;
    std::string                                                                                 sp_config_filename_;
    std::map<std::string, int>                                                                  sp_algo_map_;

public:
    explicit hipblasAlgoMap(const std::string filename, const std::string sp_config_filename = "");
    hipblasAlgoMap(const hipblasAlgoMap& map);
    ~hipblasAlgoMap();
    void loadGemmConfig();
    void loadSpGemmConfig();
    int  getSpAlgo(const int batch_count, const int m, const int n, const int k);
    bool isUseSparse(const int batch_count, const int m, const int n, const int k);

    bool isExist(const int batch_count, const int m, const int n, const int k, const hipblasDatatype_t data_type);

    hipblasLtMatmulAlgo_info
    getAlgo(const int batch_count, const int m, const int n, const int k, const hipblasDatatype_t data_type);
};

}  // namespace rocm
}  // namespace fastertransformer
