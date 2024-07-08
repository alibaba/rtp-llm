#include <hip/hip_runtime.h>
#include "hipblasAlgoMap.h"

namespace fastertransformer {
namespace rocm {

hipblasAlgoMap::hipblasAlgoMap(const std::string filename, const std::string sp_config_filename):
    config_filename_(filename), sp_config_filename_(sp_config_filename)
{
    loadGemmConfig();
    loadSpGemmConfig();
}

hipblasAlgoMap::hipblasAlgoMap(const hipblasAlgoMap& algo_map):
    config_filename_(algo_map.config_filename_),
    sp_config_filename_(algo_map.sp_config_filename_),
    algo_map_(algo_map.algo_map_),
    sp_algo_map_(algo_map.sp_algo_map_)
{
}

hipblasAlgoMap::~hipblasAlgoMap()
{
    algo_map_.clear();
}

void hipblasAlgoMap::loadGemmConfig()
{
    FILE* fd;
    fd = fopen(config_filename_.c_str(), "r");
    if (fd == NULL) {
        std::cout << "[WARNING] " << config_filename_ << " is not found; using default GEMM algo" << std::endl;
        return;
    }

    int   batchCount2, m2, n2, k2, algoId, customOption, tile, splitK_val;
    int   batch_size, seq_len, head_num, size_per_head, dataType;
    int   swizzle, reductionScheme, workspaceSize, stages;
    int   inner_shapeId, cluster_shapeId, mma_shapeId, cga_shapeId, sche_mode;
    float exec_time;
    char  tmp[1024];
    if (!fgets(tmp, 1024, fd)) {
        printf("[ERROR] fgets fail at %s:%d \n", __FILE__, __LINE__);
        exit(-1);
    }
    while (fscanf(fd,
                  "%d %d %d %d %d ### %d %d %d %d %d %d %d %d %d %d %d %d %d %d"
                  "%f\n",
                  &batch_size,
                  &seq_len,
                  &head_num,
                  &size_per_head,
                  &dataType,
                  &batchCount2,
                  &n2,
                  &m2,
                  &k2,
                  &algoId,
                  &customOption,
                  &tile,
                  &splitK_val,
                  &swizzle,
                  &reductionScheme,
                  &workspaceSize,
                  &stages,
                  &inner_shapeId,
                  &cluster_shapeId,
                  &exec_time)
           != EOF) {
        if (dataType != FLOAT_DATATYPE && dataType != HALF_DATATYPE && dataType != BFLOAT16_DATATYPE
            && dataType != INT8_DATATYPE && dataType != FP8_DATATYPE) {
            printf("[WARNING][readAlgoFromConfig] wrong dataType %d!\n", dataType);
            continue;
        }
        hipblasAlgoConfig_t markStr{batchCount2, m2, n2, k2, static_cast<hipblasDatatype_t>(dataType)};
        // workspaceSize should be zero
        if (algo_map_.find(markStr) == algo_map_.end()) {
            algo_map_[markStr].algoId          = algoId;
            algo_map_[markStr].customOption    = customOption;
            algo_map_[markStr].tile            = tile;
            algo_map_[markStr].splitK_val      = splitK_val;
            algo_map_[markStr].swizzle         = swizzle;
            algo_map_[markStr].reductionScheme = reductionScheme;
            algo_map_[markStr].workspaceSize   = workspaceSize;
            algo_map_[markStr].exec_time = exec_time;
        }
    }
    fclose(fd);
}

bool hipblasAlgoMap::isExist(
    const int batch_count, const int m, const int n, const int k, const hipblasDatatype_t data_type)
{
    hipblasAlgoConfig_t mark{batch_count, n, m, k, data_type};
    return algo_map_.find(mark) != algo_map_.end();
}

hipblasLtMatmulAlgo_info
hipblasAlgoMap::getAlgo(const int batch_count, const int m, const int n, const int k, const hipblasDatatype_t data_type)
{
    hipblasAlgoConfig_t mark{batch_count, n, m, k, data_type};
    if (algo_map_.find(mark) != algo_map_.end()) {
        return algo_map_[mark];
    }
    else {
        hipblasLtMatmulAlgo_info tmp_algo;
        tmp_algo.algoId = static_cast<int>(HIPBLAS_GEMM_DEFAULT);
        tmp_algo.customOption    = -1;
        tmp_algo.tile            = -1;
        tmp_algo.splitK_val      = -1;
        tmp_algo.swizzle         = -1;
        tmp_algo.reductionScheme = -1;
        tmp_algo.workspaceSize   = -1;
        tmp_algo.exec_time       = -1.0f;
        return tmp_algo;
    }
}

void hipblasAlgoMap::loadSpGemmConfig()
{
    if (sp_config_filename_.empty()) {
        return;
    }
    FILE* fd = fopen(sp_config_filename_.c_str(), "r");
    if (fd == NULL) {
        printf("[WARNING] %s is not found; using SPGEMM algo id 0\n", sp_config_filename_.c_str());
        return;
    }
    sp_algo_map_.clear();
    int   batch_size, seq_len, head_num, size_per_head, data_type;
    int   batchCount, m, n, k, algoId;
    float exec_time;
    char  tmp[1024];
    if (!fgets(tmp, 1024, fd)) {
        printf("[ERROR] fgets fail at %s:%d \n", __FILE__, __LINE__);
        exit(-1);
    }
    while (fscanf(fd,
                  "%d %d %d %d %d ### %d %d %d %d %d %f\n",
                  &batch_size,
                  &seq_len,
                  &head_num,
                  &size_per_head,
                  &data_type,
                  &batchCount,
                  &m,
                  &n,
                  &k,
                  &algoId,
                  &exec_time)
           != EOF) {
        char mark[256];
        sprintf(mark, "%d_%d_%d_%d", batchCount, m, n, k);
        std::string markStr(mark);
        sp_algo_map_[markStr] = algoId;
    }
    fclose(fd);
}

int hipblasAlgoMap::getSpAlgo(const int batch_count, const int m, const int n, const int k)
{
    char mark[256];
    sprintf(mark, "%d_%d_%d_%d", batch_count, m, n, k);
    if (sp_algo_map_.find(mark) != sp_algo_map_.end()) {
        return sp_algo_map_[mark];
    }
    else {
        // for remove padding, select algo 1 for simplicity
        return 0;
    }
}

bool hipblasAlgoMap::isUseSparse(const int batch_count, const int m, const int n, const int k)
{
    // not available to use cusparselt.
    if (m % 8 != 0 || n % 8 != 0 || k % 8 != 0) {
        return false;
    }
    char mark[256];
    sprintf(mark, "%d_%d_%d_%d", batch_count, m, n, k);
    if (sp_algo_map_.find(mark) != sp_algo_map_.end()) {
        return sp_algo_map_[mark] != -1;
    }
    else {
        // no gemm test case, choose sparse according to sparse flag
        return true;
    }
}

}  // namespace rocm
}  // namespace fastertransformer
