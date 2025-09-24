#include "hipblasAlgoMap.h"

#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_split.h"
#include "absl/strings/numbers.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include "rocm/include/hipblaslt/hipblaslt-ext.hpp"
#include "hip_host_utils.h"

#include <fstream>
#include <string>

namespace rtp_llm {
namespace rocm {

constexpr absl::string_view COLUMN_HEADER    = "trans_a, trans_b, m, n, k, A_data_type, lda, stride_a, "
                                               "B_data_type, ldb, stride_b, C_data_type, ldc, stride_c, "
                                               "compute_type, batch_count, algo_index";
constexpr absl::string_view COLUMN_HEADER_V2 = "trans_a, trans_b, m, n, k, A_data_type, lda, stride_a, "
                                               "B_data_type, ldb, stride_b, C_data_type, ldc, stride_c, "
                                               "compute_type, batch_count, epilogue, algo_index";

static absl::StatusOr<std::pair<hipblasLtAlgoConfig, int32_t>> parseRow(const std::string& row) {
    constexpr size_t                                  numFields = 18;
    absl::InlinedVector<absl::string_view, numFields> fields    = absl::StrSplit(row, ',');

    if (fields.size() != numFields && fields.size() != numFields - 1) {
        return absl::InvalidArgumentError("Invalid number of fields in row: " + row);
    }
    bool has_epilogue = fields.size() == numFields;

    hipblasLtAlgoConfig config;
    int32_t             algoIndex;

    auto parseOp = [](const absl::string_view& field) -> absl::StatusOr<hipblasOperation_t> {
        if (field == "T") {
            return HIPBLAS_OP_T;
        } else if (field == "N") {
            return HIPBLAS_OP_N;
        } else {
            return absl::InvalidArgumentError("Invalid hipblasOperation_t value: " + std::string(field));
        }
    };

    auto parseEpilogue = [](const absl::string_view& field) -> absl::StatusOr<hipblasLtEpilogue_t> {
        if (field == "none") {
            return HIPBLASLT_EPILOGUE_DEFAULT;
        } else if (field == "gelu_bias") {
            return HIPBLASLT_EPILOGUE_GELU_BIAS;
        } else if (field == "relu_bias") {
            return HIPBLASLT_EPILOGUE_RELU_BIAS;
        } else if (field == "bias") {
            return HIPBLASLT_EPILOGUE_BIAS;
        } else {
            return absl::InvalidArgumentError("Invalid hipblasOperation_t value: " + std::string(field));
        }
    };

    auto parseDType = [](const absl::string_view& field) -> absl::StatusOr<hipDataType> {
        if (field == "f32_r") {
            return HIP_R_32F;
        } else if (field == "f16_r") {
            return HIP_R_16F;
        } else if (field == "bf16_r") {
            return HIP_R_16BF;
        } else if (field == "f8_e4m3_fnuz_r") {
            return HIP_R_8F_E4M3_FNUZ;
        } else {
            return absl::InvalidArgumentError("Invalid hipDataType value: " + std::string(field));
        }
    };

    auto parseComputeType = [](const absl::string_view& field) -> absl::StatusOr<hipblasComputeType_t> {
        if (field == "f32_r") {
            return HIPBLAS_COMPUTE_32F;
        } else {
            return absl::InvalidArgumentError("Invalid hipblasComputeType_t value: " + std::string(field));
        }
    };

#define ASSIGN(lhs, rhs)                                                                                               \
    do {                                                                                                               \
        auto __tmp = (rhs);                                                                                            \
        if (!__tmp.ok())                                                                                               \
            return __tmp.status();                                                                                     \
        lhs = __tmp.value();                                                                                           \
    } while (0)

#define ATOI(field, member)                                                                                            \
    do {                                                                                                               \
        if (!absl::SimpleAtoi(field, &member))                                                                         \
            return absl::InvalidArgumentError("Invalid integer for " #member ": " + std::string(field));               \
    } while (0)

    ASSIGN(config.trans_a, parseOp(fields[0]));
    ASSIGN(config.trans_b, parseOp(fields[1]));
    ATOI(fields[2], config.m);
    ATOI(fields[3], config.n);
    ATOI(fields[4], config.k);
    ASSIGN(config.A_data_type, parseDType(fields[5]));
    ATOI(fields[6], config.lda);
    ATOI(fields[7], config.stride_a);
    ASSIGN(config.B_data_type, parseDType(fields[8]));
    ATOI(fields[9], config.ldb);
    ATOI(fields[10], config.stride_b);
    ASSIGN(config.C_data_type, parseDType(fields[11]));
    ATOI(fields[12], config.ldc);
    ATOI(fields[13], config.stride_c);
    ASSIGN(config.compute_type, parseComputeType(fields[14]));
    ATOI(fields[15], config.batch_count);
    if (has_epilogue) {
        ASSIGN(config.epilogue, parseEpilogue(fields[16]));
        ATOI(fields[17], algoIndex);
    } else {
        config.epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
        ATOI(fields[16], algoIndex);
    }

#undef ASSIGN
#undef ATOI

    return std::make_pair(config, algoIndex);
}

void hipblasAlgoMap::loadGemmConfig(const std::string& filename, hipblasLtHandle_t handle) {
    std::ifstream file;
    std::string   line;

    line.reserve(256);
    file.open(filename);

    if (!std::getline(file, line)) {
        std::printf("[BLAS] Gemm config not find: %s \n", filename.c_str());
        return;
    }

    if (line != COLUMN_HEADER && line != COLUMN_HEADER_V2) {
        std::printf("[BLAS] MISMATCH %s | %s\n", line.c_str(), std::string(COLUMN_HEADER).c_str());
        abort();
    }

    std::vector<int>                              algoIndices{1};
    std::vector<hipblasLtMatmulHeuristicResult_t> algos;

    while (std::getline(file, line)) {

        if (line.c_str()[0] == '#')
            continue;

        auto config_and_algoIndex = parseRow(line);

        if (!config_and_algoIndex.ok()) {
            // std::printf("%s\n", config_and_algoIndex.status().ToString().c_str());
            continue;
        }

        hipblasLtMatmulDesc_t   opDesc;
        hipblasLtMatrixLayout_t ADesc, BDesc, CDesc;

        auto [config, algoIndex] = config_and_algoIndex.value();

        hipblasLtMatmulDescCreate(&opDesc, config.compute_type, HIP_R_32F);
        hipblasLtMatmulDescSetAttribute(opDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &config.trans_a, sizeof(int32_t));
        hipblasLtMatmulDescSetAttribute(opDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &config.trans_b, sizeof(int32_t));
        if (config.epilogue != HIPBLASLT_EPILOGUE_DEFAULT) {
            hipblasLtEpilogue_t epilogue_ = config.epilogue;
            ROCM_CHECK(
                hipblasLtMatmulDescSetAttribute(opDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue_, sizeof(epilogue_)));
            int32_t bias_data_type = config.C_data_type;
            ROCM_CHECK(hipblasLtMatmulDescSetAttribute(
                opDesc, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_data_type, sizeof(bias_data_type)));
        }

        ROCM_CHECK(hipblasLtMatrixLayoutCreate(&ADesc,
                                               config.A_data_type,
                                               config.trans_a == HIPBLAS_OP_N ? config.m : config.k,
                                               config.trans_a == HIPBLAS_OP_N ? config.k : config.m,
                                               config.lda));
        ROCM_CHECK(hipblasLtMatrixLayoutCreate(&BDesc,
                                               config.B_data_type,
                                               config.trans_b == HIPBLAS_OP_N ? config.k : config.n,
                                               config.trans_b == HIPBLAS_OP_N ? config.n : config.k,
                                               config.ldb));
        ROCM_CHECK(hipblasLtMatrixLayoutCreate(&CDesc, config.C_data_type, config.m, config.n, config.ldc));

        algoIndices[0] = algoIndex;
        algos.clear();
        ROCM_CHECK(hipblaslt_ext::getAlgosFromIndex(handle, algoIndices, algos));

        if (algos.size() > 0) {
            hipblasLtMatmulInfo i;
            i.algo = algos.at(0).algo;
            i.opDesc.reset(opDesc);
            i.ADesc.reset(ADesc);
            i.CDesc.reset(CDesc);
            i.BDesc.reset(BDesc);
            algo_map_.emplace(config, std::move(i));
        }
    }
}

const hipblasLtMatmulInfo* hipblasAlgoMap::getAlgo(const hipblasOperation_t   trans_a,
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
                                                   const hipblasLtEpilogue_t  epilogue) {
    hipblasLtAlgoConfig config{trans_a,
                               trans_b,
                               m,
                               n,
                               k,
                               A_data_type,
                               lda,
                               stride_a,
                               B_data_type,
                               ldb,
                               stride_b,
                               C_data_type,
                               ldc,
                               stride_c,
                               compute_type,
                               batch_count,
                               epilogue};

    auto iter = algo_map_.find(config);
    if (iter != algo_map_.end()) {
        return &iter->second;
    }

    auto opToString = [](hipblasOperation_t op) -> const char* {
        switch (op) {
            case HIPBLAS_OP_T:
                return "T";
            case HIPBLAS_OP_N:
                return "N";
            default:
                return "<?>";
        }
    };

    auto dataTypeToString = [](hipDataType t) -> const char* {
        switch (t) {
            case HIP_R_32F:
                return "f32_r";
            case HIP_R_16F:
                return "f16_r";
            case HIP_R_16BF:
                return "bf16_r";
            case HIP_R_8F_E4M3_FNUZ:
                return "f8_e4m3_fnuz_r";
            default:
                return "<?>";
        }
    };

    auto computeTypeToString = [](hipblasComputeType_t t) -> const char* {
        switch (t) {
            case HIPBLAS_COMPUTE_32F:
                return "f32_r";
            default:
                return "<?>";
        }
    };

    auto epilogueToString = [](hipblasLtEpilogue_t t) -> const char* {
        switch (t) {
            case HIPBLASLT_EPILOGUE_DEFAULT:
                return "none";
            case HIPBLASLT_EPILOGUE_GELU_BIAS:
                return "gelu_bias";
            case HIPBLASLT_EPILOGUE_RELU_BIAS:
                return "relu_bias";
            case HIPBLASLT_EPILOGUE_BIAS:
                return "bias";
            default:
                return "<?>";
        }
    };

    /*printf("[ALGO] map size = %u\n", algo_map_.size());
    printf("[ALGO] MISSING HIPBLASLT CONFIG:\n %s,%s,%d,%d,%d,%s,%d,%lld,%s,%d,%lld,%s,%d,%lld,%s,%d,%d\n",
                   opToString(trans_a),
                   opToString(trans_b),
                   m,
                   n,
                   k,
                   dataTypeToString(A_data_type),
                   lda,
                   stride_a,
                   dataTypeToString(B_data_type),
                   ldb,
                   stride_b,
                   dataTypeToString(C_data_type),
                   ldc,
                   stride_c,
                   computeTypeToString(compute_type),
                   batch_count,
                   epilogueToString(epilogue));*/

    return nullptr;
}

}  // namespace rocm
}  // namespace rtp_llm
