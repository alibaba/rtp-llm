#include "rtp_llm/cpp/core/Types.h"

namespace rtp_llm {

typedef struct {
    void*       A_input;
    void*       B_input;
    void*       B_scales_input;
    void*       B_zeros_input;
    void*       C_input;
    size_t      M;
    size_t      N;
    size_t      K;
    size_t      Group_size;
    size_t      StrideA;
    size_t      StrideB;
    size_t      StrideC;
    hipStream_t stream;
} ckGemmParam;

class rocmCKGemmWrapper {
public:
    rocmCKGemmWrapper()  = default;
    ~rocmCKGemmWrapper() = default;
    void runCKGemm(const ckGemmParam& ckParams, DataType ADtype, DataType BDtype);
};

}  // namespace rtp_llm