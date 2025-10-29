#include "rtp_llm/cpp/core/Types.h"

namespace rtp_llm {

typedef struct {
    void*       A_input;
    void*       B_input;
    void*       D0_input;
    void*       D1_input;
    void*       E_input;
    size_t      M;
    size_t      N;
    size_t      K;
    size_t      StrideA;
    size_t      StrideB;
    size_t      StrideE;
    hipStream_t stream;
} ckW8A8GemmParam;

class rocmCKW8A8GeluGemmWrapper {
public:
    rocmCKW8A8GeluGemmWrapper()  = default;
    ~rocmCKW8A8GeluGemmWrapper() = default;
    void runCKW8A8GeluGemm(const ckW8A8GemmParam& ckParams, DataType ADtype, DataType BDtype);
};

}  // namespace rtp_llm