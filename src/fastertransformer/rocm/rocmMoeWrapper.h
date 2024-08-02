#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/utils/activation_types.h"

namespace fastertransformer {

struct rocmMoeParams {
    void*          input;
    ActivationType activation_type;

    void* gate_weight;
    void* gate_scales;
    void* gate_zeros;
    void* gate_bias;

    void* up_weight;
    void* up_scales;
    void* up_zeros;
    void* up_bias;

    void* down_weight;
    void* down_scales;
    void* down_zeros;
    void* down_bias;

    void*  output;
    void*  output_gate;
    size_t num_experts;
    int*   total_rows_before_expert_host;
    size_t N;
    size_t K;

    hipStream_t stream;
};
struct rocmGroupGEMMParams {
    void* input;

    void* B_weight;
    void* B_scales;
    void* B_zeros;
    void* B_bias;

    void*  output;
    size_t num_experts;
    int*   total_rows_before_expert_host;
    size_t N;
    size_t K;

    hipStream_t stream;
};

class rocmMoeWrapper {
private:
public:
    rocmMoeWrapper(/* args */) = default;
    ~rocmMoeWrapper()          = default;
    void runCKMoe(const rocmMoeParams& params, DataType dtype, DataType wtype);
};

}  // namespace fastertransformer
