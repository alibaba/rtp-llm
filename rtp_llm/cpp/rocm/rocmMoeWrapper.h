#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/model_utils/activation_types.h"

namespace rtp_llm {

struct rocmMoeParams {
    void* input;
    void* input_scale_ptr;
    void* gate_ptr;
    void* gate_scale_ptr;
    void* down_ptr;
    void* down_scale_ptr;
    void* smooth_scale_ptr;
    void* output_ptr;

    void* topk_ids_ptr;
    void* topk_weight_ptr;
    void* sorted_token_ids_ptr;
    void* sorted_weight_ptr;
    void* sorted_expert_ids_ptr;
    void* num_sorted_tiles_ptr;

    size_t block_m;
    size_t hidden_size;
    size_t intermediate_size;
    size_t num_tokens;
    size_t num_experts;
    size_t topk;

    size_t stride_token;

    hipStream_t stream;
};

class rocmMoeWrapper {
private:
public:
    rocmMoeWrapper(/* args */) = default;
    ~rocmMoeWrapper()          = default;
    uint32_t runCKMoe(const rocmMoeParams& params,
                      DataType             dtype,
                      DataType             wtype,
                      ActivationType       activation_type,
                      int                  fused_quant,
                      int                  gate_only);
};

}  // namespace rtp_llm
