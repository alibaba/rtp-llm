#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/cuda/nccl/nccl_utils.h"

namespace ft = fastertransformer;
namespace th = torch;
namespace torch_ext {

class NcclOp: public th::jit::CustomClassHolder {
public:
    NcclOp(const int64_t     tensor_para_size,
           const int64_t     pipeline_para_size,
           const std::string master_ip,
           const int64_t     master_port);

    ~NcclOp();

    void broadcast_tp(std::vector<th::Tensor> tensors, int64_t root);

private:
    size_t        tensor_para_size_;
    size_t        pipeline_para_size_;
    ft::NcclParam tensor_para_;
    ft::NcclParam pipeline_para_;
};

}  // namespace torch_ext
