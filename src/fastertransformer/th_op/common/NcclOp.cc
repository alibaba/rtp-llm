#include "src/fastertransformer/th_op/common/NcclOp.h"
#include <vector>

namespace th = torch;
namespace ft = fastertransformer;

namespace torch_ext {
NcclOp::NcclOp(const int64_t     tensor_para_size,
               const int64_t     pipeline_para_size,
               const std::string master_ip,
               const int64_t     master_port)
{
    ft::ftNcclInitialize(tensor_para_, pipeline_para_, tensor_para_size, pipeline_para_size, master_ip, master_port);
}

NcclOp::~NcclOp()
{
    ft::ftNcclParamDestroy(tensor_para_);
    ft::ftNcclParamDestroy(pipeline_para_);
}

void NcclOp::broadcast_tp(std::vector<th::Tensor> tensors, int64_t root)
{
    if (tensor_para_.world_size_ == 1) {
        return;
    }

    auto stream  = at::cuda::getCurrentCUDAStream().stream();

    if (tensor_para_.world_size_ > 1) {
        ft::ftNcclGroupStart();
        for (auto const& tensor : tensors) {
            char* buffer = reinterpret_cast<char*>(tensor.data_ptr());
            ft::ftNcclBroadCast(buffer, sizeBytes(tensor), root, tensor_para_, stream);
        }
        ft::ftNcclGroupEnd();
    }
    if (tensor_para_.rank_ == root){
        ft::ftNcclStreamSynchronize(tensor_para_, pipeline_para_, stream);
    }
    sync_check_cuda_error();
}

}  // namespace torch_ext

static auto fasterTransformerNcclTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::NcclOp>("FasterTransformerNcclOp")
#else
    torch::jit::class_<torch_ext::NcclOp>("FasterTransformer", "NcclOp")
#endif
        .def(torch::jit::init<int64_t, int64_t, std::string, int64_t>())
        .def("broadcast_tp", &torch_ext::NcclOp::broadcast_tp);
