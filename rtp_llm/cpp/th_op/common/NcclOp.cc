#include "rtp_llm/cpp/th_op/common/NcclOp.h"
#include <vector>

namespace th = torch;


namespace torch_ext {
NcclOp::NcclOp(const int64_t     tensor_para_size,
               const int64_t     pipeline_para_size,
               const std::string master_ip,
               const int64_t     master_port)
{
    rtp_llm::ftNcclInitialize(tensor_para_, pipeline_para_, tensor_para_size, pipeline_para_size, master_ip, master_port);
}

NcclOp::~NcclOp()
{
    rtp_llm::ftNcclParamDestroy(tensor_para_);
    rtp_llm::ftNcclParamDestroy(pipeline_para_);
}

void NcclOp::broadcast_tp(std::vector<th::Tensor> tensors, int64_t root, bool timeout)
{
    if (tensor_para_.world_size_ == 1) {
        return;
    }

    auto stream  = at::cuda::getCurrentCUDAStream().stream();

    if (tensor_para_.world_size_ > 1) {
        rtp_llm::ftNcclGroupStart();
        for (auto const& tensor : tensors) {
            char* buffer = reinterpret_cast<char*>(tensor.data_ptr());
            rtp_llm::ftNcclBroadCast(buffer, sizeBytes(tensor), root, tensor_para_, stream);
        }
        rtp_llm::ftNcclGroupEnd();
    }
    if (tensor_para_.rank_ == root){
        rtp_llm::ftNcclStreamSynchronize(tensor_para_, stream, timeout);
    }
    check_cuda_error();
}

}  // namespace torch_ext

static auto rtpLlmNcclTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::NcclOp>("RtpLlmNcclOp")
#else
    torch::jit::class_<torch_ext::NcclOp>("RtpLlm", "NcclOp")
#endif
        .def(torch::jit::init<int64_t, int64_t, std::string, int64_t>())
        .def("broadcast_tp", &torch_ext::NcclOp::broadcast_tp);
