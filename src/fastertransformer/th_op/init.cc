#include "torch/all.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"
#include "src/fastertransformer/th_op/multi_gpu_gpt/RtpEmbeddingOp.h"
#include "src/fastertransformer/th_op/multi_gpu_gpt/EmbeddingHandlerOp.h"
#include "src/fastertransformer/th_op/multi_gpu_gpt/RtpLLMOp.h"
#include "src/fastertransformer/th_op/multi_gpu_gpt/ParallelGptOp.h"

using namespace fastertransformer;

namespace torch_ext {
PYBIND11_MODULE(libth_transformer, m) {
    registerRtpEmbeddingOp(m);
    registerGptInitParameter(m);    
    registerParallelGptOp(m);    
    registerRtpLLMOp(m);
    registerEmbeddingHandler(m);
}

} // namespace torch_ext 

