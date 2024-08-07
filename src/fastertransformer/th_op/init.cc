#include "src/fastertransformer/th_op/GptInitParameter.h"
#include "src/fastertransformer/th_op/GptInitParameterRegister.h"
#include "src/fastertransformer/th_op/multi_gpu_gpt/RtpLLMOp.h"
#include "src/fastertransformer/th_op/multi_gpu_gpt/RtpEmbeddingOp.h"
#include "src/fastertransformer/th_op/multi_gpu_gpt/EmbeddingHandlerOp.h"

using namespace fastertransformer;

namespace torch_ext {
PYBIND11_MODULE(libth_transformer, m) {
    registerGptInitParameter(m);
    registerRtpLLMOp(m);
    registerRtpEmbeddingOp(m);
    registerEmbeddingHandler(m);
}

} // namespace torch_ext

