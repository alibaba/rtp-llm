#include "src/fastertransformer/th_op/GptInitParameter.h"
#include "src/fastertransformer/th_op/multi_gpu_gpt/RtpLLMOp.h"
#include "src/fastertransformer/th_op/multi_gpu_gpt/RtpEmbeddingOp.h"
#include "src/fastertransformer/th_op/multi_gpu_gpt/EmbeddingHandlerOp.h"
#if USING_CUDA
#include "src/fastertransformer/th_op/multi_gpu_gpt/ParallelGptOp.h"
# endif

using namespace fastertransformer;

namespace torch_ext {
PYBIND11_MODULE(libth_transformer, m) {
    registerGptInitParameter(m);
    registerRtpLLMOp(m);
    registerRtpEmbeddingOp(m);

#if USING_CUDA
    registerParallelGptOp(m);
    registerEmbeddingHandler(m);
#endif
}

} // namespace torch_ext

