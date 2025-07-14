#include "trt_plugins/common/trtPluginsInterface.h"
namespace tensorrt_llm::plugins {
template<typename T>
class GroupGemmPlugin {
public:
    GroupGemmPlugin();
    ~GroupGemmPlugin() = default;
    void gemm(T**          A,
              T**          B,
              T**          C,
              const int*   m,
              const int*   n,
              const int*   k,
              const float  alpha,
              const float  beta,
              const int    count,
              cudaStream_t stream);

private:
    std::shared_ptr<rtp_llm::CutlassGroupGemmRunner<T>> group_gemm_runner_;
};
}  // namespace tensorrt_llm::plugins