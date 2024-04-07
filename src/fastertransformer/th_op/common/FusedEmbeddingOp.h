#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/utils/compiler_config.h"

namespace ft = fastertransformer;
namespace th = torch;
namespace torch_ext {

class IFusedEmbedding {
public:
    virtual ~IFusedEmbedding() {}
    virtual void forward(th::Tensor& embedding_output, const th::Tensor& token_ids, th::optional<th::Tensor> position_ids,
                         th::optional<th::Tensor> token_type_ids, int token_num) = 0;
};

template<typename T>
class FusedEmbedding: public IFusedEmbedding {
public:
    FusedEmbedding(th::Tensor embedding_weight, th::optional<th::Tensor> positional_embedding_weight, th::optional<th::Tensor> token_type_embedding_weight);
    virtual void forward(th::Tensor& embedding_output, const th::Tensor& token_ids, th::optional<th::Tensor> position_ids,
                         th::optional<th::Tensor> token_type_ids, int token_num);

private:
    // 这里没有持有三个tesnor，依赖外部去避免tensor析构
    cudaStream_t stream_;
    T* embedding_weight_ptr_ = nullptr;
    T* positional_weight_ptr_ = nullptr;
    T* type_weight_ptr_ = nullptr;
    int hidden_size_;
};

class FusedEmbeddingOp: public th::jit::CustomClassHolder {
public:
    FusedEmbeddingOp(th::Tensor embedding_weight, th::optional<th::Tensor> positional_embedding_weight, th::optional<th::Tensor> token_type_embedding_weight);
    th::Tensor forward(th::Tensor& token_ids, th::optional<th::Tensor> postion_ids, th::optional<th::Tensor> token_type_ids);

private:
    IFusedEmbedding* embedding_ptr_;
    at::ScalarType   scalar_type_;
    int              hidden_size_;
};

}  // namespace torch_ext
