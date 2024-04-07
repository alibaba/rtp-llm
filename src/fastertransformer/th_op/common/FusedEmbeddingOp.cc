#include "src/fastertransformer/th_op/common/FusedEmbeddingOp.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"

namespace th = torch;
namespace ft = fastertransformer;

namespace torch_ext {

template<typename T>
FusedEmbedding<T>::FusedEmbedding(th::Tensor embedding_weight, th::optional<th::Tensor> positional_embedding_weight, th::optional<th::Tensor> token_type_embedding_weight) {
    hidden_size_ = embedding_weight.size(1);
    stream_ = at::cuda::getCurrentCUDAStream().stream();
    embedding_weight_ptr_ = get_ptr<T>(embedding_weight);
    if (positional_embedding_weight.has_value()) {
        positional_weight_ptr_ = get_ptr<T>(positional_embedding_weight.value());    
    }
    if (token_type_embedding_weight.has_value()) {
        type_weight_ptr_ = get_ptr<T>(token_type_embedding_weight.value());
    }
}

template<typename T>
void FusedEmbedding<T>::forward(th::Tensor& embedding_output, const th::Tensor& token_ids, th::optional<th::Tensor> position_ids, 
                                th::optional<th::Tensor> token_type_ids, int token_num) {
    T* embedding_output_ptr = get_ptr<T>(embedding_output);
    int* token_ids_ptr = get_ptr<int>(token_ids);
    int* position_ids_ptr = position_ids.has_value() ? get_ptr<int>(position_ids.value()) : nullptr;
    int* token_type_ids_ptr = token_type_ids.has_value() ? get_ptr<int>(token_type_ids.value()) : nullptr;
    if (positional_weight_ptr_ != nullptr) {
        FT_CHECK_WITH_INFO(position_ids_ptr != nullptr, "position_ids should not be nullptr");
    }
    if (type_weight_ptr_ != nullptr) {
        FT_CHECK_WITH_INFO(token_type_ids_ptr != nullptr, "token_type_ids should not be nullptr");
    }
    ft::invokeEmebeddingLookup(embedding_output_ptr, 
                               embedding_weight_ptr_, 
                               positional_weight_ptr_, 
                               type_weight_ptr_,
                               token_ids_ptr,
                               position_ids_ptr,
                               token_type_ids_ptr,
                               token_num,
                               hidden_size_,
                               stream_);
}

FusedEmbeddingOp::FusedEmbeddingOp(th::Tensor embedding_weight, th::optional<th::Tensor> positional_embedding_weight, th::optional<th::Tensor> token_type_embedding_weight) {
    scalar_type_ = embedding_weight.scalar_type();
    hidden_size_ = embedding_weight.size(1);
#define CREATE_INSTANCE(T_)  embedding_ptr_ = new FusedEmbedding<T_>(embedding_weight, positional_embedding_weight, token_type_embedding_weight);
    
    switch (scalar_type_) {
        case at::ScalarType::Float:
            CREATE_INSTANCE(float);
            break;
        case at::ScalarType::Half:
            CREATE_INSTANCE(half);
            break;
        case at::ScalarType::BFloat16:
            if constexpr (CompileConfig::enable_bf16) {
                CREATE_INSTANCE(__nv_bfloat16);
            }
            break;
        default:
            throw std::runtime_error("Wrong tensor type.");
    }
#undef CREATE_INSTANCE
}

#define CHECK_OPTIONAL_INPUT_SHAPE(t) \
    if (t.has_value()) {              \
        FT_CHECK_WITH_INFO(t.value().size(0) == token_num, #t" len != %d", token_num); \
    }

th::Tensor FusedEmbeddingOp::forward(th::Tensor& token_ids, th::optional<th::Tensor> position_ids, th::optional<th::Tensor> token_type_ids) {
    CHECK_INPUT(token_ids, at::ScalarType::Int);
    CHECK_OPTIONAL_INPUT(position_ids, at::ScalarType::Int);
    CHECK_OPTIONAL_INPUT(token_type_ids, at::ScalarType::Int);
    int token_num = token_ids.size(0);
    CHECK_OPTIONAL_INPUT_SHAPE(position_ids);
    CHECK_OPTIONAL_INPUT_SHAPE(token_type_ids);
    th::Tensor embedding_output =
    torch::zeros({(int64_t)(token_num), hidden_size_},
                    torch::dtype(scalar_type_).device(torch::kCUDA).requires_grad(false));

    embedding_ptr_->forward(embedding_output, token_ids, position_ids, token_type_ids, token_num);
    return embedding_output;
}

} // torch_ext

static auto fasterTransformerEmbeddingTHS =
    torch::jit::class_<torch_ext::FusedEmbeddingOp>("FasterTransformer", "FusedEmbeddingOp")
        .def(torch::jit::init<th::Tensor,
             th::optional<th::Tensor>,
             th::optional<th::Tensor>>())
        .def("forward", &torch_ext::FusedEmbeddingOp::forward);