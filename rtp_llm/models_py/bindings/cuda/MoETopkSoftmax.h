namespace torch_ext {
void moe_topk_softmax(at::Tensor& topk_weights,
     at::Tensor& topk_indices, 
     at::Tensor& token_expert_indices, 
     at::Tensor&  gating_output);
}