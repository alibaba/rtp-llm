
// #include "rtp_llm/cpp/devices/DeviceFactory.h"
// #include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
// #include "rtp_llm/cpp/devices/OpData.h"

//

// namespace unittest {

// class MlaQKVGemmOP: public torch::jit::CustomClassHolder {
// public:
//     MlaQKVGemmOP(int64_t head_num,
//                  int64_t q_lora_dim,
//                  int64_t kv_lora_dim,
//                  int64_t nope_head_dim,
//                  int64_t rope_head_dim,
//                  int64_t v_head_dim);

//     void forward(torch::Tensor input,
//                  torch::Tensor output,
//                  torch::Tensor q_a_weight,
//                  torch::Tensor q_b_weight,
//                  torch::Tensor kv_a_weight,
//                  torch::Tensor k_nope_weight,
//                  torch::Tensor k_rope_weight,
//                  torch::Tensor v_weight,
//                  torch::Tensor q_layernorm_weight,
//                  torch::Tensor kv_a_layernorm_weight);

// private:
//     DeviceBase*      device = nullptr;
//     AttentionConfigs attention_configs;
//     LayerNormConfig  layernorm_config;
// };

// MlaQKVGemmOP::MlaQKVGemmOP(int64_t head_num,
//                            int64_t q_lora_dim,
//                            int64_t kv_lora_dim,
//                            int64_t nope_head_dim,
//                            int64_t rope_head_dim,
//                            int64_t v_head_dim) {
//     ParallelismConfig parallelism_config;
//     ModelConfig model_config;
//     EPLBConfig eplb_config;
//     FMHAConfig fmha_config;
//     DeviceResourceConfig device_resource_config;
//     MoeConfig moe_config;
//     SpeculativeExecutionConfig sp_config;
//     MiscellaneousConfig misc_config;
//     ProfilingDebugLoggingConfig profiling_debug_logging_config;
//     HWKernelConfig hw_kernel_config;
//     ConcurrencyConfig concurrency_config;
//     FfnDisAggregateConfig ffn_disaggregate_config;
//     RuntimeConfig runtime_config;
//     DeviceFactory::initDevices(
//         parallelism_config,
//         model_config,
//         eplb_config,
//         fmha_config,
//         device_resource_config,
//         moe_config,
//         sp_config,
//         misc_config,
//         profiling_debug_logging_config,
//         hw_kernel_config,
//         concurrency_config,
//         ffn_disaggregate_config,
//         runtime_config);
//     device = DeviceFactory::getDefaultDevice();

//     attention_configs.use_mla       = true;
//     attention_configs.head_num      = head_num;
//     attention_configs.q_lora_rank   = q_lora_dim;
//     attention_configs.kv_lora_rank  = kv_lora_dim;
//     attention_configs.nope_head_dim = nope_head_dim;
//     attention_configs.rope_head_dim = rope_head_dim;
//     attention_configs.v_head_dim    = v_head_dim;

//     layernorm_config.eps       = 1e-6;
//     layernorm_config.norm_type = NormType::rmsnorm;
// }

// void MlaQKVGemmOP::forward(torch::Tensor input,
//                            torch::Tensor output,
//                            torch::Tensor q_a_weight,
//                            torch::Tensor q_b_weight,
//                            torch::Tensor kv_a_weight,
//                            torch::Tensor k_nope_weight,
//                            torch::Tensor k_rope_weight,
//                            torch::Tensor v_weight,
//                            torch::Tensor q_layernorm_weight,
//                            torch::Tensor kv_a_layernorm_weight) {
//     auto hidden       = torchTensor2Buffer(input);
//     auto q_a          = torchTensor2Buffer(q_a_weight);
//     auto q_b          = torchTensor2Buffer(q_b_weight);
//     auto kv_a         = torchTensor2Buffer(kv_a_weight);
//     auto k_nope       = torchTensor2Buffer(k_nope_weight);
//     auto k_rope       = torchTensor2Buffer(k_rope_weight);
//     auto v            = torchTensor2Buffer(v_weight);
//     auto q_layernorm  = torchTensor2Buffer(q_layernorm_weight);
//     auto kv_layernorm = torchTensor2Buffer(kv_a_layernorm_weight);

//     AttentionLayerWeights attention_weights;
//     attention_weights.q_a_weight.reset(new DenseWeights(q_a));
//     attention_weights.q_b_weight.reset(new DenseWeights(q_b));
//     attention_weights.kv_a_weight.reset(new DenseWeights(kv_a));
//     attention_weights.k_nope_weight.reset(new DenseWeights(k_nope));
//     attention_weights.k_rope_weight.reset(new DenseWeights(k_rope));
//     attention_weights.v_weight.reset(new DenseWeights(v));

//     BufferPtr bias = nullptr;
//     attention_weights.q_a_norm_weight.reset(new LayerNormWeights(q_layernorm, bias));
//     attention_weights.kv_a_norm_weight.reset(new LayerNormWeights(kv_layernorm, bias));

//     auto                      t1 = Buffer::emptyBuffer();
//     auto                      t2 = Buffer::emptyBuffer();
//     AttentionCommonInputs attention_common_inputs{device->clone({t1}), device->clone({t2}), };

//     // create Attention
//     AttentionLayerParams params{-1,
//                                 *hidden,
//                                 nullptr,
//                                 attention_configs,
//                                 attention_weights,
//                                 attention_common_inputs,
//                                 std::nullopt,
//                                 layernorm_config};
//     BufferPtr qkv;
//     qkv = device->mlaQKVGemm(params);

//     float* qkv_out_data = (float*)output.data_ptr();
//     cudaMemcpy(qkv_out_data, qkv->data(), qkv->sizeBytes(), cudaMemcpyDeviceToDevice);
// }
// }  // namespace unittest

// static auto MlaQKVGemmTHS = torch::jit::class_<unittest::MlaQKVGemmOP>("unittest", "MlaQKVGemmOP")
//                                 .def(torch::jit::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>())
//                                 .def("forward", &unittest::MlaQKVGemmOP::forward);
