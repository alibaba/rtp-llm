#include <assert.h>
#include <math.h>
#include <cublas_v2.h>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "src/fastertransformer/utils/DenseWeight.h"
#include "src/fastertransformer/cuda/allocator_cuda.h"
#include "src/fastertransformer/cuda/cublas/cublas.h"
#include "src/fastertransformer/cuda/cuda_utils.h"
#include "src/fastertransformer/cuda/gemm.h"
#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/cuda/memory_utils.h"
#include "src/fastertransformer/core/Tensor.h"
#include "src/fastertransformer/cuda/nvtx/nvtx_utils.h"

#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGpt.h"
#include "src/fastertransformer/models/multi_gpu_gpt/NormWrapper.h"
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptDecoderLayerWeight.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptWeight.h"
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptContextDecoder.h"
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptDecoder.h"


using namespace fastertransformer;

static const char* usage =
    "Usage: %s <path-to-weights> <test-data-path>\n"
    "Example: $test_context_decoder_layer ../models/megatron_models/c-model/345m/ ../tests/data\n";


template <typename T>
bool test_context_sharing(const std::string& weight_dir, const std::string& data_dir);
Tensor create_tensor(MemoryType memory_type, DataType data_type, std::vector<size_t> shape, bool init_zero);
void free_tensors(std::vector<Tensor> &tensors);
void free_tensors(TensorMap &tensors);
template<typename T> bool all_close(Tensor &tensor_x, Tensor &tensor_y);
Tensor tensor_to_cpu(Tensor &tensor);


int main(int argc, const char* argv[])
{
    if (argc != 3) {
        printf(usage, argv[0]);
        return 0;
    }

    bool result = true;
    result &= test_context_sharing<float>(
            argv[1], argv[2] + std::string("/gpt_context_decoder_inputs"));

    return result ? EXIT_SUCCESS: EXIT_FAILURE;
}

template <typename T>
bool test_context_sharing(const std::string& weight_dir, const std::string& data_dir)
{
    const size_t head_num = 16;
    const size_t size_per_head = 64;
    const size_t hidden_units = head_num * size_per_head;
    const size_t inter_size = 4 * hidden_units;
    const size_t decoder_layers = 2, num_layer = 2; // Reduce the number of layers for faster loading / processing
    const size_t max_seq_len = 1024;
    const size_t vocab_size = 50304;
    /* start_id = 50256 */
    /* end_id = 50256 */
    /* weight_data_type = fp32 */
    /* tensor_para_size = 1 */
    const DataType data_type = getTensorType<T>();

    NcclParam tensor_para;
    NcclParam pipeline_para;

    cudaStream_t stream;
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    check_cuda_error(cudaStreamCreate(&stream));
    check_cuda_error(cublasCreate(&cublas_handle));
    check_cuda_error(cublasLtCreate(&cublaslt_handle));
    check_cuda_error(cublasSetStream(cublas_handle, stream));

    cublasAlgoMap cublas_algo_map(GEMM_CONFIG);
    Allocator<AllocatorType::CUDA> * allocator = new Allocator<AllocatorType::CUDA>(getDevice());
    allocator->setStream(stream);

    std::mutex* cublas_wrapper_mutex = new std::mutex();
    cublasMMWrapper *cublas_wrapper = new cublasMMWrapper(cublas_handle,
                                   cublaslt_handle,
                                   stream,
                                   &cublas_algo_map,
                                   cublas_wrapper_mutex,
                                   allocator);
    if (std::is_same<T, half>::value) {
        cublas_wrapper->setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
    }
    else if (std::is_same<T, float>::value) {
        cublas_wrapper->setFP32GemmConfig();
    }

    ParallelGptWeight<T> gpt_weights(
        hidden_units, size_per_head, head_num, inter_size, vocab_size, decoder_layers, max_seq_len,
        1, // tensor_para_size
        0, // tensor_para_rank
        1, // layer_para_size
        0, // layer_para_rank
        0  // int8
    );
    gpt_weights.loadModel((weight_dir + std::string("/1-gpu")).c_str());

    AttentionType attention_type = getAttentionType<T>(size_per_head,
                                                       getSMVersion(),
                                                       false, // remove_padding
                                                       0, // gpt supports any-seq-length fmha
                                                       true, // is_fuse
                                                       false, // with_relative_position_bias
                                                       true); // causal_mask

    GptInitParameter init_parameter;
    init_parameter.head_num_ = head_num;
    init_parameter.size_per_head_ = size_per_head;
    init_parameter.inter_size_ = inter_size;
    init_parameter.num_layers_ = num_layer;
    init_parameter.expert_num_ = 0;
    init_parameter.moe_k_ = 0;
    init_parameter.moe_layer_index_ = {};
    init_parameter.layernorm_eps_ = 1e-5f;
    init_parameter.int8_mode_ = 0;

    ParallelGptContextDecoder<T> gpt_context_decoder(
            0,
            0,
            init_parameter,
            tensor_para,
            pipeline_para,
            stream,
            cublas_wrapper,
            allocator,
            false, // is_free_buffer_after_forward
            true, // is_context_qk_buf_float
            attention_type, // attention_type
            false, // sparse
            nullptr, // custom_all_reduce_comm
            false // enable_custom_all_reduce
    );

    /*************************** REFERENCE PART *********************************/

    auto decoder_inputs = TensorMap::fromNpyFolder(data_dir);

    const size_t seq_num = decoder_inputs.at("decoder_input").shape()[0];
    const size_t seq_len = decoder_inputs.at("decoder_input").shape()[1];

    const std::vector<size_t> self_k_cache_shape = {num_layer / 1,
                                                    seq_num,
                                                    head_num,
                                                    size_per_head / (16 / sizeof(T)),
                                                    max_seq_len,
                                                    16 / sizeof(T)};
    const std::vector<size_t> self_v_cache_shape = {num_layer / 1,
                                                    seq_num,
                                                    head_num,
                                                    max_seq_len,
                                                    size_per_head};

    TensorMap decoder_outputs ({
        {"decoder_output", create_tensor(MEMORY_GPU, data_type, {seq_num, (size_t)seq_len, hidden_units}, false)},
        {"key_cache", create_tensor(MEMORY_GPU, data_type, self_k_cache_shape, true)},
        {"value_cache", create_tensor(MEMORY_GPU, data_type, self_v_cache_shape, true)},
        {"last_token_hidden_units", create_tensor(MEMORY_GPU, data_type, {seq_num, hidden_units}, false)}
    });

    gpt_context_decoder.forward(
            &decoder_outputs,
            &decoder_inputs,
            &gpt_weights.decoder_layer_weights
    );

    /********************************* TEST PART *********************************/

    TensorMap decoder_outputs_test ({
        {"decoder_output", create_tensor(MEMORY_GPU,
                data_type,
                {seq_num, (size_t)seq_len, hidden_units}, false)},
        {"key_cache", create_tensor(MEMORY_GPU, data_type, self_k_cache_shape, true)},
        {"value_cache", create_tensor(MEMORY_GPU, data_type, self_v_cache_shape, true)},
        {"last_token_hidden_units", create_tensor(MEMORY_GPU, data_type, {seq_num, hidden_units}, false)}
    });

    gpt_context_decoder.forward(
            &decoder_outputs_test,
            &decoder_inputs,
            &gpt_weights.decoder_layer_weights
    );

    std::vector<std::string> keys {"decoder_output", "last_token_hidden_units", "key_cache", "value_cache"};
    for (auto key : keys) {
        all_close<T>(decoder_outputs.at(key), decoder_outputs_test.at(key));
        printf(".");
    }
    puts("");

    free_tensors(decoder_outputs);
    free_tensors(decoder_outputs_test);
    free_tensors(decoder_inputs);

    return true;
}

Tensor tensor_to_cpu(Tensor &tensor)
{
    FT_CHECK(tensor.where() == MEMORY_GPU);
    void *host_ptr = malloc(tensor.sizeBytes());
    cudaMemcpy(host_ptr, tensor.data(), tensor.sizeBytes(), cudaMemcpyDeviceToHost);

    return Tensor {MEMORY_CPU, tensor.type(), tensor.shape(), host_ptr};
}

Tensor create_tensor(MemoryType memory_type, DataType data_type, std::vector<size_t> shape, bool init_zero)
{
    auto size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    auto size_bytes = size * getTypeSize(data_type);

    void* data = nullptr;
    if (memory_type == MEMORY_GPU) {
        cudaMalloc(&data, size_bytes);
        if (init_zero) {
            cudaMemset(data, 0, size_bytes);
        }
    }
    else {
        data = malloc(size_bytes);
        if (init_zero) {
            memset(data, 0, size_bytes);
        }
    }
    return Tensor(memory_type, data_type, shape, data);
}

void free_tensors(std::vector<Tensor> &tensors)
{
    for (auto &tensor : tensors) {
        if (tensor.where() == MEMORY_GPU) {
            cudaFree((void *) tensor.data());
        }
        else {
            free((void *) tensor.data());
        }
    }
}

void free_tensors(TensorMap &tensors)
{
    for (auto &key : tensors.keys()) {
        Tensor tensor = tensors.at(key);
        if (tensor.where() == MEMORY_GPU) {
            cudaFree((void *)tensor.data());
        }
        else {
            free((void *)tensor.data());
        }
    }
}
template<typename T>
bool all_close(Tensor &tensor_x, Tensor &tensor_y)
{
    Tensor tensor_x_h = tensor_to_cpu(tensor_x);
    Tensor tensor_y_h = tensor_to_cpu(tensor_y);

    FT_CHECK(tensor_x.size() == tensor_y.size());
    size_t n_elems = tensor_x.size();

    const float r_tol = 1e-5;
    const float a_tol = 1e-8;
    for (size_t idx = 0; idx < n_elems; idx++) {
        const float x_value = tensor_x_h.getPtr<T>()[idx];
        const float y_value = tensor_y_h.getPtr<T>()[idx];

        FT_CHECK(fabsf(x_value - y_value) <= (a_tol + r_tol * fabsf(y_value)));
    }

    free((void *) tensor_x_h.data());
    free((void *) tensor_y_h.data());

    return true;
}
