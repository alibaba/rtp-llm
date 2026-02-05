#include "rtp_llm/cpp/cuda/cuda_graph_utils.h"
#include <iostream>

namespace rtp_llm {

// Helper function to print tensor info and data
void printTensorInfo(const std::string& name, const torch::Tensor& tensor, int max_print_size) {
    std::cout << "  " << name << ": defined=" << tensor.defined();
    if (tensor.defined()) {
        std::cout << ", shape=[";
        for (int i = 0; i < tensor.dim(); i++) {
            std::cout << tensor.size(i);
            if (i < tensor.dim() - 1)
                std::cout << ", ";
        }
        std::cout << "], is_cuda=" << tensor.is_cuda();
        if (!tensor.is_cuda()) {
            std::cout << ", is_pinned=" << tensor.is_pinned();
        }
        // Print data content
        if (tensor.numel() > 0) {
            auto cpu_tensor = tensor.cpu();
            int  print_size = std::min(static_cast<int>(cpu_tensor.numel()), max_print_size);
            std::cout << ", data=[";
            auto dtype = cpu_tensor.scalar_type();
            for (int i = 0; i < print_size; i++) {
                if (dtype == torch::kInt32 || dtype == torch::kInt) {
                    std::cout << cpu_tensor.data_ptr<int>()[i];
                } else if (dtype == torch::kInt64 || dtype == torch::kLong) {
                    std::cout << cpu_tensor.data_ptr<int64_t>()[i];
                } else if (dtype == torch::kFloat32 || dtype == torch::kFloat) {
                    std::cout << cpu_tensor.data_ptr<float>()[i];
                } else if (dtype == torch::kFloat16 || dtype == torch::kHalf) {
                    std::cout << static_cast<float>(cpu_tensor.data_ptr<at::Half>()[i]);
                } else {
                    std::cout << "?";
                }
                if (i < print_size - 1)
                    std::cout << ", ";
            }
            if (cpu_tensor.numel() > print_size)
                std::cout << ", ...";
            std::cout << "]";
        }
    }
    std::cout << std::endl;
}

// Helper function to print all PyModelInputs content
void debugPrintPyModelInputs(const PyModelInputs& inputs) {
    std::cout << "========== PyModelInputs Debug Info ==========" << std::endl;

    std::cout << "--- input_ids ---" << std::endl;
    std::cout << "  defined: " << inputs.input_ids.defined() << std::endl;
    if (inputs.input_ids.defined()) {
        std::cout << "  shape: [";
        for (int i = 0; i < inputs.input_ids.dim(); i++) {
            std::cout << inputs.input_ids.size(i);
            if (i < inputs.input_ids.dim() - 1)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "  dtype: " << inputs.input_ids.dtype() << std::endl;
        std::cout << "  device: " << inputs.input_ids.device() << std::endl;
        std::cout << "  is_cuda: " << inputs.input_ids.is_cuda() << std::endl;
        std::cout << "  is_pinned: " << inputs.input_ids.is_pinned() << std::endl;
        // Print data content
        if (inputs.input_ids.numel() > 0) {
            auto input_ids_cpu = inputs.input_ids.cpu();
            int  print_size    = std::min(static_cast<int>(input_ids_cpu.numel()), 20);
            std::cout << "  data (first " << print_size << " elements): [";
            for (int i = 0; i < print_size; i++) {
                std::cout << input_ids_cpu.data_ptr<int>()[i];
                if (i < print_size - 1)
                    std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
    }

    std::cout << "--- input_hiddens ---" << std::endl;
    std::cout << "  defined: " << inputs.input_hiddens.defined() << std::endl;
    if (inputs.input_hiddens.defined()) {
        std::cout << "  shape: [";
        for (int i = 0; i < inputs.input_hiddens.dim(); i++) {
            std::cout << inputs.input_hiddens.size(i);
            if (i < inputs.input_hiddens.dim() - 1)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "  dtype: " << inputs.input_hiddens.dtype() << std::endl;
        std::cout << "  device: " << inputs.input_hiddens.device() << std::endl;
        std::cout << "  is_cuda: " << inputs.input_hiddens.is_cuda() << std::endl;
        std::cout << "  is_pinned: " << inputs.input_hiddens.is_pinned() << std::endl;
    }

    std::cout << "--- attention_inputs ---" << std::endl;
    std::cout << "  is_prefill: " << inputs.attention_inputs.is_prefill << std::endl;

    printTensorInfo("prefix_lengths", inputs.attention_inputs.prefix_lengths);
    printTensorInfo("sequence_lengths", inputs.attention_inputs.sequence_lengths);
    printTensorInfo("input_lengths", inputs.attention_inputs.input_lengths);
    printTensorInfo("kv_cache_block_id_host", inputs.attention_inputs.kv_cache_block_id_host, 40);
    printTensorInfo("kv_cache_block_id_device", inputs.attention_inputs.kv_cache_block_id_device, 40);
    printTensorInfo("cu_seqlens", inputs.attention_inputs.cu_seqlens);
    printTensorInfo("cu_kv_seqlens", inputs.attention_inputs.cu_kv_seqlens);
    printTensorInfo("sequence_lengths_plus_1_d", inputs.attention_inputs.sequence_lengths_plus_1_d);
    printTensorInfo("decode_cu_seqlens_d", inputs.attention_inputs.decode_cu_seqlens_d);
    printTensorInfo("padding_offset", inputs.attention_inputs.padding_offset);

    std::cout << "  context_total_kv_length: " << inputs.attention_inputs.context_total_kv_length << std::endl;
    std::cout << "  total_tokens: " << inputs.attention_inputs.total_tokens << std::endl;
    std::cout << "=============================================" << std::endl;
}

}  // namespace rtp_llm
