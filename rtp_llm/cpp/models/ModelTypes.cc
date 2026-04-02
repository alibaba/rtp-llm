#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/core/torch_utils/TypeConvert.h"
#include "rtp_llm/cpp/core/ExecOps.h"

namespace rtp_llm {

void tpSyncModelInputs(GptModelInputs& inputs, const ParallelismConfig& parallelism_config) {
    if (parallelism_config.tp_size <= 1) {
        return;
    }
    const size_t shape_hints_size = GptModelInputIndex::gptModelInputLength;
    auto         shape_hints_t    = torch::empty({(int64_t)shape_hints_size}, torch::kInt32).pin_memory();
    auto         shape_hints_ptr  = shape_hints_t.data_ptr<int32_t>();
    shape_hints_ptr[GptModelInputIndex::comboTokens] = inputs.combo_tokens.defined() ? inputs.combo_tokens.numel() : 0;
    shape_hints_ptr[GptModelInputIndex::inputLengths] =
        inputs.input_lengths.defined() ? inputs.input_lengths.numel() : 0;
    shape_hints_ptr[GptModelInputIndex::sequenceLengths] =
        inputs.sequence_lengths.defined() ? inputs.sequence_lengths.numel() : 0;
    shape_hints_ptr[GptModelInputIndex::prefixLengths] =
        inputs.prefix_lengths.defined() ? inputs.prefix_lengths.numel() : 0;
    shape_hints_ptr[GptModelInputIndex::maxKernelBlocksPerBatch] =
        inputs.kv_cache_kernel_block_id.defined() ? inputs.kv_cache_kernel_block_id.size(2) : 0;
    shape_hints_ptr[GptModelInputIndex::maxBlocksPerBatch] =
        inputs.kv_cache_block_id.defined() ? inputs.kv_cache_block_id.size(2) : 0;
    shape_hints_ptr[GptModelInputIndex::kvCacheGroupNum] =
        inputs.kv_cache_kernel_block_id.defined() ?
            inputs.kv_cache_kernel_block_id.size(0) :
            (inputs.kv_cache_block_id.defined() ? inputs.kv_cache_block_id.size(0) : 1);
    shape_hints_ptr[GptModelInputIndex::kvCacheLayerToGroupLen] =
        inputs.kv_cache_layer_to_group.defined() ? inputs.kv_cache_layer_to_group.numel() : 0;
    shape_hints_ptr[GptModelInputIndex::kvCacheGroupTypesLen] =
        inputs.kv_cache_group_types.defined() ? inputs.kv_cache_group_types.numel() : 0;
    shape_hints_ptr[GptModelInputIndex::kvCacheUpdateCopyNum] =
        inputs.kv_cache_update_mapping.defined() ? inputs.kv_cache_update_mapping.size(0) : 0;
    shape_hints_ptr[GptModelInputIndex::lmOutputIndexes] =
        inputs.lm_output_indexes.defined() ? inputs.lm_output_indexes.numel() : 0;
    shape_hints_ptr[GptModelInputIndex::lmOutputLengthes] =
        inputs.lm_output_lengths.defined() ? inputs.lm_output_lengths.numel() : 0;
    shape_hints_ptr[GptModelInputIndex::comboPositionIds] =
        inputs.combo_position_ids.defined() ? inputs.combo_position_ids.numel() : 0;
    shape_hints_ptr[GptModelInputIndex::textTokensMask] =
        inputs.text_tokens_mask.defined() ? inputs.text_tokens_mask.numel() : 0;
    shape_hints_ptr[GptModelInputIndex::mmFeaturesLocs] =
        inputs.mm_features_locs.defined() ? inputs.mm_features_locs.numel() : 0;
    shape_hints_ptr[GptModelInputIndex::mmFeaturesNum] =
        inputs.multimodal_features.has_value() ? inputs.multimodal_features.value().size() : 0;
    shape_hints_ptr[GptModelInputIndex::mmFeaturesSize] =
        shape_hints_ptr[GptModelInputIndex::mmFeaturesNum] ? inputs.multimodal_features.value()[0].size(1) : 0;
    shape_hints_ptr[GptModelInputIndex::mmFeaturesDtype] =
        shape_hints_ptr[GptModelInputIndex::mmFeaturesNum] ?
            (std::uint8_t)torchDTypeToDataType(inputs.multimodal_features.value()[0].dtype()) :
            0;
    shape_hints_ptr[GptModelInputIndex::needAllLogits] = inputs.need_all_logits;
    shape_hints_ptr[GptModelInputIndex::mtpHiddenStates] =
        inputs.last_hidden_states.defined() ? inputs.last_hidden_states.numel() : 0;
    shape_hints_ptr[GptModelInputIndex::mtpHiddenStatesDtype] =
        inputs.last_hidden_states.defined() ? (std::uint8_t)torchDTypeToDataType(inputs.last_hidden_states.dtype()) : 0;
    shape_hints_ptr[GptModelInputIndex::skipRun] = inputs.skip_run;
    shape_hints_ptr[GptModelInputIndex::gptModelRequestLength] =
        inputs.request_id.defined() ? inputs.request_id.numel() : 0;
    shape_hints_ptr[GptModelInputIndex::isFakeStream] = inputs.is_fake_stream;
    execBroadcast({{shape_hints_t}, 0});
    execSyncCommunication(false);
    cudaSyncAndCheck();

    // multimodal features shape broadcast
    torch::Tensor mm_features_shape_t;
    int32_t*      mm_features_shape_ptr = nullptr;
    inputs.need_all_logits              = shape_hints_ptr[GptModelInputIndex::needAllLogits];
    inputs.skip_run                     = shape_hints_ptr[GptModelInputIndex::skipRun];
    inputs.is_fake_stream               = shape_hints_ptr[GptModelInputIndex::isFakeStream];
    if (inputs.skip_run) {
        return;
    }
    const size_t mm_features_num = shape_hints_ptr[GptModelInputIndex::mmFeaturesNum];
    if (mm_features_num) {
        mm_features_shape_t   = torch::empty({(int64_t)mm_features_num}, torch::kInt32).pin_memory();
        mm_features_shape_ptr = mm_features_shape_t.data_ptr<int32_t>();
        for (size_t i = 0; i < mm_features_num; ++i) {
            mm_features_shape_ptr[i] =
                inputs.multimodal_features.has_value() ? inputs.multimodal_features.value()[i].size(0) : 0;
        }
        execBroadcast({{mm_features_shape_t}, 0});
        execSyncCommunication(false);
        cudaSyncAndCheck();
    }

    auto   max_kernel_blocks       = (size_t)shape_hints_ptr[GptModelInputIndex::maxKernelBlocksPerBatch];
    auto   max_blocks              = (size_t)shape_hints_ptr[GptModelInputIndex::maxBlocksPerBatch];
    auto   kv_cache_group_num      = (size_t)shape_hints_ptr[GptModelInputIndex::kvCacheGroupNum];
    auto   layer_to_group_len      = (size_t)shape_hints_ptr[GptModelInputIndex::kvCacheLayerToGroupLen];
    auto   group_types_len         = (size_t)shape_hints_ptr[GptModelInputIndex::kvCacheGroupTypesLen];
    auto   combo_position_ids_size = shape_hints_ptr[GptModelInputIndex::comboPositionIds];
    auto   text_tokens_mask_size   = shape_hints_ptr[GptModelInputIndex::textTokensMask];
    auto   mm_features_locs_size   = shape_hints_ptr[GptModelInputIndex::mmFeaturesLocs];
    auto   hidden_states_size      = shape_hints_ptr[GptModelInputIndex::mtpHiddenStates];
    size_t request_length          = shape_hints_ptr[GptModelInputIndex::gptModelRequestLength];

    auto allocBuf = [&](rtp_llm::DataType       dtype,
                        std::vector<size_t>     dims,
                        rtp_llm::AllocationType atype = rtp_llm::AllocationType::HOST) -> torch::Tensor {
        auto torch_dtype = dataTypeToTorchType(dtype);
        auto options     = torch::TensorOptions(torch_dtype);
        if (atype == rtp_llm::AllocationType::DEVICE) {
            options = options.device(torch::kCUDA);
        }
        std::vector<int64_t> dims64(dims.begin(), dims.end());
        auto                 tensor = torch::empty(dims64, options);
        // NCCL broadcast requires pinned memory for CPU buffers
        if (atype != rtp_llm::AllocationType::DEVICE) {
            tensor = tensor.pin_memory();
        }
        return tensor;
    };

    bool is_non_root = parallelism_config.tp_rank != 0;
    if (is_non_root) {
        auto context_batch_size = (size_t)shape_hints_ptr[GptModelInputIndex::prefixLengths];

        inputs.combo_tokens =
            allocBuf(rtp_llm::DataType::TYPE_INT32, {(size_t)shape_hints_ptr[GptModelInputIndex::comboTokens]});
        inputs.input_lengths =
            allocBuf(rtp_llm::DataType::TYPE_INT32, {(size_t)shape_hints_ptr[GptModelInputIndex::inputLengths]});
        inputs.sequence_lengths =
            allocBuf(rtp_llm::DataType::TYPE_INT32, {(size_t)shape_hints_ptr[GptModelInputIndex::sequenceLengths]});
        inputs.prefix_lengths = allocBuf(rtp_llm::DataType::TYPE_INT32, {context_batch_size});
        if (max_kernel_blocks != 0) {
            inputs.kv_cache_kernel_block_id = allocBuf(
                rtp_llm::DataType::TYPE_INT32,
                {kv_cache_group_num, (size_t)shape_hints_ptr[GptModelInputIndex::inputLengths], max_kernel_blocks});
            inputs.kv_cache_update_mapping = allocBuf(
                rtp_llm::DataType::TYPE_INT32, {(size_t)shape_hints_ptr[GptModelInputIndex::kvCacheUpdateCopyNum], 2});
        }
        if (max_blocks != 0) {
            inputs.kv_cache_block_id =
                allocBuf(rtp_llm::DataType::TYPE_INT32,
                         {kv_cache_group_num, (size_t)shape_hints_ptr[GptModelInputIndex::inputLengths], max_blocks});
            if (inputs.pd_separation) {
                inputs.cache_keys = allocBuf(rtp_llm::DataType::TYPE_INT64, {context_batch_size, max_blocks});
            }
        }
        if (layer_to_group_len) {
            inputs.kv_cache_layer_to_group = allocBuf(rtp_llm::DataType::TYPE_INT32, {layer_to_group_len});
        }
        if (group_types_len) {
            inputs.kv_cache_group_types = allocBuf(rtp_llm::DataType::TYPE_INT32, {group_types_len});
        }
        inputs.request_id            = allocBuf(rtp_llm::DataType::TYPE_INT64, {request_length});
        inputs.request_pd_separation = allocBuf(rtp_llm::DataType::TYPE_BOOL, {request_length});
        inputs.lm_output_indexes =
            allocBuf(rtp_llm::DataType::TYPE_INT32, {(size_t)shape_hints_ptr[GptModelInputIndex::lmOutputIndexes]});
        inputs.lm_output_lengths =
            allocBuf(rtp_llm::DataType::TYPE_INT32, {(size_t)shape_hints_ptr[GptModelInputIndex::lmOutputLengthes]});
        if (combo_position_ids_size) {
            inputs.combo_position_ids = allocBuf(rtp_llm::DataType::TYPE_INT32, {(size_t)combo_position_ids_size});
        }
        if (shape_hints_ptr[GptModelInputIndex::mtpHiddenStates]) {
            auto hidden_states_dim0 = (size_t)shape_hints_ptr[GptModelInputIndex::comboTokens];
            auto hidden_states_dim1 = (size_t)hidden_states_size / hidden_states_dim0;
            RTP_LLM_CHECK(hidden_states_size % hidden_states_dim0 == 0);
            inputs.last_hidden_states =
                allocBuf((rtp_llm::DataType)shape_hints_ptr[GptModelInputIndex::mtpHiddenStatesDtype],
                         {hidden_states_dim0, hidden_states_dim1},
                         rtp_llm::AllocationType::DEVICE);
        }
        if (text_tokens_mask_size) {
            inputs.text_tokens_mask = allocBuf(rtp_llm::DataType::TYPE_INT32, {(size_t)text_tokens_mask_size});
        }
        if (mm_features_locs_size) {
            inputs.mm_features_locs = allocBuf(rtp_llm::DataType::TYPE_INT32, {(size_t)mm_features_locs_size});
        }
        if (mm_features_num) {
            std::vector<torch::Tensor> mm_features;
            auto                       mm_dtype =
                dataTypeToTorchType((rtp_llm::DataType)shape_hints_ptr[GptModelInputIndex::mmFeaturesDtype]);
            for (auto mm_index = 0; mm_index < mm_features_num; ++mm_index) {
                mm_features.emplace_back(torch::empty({(int64_t)mm_features_shape_ptr[mm_index],
                                                       (int64_t)shape_hints_ptr[GptModelInputIndex::mmFeaturesSize]},
                                                      torch::TensorOptions().dtype(mm_dtype).device(torch::kCUDA)));
            }
            inputs.multimodal_features = std::move(mm_features);
        }
    }

    // Collect all tensors that participate in broadcast.
    // The collect order must be deterministic and identical across all ranks.
    std::vector<torch::Tensor*> tensor_ptrs;
    auto                        collect = [&](torch::Tensor& t) {
        if (t.defined() && t.numel() > 0) {
            tensor_ptrs.push_back(&t);
        }
    };

    collect(inputs.combo_tokens);
    collect(inputs.input_lengths);
    collect(inputs.sequence_lengths);
    collect(inputs.prefix_lengths);
    if (max_kernel_blocks || max_blocks) {
        collect(inputs.kv_cache_kernel_block_id);
        collect(inputs.kv_cache_block_id);
        if (layer_to_group_len) {
            collect(inputs.kv_cache_layer_to_group);
        }
        if (group_types_len) {
            collect(inputs.kv_cache_group_types);
        }
        if (inputs.pd_separation) {
            collect(inputs.cache_keys);
        }
        collect(inputs.kv_cache_update_mapping);
    }
    collect(inputs.request_id);
    collect(inputs.request_pd_separation);
    collect(inputs.lm_output_indexes);
    collect(inputs.lm_output_lengths);
    if (combo_position_ids_size) {
        collect(inputs.combo_position_ids);
    }
    if (text_tokens_mask_size) {
        collect(inputs.text_tokens_mask);
    }
    if (mm_features_locs_size) {
        collect(inputs.mm_features_locs);
    }
    if (mm_features_num) {
        for (auto& f : inputs.multimodal_features.value()) {
            collect(f);
        }
    }
    if (hidden_states_size) {
        collect(inputs.last_hidden_states);
    }

    // Classify tensors by device type (runtime check) and calculate packed sizes.
    // Align each entry to 16 bytes so that typed access at any offset is safe
    // and GPU memory coalescing / NCCL transfers stay on fast paths.
    constexpr int64_t kPackAlignment = 16;
    auto              align_up       = [](int64_t size, int64_t alignment) -> int64_t {
        return (size + alignment - 1) & ~(alignment - 1);
    };

    struct PackEntry {
        torch::Tensor* tensor;
        int64_t        offset;
        int64_t        nbytes;
    };
    std::vector<PackEntry> cpu_entries, gpu_entries;
    int64_t                cpu_total_bytes = 0, gpu_total_bytes = 0;

    for (auto* tp : tensor_ptrs) {
        auto nb = static_cast<int64_t>(tp->nbytes());
        if (tp->is_cuda()) {
            gpu_entries.push_back({tp, gpu_total_bytes, nb});
            gpu_total_bytes += align_up(nb, kPackAlignment);
        } else {
            cpu_entries.push_back({tp, cpu_total_bytes, nb});
            cpu_total_bytes += align_up(nb, kPackAlignment);
        }
    }

    bool is_root = parallelism_config.tp_rank == 0;

    // Allocate one packed buffer per device type.
    // CPU buffer uses pinned memory (required by NCCL for host-side broadcast).
    torch::Tensor cpu_packed, gpu_packed;

    if (cpu_total_bytes > 0) {
        cpu_packed = torch::empty({cpu_total_bytes}, torch::kUInt8).pin_memory();
        if (is_root) {
            auto* base = static_cast<uint8_t*>(cpu_packed.data_ptr());
            for (auto& e : cpu_entries) {
                auto contig = e.tensor->contiguous();
                std::memcpy(base + e.offset, contig.data_ptr(), e.nbytes);
            }
        }
    }

    if (gpu_total_bytes > 0) {
        gpu_packed = torch::empty({gpu_total_bytes}, torch::TensorOptions(torch::kUInt8).device(torch::kCUDA));
        if (is_root) {
            for (auto& e : gpu_entries) {
                auto contig    = e.tensor->contiguous();
                auto src_bytes = torch::from_blob(
                    contig.data_ptr(), {e.nbytes}, torch::TensorOptions(torch::kUInt8).device(contig.device()));
                gpu_packed.narrow(0, e.offset, e.nbytes).copy_(src_bytes);
            }
        }
    }

    // Broadcast at most 2 packed buffers instead of N individual tensors.
    std::vector<torch::Tensor> packed_buffers;
    if (cpu_packed.defined()) {
        packed_buffers.push_back(cpu_packed);
    }
    if (gpu_packed.defined()) {
        packed_buffers.push_back(gpu_packed);
    }
    execBroadcast({packed_buffers, 0});
    cudaSyncAndCheck();

    // Unpack from packed buffers back to each tensor's original storage.
    if (!is_root) {
        if (cpu_total_bytes > 0) {
            auto* base = static_cast<const uint8_t*>(cpu_packed.data_ptr());
            for (auto& e : cpu_entries) {
                std::memcpy(e.tensor->data_ptr(), base + e.offset, e.nbytes);
            }
        }
        if (gpu_total_bytes > 0) {
            for (auto& e : gpu_entries) {
                auto dst_bytes = torch::from_blob(
                    e.tensor->data_ptr(), {e.nbytes}, torch::TensorOptions(torch::kUInt8).device(e.tensor->device()));
                dst_bytes.copy_(gpu_packed.narrow(0, e.offset, e.nbytes));
            }
        }
    }
}

}  // namespace rtp_llm
