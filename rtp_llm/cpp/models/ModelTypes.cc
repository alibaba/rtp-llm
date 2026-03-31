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

    // NCCL requires pinned memory for CPU buffers. Rank 0's tensors may be
    // regular (non-pinned) CPU memory from NormalBatchStreamProcessor, so we
    // pin them here. pinned_holders keeps the pinned copies alive until after
    // broadcast + syncAndCheck completes.
    std::vector<torch::Tensor> pinned_holders;
    auto                       ensurePinned = [&pinned_holders](const torch::Tensor& t) -> torch::Tensor {
        if (!t.defined())
            return t;
        if (t.is_cpu() && !t.is_pinned()) {
            auto pinned = t.pin_memory();
            pinned_holders.push_back(pinned);
            return pinned;
        }
        return t;
    };

    std::vector<torch::Tensor> buffers;
    buffers.emplace_back(ensurePinned(inputs.combo_tokens));
    buffers.emplace_back(ensurePinned(inputs.input_lengths));
    buffers.emplace_back(ensurePinned(inputs.sequence_lengths));
    buffers.emplace_back(ensurePinned(inputs.prefix_lengths));
    if (max_kernel_blocks || max_blocks) {
        if (inputs.kv_cache_kernel_block_id.defined()) {
            buffers.emplace_back(ensurePinned(inputs.kv_cache_kernel_block_id));
        }
        if (inputs.kv_cache_block_id.defined()) {
            buffers.emplace_back(ensurePinned(inputs.kv_cache_block_id));
        }
        if (layer_to_group_len) {
            buffers.emplace_back(ensurePinned(inputs.kv_cache_layer_to_group));
        }
        if (group_types_len) {
            buffers.emplace_back(ensurePinned(inputs.kv_cache_group_types));
        }
        if (inputs.pd_separation) {
            buffers.emplace_back(ensurePinned(inputs.cache_keys));
        }
        if (inputs.kv_cache_update_mapping.defined()) {
            buffers.emplace_back(ensurePinned(inputs.kv_cache_update_mapping));
        }
    }
    buffers.emplace_back(ensurePinned(inputs.request_id));
    buffers.emplace_back(ensurePinned(inputs.request_pd_separation));
    buffers.emplace_back(ensurePinned(inputs.lm_output_indexes));
    buffers.emplace_back(ensurePinned(inputs.lm_output_lengths));
    if (combo_position_ids_size) {
        buffers.emplace_back(ensurePinned(inputs.combo_position_ids));
    }
    if (text_tokens_mask_size) {
        buffers.emplace_back(ensurePinned(inputs.text_tokens_mask));
    }
    if (mm_features_locs_size) {
        buffers.emplace_back(ensurePinned(inputs.mm_features_locs));
    }
    if (mm_features_num) {
        for (auto& mm_feature : inputs.multimodal_features.value()) {
            buffers.emplace_back(mm_feature);  // already on CUDA
        }
    }
    if (hidden_states_size) {
        buffers.emplace_back(ensurePinned(inputs.last_hidden_states));
    }
    execBroadcast({buffers, 0});
    cudaSyncAndCheck();
}

}  // namespace rtp_llm
