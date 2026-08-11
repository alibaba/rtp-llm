#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/models_py/bindings/core/torch_utils/TypeConvert.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

namespace rtp_llm {

GptModelInputShapeHints getModelInputShapeHints(const GptModelInputs& inputs) {
    GptModelInputShapeHints shape_hints{};
    shape_hints[GptModelInputIndex::comboTokens] = inputs.combo_tokens.defined() ? inputs.combo_tokens.numel() : 0;
    shape_hints[GptModelInputIndex::inputLengths] = inputs.input_lengths.defined() ? inputs.input_lengths.numel() : 0;
    shape_hints[GptModelInputIndex::sequenceLengths] =
        inputs.sequence_lengths.defined() ? inputs.sequence_lengths.numel() : 0;
    shape_hints[GptModelInputIndex::prefixLengths] =
        inputs.prefix_lengths.defined() ? inputs.prefix_lengths.numel() : 0;
    shape_hints[GptModelInputIndex::maxKernelBlocksPerBatch] =
        inputs.kv_cache_kernel_block_id.defined() ? inputs.kv_cache_kernel_block_id.size(2) : 0;
    shape_hints[GptModelInputIndex::maxBlocksPerBatch] =
        inputs.kv_cache_block_id.defined() ? inputs.kv_cache_block_id.size(2) : 0;
    shape_hints[GptModelInputIndex::cacheKeysWidth] =
        inputs.cache_keys.defined() && inputs.cache_keys.dim() >= 2 ? inputs.cache_keys.size(1) : 0;
    shape_hints[GptModelInputIndex::kvCacheGroupNum] =
        inputs.kv_cache_kernel_block_id.defined() ?
            inputs.kv_cache_kernel_block_id.size(0) :
            (inputs.kv_cache_block_id.defined() ? inputs.kv_cache_block_id.size(0) : 1);
    shape_hints[GptModelInputIndex::kvCacheLayerToGroupLen] =
        inputs.kv_cache_layer_to_group.defined() ? inputs.kv_cache_layer_to_group.numel() : 0;
    shape_hints[GptModelInputIndex::kvCacheGroupTypesLen] =
        inputs.kv_cache_group_types.defined() ? inputs.kv_cache_group_types.numel() : 0;
    shape_hints[GptModelInputIndex::kvCacheUpdateCopyNum] =
        inputs.kv_cache_update_mapping.defined() ? inputs.kv_cache_update_mapping.size(0) : 0;
    shape_hints[GptModelInputIndex::lmOutputIndexes] =
        inputs.lm_output_indexes.defined() ? inputs.lm_output_indexes.numel() : 0;
    shape_hints[GptModelInputIndex::comboPositionIds] =
        inputs.combo_position_ids.defined() ? inputs.combo_position_ids.numel() : 0;
    shape_hints[GptModelInputIndex::textTokensMask] =
        inputs.text_tokens_mask.defined() ? inputs.text_tokens_mask.numel() : 0;
    shape_hints[GptModelInputIndex::mmFeaturesLocs] =
        inputs.mm_features_locs.defined() ? inputs.mm_features_locs.numel() : 0;
    shape_hints[GptModelInputIndex::mmFeaturesNum] =
        inputs.multimodal_features.has_value() ? static_cast<int64_t>(inputs.multimodal_features.value().size()) : 0;
    shape_hints[GptModelInputIndex::mmFeaturesSize] =
        shape_hints[GptModelInputIndex::mmFeaturesNum] ? inputs.multimodal_features.value()[0].size(1) : 0;
    shape_hints[GptModelInputIndex::mmFeaturesDtype] =
        shape_hints[GptModelInputIndex::mmFeaturesNum] ?
            static_cast<int64_t>(torchDTypeToDataType(inputs.multimodal_features.value()[0].dtype())) :
            0;
    shape_hints[GptModelInputIndex::needAllLogits]       = inputs.need_all_logits;
    shape_hints[GptModelInputIndex::needAllHiddenStates] = inputs.need_all_hidden_states;
    shape_hints[GptModelInputIndex::mtpHiddenStates] =
        inputs.last_hidden_states.defined() ? inputs.last_hidden_states.numel() : 0;
    shape_hints[GptModelInputIndex::mtpHiddenStatesDtype] =
        inputs.last_hidden_states.defined() ?
            static_cast<int64_t>(torchDTypeToDataType(inputs.last_hidden_states.dtype())) :
            0;
    shape_hints[GptModelInputIndex::skipRun] = inputs.skip_run;
    shape_hints[GptModelInputIndex::gptModelRequestLength] =
        inputs.request_id.defined() ? inputs.request_id.numel() : 0;
    shape_hints[GptModelInputIndex::isFakeStream] = inputs.is_fake_stream;
    shape_hints[GptModelInputIndex::mtpHiddenStatesRows] =
        inputs.last_hidden_states.defined() ? inputs.last_hidden_states.size(0) : 0;

    uint32_t device_bits = 0;
    if (inputs.combo_tokens.defined() && inputs.combo_tokens.is_cuda()) {
        device_bits |= GptModelInputDeviceBit::kDeviceBitComboTokens;
    }
    if (inputs.input_lengths.defined() && inputs.input_lengths.is_cuda()) {
        device_bits |= GptModelInputDeviceBit::kDeviceBitInputLengths;
    }
    if (inputs.sequence_lengths.defined() && inputs.sequence_lengths.is_cuda()) {
        device_bits |= GptModelInputDeviceBit::kDeviceBitSequenceLengths;
    }
    if (inputs.prefix_lengths.defined() && inputs.prefix_lengths.is_cuda()) {
        device_bits |= GptModelInputDeviceBit::kDeviceBitPrefixLengths;
    }
    if (inputs.lm_output_indexes.defined() && inputs.lm_output_indexes.is_cuda()) {
        device_bits |= GptModelInputDeviceBit::kDeviceBitLmOutputIndexes;
    }
    shape_hints[GptModelInputIndex::tensorDeviceMap] = static_cast<int64_t>(device_bits);
    return shape_hints;
}

torch::Tensor makeModelInputShapeHintsTensor(const GptModelInputs& inputs) {
    const auto shape_hints = getModelInputShapeHints(inputs);
    auto       tensor = torch::empty({static_cast<int64_t>(shape_hints.size())}, torch::kInt64).pin_memory();
    std::copy(shape_hints.begin(), shape_hints.end(), tensor.data_ptr<int64_t>());
    return tensor;
}

std::array<int64_t, 2> decodeMtpHiddenStatesShape(int64_t total_numel, int64_t rows) {
    RTP_LLM_CHECK_WITH_INFO(total_numel >= 0,
                            "last_hidden_states total_numel must be non-negative, got %lld",
                            static_cast<long long>(total_numel));
    RTP_LLM_CHECK_WITH_INFO(
        rows >= 0, "last_hidden_states rows must be non-negative, got %lld", static_cast<long long>(rows));
    if (total_numel == 0) {
        RTP_LLM_CHECK_WITH_INFO(
            rows == 0, "empty last_hidden_states must have zero rows, got %lld", static_cast<long long>(rows));
        return {0, 0};
    }
    RTP_LLM_CHECK_WITH_INFO(rows > 0, "non-empty last_hidden_states must have a positive row count");
    RTP_LLM_CHECK_WITH_INFO(total_numel % rows == 0,
                            "last_hidden_states numel %lld is not divisible by rows %lld",
                            static_cast<long long>(total_numel),
                            static_cast<long long>(rows));
    const int64_t cols = total_numel / rows;
    RTP_LLM_CHECK_WITH_INFO(cols > 0, "non-empty last_hidden_states must have a positive column count");
    return {rows, cols};
}

void tpSyncModelInputs(GptModelInputs& inputs, const ParallelismConfig& parallelism_config) {
    if (parallelism_config.tp_size <= 1) {
        return;
    }

    // The UDS-backed CPU broadcaster (used by execBroadcastCpu below) is
    // bootstrapped from Python in collective_torch._register_process_groups_to_cpp,
    // which guarantees deterministic timing across TP siblings. Cross-node TP
    // skips the init and falls back to NCCL automatically inside execBroadcastCpu.

    // first sync stage: shape hints
    auto shape_hints_t   = makeModelInputShapeHintsTensor(inputs);
    auto shape_hints_ptr = shape_hints_t.data_ptr<int64_t>();

    // CPU broadcast: routed through CpuTpBroadcaster (UDS) when intra-node;
    // execBroadcastCpu's fallback path keeps the NCCL+cudaSyncAndCheck
    // contract for cross-node TP.
    execBroadcastCpu({{shape_hints_t}, 0});

    // multimodal features shape broadcast
    torch::Tensor mm_features_shape_t;
    int64_t*      mm_features_shape_ptr = nullptr;
    inputs.need_all_logits              = shape_hints_ptr[GptModelInputIndex::needAllLogits];
    inputs.need_all_hidden_states       = shape_hints_ptr[GptModelInputIndex::needAllHiddenStates];
    inputs.skip_run                     = shape_hints_ptr[GptModelInputIndex::skipRun];
    inputs.is_fake_stream               = shape_hints_ptr[GptModelInputIndex::isFakeStream];
    if (inputs.skip_run) {
        return;
    }
    const int64_t mm_features_num = shape_hints_ptr[GptModelInputIndex::mmFeaturesNum];
    RTP_LLM_CHECK_WITH_INFO(mm_features_num >= 0,
                            "mm_features_num must be non-negative, got %lld",
                            static_cast<long long>(mm_features_num));
    if (mm_features_num) {
        mm_features_shape_t   = torch::empty({mm_features_num}, torch::kInt64).pin_memory();
        mm_features_shape_ptr = mm_features_shape_t.data_ptr<int64_t>();
        for (int64_t i = 0; i < mm_features_num; ++i) {
            mm_features_shape_ptr[i] =
                inputs.multimodal_features.has_value() ? inputs.multimodal_features.value()[i].size(0) : 0;
        }
        // CPU broadcast (UDS path; fallback handles cudaSyncAndCheck).
        execBroadcastCpu({{mm_features_shape_t}, 0});
    }

    auto checkedHint = [&](GptModelInputIndex index, const char* name) -> int64_t {
        const auto value = shape_hints_ptr[index];
        RTP_LLM_CHECK_WITH_INFO(
            value >= 0, "tpSyncModelInputs received negative %s shape hint: %lld", name, static_cast<long long>(value));
        return value;
    };
    const auto max_kernel_blocks  = checkedHint(GptModelInputIndex::maxKernelBlocksPerBatch, "maxKernelBlocksPerBatch");
    const auto max_blocks = checkedHint(GptModelInputIndex::maxBlocksPerBatch, "maxBlocksPerBatch");
    const auto cache_keys_width = checkedHint(GptModelInputIndex::cacheKeysWidth, "cacheKeysWidth");
    const auto kv_cache_group_num = checkedHint(GptModelInputIndex::kvCacheGroupNum, "kvCacheGroupNum");
    const auto layer_to_group_len = checkedHint(GptModelInputIndex::kvCacheLayerToGroupLen, "kvCacheLayerToGroupLen");
    const auto group_types_len = checkedHint(GptModelInputIndex::kvCacheGroupTypesLen, "kvCacheGroupTypesLen");
    const auto combo_position_ids_size = checkedHint(GptModelInputIndex::comboPositionIds, "comboPositionIds");
    const auto text_tokens_mask_size = checkedHint(GptModelInputIndex::textTokensMask, "textTokensMask");
    const auto mm_features_locs_size = checkedHint(GptModelInputIndex::mmFeaturesLocs, "mmFeaturesLocs");
    const auto hidden_states_size = checkedHint(GptModelInputIndex::mtpHiddenStates, "mtpHiddenStates");
    const auto request_length = checkedHint(GptModelInputIndex::gptModelRequestLength, "gptModelRequestLength");

    auto allocBuf = [&](rtp_llm::DataType       dtype,
                        std::vector<int64_t>    dims,
                        rtp_llm::AllocationType atype = rtp_llm::AllocationType::HOST) -> torch::Tensor {
        auto torch_dtype = dataTypeToTorchType(dtype);
        int64_t numel     = 1;
        for (const auto dim : dims) {
            RTP_LLM_CHECK_WITH_INFO(
                dim >= 0, "tpSyncModelInputs cannot allocate a negative dimension: %lld", static_cast<long long>(dim));
            if (dim == 0) {
                numel = 0;
                continue;
            }
            RTP_LLM_CHECK_WITH_INFO(numel <= std::numeric_limits<int64_t>::max() / dim,
                                    "tpSyncModelInputs tensor shape numel overflow");
            numel *= dim;
        }
        const auto element_size = static_cast<int64_t>(torch::elementSize(torch_dtype));
        RTP_LLM_CHECK_WITH_INFO(element_size > 0 && numel <= std::numeric_limits<int64_t>::max() / element_size,
                                "tpSyncModelInputs tensor storage size overflow");
        auto options     = torch::TensorOptions(torch_dtype);
        if (atype == rtp_llm::AllocationType::DEVICE) {
            options = options.device(torch::kCUDA);
        }
        auto tensor = torch::empty(dims, options);
        // NCCL broadcast requires pinned memory for CPU buffers
        if (atype != rtp_llm::AllocationType::DEVICE) {
            tensor = tensor.pin_memory();
        }
        return tensor;
    };

    bool is_non_root = parallelism_config.tp_rank != 0;
    if (is_non_root) {
        const auto context_batch_size = checkedHint(GptModelInputIndex::prefixLengths, "prefixLengths");

        // Respect the root-side device bitmap so all ranks classify tensors the
        // same way and preserve NCCL broadcast ordering.
        const uint32_t device_bits = static_cast<uint32_t>(shape_hints_ptr[GptModelInputIndex::tensorDeviceMap]);
        auto           pickAlloc   = [&](GptModelInputDeviceBit bit) {
            return (device_bits & bit) ? rtp_llm::AllocationType::DEVICE : rtp_llm::AllocationType::HOST;
        };

        inputs.combo_tokens = allocBuf(rtp_llm::DataType::TYPE_INT32,
                                       {checkedHint(GptModelInputIndex::comboTokens, "comboTokens")},
                                       pickAlloc(GptModelInputDeviceBit::kDeviceBitComboTokens));
        inputs.input_lengths = allocBuf(rtp_llm::DataType::TYPE_INT32,
                                        {checkedHint(GptModelInputIndex::inputLengths, "inputLengths")},
                                        pickAlloc(GptModelInputDeviceBit::kDeviceBitInputLengths));
        inputs.sequence_lengths = allocBuf(rtp_llm::DataType::TYPE_INT32,
                                           {checkedHint(GptModelInputIndex::sequenceLengths, "sequenceLengths")},
                                           pickAlloc(GptModelInputDeviceBit::kDeviceBitSequenceLengths));
        inputs.prefix_lengths = allocBuf(rtp_llm::DataType::TYPE_INT32,
                                         {context_batch_size},
                                         pickAlloc(GptModelInputDeviceBit::kDeviceBitPrefixLengths));
        if (max_kernel_blocks != 0) {
            // kv_cache_kernel_block_id is now device-resident on the producer (rank 0). Allocate
            // the matching buffer on CUDA for non-root ranks so the gpu_packed branch below
            // classifies it identically across ranks (otherwise pack/unpack drifts off-by-tensor).
            inputs.kv_cache_kernel_block_id = allocBuf(
                rtp_llm::DataType::TYPE_INT32,
                {kv_cache_group_num, checkedHint(GptModelInputIndex::inputLengths, "inputLengths"), max_kernel_blocks},
                rtp_llm::AllocationType::DEVICE);
            inputs.kv_cache_update_mapping =
                allocBuf(rtp_llm::DataType::TYPE_INT32,
                {checkedHint(GptModelInputIndex::kvCacheUpdateCopyNum, "kvCacheUpdateCopyNum"), 2});
        }
        if (max_blocks != 0) {
            inputs.kv_cache_block_id = allocBuf(
                rtp_llm::DataType::TYPE_INT32,
                {kv_cache_group_num, checkedHint(GptModelInputIndex::inputLengths, "inputLengths"), max_blocks});
            if (inputs.pd_separation) {
                inputs.cache_keys = allocBuf(rtp_llm::DataType::TYPE_INT64,
                                             {context_batch_size, cache_keys_width ? cache_keys_width : max_blocks});
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
        inputs.lm_output_indexes = allocBuf(rtp_llm::DataType::TYPE_INT32,
                                            {checkedHint(GptModelInputIndex::lmOutputIndexes, "lmOutputIndexes")},
                                            pickAlloc(GptModelInputDeviceBit::kDeviceBitLmOutputIndexes));
        if (combo_position_ids_size) {
            inputs.combo_position_ids = allocBuf(rtp_llm::DataType::TYPE_INT32, {combo_position_ids_size});
        }
        const auto hidden_shape = decodeMtpHiddenStatesShape(
            hidden_states_size, checkedHint(GptModelInputIndex::mtpHiddenStatesRows, "mtpHiddenStatesRows"));
        if (hidden_states_size) {
            inputs.last_hidden_states =
                allocBuf((rtp_llm::DataType)shape_hints_ptr[GptModelInputIndex::mtpHiddenStatesDtype],
                         {hidden_shape[0], hidden_shape[1]},
                         rtp_llm::AllocationType::DEVICE);
        } else {
            inputs.last_hidden_states = torch::Tensor();
        }
        if (text_tokens_mask_size) {
            inputs.text_tokens_mask = allocBuf(rtp_llm::DataType::TYPE_INT32, {text_tokens_mask_size});
        }
        if (mm_features_locs_size) {
            inputs.mm_features_locs = allocBuf(rtp_llm::DataType::TYPE_INT32, {mm_features_locs_size});
        }
        if (mm_features_num) {
            std::vector<torch::Tensor> mm_features;
            const auto mm_data_type =
                static_cast<rtp_llm::DataType>(shape_hints_ptr[GptModelInputIndex::mmFeaturesDtype]);
            for (int64_t mm_index = 0; mm_index < mm_features_num; ++mm_index) {
                const auto mm_rows = mm_features_shape_ptr[mm_index];
                const auto mm_cols = checkedHint(GptModelInputIndex::mmFeaturesSize, "mmFeaturesSize");
                mm_features.emplace_back(allocBuf(mm_data_type, {mm_rows, mm_cols}, rtp_llm::AllocationType::DEVICE));
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
        RTP_LLM_CHECK_WITH_INFO(size >= 0 && alignment > 0, "tpSyncModelInputs invalid packed-buffer alignment input");
        RTP_LLM_CHECK_WITH_INFO(size <= std::numeric_limits<int64_t>::max() - (alignment - 1),
                                "tpSyncModelInputs packed-buffer alignment overflow");
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
        const auto raw_nbytes = tp->nbytes();
        RTP_LLM_CHECK_WITH_INFO(raw_nbytes <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                                "tpSyncModelInputs tensor byte size exceeds int64");
        const auto nb      = static_cast<int64_t>(raw_nbytes);
        const auto aligned = align_up(nb, kPackAlignment);
        if (tp->is_cuda()) {
            gpu_entries.push_back({tp, gpu_total_bytes, nb});
            RTP_LLM_CHECK_WITH_INFO(gpu_total_bytes <= std::numeric_limits<int64_t>::max() - aligned,
                                    "tpSyncModelInputs GPU packed-buffer size overflow");
            gpu_total_bytes += aligned;
        } else {
            cpu_entries.push_back({tp, cpu_total_bytes, nb});
            RTP_LLM_CHECK_WITH_INFO(cpu_total_bytes <= std::numeric_limits<int64_t>::max() - aligned,
                                    "tpSyncModelInputs CPU packed-buffer size overflow");
            cpu_total_bytes += aligned;
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
            auto*              packed_base = static_cast<uint8_t*>(gpu_packed.data_ptr());
            FusedD2DCopyParams fused_params;
            auto               flush_fused_copy = [&]() {
                if (fused_params.num_copies > 0) {
                    fusedCopy(fused_params);
                    fused_params.clear();
                }
            };
            for (auto& e : gpu_entries) {
                if (e.tensor->is_contiguous()) {
                    if (fused_params.num_copies == MAX_FUSED_D2D_COPIES) {
                        flush_fused_copy();
                    }
                    fused_params.add(e.tensor->data_ptr(), packed_base + e.offset, static_cast<size_t>(e.nbytes));
                    continue;
                }

                // Preserve the old logical-order copy for rare non-contiguous tensors.
                flush_fused_copy();
                auto contig    = e.tensor->contiguous();
                auto src_bytes = torch::from_blob(
                    contig.data_ptr(), {e.nbytes}, torch::TensorOptions(torch::kUInt8).device(contig.device()));
                gpu_packed.narrow(0, e.offset, e.nbytes).copy_(src_bytes);
            }
            flush_fused_copy();
        }
    }

    // Broadcast at most 2 packed buffers instead of N individual tensors.
    if (cpu_packed.defined()) {
        execBroadcastCpu({{cpu_packed}, 0});
    }

    if (gpu_packed.defined()) {
        // gpu no need to sync communication
        execBroadcast({{gpu_packed}, 0});
    }

    // Unpack from packed buffers back to each tensor's original storage.
    if (!is_root) {
        if (cpu_total_bytes > 0) {
            auto* base = static_cast<const uint8_t*>(cpu_packed.data_ptr());
            for (auto& e : cpu_entries) {
                std::memcpy(e.tensor->data_ptr(), base + e.offset, e.nbytes);
            }
        }
        if (gpu_total_bytes > 0) {
            auto*              packed_base = static_cast<uint8_t*>(gpu_packed.data_ptr());
            FusedD2DCopyParams fused_params;
            auto               flush_fused_copy = [&]() {
                if (fused_params.num_copies > 0) {
                    fusedCopy(fused_params);
                    fused_params.clear();
                }
            };
            for (auto& e : gpu_entries) {
                if (e.tensor->is_contiguous()) {
                    if (fused_params.num_copies == MAX_FUSED_D2D_COPIES) {
                        flush_fused_copy();
                    }
                    fused_params.add(packed_base + e.offset, e.tensor->data_ptr(), static_cast<size_t>(e.nbytes));
                    continue;
                }

                flush_fused_copy();
                auto src_tensor = torch::from_blob(packed_base + e.offset, e.tensor->sizes(), e.tensor->options());
                e.tensor->copy_(src_tensor);
            }
            flush_fused_copy();
        }
    }
}

}  // namespace rtp_llm
