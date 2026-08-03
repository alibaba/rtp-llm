#include "rtp_llm/cpp/models/ModelTypes.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include "rtp_llm/models_py/bindings/core/torch_utils/TypeConvert.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {

void tpSyncModelInputs(GptModelInputs& inputs, const ParallelismConfig& parallelism_config) {
    if (parallelism_config.tp_size <= 1) {
        return;
    }
    constexpr size_t kMaxCacheGroups  = 64;
    constexpr size_t kMaxCacheTagSize = 128;
    constexpr size_t kTagWords        = kMaxCacheTagSize / sizeof(int32_t);
    // tag length, type, physical width, kernel width, physical span, kernel span,
    // KV stride, scale stride, followed by tag bytes.
    constexpr size_t kGroupHintWords  = 8 + kTagWords;
    const size_t     shape_hints_size = GptModelInputIndex::gptModelInputLength + kMaxCacheGroups * kGroupHintWords;
    auto             shape_hints_t    = torch::empty({(int64_t)shape_hints_size}, torch::kInt32).pin_memory();
    auto             shape_hints_ptr  = shape_hints_t.data_ptr<int32_t>();
    shape_hints_ptr[GptModelInputIndex::comboTokens] = inputs.combo_tokens.defined() ? inputs.combo_tokens.numel() : 0;
    shape_hints_ptr[GptModelInputIndex::inputLengths] =
        inputs.input_lengths.defined() ? inputs.input_lengths.numel() : 0;
    shape_hints_ptr[GptModelInputIndex::sequenceLengths] =
        inputs.sequence_lengths.defined() ? inputs.sequence_lengths.numel() : 0;
    shape_hints_ptr[GptModelInputIndex::prefixLengths] =
        inputs.prefix_lengths.defined() ? inputs.prefix_lengths.numel() : 0;
    int32_t max_kernel_blocks_hint = 0;
    int32_t max_blocks_hint        = 0;
    RTP_LLM_CHECK_WITH_INFO(inputs.block_tables_by_group.size() <= kMaxCacheGroups,
                            "too many tagged KV cache groups: %zu > %zu",
                            inputs.block_tables_by_group.size(),
                            kMaxCacheGroups);
    size_t group_hint_offset = GptModelInputIndex::gptModelInputLength;
    for (const auto& [tag, table] : inputs.block_tables_by_group) {
        RTP_LLM_CHECK_WITH_INFO(table.tag == tag, "block table key/tag mismatch for tag=%s", tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(
            tag.size() <= kMaxCacheTagSize, "KV cache tag is too long: tag=%s length=%zu", tag.c_str(), tag.size());
        RTP_LLM_CHECK_WITH_INFO(table.block_ids.dim() == 2 && table.kernel_block_ids.dim() == 2,
                                "KV cache tables must be two-dimensional for tag=%s",
                                tag.c_str());
        shape_hints_ptr[group_hint_offset]     = static_cast<int32_t>(tag.size());
        shape_hints_ptr[group_hint_offset + 1] = static_cast<int32_t>(table.type);
        shape_hints_ptr[group_hint_offset + 2] = static_cast<int32_t>(table.block_ids.size(1));
        shape_hints_ptr[group_hint_offset + 3] = static_cast<int32_t>(table.kernel_block_ids.size(1));
        RTP_LLM_CHECK_WITH_INFO(
            table.seq_size_per_block <= static_cast<size_t>(std::numeric_limits<int32_t>::max())
                && table.kernel_seq_size_per_block <= static_cast<size_t>(std::numeric_limits<int32_t>::max())
                && table.kv_block_stride_bytes <= static_cast<size_t>(std::numeric_limits<int32_t>::max())
                && table.kv_scale_stride_bytes <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
            "tagged KV cache metadata exceeds TP wire range for tag=%s",
            tag.c_str());
        shape_hints_ptr[group_hint_offset + 4] = static_cast<int32_t>(table.seq_size_per_block);
        shape_hints_ptr[group_hint_offset + 5] = static_cast<int32_t>(table.kernel_seq_size_per_block);
        shape_hints_ptr[group_hint_offset + 6] = static_cast<int32_t>(table.kv_block_stride_bytes);
        shape_hints_ptr[group_hint_offset + 7] = static_cast<int32_t>(table.kv_scale_stride_bytes);
        std::memset(shape_hints_ptr + group_hint_offset + 8, 0, kMaxCacheTagSize);
        std::memcpy(shape_hints_ptr + group_hint_offset + 8, tag.data(), tag.size());
        max_blocks_hint        = std::max(max_blocks_hint, shape_hints_ptr[group_hint_offset + 2]);
        max_kernel_blocks_hint = std::max(max_kernel_blocks_hint, shape_hints_ptr[group_hint_offset + 3]);
        group_hint_offset += kGroupHintWords;
    }
    shape_hints_ptr[GptModelInputIndex::maxKernelBlocksPerBatch] = max_kernel_blocks_hint;
    shape_hints_ptr[GptModelInputIndex::maxBlocksPerBatch]       = max_blocks_hint;
    shape_hints_ptr[GptModelInputIndex::cacheKeysWidth] =
        inputs.cache_keys.defined() && inputs.cache_keys.dim() >= 2 ? inputs.cache_keys.size(1) : 0;
    shape_hints_ptr[GptModelInputIndex::kvCacheGroupNum] = inputs.block_tables_by_group.size();
    // Kept as a reserved zero-valued slot for shape-hint wire compatibility.
    shape_hints_ptr[GptModelInputIndex::kvCacheLayerToGroupLen] = 0;
    shape_hints_ptr[GptModelInputIndex::kvCacheGroupTypesLen]   = 0;
    shape_hints_ptr[GptModelInputIndex::kvCacheUpdateCopyNum]   = inputs.kv_cache_update_mapping.size();
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
    shape_hints_ptr[GptModelInputIndex::mmHasExtraInput] =
        inputs.mm_extra_input.has_value() ? inputs.mm_extra_input.value().size() : 0;
    shape_hints_ptr[GptModelInputIndex::mmExtraInputDtype] =
        (inputs.mm_extra_input.has_value() && !inputs.mm_extra_input.value().empty()) ?
            (std::uint8_t)torchDTypeToDataType(inputs.mm_extra_input.value()[0].dtype()) :
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
    // extra-input (model-specific, treated as opaque flat 1-D tensors) per-tensor element count
    torch::Tensor mm_extra_input_shape_t;
    int64_t*      mm_extra_input_shape_ptr = nullptr;
    inputs.need_all_logits                 = shape_hints_ptr[GptModelInputIndex::needAllLogits];
    inputs.skip_run                        = shape_hints_ptr[GptModelInputIndex::skipRun];
    inputs.is_fake_stream                  = shape_hints_ptr[GptModelInputIndex::isFakeStream];
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

    // extra-input element counts broadcast: each extra-input is an opaque flat 1-D tensor,
    // so we send its element count first ("先传shape") and allocate a 1-D buffer on non-root.
    const size_t mm_extra_input_num = (size_t)shape_hints_ptr[GptModelInputIndex::mmHasExtraInput];
    if (mm_extra_input_num) {
        mm_extra_input_shape_t   = torch::empty({(int64_t)mm_extra_input_num}, torch::kInt64).pin_memory();
        mm_extra_input_shape_ptr = mm_extra_input_shape_t.data_ptr<int64_t>();
        for (size_t i = 0; i < mm_extra_input_num; ++i) {
            mm_extra_input_shape_ptr[i] =
                inputs.mm_extra_input.has_value() ? inputs.mm_extra_input.value()[i].numel() : 0;
        }
        execBroadcast({{mm_extra_input_shape_t}, 0});
        execSyncCommunication(false);
        cudaSyncAndCheck();
    }

    auto   max_kernel_blocks       = (size_t)shape_hints_ptr[GptModelInputIndex::maxKernelBlocksPerBatch];
    auto   max_blocks              = (size_t)shape_hints_ptr[GptModelInputIndex::maxBlocksPerBatch];
    auto   cache_keys_width        = (size_t)shape_hints_ptr[GptModelInputIndex::cacheKeysWidth];
    auto   kv_cache_group_num      = (size_t)shape_hints_ptr[GptModelInputIndex::kvCacheGroupNum];
    auto   combo_position_ids_size = shape_hints_ptr[GptModelInputIndex::comboPositionIds];
    auto   text_tokens_mask_size   = shape_hints_ptr[GptModelInputIndex::textTokensMask];
    auto   mm_features_locs_size   = shape_hints_ptr[GptModelInputIndex::mmFeaturesLocs];
    auto   hidden_states_size      = shape_hints_ptr[GptModelInputIndex::mtpHiddenStates];
    size_t request_length          = shape_hints_ptr[GptModelInputIndex::gptModelRequestLength];

    struct GroupDescriptor {
        std::string    tag;
        CacheGroupType type;
        size_t         width;
        size_t         kernel_width;
        size_t         seq_size_per_block;
        size_t         kernel_seq_size_per_block;
        size_t         kv_block_stride_bytes;
        size_t         kv_scale_stride_bytes;
    };
    std::vector<GroupDescriptor> group_descriptors;
    group_descriptors.reserve(kv_cache_group_num);
    group_hint_offset = GptModelInputIndex::gptModelInputLength;
    for (size_t i = 0; i < kv_cache_group_num; ++i) {
        const auto tag_size = static_cast<size_t>(shape_hints_ptr[group_hint_offset]);
        RTP_LLM_CHECK_WITH_INFO(tag_size <= kMaxCacheTagSize, "invalid broadcast KV cache tag length=%zu", tag_size);
        const char* tag_data = reinterpret_cast<const char*>(shape_hints_ptr + group_hint_offset + 8);
        group_descriptors.push_back(GroupDescriptor{std::string(tag_data, tag_size),
                                                    static_cast<CacheGroupType>(shape_hints_ptr[group_hint_offset + 1]),
                                                    static_cast<size_t>(shape_hints_ptr[group_hint_offset + 2]),
                                                    static_cast<size_t>(shape_hints_ptr[group_hint_offset + 3]),
                                                    static_cast<size_t>(shape_hints_ptr[group_hint_offset + 4]),
                                                    static_cast<size_t>(shape_hints_ptr[group_hint_offset + 5]),
                                                    static_cast<size_t>(shape_hints_ptr[group_hint_offset + 6]),
                                                    static_cast<size_t>(shape_hints_ptr[group_hint_offset + 7])});
        group_hint_offset += kGroupHintWords;
    }

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
        if (max_kernel_blocks != 0 || max_blocks != 0) {
            const size_t batch_size     = shape_hints_ptr[GptModelInputIndex::inputLengths];
            size_t       physical_numel = 0;
            size_t       kernel_numel   = 0;
            for (const auto& descriptor : group_descriptors) {
                physical_numel += batch_size * descriptor.width;
                kernel_numel += batch_size * descriptor.kernel_width;
            }
            auto   physical_backing = allocBuf(rtp_llm::DataType::TYPE_INT32, {physical_numel});
            auto   kernel_backing   = allocBuf(rtp_llm::DataType::TYPE_INT32, {kernel_numel});
            size_t physical_offset  = 0;
            size_t kernel_offset    = 0;
            inputs.block_tables_by_group.clear();
            for (const auto& descriptor : group_descriptors) {
                GroupBlockTable table;
                table.tag                       = descriptor.tag;
                table.type                      = descriptor.type;
                table.seq_size_per_block        = descriptor.seq_size_per_block;
                table.kernel_seq_size_per_block = descriptor.kernel_seq_size_per_block;
                table.kv_block_stride_bytes     = descriptor.kv_block_stride_bytes;
                table.kv_scale_stride_bytes     = descriptor.kv_scale_stride_bytes;
                table.block_ids = physical_backing.narrow(0, physical_offset, batch_size * descriptor.width)
                                      .view({static_cast<int64_t>(batch_size), static_cast<int64_t>(descriptor.width)});
                table.kernel_block_ids =
                    kernel_backing.narrow(0, kernel_offset, batch_size * descriptor.kernel_width)
                        .view({static_cast<int64_t>(batch_size), static_cast<int64_t>(descriptor.kernel_width)});
                physical_offset += batch_size * descriptor.width;
                kernel_offset += batch_size * descriptor.kernel_width;
                const auto [it, inserted] = inputs.block_tables_by_group.emplace(descriptor.tag, std::move(table));
                (void)it;
                RTP_LLM_CHECK_WITH_INFO(inserted, "duplicate broadcast KV cache tag=%s", descriptor.tag.c_str());
            }
            if (inputs.pd_separation) {
                inputs.cache_keys = allocBuf(rtp_llm::DataType::TYPE_INT64,
                                             {context_batch_size, cache_keys_width ? cache_keys_width : max_blocks});
            }
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
        if (mm_extra_input_num) {
            std::vector<torch::Tensor> mm_extra_input;
            auto                       extra_dtype =
                dataTypeToTorchType((rtp_llm::DataType)shape_hints_ptr[GptModelInputIndex::mmExtraInputDtype]);
            for (size_t i = 0; i < mm_extra_input_num; ++i) {
                mm_extra_input.emplace_back(
                    torch::empty({(int64_t)mm_extra_input_shape_ptr[i]},
                                 torch::TensorOptions().dtype(extra_dtype).device(torch::kCUDA)));
            }
            inputs.mm_extra_input = std::move(mm_extra_input);
        }
    }

    constexpr size_t kUpdateWireRecordSize = kMaxCacheTagSize + 2 * sizeof(int32_t);
    const size_t     update_count = static_cast<size_t>(shape_hints_ptr[GptModelInputIndex::kvCacheUpdateCopyNum]);
    torch::Tensor    update_wire;
    if (update_count > 0) {
        update_wire = torch::zeros({static_cast<int64_t>(update_count * kUpdateWireRecordSize)},
                                   torch::TensorOptions(torch::kUInt8))
                          .pin_memory();
        if (!is_non_root) {
            RTP_LLM_CHECK_WITH_INFO(inputs.kv_cache_update_mapping.size() == update_count,
                                    "KV cache update mapping count changed during TP sync");
            auto* wire = update_wire.data_ptr<uint8_t>();
            for (size_t i = 0; i < update_count; ++i) {
                const auto& mapping = inputs.kv_cache_update_mapping[i];
                RTP_LLM_CHECK_WITH_INFO(mapping.tag.size() <= kMaxCacheTagSize,
                                        "KV cache update tag is too long: tag=%s",
                                        mapping.tag.c_str());
                auto* record = wire + i * kUpdateWireRecordSize;
                std::memcpy(record, mapping.tag.data(), mapping.tag.size());
                std::memcpy(record + kMaxCacheTagSize, &mapping.src, sizeof(mapping.src));
                std::memcpy(record + kMaxCacheTagSize + sizeof(mapping.src), &mapping.dst, sizeof(mapping.dst));
            }
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
        for (auto& [tag, table] : inputs.block_tables_by_group) {
            (void)tag;
            collect(table.kernel_block_ids);
            collect(table.block_ids);
        }
        if (inputs.pd_separation) {
            collect(inputs.cache_keys);
        }
    }
    if (update_wire.defined()) {
        collect(update_wire);
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
    if (mm_extra_input_num) {
        for (auto& e : inputs.mm_extra_input.value()) {
            collect(e);
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

    if (is_non_root) {
        inputs.kv_cache_update_mapping.clear();
        inputs.kv_cache_update_mapping.reserve(update_count);
        const auto* wire = update_wire.defined() ? update_wire.data_ptr<uint8_t>() : nullptr;
        for (size_t i = 0; i < update_count; ++i) {
            const auto*      record   = wire + i * kUpdateWireRecordSize;
            const auto*      end      = static_cast<const uint8_t*>(std::memchr(record, '\0', kMaxCacheTagSize));
            const size_t     tag_size = end ? static_cast<size_t>(end - record) : kMaxCacheTagSize;
            GroupBlockIdPair mapping;
            mapping.tag.assign(reinterpret_cast<const char*>(record), tag_size);
            std::memcpy(&mapping.src, record + kMaxCacheTagSize, sizeof(mapping.src));
            std::memcpy(&mapping.dst, record + kMaxCacheTagSize + sizeof(mapping.src), sizeof(mapping.dst));
            inputs.kv_cache_update_mapping.push_back(std::move(mapping));
        }
    }
}

}  // namespace rtp_llm
