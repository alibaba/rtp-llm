#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/models_py/bindings/core/torch_utils/TypeConvert.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

namespace rtp_llm {

namespace {

// RTP_LLM_SCORING_PARTIAL_CHUNK_ROWS sets the row budget for the tp>1 scoring
// all-gather/merge (an internal debug/test tuning knob). Every rank parses its own env
// here, but only rank 0's parsed value is broadcast through the shape hint — so a
// malformed non-root value can at worst log a local warning, and no rank-local value
// can make a single rank throw or pick a different chunk size than its peers.
// Empty/unparseable/non-positive input falls back to kDefaultScoringPartialChunkRows.
int64_t scoringPartialChunkRowsFromEnv() {
    static const int64_t parsed = []() {
        const char* env = std::getenv("RTP_LLM_SCORING_PARTIAL_CHUNK_ROWS");
        if (env == nullptr || *env == '\0') {
            return kDefaultScoringPartialChunkRows;
        }
        errno                 = 0;
        char*           end   = nullptr;
        const long long value = std::strtoll(env, &end, 10);
        if (errno != 0 || end == env || *end != '\0' || value <= 0 || value > std::numeric_limits<int32_t>::max()) {
            RTP_LLM_LOG_WARNING("invalid RTP_LLM_SCORING_PARTIAL_CHUNK_ROWS='%s', using default %ld",
                                env,
                                kDefaultScoringPartialChunkRows);
            return kDefaultScoringPartialChunkRows;
        }
        return static_cast<int64_t>(value);
    }();
    return parsed;
}

}  // namespace

GptModelInputShapeHints getModelInputShapeHints(const GptModelInputs& inputs) {
    GptModelInputShapeHints shape_hints{};
    shape_hints[GptModelInputIndex::comboTokens]  = inputs.combo_tokens.defined() ? inputs.combo_tokens.numel() : 0;
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
    // Kept as a reserved zero-valued slot for shape-hint wire compatibility.
    // Per-layer group mapping is carried by GroupedCacheLayerLayout now, not by
    // a broadcast tensor.
    shape_hints[GptModelInputIndex::kvCacheLayerToGroupLen] = 0;
    shape_hints[GptModelInputIndex::kvCacheGroupTypesLen] =
        inputs.kv_cache_group_types.defined() ? inputs.kv_cache_group_types.numel() : 0;
    shape_hints[GptModelInputIndex::kvCacheUpdateCopyNum] =
        inputs.kv_cache_update_mapping.defined() ? inputs.kv_cache_update_mapping.size(0) : 0;
    shape_hints[GptModelInputIndex::lmOutputIndexes] =
        inputs.lm_output_indexes.defined() ? inputs.lm_output_indexes.numel() : 0;
    shape_hints[GptModelInputIndex::comboPositionIds] =
        inputs.combo_position_ids.defined() ? inputs.combo_position_ids.numel() : 0;
    shape_hints[GptModelInputIndex::comboTokenTypeIds] =
        inputs.combo_tokens_type_ids.defined() ? inputs.combo_tokens_type_ids.numel() : 0;
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
    shape_hints[GptModelInputIndex::mmHasExtraInput] =
        inputs.mm_extra_input.has_value() ? static_cast<int64_t>(inputs.mm_extra_input.value().size()) : 0;
    shape_hints[GptModelInputIndex::mmExtraInputDtype] =
        (inputs.mm_extra_input.has_value() && !inputs.mm_extra_input.value().empty()) ?
            static_cast<int64_t>(torchDTypeToDataType(inputs.mm_extra_input.value()[0].dtype())) :
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
    shape_hints[GptModelInputIndex::allLogitsRowIndexes] =
        inputs.all_logits_row_indexes.defined() ? inputs.all_logits_row_indexes.numel() : 0;
    shape_hints[GptModelInputIndex::allLogitsTopK] = inputs.all_logits_top_k;
    const bool row_idx_defined                     = inputs.all_logits_row_indexes.defined();
    const bool row_lbl_defined                     = inputs.all_logits_row_labels.defined();
    const bool row_len_equal                       = row_idx_defined && row_lbl_defined
                               && inputs.all_logits_row_indexes.numel() == inputs.all_logits_row_labels.numel();
    shape_hints[GptModelInputIndex::allLogitsRowMeta] =
        (row_idx_defined ? 2 : 0) | (row_lbl_defined ? 1 : 0) | (row_len_equal ? 4 : 0);
    shape_hints[GptModelInputIndex::scoringChunkRows] =
        inputs.scoring_partial_chunk_rows > 0 ? inputs.scoring_partial_chunk_rows : scoringPartialChunkRowsFromEnv();

    // encode root-side tensor device for fields that may live on GPU on the
    // PDFUSION fast path, so non-root ranks can allocate matching GPU buffers
    // and tpSync's pack/unpack stays in lockstep.
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
    if (inputs.kv_cache_kernel_block_id.defined() && inputs.kv_cache_kernel_block_id.is_cuda()) {
        device_bits |= GptModelInputDeviceBit::kDeviceBitKernelBlockId;
    }
    if (inputs.combo_tokens_type_ids.defined() && inputs.combo_tokens_type_ids.is_cuda()) {
        device_bits |= GptModelInputDeviceBit::kDeviceBitTokenTypeIds;
    }
    shape_hints[GptModelInputIndex::tensorDeviceMap] = static_cast<int64_t>(device_bits);
    return shape_hints;
}

torch::Tensor makeModelInputShapeHintsTensor(const GptModelInputs& inputs) {
    const auto shape_hints = getModelInputShapeHints(inputs);
    auto       tensor      = torch::empty({static_cast<int64_t>(shape_hints.size())}, torch::kInt64).pin_memory();
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

void checkScoringRowMetaConsistent(bool need_all_logits, int64_t row_meta) {
    const bool indexes_defined = (row_meta & 2) != 0;
    const bool labels_defined  = (row_meta & 1) != 0;
    const bool lengths_equal   = (row_meta & 4) != 0;
    RTP_LLM_CHECK_WITH_INFO(row_meta >= 0 && row_meta <= 7,
                            "tpSyncModelInputs: invalid allLogitsRowMeta hint %lld",
                            static_cast<long long>(row_meta));
    // bit2 (lengths equal) may only be set when both tensors are defined.
    RTP_LLM_CHECK_WITH_INFO(!lengths_equal || (indexes_defined && labels_defined),
                            "tpSyncModelInputs: allLogitsRowMeta hint %lld sets the "
                            "lengths-equal bit without both tensors defined",
                            static_cast<long long>(row_meta));
    // Producer contract: the root attaches the (possibly empty) row set iff need_all_logits.
    // A producer that diverges — flag without tensor or tensor without flag — would make the
    // all-gather row counts disagree across ranks and hang NCCL, so fail fast before
    // broadcasting.
    RTP_LLM_CHECK_WITH_INFO(need_all_logits == indexes_defined,
                            "tpSyncModelInputs: need_all_logits=%d but all_logits_row_indexes is %s; "
                            "attach the (possibly empty) row tensor iff need_all_logits is set",
                            static_cast<int>(need_all_logits),
                            indexes_defined ? "defined" : "undefined");
    // The sharded scoring reduction needs one label per scoring row on every rank; it is
    // broadcast together with the row indexes, so the producer must attach both or neither.
    RTP_LLM_CHECK_WITH_INFO(indexes_defined == labels_defined,
                            "tpSyncModelInputs: all_logits_row_indexes is %s but "
                            "all_logits_row_labels is %s; attach both or neither",
                            indexes_defined ? "defined" : "undefined",
                            labels_defined ? "defined" : "undefined");
    // A defined row set whose label count differs from its index count would make root and
    // non-root disagree on the packed broadcast size, so reject it before packing.
    RTP_LLM_CHECK_WITH_INFO(!indexes_defined || lengths_equal,
                            "tpSyncModelInputs: all_logits_row_labels row count differs from "
                            "all_logits_row_indexes; label rows must align 1:1 with index rows");
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
    // extra-input (model-specific, treated as opaque flat 1-D tensors) per-tensor element count
    torch::Tensor mm_extra_input_shape_t;
    int64_t*      mm_extra_input_shape_ptr = nullptr;
    inputs.need_all_logits                 = shape_hints_ptr[GptModelInputIndex::needAllLogits];
    inputs.all_logits_top_k                = shape_hints_ptr[GptModelInputIndex::allLogitsTopK];
    inputs.need_all_hidden_states          = shape_hints_ptr[GptModelInputIndex::needAllHiddenStates];
    inputs.skip_run                        = shape_hints_ptr[GptModelInputIndex::skipRun];
    inputs.is_fake_stream                  = shape_hints_ptr[GptModelInputIndex::isFakeStream];
    inputs.scoring_partial_chunk_rows      = shape_hints_ptr[GptModelInputIndex::scoringChunkRows];
    if (inputs.skip_run) {
        return;
    }

    auto checkedHint = [&](GptModelInputIndex index, const char* name) -> int64_t {
        const auto value = shape_hints_ptr[index];
        RTP_LLM_CHECK_WITH_INFO(
            value >= 0, "tpSyncModelInputs received negative %s shape hint: %lld", name, static_cast<long long>(value));
        return value;
    };

    const auto mm_features_num = checkedHint(GptModelInputIndex::mmFeaturesNum, "mmFeaturesNum");
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

    // extra-input element counts broadcast: each extra-input is an opaque flat 1-D tensor,
    // so we send its element count first ("先传shape") and allocate a 1-D buffer on non-root.
    const auto mm_extra_input_num = checkedHint(GptModelInputIndex::mmHasExtraInput, "mmHasExtraInput");
    if (mm_extra_input_num) {
        mm_extra_input_shape_t   = torch::empty({mm_extra_input_num}, torch::kInt64).pin_memory();
        mm_extra_input_shape_ptr = mm_extra_input_shape_t.data_ptr<int64_t>();
        for (int64_t i = 0; i < mm_extra_input_num; ++i) {
            mm_extra_input_shape_ptr[i] =
                inputs.mm_extra_input.has_value() ? inputs.mm_extra_input.value()[i].numel() : 0;
        }
        execBroadcast({{mm_extra_input_shape_t}, 0});
        execSyncCommunication(false);
        cudaSyncAndCheck();
    }

    const auto max_kernel_blocks  = checkedHint(GptModelInputIndex::maxKernelBlocksPerBatch, "maxKernelBlocksPerBatch");
    const auto max_blocks         = checkedHint(GptModelInputIndex::maxBlocksPerBatch, "maxBlocksPerBatch");
    const auto cache_keys_width   = checkedHint(GptModelInputIndex::cacheKeysWidth, "cacheKeysWidth");
    const auto kv_cache_group_num = checkedHint(GptModelInputIndex::kvCacheGroupNum, "kvCacheGroupNum");
    const auto group_types_len    = checkedHint(GptModelInputIndex::kvCacheGroupTypesLen, "kvCacheGroupTypesLen");
    const auto combo_position_ids_size = checkedHint(GptModelInputIndex::comboPositionIds, "comboPositionIds");
    const auto token_type_ids_size     = checkedHint(GptModelInputIndex::comboTokenTypeIds, "comboTokenTypeIds");
    const auto text_tokens_mask_size   = checkedHint(GptModelInputIndex::textTokensMask, "textTokensMask");
    const auto mm_features_locs_size   = checkedHint(GptModelInputIndex::mmFeaturesLocs, "mmFeaturesLocs");
    const auto hidden_states_size      = checkedHint(GptModelInputIndex::mtpHiddenStates, "mtpHiddenStates");
    const auto request_length = checkedHint(GptModelInputIndex::gptModelRequestLength, "gptModelRequestLength");

    auto allocBuf = [&](rtp_llm::DataType       dtype,
                        std::vector<int64_t>    dims,
                        rtp_llm::AllocationType atype = rtp_llm::AllocationType::HOST) -> torch::Tensor {
        auto    torch_dtype = dataTypeToTorchType(dtype);
        int64_t numel       = 1;
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
        auto options = torch::TensorOptions(torch_dtype);
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

        inputs.combo_tokens     = allocBuf(rtp_llm::DataType::TYPE_INT32,
                                           {checkedHint(GptModelInputIndex::comboTokens, "comboTokens")},
                                       pickAlloc(GptModelInputDeviceBit::kDeviceBitComboTokens));
        inputs.input_lengths    = allocBuf(rtp_llm::DataType::TYPE_INT32,
                                           {checkedHint(GptModelInputIndex::inputLengths, "inputLengths")},
                                        pickAlloc(GptModelInputDeviceBit::kDeviceBitInputLengths));
        inputs.sequence_lengths = allocBuf(rtp_llm::DataType::TYPE_INT32,
                                           {checkedHint(GptModelInputIndex::sequenceLengths, "sequenceLengths")},
                                           pickAlloc(GptModelInputDeviceBit::kDeviceBitSequenceLengths));
        inputs.prefix_lengths   = allocBuf(rtp_llm::DataType::TYPE_INT32,
                                           {context_batch_size},
                                         pickAlloc(GptModelInputDeviceBit::kDeviceBitPrefixLengths));
        if (max_kernel_blocks != 0) {
            // kv_cache_kernel_block_id residency follows the producer (rank 0): device only when
            // RTP_LLM_DEVICE_INPUT publishes it to CUDA. Follow the root bitmap so the packed-buffer
            // classification below stays identical across ranks (otherwise pack/unpack drifts
            // off-by-tensor and non-root unpacks garbage).
            inputs.kv_cache_kernel_block_id = allocBuf(
                rtp_llm::DataType::TYPE_INT32,
                {kv_cache_group_num, checkedHint(GptModelInputIndex::inputLengths, "inputLengths"), max_kernel_blocks},
                pickAlloc(GptModelInputDeviceBit::kDeviceBitKernelBlockId));
            inputs.kv_cache_update_mapping =
                allocBuf(rtp_llm::DataType::TYPE_INT32,
                         {checkedHint(GptModelInputIndex::kvCacheUpdateCopyNum, "kvCacheUpdateCopyNum"), 3});
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
        if (group_types_len) {
            inputs.kv_cache_group_types = allocBuf(rtp_llm::DataType::TYPE_INT32, {group_types_len});
        }
        inputs.request_id            = allocBuf(rtp_llm::DataType::TYPE_INT64, {request_length});
        inputs.request_pd_separation = allocBuf(rtp_llm::DataType::TYPE_BOOL, {request_length});
        inputs.lm_output_indexes     = allocBuf(rtp_llm::DataType::TYPE_INT32,
                                                {checkedHint(GptModelInputIndex::lmOutputIndexes, "lmOutputIndexes")},
                                            pickAlloc(GptModelInputDeviceBit::kDeviceBitLmOutputIndexes));
        if (inputs.need_all_logits) {
            // Keep definedness in lockstep with the root rank (which attaches a possibly-empty
            // tensor iff need_all_logits): the PyWrappedModel row-select decision keys off
            // defined() and must agree across ranks or the TP all-gather sizes would diverge.
            inputs.all_logits_row_indexes =
                allocBuf(rtp_llm::DataType::TYPE_INT32,
                         {checkedHint(GptModelInputIndex::allLogitsRowIndexes, "allLogitsRowIndexes")});
            inputs.all_logits_row_labels =
                allocBuf(rtp_llm::DataType::TYPE_INT32,
                         {checkedHint(GptModelInputIndex::allLogitsRowIndexes, "allLogitsRowIndexes")});
        }
        if (combo_position_ids_size) {
            inputs.combo_position_ids = allocBuf(rtp_llm::DataType::TYPE_INT32, {combo_position_ids_size});
        }
        inputs.combo_tokens_type_ids = token_type_ids_size ?
                                           allocBuf(rtp_llm::DataType::TYPE_INT32,
                                                    {token_type_ids_size},
                                                    pickAlloc(GptModelInputDeviceBit::kDeviceBitTokenTypeIds)) :
                                           torch::Tensor();
        if (hidden_states_size) {
            // DSpARK prefill seeding makes last_hidden_states' row count
            // independent from combo_tokens, so the row count travels as its own
            // shape hint instead of being derived from the token count.
            const auto hidden_shape = decodeMtpHiddenStatesShape(
                hidden_states_size, checkedHint(GptModelInputIndex::mtpHiddenStatesRows, "mtpHiddenStatesRows"));
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
            const auto                 mm_data_type =
                static_cast<rtp_llm::DataType>(shape_hints_ptr[GptModelInputIndex::mmFeaturesDtype]);
            const auto mm_cols = checkedHint(GptModelInputIndex::mmFeaturesSize, "mmFeaturesSize");
            for (int64_t mm_index = 0; mm_index < mm_features_num; ++mm_index) {
                const auto mm_rows = mm_features_shape_ptr[mm_index];
                mm_features.emplace_back(allocBuf(mm_data_type, {mm_rows, mm_cols}, rtp_llm::AllocationType::DEVICE));
            }
            inputs.multimodal_features = std::move(mm_features);
        }
        if (mm_extra_input_num) {
            std::vector<torch::Tensor> mm_extra_input;
            const auto                 extra_data_type =
                static_cast<rtp_llm::DataType>(shape_hints_ptr[GptModelInputIndex::mmExtraInputDtype]);
            for (int64_t i = 0; i < mm_extra_input_num; ++i) {
                mm_extra_input.emplace_back(
                    allocBuf(extra_data_type, {mm_extra_input_shape_ptr[i]}, rtp_llm::AllocationType::DEVICE));
            }
            inputs.mm_extra_input = std::move(mm_extra_input);
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
    // Producer contract for the scoring-row metadata, evaluated from the broadcast shape
    // hints so every rank checks the identical value and aborts at the same point — a
    // root-only check could leave non-root ranks waiting inside the packed broadcast
    // after root already failed.
    checkScoringRowMetaConsistent(inputs.need_all_logits, shape_hints_ptr[GptModelInputIndex::allLogitsRowMeta]);
    collect(inputs.all_logits_row_indexes);
    collect(inputs.all_logits_row_labels);
    if (combo_position_ids_size) {
        collect(inputs.combo_position_ids);
    }
    if (token_type_ids_size) {
        collect(inputs.combo_tokens_type_ids);
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
