#include "rtp_llm/cpp/cuda_graph/cuda_graph_runner.h"
#include "rtp_llm/cpp/cuda_graph/combo_position_ids_validation.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <c10/core/InferenceMode.h>
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#if USING_CUDA
#include "rtp_llm/models_py/bindings/cuda/kernels/cuda_graph_prepare.h"
#endif
using namespace torch_ext;
namespace rtp_llm {

BlockIdxType CudaGraphRunner::safeKernelBlockIdForTag(std::string_view tag) const {
    return cudaGraphSafeKernelBlockId(cache_config_->group(tag));
}

BlockIdxType CudaGraphRunner::safeKernelBlockIdForFlatTable() const {
    RTP_LLM_CHECK_WITH_INFO(kv_cache_group_tags_.size() == 1,
                            "flat CUDA graph block table requires one cache group, got %zu",
                            kv_cache_group_tags_.size());
    return safeKernelBlockIdForTag(kv_cache_group_tags_.front());
}

BlockIdxType CudaGraphRunner::safeKernelBlockIdForPrimaryTable() const {
    RTP_LLM_CHECK_WITH_INFO(!kv_cache_group_tags_.empty(), "CUDA graph cache requires at least one group");
    return safeKernelBlockIdForTag(kv_cache_group_tags_.front());
}

namespace {

class ScopedEnvFlag {
public:
    ScopedEnvFlag(const char* name, const char* value): name_(name) {
        const char* old_value = std::getenv(name_);
        if (old_value != nullptr) {
            had_old_value_ = true;
            old_value_     = old_value;
        }
        setenv(name_, value, 1);
    }

    ~ScopedEnvFlag() {
        if (had_old_value_) {
            setenv(name_, old_value_.c_str(), 1);
        } else {
            unsetenv(name_);
        }
    }

private:
    const char* name_;
    bool        had_old_value_ = false;
    std::string old_value_;
};

// The stream-async pipelines (RTP_LLM_STREAM_ASYNC / RTP_LLM_MTP_ASYNC_PREPARE
// all set to 1 on the reference deployment) run the syncing prepare on a
// dedicated worker (PyWrappedModel::prepareAttentionInputs with
// skip_forward_event_sync=false) and keep every replay-prep mutation
// stream-ordered, so skipping the forward-event wait there is safe. The
// default host pipeline mutates pinned capture buffers directly from the CPU
// (host memcpy / fill_params / fa2 replan staging), exactly like main, and
// therefore must wait for the previous replay before touching them — main did
// this unconditionally in prepareInputs().
bool streamAsyncReplayPrepEnabled() {
    static const bool enabled = []() {
        auto is_on = [](const char* name) {
            const char* value = std::getenv(name);
            return value != nullptr && std::string(value) == "1";
        };
        return is_on("RTP_LLM_STREAM_ASYNC") || is_on("RTP_LLM_MTP_ASYNC_PREPARE");
    }();
    return enabled;
}

void callPrepareCudaGraph(py::object attn_pyobj, PyModelInputs& inputs) {
    if (!attn_pyobj || attn_pyobj.is_none()) {
        return;
    }

    if (inputs.attention_inputs_by_tag.empty()) {
        if (py::hasattr(attn_pyobj, "prepare_cuda_graph")) {
            attn_pyobj.attr("prepare_cuda_graph")(inputs.attention_inputs);
        }
        return;
    }

    if (py::isinstance<py::dict>(attn_pyobj)) {
        auto impls = attn_pyobj.cast<py::dict>();
        for (auto item : impls) {
            const auto tag = py::cast<std::string>(item.first);
            const auto it  = inputs.attention_inputs_by_tag.find(tag);
            RTP_LLM_CHECK_WITH_INFO(it != inputs.attention_inputs_by_tag.end(),
                                    "missing CUDA graph attention inputs for implementation tag=%s",
                                    tag.c_str());
            auto impl = item.second;
            RTP_LLM_CHECK_WITH_INFO(py::hasattr(impl, "prepare_cuda_graph"),
                                    "attention implementation for tag=%s has no prepare_cuda_graph",
                                    tag.c_str());
            impl.attr("prepare_cuda_graph")(it->second);
        }
        return;
    }

    if (py::hasattr(attn_pyobj, "prepare_cuda_graph")) {
        attn_pyobj.attr("prepare_cuda_graph")(inputs.attention_inputs_by_tag);
    }
}

#if USING_CUDA
void addCudaGraphPrepareFillRegion(
    CudaGraphPrepareFillParams& params, torch::Tensor& tensor, int64_t start, int64_t end, int32_t value) {
    if (!tensor.defined() || !tensor.is_cuda() || end <= start) {
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(tensor.scalar_type() == torch::kInt32, "cuda graph prepare fill expects int32 CUDA tensor");
    RTP_LLM_CHECK_WITH_INFO(tensor.is_contiguous(), "cuda graph prepare fill expects contiguous tensor");
    RTP_LLM_CHECK_WITH_INFO(start >= 0 && end <= tensor.numel(),
                            "cuda graph prepare fill range [%ld, %ld) exceeds tensor numel %ld",
                            start,
                            end,
                            tensor.numel());
    RTP_LLM_CHECK_WITH_INFO(params.region_count < kMaxCudaGraphPrepareFillRegions,
                            "too many cuda graph prepare fill regions: %d",
                            params.region_count);
    auto& region = params.regions[params.region_count++];
    region.ptr   = tensor.data_ptr<int32_t>() + start;
    region.count = end - start;
    region.value = value;
}

void addCudaGraphPrepareFillRegionFromDevice(CudaGraphPrepareFillParams& params,
                                             torch::Tensor&              tensor,
                                             int64_t                     start,
                                             int64_t                     end,
                                             const torch::Tensor&        value_tensor,
                                             int64_t                     value_index) {
    const int32_t old_region_count = params.region_count;
    addCudaGraphPrepareFillRegion(params, tensor, start, end, 0);
    if (params.region_count == old_region_count) {
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(value_tensor.defined() && value_tensor.is_cuda()
                                && value_tensor.scalar_type() == torch::kInt32 && value_index >= 0
                                && value_index < value_tensor.numel(),
                            "cuda graph prepare fill source must be a valid CUDA int32 element");
    params.regions[params.region_count - 1].value_ptr = value_tensor.data_ptr<int32_t>() + value_index;
}
#endif

int inferTotalTokensNoSync(const PyModelInputs& inputs) {
    if (inputs.input_ids.defined() && inputs.input_ids.numel() > 0) {
        return static_cast<int>(inputs.input_ids.size(0));
    }
    return inputs.attention_inputs.total_tokens > 0 ? inputs.attention_inputs.total_tokens : 0;
}

}  // namespace

// clang-format off
// CUDA Graph Mode Configuration Table:
// +--------------------------------+-----------------------------+--------------------------------------+--------------+
// | Model Type                     | is_prefill_cuda_graph_mode_ | num_tokens_per_bs_                   | 是否已经支持   |
// +--------------------------------+-----------------------------+--------------------------------------+--------------+
// | Draft Model (prefill)          | true                        | gen_num_per_cycle + 1                | yes          |
// | Target Model (verify)          | false                       | gen_num_per_cycle + 1                | yes          |
// | Draft Model (decode)           | false                       | 1                                    | yes          |
// | Embedding Model (prefill)      | true                        | max_seq_len                          | yes          |
// | Normal Model (decode)          | false                       | 1                                    | yes          |
// +--------------------------------+-----------------------------+--------------------------------------+--------------+
// Notes:
// - Speculative sampling: model_id == 0 (target), model_id == 1 (draft)
// - Target verify uses context-style attention inputs but selects/replays CUDA
//   graphs by decode batch size.
// clang-format on

// Helper function for optimized tensor copy using async operations with current CUDA stream
void optimizedCopyAsync(const torch::Tensor& src, torch::Tensor& dst, size_t size) {
    if (!src.defined() || !dst.defined() || src.numel() <= 0) {
        return;
    }

    RTP_LLM_PROFILE_SCOPE("optimizedCopyAsync");

    void* stream = reinterpret_cast<void*>(cuda_graph::graphGetCurrentStream().stream());
    if (src.is_cuda() && dst.is_cuda()) {
        cuda_graph::graphMemcpyAsync(dst.data_ptr(), src.data_ptr(), size, cuda_graph::GraphMemcpyKind::D2D, stream);
    } else if (!src.is_cuda() && !dst.is_cuda()) {
        std::memcpy(dst.data_ptr(), src.data_ptr(), size);
    } else if (src.is_cuda() && !dst.is_cuda()) {
        cuda_graph::graphMemcpyAsync(dst.data_ptr(), src.data_ptr(), size, cuda_graph::GraphMemcpyKind::D2H, stream);
    } else {
        cuda_graph::graphMemcpyAsync(dst.data_ptr(), src.data_ptr(), size, cuda_graph::GraphMemcpyKind::H2D, stream);
    }
}

void CudaGraphRunner::prepareInputs(const PyModelInputs& inputs, CudaGraphState& state) {
    RTP_LLM_PROFILE_SCOPE("cuda_graph.prepareInputs");
    prepareInputData(inputs, state);
    prepareAttentionInputs(inputs, state, /*skip_forward_event_sync=*/true);
}

void CudaGraphRunner::prepareInputData(const PyModelInputs& inputs, CudaGraphState& state) {
    RTP_LLM_PROFILE_SCOPE("cuda_graph.prepareInputData");
    const size_t graph_idx =
        is_prefill_cuda_graph_mode_ ? state.current_real_graph_seq_len : state.current_real_graph_bs;
    auto&     py_model_inputs = graph_instances_[graph_idx].mem_hold_.py_model_inputs_;
    const int token_num       = is_prefill_cuda_graph_mode_ ? state.current_seq_len : inputs.input_ids.size(0);

    optimizedCopyAsync(inputs.input_ids, py_model_inputs.input_ids, token_num * sizeof(int));

    // check size and dtype. The copy below is a raw byte memcpy into the captured
    // buffer, so the row width must match exactly: a DSpARK draft graph rows are
    // len(target_layer_ids) * hidden_size wide while a plain MTP graph is
    // hc_mult * hidden_size, and a numel-only check would silently accept a
    // reshaped view of the wrong layout.
    if (inputs.input_hiddens.defined() && inputs.input_hiddens.numel() > 0) {
        RTP_LLM_CHECK_WITH_INFO(inputs.input_hiddens.is_contiguous(),
                                "input_hiddens must be contiguous for the raw byte copy into the graph buffer");
        RTP_LLM_CHECK_WITH_INFO(inputs.input_hiddens.dim() == 2 && py_model_inputs.input_hiddens.dim() == 2,
                                "input_hiddens must be rank 2, got input dim=%ld capture dim=%ld",
                                inputs.input_hiddens.dim(),
                                py_model_inputs.input_hiddens.dim());
        RTP_LLM_CHECK_WITH_INFO(inputs.input_hiddens.size(1) == py_model_inputs.input_hiddens.size(1),
                                "input_hiddens row width mismatch: input=%ld capture=%ld",
                                inputs.input_hiddens.size(1),
                                py_model_inputs.input_hiddens.size(1));
        RTP_LLM_CHECK_WITH_INFO(inputs.input_hiddens.size(0) <= py_model_inputs.input_hiddens.size(0),
                                "input_hiddens row count exceeds graph capacity: input=%ld capture=%ld",
                                inputs.input_hiddens.size(0),
                                py_model_inputs.input_hiddens.size(0));
        RTP_LLM_CHECK_WITH_INFO(inputs.input_hiddens.dtype() == py_model_inputs.input_hiddens.dtype(),
                                "input_hiddens dtype mismatch: %s != %s",
                                inputs.input_hiddens.dtype().name(),
                                py_model_inputs.input_hiddens.dtype().name());
        optimizedCopyAsync(inputs.input_hiddens,
                           py_model_inputs.input_hiddens,
                           inputs.input_hiddens.numel() * inputs.input_hiddens.element_size());
    }
}

void CudaGraphRunner::prepareAttentionInputs(const PyModelInputs& inputs,
                                             CudaGraphState&      state,
                                             bool                 skip_forward_event_sync) {
    RTP_LLM_PROFILE_SCOPE("cuda_graph.prepareAttentionInputs");
    // 1. non spec cuda graph:
    // is_prefill_cuda_graph_mode_ is set true only when use embedding model
    // 2. spec cuda graph:
    // 2.1 spec hold target model and draft model. when the user prompt first comes in, the target model
    // adn draft model will do real "prefill forward". And for this phase, we don't support cuda graph
    // 2.2 after real "prefill forward", it is consisted of three parts:
    // 2.2.1 target model score(verfiy)
    // 2.2.2 draft model do first forward (input is from 2.2.1)
    // 2.2.3 draft model do auto-agressive forward
    // for now we only support 2.2.1 and 2.2.3 in deocode cuda graph, and 2.2.2 will be support in prefill cuda graph.

    if (!skip_forward_event_sync || !streamAsyncReplayPrepEnabled()) {
        RTP_LLM_PROFILE_SCOPE("cuda_graph.prepareAttentionInputs(wait_forward_event)");
        forward_event_.synchronize();
    }
    prepared_attention_inputs_.store(true, std::memory_order_release);

    const size_t graph_idx =
        is_prefill_cuda_graph_mode_ ? state.current_real_graph_seq_len : state.current_real_graph_bs;
    auto&      py_model_inputs_ = graph_instances_[graph_idx].mem_hold_.py_model_inputs_;
    auto       attn_pyobj       = graph_instances_[graph_idx].mem_hold_.attn_pyobj_;
    const bool has_tagged_cache = !inputs.attention_inputs_by_tag.empty();

    // Per-launch capacity contract: see fuse_copy_util.h sizing rationale.
    // Worst case here is ~8 contiguous + (1 + group_count) strided copies,
    // batched into one launch each. If new copies are added below — or if the
    // hybrid KV-cache group_count grows materially — re-check MAX_FUSED_*_COPIES.
    FusedD2DCopyParams     d2d_copies;
    FusedStridedCopyParams strided_d2d_copies;

    auto tryAddD2DCopy = [&d2d_copies](const torch::Tensor& src, torch::Tensor& dst, size_t bytes) {
        if (src.defined() && src.numel() > 0) {
            d2d_copies.add(src.data_ptr(), dst.data_ptr(), bytes);
        }
    };

    // Collect a strided 2D D2D copy: copies src[0..rows, 0..cols] into dst[0..rows, 0..cols]
    // where src and dst may have different column strides (copySmallerIntoLarger semantics).
    // For 1D tensors, falls back to a contiguous D2D copy to avoid silent data loss.
    auto tryAddStridedD2DCopy = [&strided_d2d_copies, &d2d_copies](const torch::Tensor& src, torch::Tensor& dst) {
        if (!src.defined() || src.numel() <= 0)
            return;
        if (src.dim() < 2) {
            d2d_copies.add(src.data_ptr(), dst.data_ptr(), src.numel() * src.element_size());
            return;
        }
        strided_d2d_copies.add(src.data_ptr(),
                               dst.data_ptr(),
                               src.size(0),
                               src.size(1) * src.element_size(),
                               src.stride(0) * src.element_size(),
                               dst.stride(0) * dst.element_size());
    };

    // Host block-table mirror refresh. Main's host pipeline guaranteed that
    // the pinned host mirrors reflect the current step before prepare_cuda_graph
    // runs (the host fillParams/plan paths read them). The source table may be
    // CUDA-resident even in the default host pipeline (e.g. the linear-attention
    // regather or the MTP device-state fast path), so a CUDA source becomes an
    // async strided D2H copy that the pre-callPrepareCudaGraph synchronize below
    // waits for; host sources keep main's synchronous row-by-row memcpy.
    bool pending_host_mirror_d2h = false;
    auto stridedCopyHost         = [&pending_host_mirror_d2h](const torch::Tensor& src, torch::Tensor& dst) {
        if (!src.defined() || src.numel() <= 0 || !dst.defined() || dst.is_cuda())
            return;
        if (src.is_cuda()) {
            RTP_LLM_PROFILE_SCOPE("stridedCopyHost(D2H)");
            if (src.dim() < 2) {
                dst.view({-1}).narrow(0, 0, src.numel()).copy_(src, /*non_blocking=*/true);
            } else {
                dst.narrow(0, 0, src.size(0)).narrow(1, 0, src.size(1)).copy_(src, /*non_blocking=*/true);
            }
            pending_host_mirror_d2h = true;
            return;
        }
        RTP_LLM_PROFILE_SCOPE("stridedCopyHost");
        if (src.dim() < 2) {
            memcpy(dst.data_ptr(), src.data_ptr(), src.numel() * src.element_size());
            return;
        }
        const size_t nrows      = src.size(0);
        const size_t row_bytes  = src.size(1) * src.element_size();
        const size_t src_stride = src.stride(0) * src.element_size();
        const size_t dst_stride = dst.stride(0) * dst.element_size();
        const char*  src_ptr    = reinterpret_cast<const char*>(src.data_ptr());
        char*        dst_ptr    = reinterpret_cast<char*>(dst.data_ptr());
        for (size_t r = 0; r < nrows; ++r) {
            memcpy(dst_ptr + r * dst_stride, src_ptr + r * src_stride, row_bytes);
        }
    };

    const int selected_graph_batch_size =
        is_prefill_cuda_graph_mode_ ? static_cast<int>(max_bs_) : state.current_real_graph_bs;

    // Clear stale device ranges in one launch before copying the live portions.
#if USING_CUDA
    {
        RTP_LLM_PROFILE_SCOPE("cuda_graph.prepareAttentionInputs(fused_fill)");
        CudaGraphPrepareFillParams fill_params;
        if (!has_tagged_cache) {
            addCudaGraphPrepareFillRegion(fill_params,
                                          py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_device,
                                          0,
                                          py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_device.numel(),
                                          safeKernelBlockIdForFlatTable());
        } else {
            for (auto& [tag, dst_inputs] : py_model_inputs_.attention_inputs_by_tag) {
                addCudaGraphPrepareFillRegion(fill_params,
                                              dst_inputs.kv_cache_kernel_block_id_device,
                                              0,
                                              dst_inputs.kv_cache_kernel_block_id_device.numel(),
                                              safeKernelBlockIdForTag(tag));
            }
        }
        if (is_prefill_cuda_graph_mode_) {
            addCudaGraphPrepareFillRegion(fill_params,
                                          py_model_inputs_.attention_inputs.prefix_lengths_device,
                                          state.current_batch_size,
                                          max_bs_,
                                          0);
            addCudaGraphPrepareFillRegion(fill_params,
                                          py_model_inputs_.attention_inputs.input_lengths_device,
                                          state.current_batch_size,
                                          max_bs_,
                                          0);
            addCudaGraphPrepareFillRegion(fill_params,
                                          py_model_inputs_.attention_inputs.cu_seqlens_device,
                                          state.current_batch_size + 1,
                                          max_bs_ + 1,
                                          state.current_seq_len);
            addCudaGraphPrepareFillRegionFromDevice(fill_params,
                                                    py_model_inputs_.attention_inputs.cu_kv_seqlens_device,
                                                    state.current_batch_size + 1,
                                                    max_bs_ + 1,
                                                    inputs.attention_inputs.cu_kv_seqlens_device,
                                                    state.current_batch_size);
        }
        // Target-verify padding (input_lengths / prefix_lengths / cu_*) is cleared by
        // the shared tail block below, which covers both the host mirrors and the
        // device buffers in one place.
        invokeCudaGraphPrepareFill(fill_params, cuda_graph::graphGetCurrentStream().stream());
    }
#else
    if (!has_tagged_cache) {
        py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_device.fill_(safeKernelBlockIdForFlatTable());
    } else {
        for (auto& [tag, dst_inputs] : py_model_inputs_.attention_inputs_by_tag) {
            dst_inputs.kv_cache_kernel_block_id_device.fill_(safeKernelBlockIdForTag(tag));
        }
    }
#endif

    if (!has_tagged_cache) {
        // The host mirror must be cleared with the device block table. fillParams
        // walks every graph-batch row and may dereference padding rows when a
        // backend keeps input_lengths uniform for graph-stable cu_seqlens. Without
        // this reset, a padding row can retain a previous replay's block ID and
        // route a KV write into a live request's block. Block 0 is reserved and is
        // therefore the safe destination for padding rows.
        py_model_inputs_.attention_inputs.kv_cache_kernel_block_id.fill_(0);
    }

    // NOTE: kv_cache_block_id_{host,device} are physical block IDs dedicated for cache store
    // (see OpDefs.h). They are NOT consumed by any GPU attention kernel during CUDA graph replay;
    // attention kernels only use kv_cache_kernel_block_id_{host,device}. Cache store operations
    // run outside the CUDA graph and read from the original (non-graph) inputs directly.

    // input_ids / input_hiddens are handled by prepareInputData. They MUST NOT be touched here
    // because the async-prepare path (PyWrappedModel::prepareAttentionInputs) calls this with
    // undefined empty tensors for those slots, which would crash on element_size().

    tryAddD2DCopy(inputs.attention_inputs.cu_seqlens_device,
                  py_model_inputs_.attention_inputs.cu_seqlens_device,
                  (state.current_batch_size + 1) * sizeof(int));
    tryAddD2DCopy(inputs.attention_inputs.cu_kv_seqlens_device,
                  py_model_inputs_.attention_inputs.cu_kv_seqlens_device,
                  (state.current_batch_size + 1) * sizeof(int));
    tryAddD2DCopy(inputs.attention_inputs.input_lengths_device,
                  py_model_inputs_.attention_inputs.input_lengths_device,
                  state.current_batch_size * sizeof(int));
    tryAddD2DCopy(inputs.attention_inputs.prefix_lengths_device,
                  py_model_inputs_.attention_inputs.prefix_lengths_device,
                  state.current_batch_size * sizeof(int));
    if (!has_tagged_cache) {
        // Strided 2D D2D copy for flat kv_cache_block_id
        tryAddStridedD2DCopy(inputs.attention_inputs.kv_cache_kernel_block_id_device,
                             py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_device);
    }

    if (position_id_len_factor_ > 0) {
        size_t copy_numel = 0;
        RTP_LLM_CHECK_WITH_INFO(
            validateComboPositionIds(inputs, state, py_model_inputs_.combo_position_ids, copy_numel),
            "invalid combo_position_ids before CUDA graph replay: factor=%d, src_numel=%lld, dst_numel=%lld",
            position_id_len_factor_,
            inputs.combo_position_ids.defined() ? static_cast<long long>(inputs.combo_position_ids.numel()) : -1LL,
            py_model_inputs_.combo_position_ids.defined() ?
                static_cast<long long>(py_model_inputs_.combo_position_ids.numel()) :
                -1LL);
        tryAddD2DCopy(inputs.combo_position_ids,
                      py_model_inputs_.combo_position_ids,
                      copy_numel * inputs.combo_position_ids.element_size());
    }

    if (!is_prefill_cuda_graph_mode_) {
        // D2D copies — collected for single batched kernel launch
        tryAddD2DCopy(inputs.attention_inputs.sequence_lengths_plus_1_device,
                      py_model_inputs_.attention_inputs.sequence_lengths_plus_1_device,
                      state.current_batch_size * sizeof(int));
        tryAddD2DCopy(inputs.attention_inputs.decode_cu_seqlens_device,
                      py_model_inputs_.attention_inputs.decode_cu_seqlens_device,
                      (state.current_batch_size + 1) * sizeof(int));
    } else {
        // D2D copy
        if (inputs.bert_embedding_inputs.position_encoding.numel() > 0) {
            tryAddD2DCopy(inputs.bert_embedding_inputs.combo_position_ids,
                          py_model_inputs_.bert_embedding_inputs.combo_position_ids,
                          state.current_seq_len * sizeof(int));
            tryAddD2DCopy(inputs.bert_embedding_inputs.combo_tokens_type_ids,
                          py_model_inputs_.bert_embedding_inputs.combo_tokens_type_ids,
                          state.current_seq_len * sizeof(int));
        }
    }

    // Multi-group cache: collect group-local block tables by stable topology tag.
    if (has_tagged_cache) {
        RTP_LLM_CHECK_WITH_INFO(inputs.attention_inputs_by_tag.size()
                                    == py_model_inputs_.attention_inputs_by_tag.size(),
                                "tagged attention input size mismatch");
        for (const auto& [tag, src_inputs] : inputs.attention_inputs_by_tag) {
            auto dst_it = py_model_inputs_.attention_inputs_by_tag.find(tag);
            RTP_LLM_CHECK_WITH_INFO(dst_it != py_model_inputs_.attention_inputs_by_tag.end(),
                                    "CUDA graph capture has no attention input for tag=%s",
                                    tag.c_str());
            auto& dst_inputs = dst_it->second;
            if (dst_inputs.kv_cache_kernel_block_id.defined() && !dst_inputs.kv_cache_kernel_block_id.is_cuda()) {
                dst_inputs.kv_cache_kernel_block_id.fill_(safeKernelBlockIdForTag(tag));
            }
            tryAddStridedD2DCopy(src_inputs.kv_cache_kernel_block_id_device,
                                 dst_inputs.kv_cache_kernel_block_id_device);
        }
    }

    // Launch ALL D2D copies (contiguous + strided) in two fused kernels
    {
        RTP_LLM_PROFILE_SCOPE("cuda_graph.prepareAttentionInputs(fused_d2d_copy)");
        fusedCopy(d2d_copies);
        fusedStridedCopy(strided_d2d_copies);
    }

    // NOTE: we do H2H after D2D copies to let GPU finish the D2D copies as soon as possible,
    // so that the GPU can start the kernel launch as soon as possible.

    {
        RTP_LLM_PROFILE_SCOPE("cuda_graph.prepareAttentionInputs(host_mirror_copy)");

        // H2H copies (common to both modes)
        //
        // When the incoming metadata is CUDA-resident (the MTP decode path
        // publishes sequence/input lengths on CUDA even in the default host
        // pipeline) while the capture mirrors are host pinned, optimizedCopyAsync
        // becomes an async D2H. callPrepareCudaGraph below consumes those host
        // mirrors synchronously on the CPU (host fill_params / fa2 replan), so the
        // copies must land first. Main's host pipeline used synchronous host
        // memcpy here and never observed stale values.
        auto copyToHostMirror = [&pending_host_mirror_d2h](const torch::Tensor& src, torch::Tensor& dst, size_t size) {
            optimizedCopyAsync(src, dst, size);
            if (src.defined() && src.numel() > 0 && src.is_cuda() && dst.defined() && !dst.is_cuda()) {
                pending_host_mirror_d2h = true;
            }
        };

        copyToHostMirror(inputs.attention_inputs.cu_seqlens,
                         py_model_inputs_.attention_inputs.cu_seqlens,
                         (state.current_batch_size + 1) * sizeof(int));

        copyToHostMirror(inputs.attention_inputs.input_lengths,
                         py_model_inputs_.attention_inputs.input_lengths,
                         state.current_batch_size * sizeof(int));

        copyToHostMirror(inputs.attention_inputs.prefix_lengths,
                         py_model_inputs_.attention_inputs.prefix_lengths,
                         state.current_batch_size * sizeof(int));

        if (!has_tagged_cache) {
            // Common H2H strided copies for kv_cache block tables (both decode & prefill)
            py_model_inputs_.attention_inputs.kv_cache_kernel_block_id.fill_(safeKernelBlockIdForFlatTable());
            stridedCopyHost(inputs.attention_inputs.kv_cache_kernel_block_id,
                            py_model_inputs_.attention_inputs.kv_cache_kernel_block_id);
        }

        if (!is_prefill_cuda_graph_mode_) {
            copyToHostMirror(inputs.attention_inputs.sequence_lengths,
                             py_model_inputs_.attention_inputs.sequence_lengths,
                             state.current_batch_size * sizeof(int));
            // When the live inputs carry no sequence_lengths at all (target verify
            // feeds the prefill attention path), nothing was copied above, so the
            // whole captured range - not just the padding tail - would otherwise
            // keep the capture-time max_seq_len values.
            const bool has_live_sequence_lengths = inputs.attention_inputs.sequence_lengths.defined()
                                                   && inputs.attention_inputs.sequence_lengths.numel() > 0;
            const int fill_start = has_live_sequence_lengths ? state.current_batch_size : 0;
            if (fill_start < selected_graph_batch_size) {
                py_model_inputs_.attention_inputs.sequence_lengths.slice(0, fill_start, selected_graph_batch_size)
                    .fill_(0);
            }
        } else {
            if (isEmbeddingStylePrefillCudaGraph()) {
                // This loop consumes input_lengths on the CPU. When the live tensor is
                // CUDA-resident (e.g. tests or device-side metadata pipelines), reading its
                // data_ptr on the host is invalid; use the host-pinned capture mirror filled
                // by the async D2H copy above and wait for that copy to land first.
                const bool live_input_lengths_on_cuda = inputs.attention_inputs.input_lengths.is_cuda();
                if (live_input_lengths_on_cuda && pending_host_mirror_d2h) {
                    RTP_LLM_PROFILE_SCOPE("cuda_graph.prepareAttentionInputs(wait_input_lengths_d2h)");
                    cuda_graph::graphGetCurrentStream().synchronize();
                    pending_host_mirror_d2h = false;
                }
                const auto& input_lengths_host = live_input_lengths_on_cuda ?
                                                     py_model_inputs_.attention_inputs.input_lengths :
                                                     inputs.attention_inputs.input_lengths;
                auto*       input_lengths      = input_lengths_host.data_ptr<int32_t>();
                auto*       padding_offset     = py_model_inputs_.attention_inputs.padding_offset.data_ptr<int32_t>();
                int         cumulative_padding = 0;
                int         token_idx          = 0;
                for (int batch_idx = 0; batch_idx < state.current_batch_size; ++batch_idx) {
                    const int input_length = input_lengths[batch_idx];
                    std::fill_n(padding_offset + token_idx, input_length, cumulative_padding);
                    token_idx += input_length;
                    cumulative_padding += state.current_real_graph_seq_len - input_length;
                }
            } else {
                copyToHostMirror(inputs.attention_inputs.padding_offset,
                                 py_model_inputs_.attention_inputs.padding_offset,
                                 state.current_seq_len * sizeof(int));
            }

            if (py_model_inputs_.attention_inputs.prefill_cuda_graph_copy_params) {
                auto* batch_size_ptr = py_model_inputs_.attention_inputs.prefill_cuda_graph_copy_params
                                           ->cuda_graph_prefill_batch_size.data_ptr<int>();
                *batch_size_ptr = state.current_batch_size;
            }
        }

        // Multi-group cache: H2H strided copies for group-local block tables.
        if (has_tagged_cache) {
            for (const auto& [tag, src_inputs] : inputs.attention_inputs_by_tag) {
                auto& dst_inputs = py_model_inputs_.attention_inputs_by_tag.at(tag);
                dst_inputs.kv_cache_kernel_block_id.fill_(safeKernelBlockIdForTag(tag));
                stridedCopyHost(src_inputs.kv_cache_kernel_block_id, dst_inputs.kv_cache_kernel_block_id);
            }
        }
    }

    // Prefill attention consumes cumulative Q/KV lengths. Target verify uses the
    // same attention path even though it is replayed by the decode graph runner;
    // it is covered by num_tokens_per_bs_ > 1 (a verify step always scores
    // gen_num_per_cycle + 1 tokens per request), which is also the condition under
    // which initCaptureAttentionInputs allocated prefix_lengths{,_device} at all.
    // Clear graph padding so rounded-up batch slots do not retain capture-time
    // max sequence lengths and trigger unnecessary attention work.
    if ((is_prefill_cuda_graph_mode_ || num_tokens_per_bs_ > 1)
        && state.current_batch_size < selected_graph_batch_size) {
        py_model_inputs_.attention_inputs.prefix_lengths.slice(0, state.current_batch_size, selected_graph_batch_size)
            .fill_(0);
        py_model_inputs_.attention_inputs.input_lengths.slice(0, state.current_batch_size, selected_graph_batch_size)
            .fill_(0);
        py_model_inputs_.attention_inputs.prefix_lengths_device
            .slice(0, state.current_batch_size, selected_graph_batch_size)
            .fill_(0);
        py_model_inputs_.attention_inputs.input_lengths_device
            .slice(0, state.current_batch_size, selected_graph_batch_size)
            .fill_(0);

        const int last_valid_q  = is_prefill_cuda_graph_mode_ ? state.current_seq_len : state.seq_len_sum;
        const int last_valid_kv = inputs.attention_inputs.context_total_kv_length;
        py_model_inputs_.attention_inputs.cu_seqlens
            .slice(0, state.current_batch_size + 1, selected_graph_batch_size + 1)
            .fill_(last_valid_q);
        py_model_inputs_.attention_inputs.cu_seqlens_device
            .slice(0, state.current_batch_size + 1, selected_graph_batch_size + 1)
            .fill_(last_valid_q);
        py_model_inputs_.attention_inputs.cu_kv_seqlens_device
            .slice(0, state.current_batch_size + 1, selected_graph_batch_size + 1)
            .fill_(last_valid_kv);
    }

    // launch prepare_cuda_graph when attention inputs are ready.
    // GIL is required: this function may be invoked from an AsyncRunner worker thread
    // (MtpExecutor::decodeStep) and from the engine main thread (PyWrappedModel::forward),
    // neither of which holds the GIL on entry. pybind11's attr() and call operator construct
    // a Python args tuple via PyTuple_New, which segfaults without the GIL.
    {
        RTP_LLM_PROFILE_SCOPE("cuda_graph.prepareAttentionInputs(prepare_cuda_graph)");
        if (pending_host_mirror_d2h) {
            // Wait for the async D2H metadata copies above so the CPU-side
            // fill/plan paths inside prepare_cuda_graph read this step's
            // values instead of the previous step's.
            RTP_LLM_PROFILE_SCOPE("cuda_graph.prepareAttentionInputs(wait_host_mirror_d2h)");
            cuda_graph::graphGetCurrentStream().synchronize();
        }
        py::gil_scoped_acquire gil;
        callPrepareCudaGraph(attn_pyobj, py_model_inputs_);
    }
}

void CudaGraphRunner::updateKVCacheKernelBlockId(const PyModelInputs& inputs, CudaGraphState& state) {
    RTP_LLM_PROFILE_SCOPE("cuda_graph.updateKVCacheKernelBlockId");
    const size_t graph_idx =
        is_prefill_cuda_graph_mode_ ? state.current_real_graph_seq_len : state.current_real_graph_bs;
    auto& py_model_inputs = graph_instances_[graph_idx].mem_hold_.py_model_inputs_;

    FusedD2DCopyParams     d2d_copies;
    FusedStridedCopyParams strided_d2d_copies;
    auto add_block_table = [&d2d_copies, &strided_d2d_copies](const torch::Tensor& src, torch::Tensor& dst) {
        if (!src.defined() || src.numel() == 0) {
            return;
        }
        if (src.dim() < 2) {
            d2d_copies.add(src.data_ptr(), dst.data_ptr(), src.numel() * src.element_size());
            return;
        }
        strided_d2d_copies.add(src.data_ptr(),
                               dst.data_ptr(),
                               src.size(0),
                               src.size(1) * src.element_size(),
                               src.stride(0) * src.element_size(),
                               dst.stride(0) * dst.element_size());
    };

    if (inputs.attention_inputs_by_tag.empty()) {
        py_model_inputs.attention_inputs.kv_cache_kernel_block_id_device.fill_(safeKernelBlockIdForFlatTable());
        add_block_table(inputs.attention_inputs.kv_cache_kernel_block_id_device,
                        py_model_inputs.attention_inputs.kv_cache_kernel_block_id_device);
    } else {
        RTP_LLM_CHECK_WITH_INFO(inputs.attention_inputs_by_tag.size() == py_model_inputs.attention_inputs_by_tag.size(),
                                "tagged attention input size mismatch while refreshing CUDA graph block tables");
        for (const auto& [tag, src_inputs] : inputs.attention_inputs_by_tag) {
            auto dst_it = py_model_inputs.attention_inputs_by_tag.find(tag);
            RTP_LLM_CHECK_WITH_INFO(dst_it != py_model_inputs.attention_inputs_by_tag.end(),
                                    "CUDA graph capture has no attention input for tag=%s",
                                    tag.c_str());
            dst_it->second.kv_cache_kernel_block_id_device.fill_(safeKernelBlockIdForTag(tag));
            add_block_table(src_inputs.kv_cache_kernel_block_id_device, dst_it->second.kv_cache_kernel_block_id_device);
        }
    }
    fusedCopy(d2d_copies);
    fusedStridedCopy(strided_d2d_copies);
}

PyModelOutputs CudaGraphRunner::forward(const PyModelInputs& inputs, CudaGraphState& state) {
    c10::InferenceMode inference_guard(true);
    PyModelOutputs     outputs;

    struct PreparedFlagGuard {
        std::atomic<bool>& flag;
        ~PreparedFlagGuard() {
            flag.store(false, std::memory_order_release);
        }
    } flag_guard{prepared_attention_inputs_};

    // decode or embedding model only
    RTP_LLM_LOG_DEBUG("Replay Start");
    if (!prepared_attention_inputs_.load(std::memory_order_acquire)) {
        prepareInputs(inputs, state);
    } else {
        prepareInputData(inputs, state);
    }
    if (is_prefill_cuda_graph_mode_) {
        {
            RTP_LLM_PROFILE_SCOPE("cuda_graph.forward(replayPrefill)");
            replayPrefill(state.current_real_graph_seq_len);
        }
        outputs.hidden_states =
            graph_instances_[state.current_real_graph_seq_len].mem_hold_.decoder_layer_hidden_states_.slice(
                0, 0, state.current_seq_len);
    } else {
        {
            RTP_LLM_PROFILE_SCOPE("cuda_graph.forward(replayDecode)");
            replayDecode(state.current_real_graph_bs);
        }
        outputs.hidden_states =
            graph_instances_[state.current_real_graph_bs].mem_hold_.decoder_layer_hidden_states_.slice(
                0, 0, state.seq_len_sum);
    }
    // record forward done event
    forward_event_.record(cuda_graph::graphGetCurrentStream());
    RTP_LLM_LOG_DEBUG("Replay End");
    return outputs;
}

bool CudaGraphRunner::tryGetRealGraphPrefillSeqLen(const PyModelInputs& inputs, CudaGraphState& state) {
    state.current_seq_len = inferTotalTokensNoSync(inputs);
    if (state.current_seq_len <= 0) {
        RTP_LLM_LOG_WARNING("prefill cuda graph: total token count is unavailable, fallback to normal run");
        return false;
    }
    if (capture_range_.empty()) {
        RTP_LLM_LOG_WARNING("prefill cuda graph: capture_range_ is empty, cannot run");
        return false;
    }
    auto it = std::lower_bound(capture_range_.begin(), capture_range_.end(), state.current_seq_len);
    // No captured graph for seq_len >= current (all captures smaller than requested)
    if (it == capture_range_.end()) {
        RTP_LLM_LOG_WARNING("prefill seq_len %d exceeds max captured %d, fallback to normal run",
                            state.current_seq_len,
                            capture_range_.back());
        return false;
    }
    state.current_real_graph_seq_len = *it;
    state.current_batch_size         = inputs.attention_inputs.input_lengths.size(0);
    return true;
}

bool CudaGraphRunner::tryGetRealGraphDecodeBatchSize(const PyModelInputs& inputs, CudaGraphState& state) {
    int cuda_graph_bs        = inputs.attention_inputs.input_lengths.size(0);
    state.current_batch_size = cuda_graph_bs;
    RTP_LLM_LOG_DEBUG("canRun judge for batch size: %d", cuda_graph_bs);
    if (capture_range_.empty()) {
        RTP_LLM_LOG_WARNING("decode cuda graph: capture_range_ is empty, cannot run");
        return false;
    }
    auto it = std::lower_bound(capture_range_.begin(), capture_range_.end(), state.current_batch_size);
    // No captured graph for batch >= current (all captures smaller)
    if (it == capture_range_.end()) {
        RTP_LLM_LOG_WARNING("decode batch size %d exceeds max captured %d, fallback to normal run",
                            state.current_batch_size,
                            capture_range_.back());
        return false;
    }
    state.current_real_graph_bs = *it;
    RTP_LLM_LOG_DEBUG(
        "batch size used in replay: %d (graph key %d)", state.current_batch_size, state.current_real_graph_bs);

    if (inputs.attention_inputs.is_prefill) {
        state.seq_len_sum = inferTotalTokensNoSync(inputs);
        if (state.seq_len_sum <= 0) {
            RTP_LLM_LOG_WARNING("decode cuda graph: prefill token count is unavailable, fallback to normal run");
            return false;
        }
    } else {
        state.seq_len_sum = cuda_graph_bs;
    }
    RTP_LLM_LOG_DEBUG("can run cuda graph for decode");
    return true;
}

bool CudaGraphRunner::validateComboPositionIds(const PyModelInputs&  inputs,
                                               const CudaGraphState& state,
                                               const torch::Tensor&  captured_position_ids,
                                               size_t&               copy_numel) const {
    const int token_count = is_prefill_cuda_graph_mode_ ? state.current_seq_len : state.seq_len_sum;
    return validateComboPositionIdsForReplay(
        position_id_len_factor_, token_count, inputs.combo_position_ids, captured_position_ids, copy_numel);
}

bool CudaGraphRunner::canReplaySelectedGraph(const PyModelInputs& inputs, const CudaGraphState& state) const {
    const int  graph_key = is_prefill_cuda_graph_mode_ ? state.current_real_graph_seq_len : state.current_real_graph_bs;
    const auto graph_it  = graph_instances_.find(graph_key);
    if (graph_it == graph_instances_.end()) {
        RTP_LLM_LOG_WARNING("CUDA graph key %d was not captured, fallback to normal run", graph_key);
        return false;
    }

    size_t      copy_numel            = 0;
    const auto& captured_position_ids = graph_it->second.mem_hold_.py_model_inputs_.combo_position_ids;
    if (!validateComboPositionIds(inputs, state, captured_position_ids, copy_numel)) {
        const uint64_t fallback_count = combo_position_fallback_count_.fetch_add(1, std::memory_order_relaxed) + 1;
        // Log the first fallback and then at powers of two. This keeps the
        // decode hot path quiet while retaining a monotonic, observable count.
        if ((fallback_count & (fallback_count - 1)) == 0) {
            RTP_LLM_LOG_WARNING(
                "combo_position_ids are incompatible with CUDA graph key %d: factor=%d, src_numel=%lld, "
                "dst_numel=%lld; fallback to normal run (fallback_count=%llu)",
                graph_key,
                position_id_len_factor_,
                inputs.combo_position_ids.defined() ? static_cast<long long>(inputs.combo_position_ids.numel()) : -1LL,
                captured_position_ids.defined() ? static_cast<long long>(captured_position_ids.numel()) : -1LL,
                static_cast<unsigned long long>(fallback_count));
        }
        return false;
    }
    return true;
}

bool CudaGraphRunner::canRun(const PyModelInputs& inputs, CudaGraphState& state) {
    RTP_LLM_PROFILE_SCOPE("cuda_graph.canRun");
    if (!enable_cuda_graph_) {
        return false;
    }
    if (kv_cache_group_tags_.size() > 1) {
        if (inputs.attention_inputs_by_tag.size() != kv_cache_group_tags_.size()) {
            RTP_LLM_LOG_WARNING("Tagged kv cache size mismatch: inputs=%zu, captured=%zu, fallback to normal run.",
                                inputs.attention_inputs_by_tag.size(),
                                kv_cache_group_tags_.size());
            return false;
        }
        for (const auto& tag : kv_cache_group_tags_) {
            if (inputs.attention_inputs_by_tag.find(tag) == inputs.attention_inputs_by_tag.end()) {
                RTP_LLM_LOG_WARNING("Tagged kv cache is missing tag=%s, fallback to normal run.", tag.c_str());
                return false;
            }
        }
    } else if (!inputs.attention_inputs_by_tag.empty()) {
        RTP_LLM_LOG_WARNING("Tagged kv cache input does not match a single-group CUDA graph, fallback to normal run.");
        return false;
    }

    if (is_target_verify_) {
        if (!inputs.attention_inputs.is_target_verify || !inputs.attention_inputs.is_prefill) {
            return false;
        }
        if (!tryGetRealGraphDecodeBatchSize(inputs, state)) {
            return false;
        }
        const int expected_tokens = state.current_batch_size * num_tokens_per_bs_;
        RTP_LLM_CHECK_WITH_INFO(state.seq_len_sum == expected_tokens,
                                "target-verify decode graph expects %d tokens (%d batches * %d), got %d",
                                expected_tokens,
                                state.current_batch_size,
                                num_tokens_per_bs_,
                                state.seq_len_sum);
        if (inputs.input_hiddens.defined() && inputs.input_hiddens.numel() > 0
            && inputs.input_hiddens.size(0) != expected_tokens) {
            RTP_LLM_FAIL("target-verify decode graph expects %d input-hidden rows, got %ld",
                         expected_tokens,
                         inputs.input_hiddens.size(0));
        }
        return canReplaySelectedGraph(inputs, state);
    }

    if (inputs.attention_inputs.is_prefill && !is_prefill_cuda_graph_mode_) {
        return false;
    }

    if (is_prefill_cuda_graph_mode_) {
        if (!tryGetRealGraphPrefillSeqLen(inputs, state)) {
            return false;
        }
        // current_real_graph_seq_len is always *it from lower_bound within capture_range_
        RTP_LLM_LOG_DEBUG("prefill cuda graph replay seq_len key %d", state.current_real_graph_seq_len);
    } else if (!tryGetRealGraphDecodeBatchSize(inputs, state)) {
        return false;
    }
    return canReplaySelectedGraph(inputs, state);
}

void CudaGraphRunner::initKernelInternalMemory() {
    torch::Tensor cu_seqlens =
        torch::zeros({int(max_bs_ + 1)}, torch::TensorOptions(torch::kInt32).device(torch::kCPU)).pin_memory();
    torch::Tensor cu_kv_seqlens =
        torch::zeros({int(max_bs_ + 1)}, torch::TensorOptions(torch::kInt32).device(torch::kCPU).pinned_memory(true));
    auto input_lengths  = capture_mem_hold_.py_model_inputs_.attention_inputs.input_lengths;
    auto prefix_lengths = capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths;

    cu_seqlens.slice(0, 1, max_bs_ + 1) = input_lengths.cumsum(0);
    if (prefix_lengths.defined() && prefix_lengths.size(0) > 0) {
        cu_kv_seqlens.slice(0, 1, max_bs_ + 1) = input_lengths.add(prefix_lengths).cumsum(0);
    }
    capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens           = cu_seqlens;
    capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens_device    = cu_seqlens.cuda();
    capture_mem_hold_.py_model_inputs_.attention_inputs.cu_kv_seqlens_device = cu_kv_seqlens.cuda();
    refreshTaggedAttentionInputs(capture_mem_hold_.py_model_inputs_);
}

int CudaGraphRunner::getCurrentRealGraphBs(const CudaGraphState& state) const {
    return state.current_real_graph_bs;
}

void CudaGraphRunner::initCaptureAttentionInputs(PyModelInputs& inputs, int max_bs, int num_tokens_per_bs) {
    inputs.attention_inputs.is_target_verify = is_target_verify_;
    inputs.attention_inputs.is_prefill       = is_prefill_cuda_graph_mode_ || is_target_verify_;

    // input_ids [tokens_nums] = [batch_size * num_tokens_per_bs]
    inputs.input_ids = torch::zeros({max_num_token_}, options_cuda_int32_);
    // input_lengths [batch_size, int32] (decode only)
    inputs.attention_inputs.input_lengths        = torch::full({int(max_bs_)}, num_tokens_per_bs_, options_cpu_int32_);
    inputs.attention_inputs.input_lengths        = inputs.attention_inputs.input_lengths.pin_memory();
    inputs.attention_inputs.input_lengths_device = inputs.attention_inputs.input_lengths.cuda();
    // sequence_lengths [batch_size, int32] (decode only)
    // sequence_length should in pinned memory
    inputs.attention_inputs.sequence_lengths = torch::ones({int(max_bs_)}, options_cpu_int32_);
    inputs.attention_inputs.sequence_lengths.fill_(max_seq_len_ - num_tokens_per_bs - 1);
    inputs.attention_inputs.sequence_lengths = inputs.attention_inputs.sequence_lengths.pin_memory();

    const auto logical_tokens_per_block = cache_config_->seq_size_per_block;
    RTP_LLM_CHECK_WITH_INFO(max_seq_len_ >= 0 && sp_steps_ >= 0,
                            "CUDA graph requires non-negative max_seq_len/sp_steps, got %d/%d",
                            max_seq_len_,
                            sp_steps_);
    const auto max_seq_len = static_cast<size_t>(max_seq_len_);
    RTP_LLM_CHECK_WITH_INFO(max_seq_len <= std::numeric_limits<size_t>::max() - (logical_tokens_per_block - 1),
                            "CUDA graph logical block count overflow");
    const size_t base_logical_blocks = (max_seq_len + logical_tokens_per_block - 1) / logical_tokens_per_block;
    RTP_LLM_CHECK_WITH_INFO(base_logical_blocks <= std::numeric_limits<size_t>::max() - static_cast<size_t>(sp_steps_),
                            "CUDA graph logical block slack overflow");
    const size_t logical_blocks  = base_logical_blocks + static_cast<size_t>(sp_steps_);
    size_t       max_group_ratio = 1;
    for (const auto& group : cache_config_->groups()) {
        max_group_ratio = std::max(max_group_ratio, group.storedKernelBlocksPerKvBlock());
    }
    RTP_LLM_CHECK_WITH_INFO(logical_blocks <= std::numeric_limits<size_t>::max() / max_group_ratio,
                            "CUDA graph kernel block capacity overflow: logical_blocks=%zu ratio=%zu",
                            logical_blocks,
                            max_group_ratio);
    const size_t max_blocks_size = logical_blocks * max_group_ratio;
    RTP_LLM_CHECK_WITH_INFO(max_blocks_size <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                            "CUDA graph kernel block capacity exceeds int64: %zu",
                            max_blocks_size);

    // Allocate combo_position_ids capture buffer only when the model actually uses
    // combo position ids (Mrope etc.). The factor is sourced from the C++ rope_config
    // by PyWrappedModel — 0 means "no combo_position_ids" and the buffer stays unset
    // (non-Mrope models pay zero memory and the captured graph never references it).
    if (position_id_len_factor_ > 0) {
        inputs.combo_position_ids =
            torch::ones({int(max_bs_) * num_tokens_per_bs_ * position_id_len_factor_}, options_cuda_int32_);
        inputs.attention_inputs.combo_position_ids = inputs.combo_position_ids;
    }

    const int64_t max_blocks = static_cast<int64_t>(max_blocks_size);
    // kv_cache_kernel_block_id_device [batch_size, block_num]
    inputs.attention_inputs.kv_cache_kernel_block_id_device =
        torch::full({int(max_bs_), max_blocks}, safeKernelBlockIdForPrimaryTable(), options_cuda_int32_);

    inputs.attention_inputs.kv_cache_kernel_block_id =
        torch::full({int(max_bs_), max_blocks}, safeKernelBlockIdForPrimaryTable(), options_cpu_int32_).pin_memory();

    // Target verify keeps multi-token attention geometry while selecting
    // graphs by batch.
    // Plain one-token decode must leave prefix_lengths undefined.
    if (is_target_verify_) {
        inputs.attention_inputs.prefix_lengths =
            torch::full({int(max_bs_)}, max_seq_len_ - num_tokens_per_bs_, options_cpu_int32_).pin_memory();
        inputs.attention_inputs.prefix_lengths_device = inputs.attention_inputs.prefix_lengths.cuda();
    } else if (is_prefill_cuda_graph_mode_) {
        // ROCm needs prefix>0 here for AiterPrefillImplPaged.support(); CUDA keeps prefix=0.
#if USING_ROCM
        const int prefix_init = isMtpDraftPrefillCudaGraph() ? max_seq_len_ : 0;
#else
        const int prefix_init = 0;
#endif
        inputs.attention_inputs.prefix_lengths =
            torch::full({int(max_bs_)}, prefix_init, options_cpu_int32_).pin_memory();
        inputs.attention_inputs.prefix_lengths_device = inputs.attention_inputs.prefix_lengths.cuda();
    } else {
        // Decode CUDA graph mode: prefix_lengths should be empty tensor
        inputs.attention_inputs.prefix_lengths = torch::empty({0}, options_cpu_int32_).pin_memory();
    }
    // padding_offset [max_num_token_, int32] (for attention padding)
    inputs.attention_inputs.padding_offset = torch::zeros({int(max_seq_len_ * max_bs_)}, options_cpu_int32_);
    inputs.attention_inputs.padding_offset = inputs.attention_inputs.padding_offset.pin_memory();
    inputs.attention_inputs.dtype          = model_data_type_;
    inputs.attention_inputs.is_s_padded    = true;
    auto sequence_lengths_plus_1           = inputs.attention_inputs.sequence_lengths.add(1).pin_memory();
    inputs.attention_inputs.sequence_lengths_plus_1_device = sequence_lengths_plus_1.cuda();
    // Step=1 is intentional: when num_tokens_per_bs_ > 1 (target verify), is_prefill is set to true
    // so the factory selects PREFILL impls (which use cu_seqlens, not decode_cu_seqlens).
    // XQADecodeImpl/XQAWrapper (the consumers of decode_cu_seqlens_host) are never reached in that path.
    inputs.attention_inputs.decode_cu_seqlens_device =
        torch::arange(0, max_bs_ + 1, 1, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
    inputs.attention_inputs.decode_cu_seqlens = torch::arange(0, max_bs_ + 1, 1, options_cpu_int32_).pin_memory();

    inputs.attention_inputs_by_tag.clear();
    if (kv_cache_group_tags_.size() > 1) {
        // Boundary adapter: capture buffers are addressed by an adapter-local
        // group_ordinal taken from the canonical sorted tag order, so which tag
        // reuses the shared fast-path buffers does not depend on the topology's
        // record order. The ordinal never leaves this function.
        const auto sorted_tags = sortedCacheGroupTags(kv_cache_group_tags_, "CUDA graph KV cache");
        for (size_t group_ordinal = 0; group_ordinal < sorted_tags.size(); ++group_ordinal) {
            auto tagged_inputs = inputs.attention_inputs;
            if (group_ordinal > 0) {
                const auto safe_kernel_id = safeKernelBlockIdForTag(sorted_tags[group_ordinal]);
                tagged_inputs.kv_cache_kernel_block_id_device =
                    torch::full({int(max_bs_), max_blocks}, safe_kernel_id, options_cuda_int32_);
                tagged_inputs.kv_cache_kernel_block_id =
                    torch::full({int(max_bs_), max_blocks}, safe_kernel_id, options_cpu_int32_).pin_memory();
            }
            const auto [it, inserted] =
                inputs.attention_inputs_by_tag.emplace(sorted_tags[group_ordinal], std::move(tagged_inputs));
            (void)it;
            RTP_LLM_CHECK_WITH_INFO(
                inserted, "duplicate CUDA graph KV cache tag=%s", sorted_tags[group_ordinal].c_str());
        }
    }
}

void CudaGraphRunner::initCaptureAttentionInputsPost() {
    auto&         inputs                        = capture_mem_hold_.py_model_inputs_;
    torch::Tensor cuda_graph_prefill_batch_size = torch::zeros({1}, options_cpu_int32_).pin_memory();
    // as one batch to capture
    cuda_graph_prefill_batch_size.fill_(1);
    RTP_LLM_CHECK_WITH_INFO(cuda_graph_prefill_batch_size.is_pinned(),
                            "capture_mem_hold_ cuda_graph_prefill_batch_size is not pinned memory");

    // draft model prefill but not embedding model
    if (num_tokens_per_bs_ > 1 && num_tokens_per_bs_ != max_seq_len_) {
        inputs.attention_inputs.prefill_cuda_graph_copy_params =
            PyPrefillCudaGaphCopyParams{cuda_graph_prefill_batch_size, num_tokens_per_bs_, int(max_bs_)};
    } else {
        inputs.attention_inputs.prefill_cuda_graph_copy_params =
            PyPrefillCudaGaphCopyParams{cuda_graph_prefill_batch_size, max_seq_len_, int(max_bs_)};
    }
    refreshTaggedAttentionInputs(inputs);
}

void CudaGraphRunner::setPositionEncoding(torch::Tensor position_encoding) {
    position_encoding_ = position_encoding;
}

void CudaGraphRunner::setTokenTypeEmbedding(torch::Tensor token_type_embedding) {
    token_type_embedding_ = token_type_embedding;
}

void CudaGraphRunner::setInputEmbeddingScalar(float input_embedding_scalar) {
    input_embedding_scalar_ = input_embedding_scalar;
}

void CudaGraphRunner::initCaptureBertEmbeddingInputs(PyModelInputs& inputs, int max_bs, int max_num_token) {
    auto options_cuda_int32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);
    // Initialize BertEmbeddingInputs for capture
    // combo_position_ids: empty tensor for capture (will be filled during actual forward)
    inputs.bert_embedding_inputs.combo_position_ids = torch::zeros({max_seq_len_ * max_bs}, options_cuda_int32);

    // position_encoding: from weights
    inputs.bert_embedding_inputs.position_encoding = position_encoding_;

    // combo_tokens_type_ids: empty tensor for capture (will be filled during actual forward)
    inputs.bert_embedding_inputs.combo_tokens_type_ids = torch::zeros({max_seq_len_ * max_bs}, options_cuda_int32);

    // token_type_embedding: from weights
    inputs.bert_embedding_inputs.token_type_embedding = token_type_embedding_;

    // input_embedding_scalar: fixed value
    inputs.bert_embedding_inputs.input_embedding_scalar = input_embedding_scalar_;
}

void CudaGraphRunner::logCudaGraphPoolMemory(const char* phase) {
    size_t free_bytes  = 0;
    size_t total_bytes = 0;
    cuda_graph::graphMemGetInfo(&free_bytes, &total_bytes);
    const size_t used_bytes        = total_bytes - free_bytes;
    const size_t pytorch_allocated = cuda_graph::graphAllocatedBytes();
    const size_t pytorch_reserved  = cuda_graph::graphReservedBytes();
    const size_t pool_overhead     = pytorch_reserved > pytorch_allocated ? pytorch_reserved - pytorch_allocated : 0;

    RTP_LLM_LOG_INFO("[CudaGraph Memory][%s] cudaMemGetInfo: used=%zu MiB, free=%zu MiB, total=%zu MiB | "
                     "PyTorch: allocated=%zu MiB, reserved=%zu MiB, pool_overhead=%zu MiB",
                     phase,
                     used_bytes / 1024 / 1024,
                     free_bytes / 1024 / 1024,
                     total_bytes / 1024 / 1024,
                     pytorch_allocated / 1024 / 1024,
                     pytorch_reserved / 1024 / 1024,
                     pool_overhead / 1024 / 1024);
}

void CudaGraphRunner::initCapture() {
    c10::InferenceMode inference_guard(true);

    if (enable_cuda_graph_) {
        RTP_LLM_LOG_INFO("CUDA graph capture is enabled");
        shared_graph_pool_ = cuda_graph::graphPoolHandle();
        if (is_prefill_cuda_graph_mode_) {
            RTP_LLM_LOG_INFO("CUDA graph capture for prefill, num_tokens_per_bs_: %d", num_tokens_per_bs_);
        }
        max_num_token_ = max_bs_ * num_tokens_per_bs_;
        if (is_prefill_cuda_graph_mode_) {
            capture_range_ = getPrefillSequenceLengthsToCapture();
        } else {
            capture_range_ = getDecodeBatchSizesToCapture();
        }

        PyModelInputs inputs;
        // input_ids [tokens_nums] = [batch_size * num_tokens_per_bs]
        inputs.input_ids = torch::zeros({max_num_token_}, options_cuda_int32_);
        // input_hidden_size_ is the width of one input_hiddens row. PyWrappedModel sets it
        // to hidden_size * hc_mult for regular (MTP) graphs and to
        // len(target_layer_ids) * hidden_size for a DSpARK draft graph, so it must be used
        // instead of recomputing hc_mult_ * hidden_size_ here.
        RTP_LLM_CHECK_WITH_INFO(
            input_hidden_size_ > 0, "CUDA graph input_hidden_size must be positive, got %zu", input_hidden_size_);
        inputs.input_hiddens =
            torch::zeros({max_num_token_, static_cast<int64_t>(input_hidden_size_)}, options_cuda_float_);
        // Setup attention inputs using the extracted function
        initCaptureAttentionInputs(inputs, max_bs_, num_tokens_per_bs_);

        // Setup BertEmbedding inputs using the extracted function
        initCaptureBertEmbeddingInputs(inputs, max_bs_, max_num_token_);

        torch::Tensor output;
        capture_mem_hold_ = CaptureMemoryHold(output, inputs, is_prefill_cuda_graph_mode_);
        initKernelInternalMemory();

        // get real output data type (params already prepared in attn impl __init__/create_params)
        py::object attn_pyobj;
        try {
            attn_pyobj = py_attn_pyobj_method_(capture_mem_hold_.py_model_inputs_, true);
        } catch (const py::error_already_set& e) {
            RTP_LLM_LOG_ERROR("initCapture prepare_fmha_impl failed: %s", e.what());
            throw;
        }
        RTP_LLM_LOG_INFO("initCapture forward for output datatype start");
        try {
            // Distributed model implementations may rendezvous during this
            // eager warmup forward. The flag is scoped so real graph capture
            // and replay never contain that synchronization.
            ScopedEnvFlag cuda_graph_warmup("RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD", "1");
            py_forward_method_(capture_mem_hold_.py_model_inputs_, attn_pyobj);
        } catch (const py::error_already_set& e) {
            RTP_LLM_LOG_ERROR("initCapture forward for output datatype failed with Python exception: %s", e.what());
            throw;
        } catch (const std::exception& e) {
            RTP_LLM_LOG_ERROR("initCapture forward for output datatype failed with C++ exception: %s", e.what());
            throw;
        }
        RTP_LLM_LOG_INFO("initCapture forward for output datatype end");
        output = torch::zeros({max_num_token_, hidden_size_}, options_cuda_float_);
        capture_mem_hold_.setHiddenStates(output);
        initCaptureAttentionInputsPost();
        logCudaGraphPoolMemory("before_capture");

        if (is_prefill_cuda_graph_mode_) {
            RTP_LLM_CHECK_WITH_INFO(isEmbeddingStylePrefillCudaGraph() || isMtpDraftPrefillCudaGraph(),
                                    "prefill cuda graph: expected embedding-style or MTP draft layout");
            capturePrefill();
        } else {
            captureDecode();
        }
        logCudaGraphPoolMemory("after_capture");
    } else {
        initKernelInternalMemory();
        RTP_LLM_LOG_INFO("CUDA graph capture is not enabled, skipping initialization");
    }
}

void CudaGraphRunner::replayGraph(int key) {
    graph_instances_[key].graph_.replay();
}

void CudaGraphRunner::captureOneGraphInstance(int key, const char* key_type) {
    auto inputs = graph_instances_[key].mem_hold_.py_model_inputs_;

    size_t pre_capture_reserved = cuda_graph::graphReservedBytes();

    // WarmUp twice (params already prepared in attn impl __init__/create_params when instance was created)
    RTP_LLM_LOG_INFO("WarmUp for %s %d start.", key_type, key);
    auto attn_pyobj = graph_instances_[key].mem_hold_.attn_pyobj_;
    try {
        py_forward_method_(inputs, attn_pyobj);
        py_forward_method_(inputs, attn_pyobj);
    } catch (const py::error_already_set& e) {
        RTP_LLM_LOG_ERROR("WarmUp forward failed for %s %d: %s", key_type, key, e.what());
        throw;
    }
    RTP_LLM_LOG_INFO("WarmUp for %s %d successfully.", key_type, key);

    {
        // sync before capture
        cuda_graph::graphDeviceSynchronize();

        CudaGraphStreamLife stream_life(capture_stream_);
        auto&               graph               = graph_instances_[key].graph_;
        std::string         output_dot_filename = "";
        if (enable_cuda_graph_debug_mode_) {
            graph.enable_debug_mode();
            std::string key_type_str = std::string(key_type);
            std::replace(key_type_str.begin(), key_type_str.end(), ' ', '_');
            output_dot_filename = "cuda_graph_tokens" + std::to_string(num_tokens_per_bs_) + "_" + key_type_str + "_"
                                  + std::to_string(key) + "_visualization.dot";
            RTP_LLM_LOG_INFO("CUDA Graph debug mode enabled, output file: %s", output_dot_filename.c_str());
        }
        RTP_LLM_LOG_INFO("Capture for %s %d begin.", key_type, key);
        PyModelOutputs outputs;
        {
            cuda_graph::graphCaptureBegin(graph, shared_graph_pool_);
            cuda_graph::GraphNcclCaptureContext capture_ctx;
            CudaGraphCaptureGuard               capture_guard(&capture_ctx);
            try {
                auto py_outputs_obj = py_forward_method_(inputs, attn_pyobj);
                outputs             = py_outputs_obj.cast<PyModelOutputs>();
            } catch (const py::error_already_set& e) {
                RTP_LLM_LOG_ERROR("Capture forward failed for %s %d: %s", key_type, key, e.what());
                throw;
            }
            graph_instances_[key].mem_hold_.decoder_layer_hidden_states_.copy_(outputs.hidden_states);
            graph.capture_end();
        }

        if (enable_cuda_graph_debug_mode_) {
            RTP_LLM_LOG_INFO("Calling debug_dump to generate: %s", output_dot_filename.c_str());
            graph.debug_dump(output_dot_filename.c_str());
            RTP_LLM_LOG_INFO("debug_dump completed for: %s", output_dot_filename.c_str());
        }

        size_t post_capture_reserved = cuda_graph::graphReservedBytes();
        size_t graph_pool_delta =
            post_capture_reserved > pre_capture_reserved ? post_capture_reserved - pre_capture_reserved : 0;
        RTP_LLM_LOG_INFO("[CudaGraph Memory] captured %s %d: pool_delta=%zu MiB, total_reserved=%zu MiB",
                         key_type,
                         key,
                         graph_pool_delta / 1024 / 1024,
                         post_capture_reserved / 1024 / 1024);
    }
}

void CudaGraphRunner::replayAndSyncCheck(int key, const char* key_type) {
    RTP_LLM_LOG_INFO("replay start check for %s %d", key_type, key);
    replayGraph(key);
    cuda_graph::graphDeviceSynchronize();
    RTP_LLM_LOG_INFO("replay end check for %s %d", key_type, key);
}

void CudaGraphRunner::prepareCaptureInputs(PyModelInputs& inputs, int batch_size, int seq_len_or_tokens) {
    // Common slice operations for input_ids and padding_offset
    inputs.attention_inputs.is_prefill       = is_prefill_cuda_graph_mode_ || is_target_verify_;
    inputs.attention_inputs.is_target_verify = is_target_verify_;
    // HC-shaped MTP draft prefill executes a fixed-capacity Python path. Other
    // MTP models must slice to the current graph key so FlashInfer's batch
    // indices length remains equal to the query nnz.
    const bool fixed_capacity_draft_prefill = usesFixedCapacityMtpDraftPrefillCudaGraph();
    const int  token_slice_len = fixed_capacity_draft_prefill ? max_bs_ * num_tokens_per_bs_ : seq_len_or_tokens;
    inputs.input_ids           = capture_mem_hold_.py_model_inputs_.input_ids.slice(0, 0, token_slice_len);
    inputs.input_hiddens       = capture_mem_hold_.py_model_inputs_.input_hiddens.slice(0, 0, token_slice_len);
    inputs.attention_inputs.input_lengths =
        capture_mem_hold_.py_model_inputs_.attention_inputs.input_lengths.slice(0, 0, batch_size);
    inputs.attention_inputs.input_lengths_device =
        capture_mem_hold_.py_model_inputs_.attention_inputs.input_lengths_device.slice(0, 0, batch_size);
    inputs.attention_inputs.padding_offset =
        capture_mem_hold_.py_model_inputs_.attention_inputs.padding_offset.slice(0, 0, seq_len_or_tokens);

    // Common slice operations for attention inputs
    if (capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths.defined()) {
        if (capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths.size(0) > 0) {
            inputs.attention_inputs.prefix_lengths =
                capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths.slice(0, 0, batch_size);
            inputs.attention_inputs.prefix_lengths_device =
                capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths_device.slice(0, 0, batch_size);
        } else {
            // For decode CUDA graph mode: prefix_lengths is empty tensor
            inputs.attention_inputs.prefix_lengths = capture_mem_hold_.py_model_inputs_.attention_inputs.prefix_lengths;
        }
    }
    inputs.attention_inputs.sequence_lengths =
        capture_mem_hold_.py_model_inputs_.attention_inputs.sequence_lengths.slice(0, 0, batch_size);
    if (capture_mem_hold_.py_model_inputs_.combo_position_ids.defined()) {
        // Buffer was allocated as max_bs_ * num_tokens_per_bs_ * position_id_len_factor_;
        // slice proportionally with current batch_size using the same factor.
        inputs.combo_position_ids = capture_mem_hold_.py_model_inputs_.combo_position_ids.slice(
            0, 0, batch_size * num_tokens_per_bs_ * position_id_len_factor_);
        inputs.attention_inputs.combo_position_ids = inputs.combo_position_ids;
    }

    inputs.attention_inputs.kv_cache_kernel_block_id_device =
        capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_kernel_block_id_device.slice(0, 0, batch_size);
    inputs.attention_inputs.kv_cache_kernel_block_id =
        capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_kernel_block_id.slice(0, 0, batch_size);
    inputs.attention_inputs.kv_cache_block_id_device =
        capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_block_id_device.defined() ?
            capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_block_id_device.slice(0, 0, batch_size) :
            torch::Tensor();
    inputs.attention_inputs.kv_cache_block_id =
        capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_block_id.defined() ?
            capture_mem_hold_.py_model_inputs_.attention_inputs.kv_cache_block_id.slice(0, 0, batch_size) :
            torch::Tensor();
    inputs.attention_inputs.cu_seqlens =
        capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens.slice(0, 0, batch_size + 1);
    inputs.attention_inputs.cu_seqlens_device =
        capture_mem_hold_.py_model_inputs_.attention_inputs.cu_seqlens_device.slice(0, 0, batch_size + 1);
    inputs.attention_inputs.cu_kv_seqlens_device =
        capture_mem_hold_.py_model_inputs_.attention_inputs.cu_kv_seqlens_device.slice(0, 0, batch_size + 1);
    inputs.attention_inputs.decode_cu_seqlens_device =
        capture_mem_hold_.py_model_inputs_.attention_inputs.decode_cu_seqlens_device.slice(0, 0, batch_size + 1);
    inputs.attention_inputs.decode_cu_seqlens =
        capture_mem_hold_.py_model_inputs_.attention_inputs.decode_cu_seqlens.slice(0, 0, batch_size + 1);
    inputs.attention_inputs.sequence_lengths_plus_1_device =
        capture_mem_hold_.py_model_inputs_.attention_inputs.sequence_lengths_plus_1_device.slice(0, 0, batch_size);

    inputs.attention_inputs_by_tag.clear();
    for (const auto& [tag, cap_attn] : capture_mem_hold_.py_model_inputs_.attention_inputs_by_tag) {
        auto tagged_inputs = inputs.attention_inputs;
        tagged_inputs.kv_cache_kernel_block_id_device =
            cap_attn.kv_cache_kernel_block_id_device.slice(0, 0, batch_size);
        tagged_inputs.kv_cache_kernel_block_id = cap_attn.kv_cache_kernel_block_id.slice(0, 0, batch_size);
        if (cap_attn.kv_cache_block_id_device.defined()) {
            tagged_inputs.kv_cache_block_id_device = cap_attn.kv_cache_block_id_device.slice(0, 0, batch_size);
        }
        if (cap_attn.kv_cache_block_id.defined()) {
            tagged_inputs.kv_cache_block_id = cap_attn.kv_cache_block_id.slice(0, 0, batch_size);
        }
        inputs.attention_inputs_by_tag.emplace(tag, std::move(tagged_inputs));
    }

    // Common direct assignments (no slice needed)
    inputs.attention_inputs.dtype       = capture_mem_hold_.py_model_inputs_.attention_inputs.dtype;
    inputs.bert_embedding_inputs        = capture_mem_hold_.py_model_inputs_.bert_embedding_inputs;
    inputs.attention_inputs.is_s_padded = true;
    refreshTaggedAttentionInputs(inputs);
}

CaptureMemoryHold CudaGraphRunner::createCaptureMemoryHold(PyModelInputs& inputs, int tokens_count) {
    // only when prefill or target model score phase, the num_tokens_per_bs_ > 1
    return CaptureMemoryHold(capture_mem_hold_.decoder_layer_hidden_states_.slice(0, 0, tokens_count),
                             inputs,
                             is_prefill_cuda_graph_mode_ || num_tokens_per_bs_ > 1);
}

CudaGraphRunner* CudaGraphRunner::createForPrefill(py::object py_instance, GraphParams params) {
    params.enable_cuda_graph = true;
    if (params.num_tokens_per_bs == 0) {
        params.num_tokens_per_bs = params.max_seq_len;
    }
    CudaGraphRunner* runner = new CudaGraphRunner(params, std::move(py_instance));
    runner->initCapture();
    return runner;
}

CudaGraphRunner* CudaGraphRunner::createForDecode(py::object py_instance, GraphParams params) {
    params.enable_cuda_graph = true;
    if (params.num_tokens_per_bs == 0) {
        params.num_tokens_per_bs = 1;
    }
    CudaGraphRunner* runner = new CudaGraphRunner(params, std::move(py_instance));
    runner->initCapture();
    return runner;
}

}  // namespace rtp_llm
