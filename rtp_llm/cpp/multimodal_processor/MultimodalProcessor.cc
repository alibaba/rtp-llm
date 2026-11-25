
#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include <mutex>
#include <optional>
#include <string>
#include <vector>
#include <torch/python.h>
#if USING_CUDA
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#endif
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/multimodal_processor/FeatureHashOp.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalError.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalTokenUtils.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"
#include "rtp_llm/cpp/model_rpc/MultimodalPbConverter.h"

namespace rtp_llm {

namespace {

#if USING_CUDA
c10::cuda::CUDAStream multimodalCudaStream(c10::DeviceIndex device) {
    static std::mutex                                        mutex;
    static std::map<c10::DeviceIndex, c10::cuda::CUDAStream> streams;
    std::lock_guard<std::mutex>                              lock(mutex);
    if (const auto found = streams.find(device); found != streams.end()) {
        return found->second;
    }
    const c10::cuda::CUDAGuard device_guard(device);
    cudaStream_t               raw = nullptr;
    C10_CUDA_CHECK(cudaStreamCreateWithFlags(&raw, cudaStreamNonBlocking));
    try {
        const auto stream = c10::cuda::getStreamFromExternal(raw, device);
        streams.emplace(device, stream);
        // Keep one stream per device for the CUDA context's lifetime, like
        // PyTorch's own pool. GPU allocations retain their allocation stream
        // in the caching allocator even after this request releases them.
        // Never borrow a pool entry that engine/collective workers may use.
        return stream;
    } catch (...) {
        (void)cudaStreamDestroy(raw);
        throw;
    }
}

class ScopedMultimodalCudaStream {
public:
    ScopedMultimodalCudaStream() {
        if (c10::cuda::device_count() == 0) {
            return;
        }
        guard_.emplace(multimodalCudaStream(c10::cuda::current_device()));
    }

    ScopedMultimodalCudaStream(const ScopedMultimodalCudaStream&)            = delete;
    ScopedMultimodalCudaStream& operator=(const ScopedMultimodalCudaStream&) = delete;

private:
    std::optional<c10::cuda::CUDAStreamGuard> guard_;
};
#endif

ErrorInfo pinMultimodalTensors(std::vector<torch::Tensor>& tensors, const char* tensor_name) {
#if USING_CUDA
    try {
        for (auto& tensor : tensors) {
            if (!tensor.defined() || tensor.is_pinned()) {
                continue;
            }
            if (tensor.is_cuda()) {
                c10::cuda::CUDAStreamGuard copy_guard(multimodalCudaStream(tensor.get_device()));
                auto                       options =
                    torch::TensorOptions().dtype(tensor.scalar_type()).device(torch::kCPU).pinned_memory(true);
                // Allocate the pinned destination explicitly. Tensor::to()
                // overrides TensorOptions::pinned_memory with a pin_out value
                // derived from non_blocking; to(pinned_cpu_options, false)
                // therefore allocates pageable output in current PyTorch.
                // Pageable D2H can wait inside the driver while holding its
                // context lock, preventing another thread from submitting the
                // GPU work needed to satisfy that wait.
                auto host = tensor.is_non_overlapping_and_dense() ?
                                torch::empty_strided(tensor.sizes(), tensor.strides(), options) :
                                torch::empty(tensor.sizes(), options);
                // The embedding adapter publishes its producer dependency to
                // this consumer stream; receiving-side materialization and
                // this copy stay in the same isolated stream's program order.
                // Do not add a model-default-stream event here: that stream
                // may be waiting for another TP rank to enqueue its work.
                host.copy_(tensor, /*non_blocking=*/true);
                at::cuda::CUDAEvent copied;
                copied.record(c10::cuda::getCurrentCUDAStream(tensor.get_device()));
                // TP serialization reads CPU data immediately after return.
                // Wait only for the copy's completion, not the device/default
                // stream or any unrelated model work.
                copied.synchronize();
                tensor = std::move(host);
            } else {
                tensor = tensor.pin_memory();
            }
        }
    } catch (const std::exception& e) {
        return ErrorInfo(ErrorCode::MM_PROCESS_ERROR,
                         std::string("failed to move multimodal ") + tensor_name + " to pinned CPU: " + e.what());
    }
#else
    (void)tensors;
    (void)tensor_name;
#endif
    return ErrorInfo::OkStatus();
}

}  // namespace

ErrorResult<ExpandedOutput>
MultimodalProcessor::expandTokenIds(const std::vector<torch::Tensor>&                mm_embedding,
                                    const torch::Tensor&                             token_ids,
                                    torch::Tensor                                    token_type_ids,
                                    const std::optional<std::vector<torch::Tensor>>& feature_hashes) {
    if (mm_embedding.size() == 0) {
        return ExpandedOutput(token_ids, token_type_ids);
    }
    if (semantic_mm_token_id_ < -1 || semantic_mm_token_id_ > std::numeric_limits<int32_t>::max()) {
        return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "semantic multimodal token ID is outside int32 range");
    }

    assert(token_ids.dim() == 1);
    int expanded_len = token_ids.size(0);
    CHECK_AND_RETURN_REF(locs, getMultimodalTags(token_ids));
    torch::Tensor expanded_token_type_ids;
    int           mm_num = mm_embedding.size();
    if (feature_hashes.has_value() && feature_hashes->size() != mm_embedding.size()) {
        return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "multimodal feature hash segment count mismatch");
    }
    if (locs.size() != mm_num) {
        std::stringstream exception_str;
        exception_str << "number of multimodal tags and multimodal input not matched, expect " << locs.size()
                      << ", get " << mm_num;
        return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, exception_str.str());
    }
    for (int i = 0; i < mm_num; i++) {
        expanded_len += mm_embedding[i].sizes()[0] - locs[i].second + locs[i].first;
    }

    auto expanded_ids = torch::empty({(int64_t)expanded_len}, torch::kInt32);
    auto token_masks  = torch::empty({(int64_t)expanded_len}, torch::kInt32);
    auto new_locs     = torch::empty({(int64_t)mm_num}, torch::kInt32);
    memset(expanded_ids.data_ptr(), -1, expanded_ids.nbytes());
    std::fill(token_masks.data_ptr<int32_t>(), token_masks.data_ptr<int32_t>() + token_masks.numel(), 1);
    if (token_type_ids.defined()) {
        expanded_token_type_ids = torch::empty({(int64_t)expanded_len}, torch::kInt32);
        std::fill(expanded_token_type_ids.data_ptr<int32_t>(),
                  expanded_token_type_ids.data_ptr<int32_t>() + expanded_token_type_ids.numel(),
                  0);
    }
    auto cache_spans = semantic_mm_token_id_ >= 0 ? std::make_shared<MultimodalCacheSpans>() : nullptr;
    if (cache_spans) {
        cache_spans->reserve(mm_num);
    }
    int new_loc_idx = 0, old_loc_idx = 0;
    for (int i = 0; i < mm_num; i++) {
        auto& loc      = locs[i];
        int   copy_len = loc.first - old_loc_idx;
        memcpy(expanded_ids.data_ptr<int32_t>() + new_loc_idx,
               token_ids.data_ptr<int32_t>() + old_loc_idx,
               sizeof(int32_t) * copy_len);
        memset(
            token_masks.data_ptr<int32_t>() + new_loc_idx + copy_len, 0, mm_embedding[i].sizes()[0] * sizeof(int32_t));
        if (token_type_ids.defined()) {
            memcpy(expanded_token_type_ids.data_ptr<int32_t>() + new_loc_idx,
                   token_type_ids.data_ptr<int32_t>() + old_loc_idx,
                   sizeof(int32_t) * copy_len);
        }
        *(new_locs.data_ptr<int32_t>() + i) = copy_len + new_loc_idx;

        torch::Tensor hashes;
        try {
            hashes = feature_hashes.has_value() ? feature_hashes->at(i) : getMultimodalFeatureHash(mm_embedding[i]);
        } catch (const std::exception& error) {
            return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, error.what());
        }
        if (!hashes.defined() || !hashes.device().is_cpu() || hashes.scalar_type() != torch::kInt32 || hashes.dim() != 1
            || hashes.numel() != mm_embedding[i].size(0)) {
            return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "invalid multimodal feature hashes");
        }
        hashes = hashes.contiguous();

        const int32_t output_offset = new_loc_idx + copy_len;
        auto*         target        = expanded_ids.data_ptr<int32_t>() + output_offset;
        if (cache_spans) {
            std::fill(target, target + hashes.numel(), static_cast<int32_t>(semantic_mm_token_id_));
            const auto* begin = hashes.data_ptr<int32_t>();
            cache_spans->push_back({output_offset, std::vector<int32_t>(begin, begin + hashes.numel())});
        } else {
            memcpy(target, hashes.data_ptr<int32_t>(), hashes.nbytes());
        }

        new_loc_idx += copy_len + mm_embedding[i].sizes()[0];
        old_loc_idx = loc.second;
    }
    if (expanded_ids.size(0) - new_loc_idx != token_ids.size(0) - old_loc_idx) {
        return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "expanded length calculate error");
    }
    memcpy(expanded_ids.data_ptr<int32_t>() + new_loc_idx,
           token_ids.data_ptr<int32_t>() + old_loc_idx,
           sizeof(int32_t) * (expanded_ids.size(0) - new_loc_idx));
    if (token_type_ids.defined()) {
        memcpy(expanded_token_type_ids.data_ptr<int32_t>() + new_loc_idx,
               token_type_ids.data_ptr<int32_t>() + old_loc_idx,
               sizeof(int32_t) * (expanded_ids.size(0) - new_loc_idx));
    }
    return ExpandedOutput(std::move(expanded_ids),
                          std::move(expanded_token_type_ids),
                          std::move(token_masks),
                          std::move(new_locs),
                          std::move(cache_spans));
}

ErrorResult<std::vector<std::pair<int32_t, int32_t>>>
MultimodalProcessor::getMultimodalTags(const torch::Tensor& token_ids) {
    try {
        const auto* data = token_ids.data_ptr<int32_t>();
        return getMultimodalTokenSpans({data, data + token_ids.numel()}, sep_token_ids_, include_sep_tokens_);
    } catch (const std::exception& error) {
        return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, error.what());
    }
}

ErrorInfo MultimodalProcessor::checkExpandLength(const ExpandedOutput& expand_output) {
    if (expand_output.expanded_ids.numel() >= max_seq_len_) {
        std::stringstream exception_str;
        exception_str << "input after multimodal process is " << expand_output.expanded_ids.numel() << " > max_seq_len("
                      << max_seq_len_ << ")";
        return ErrorInfo(ErrorCode::MM_LONG_PROMPT_ERROR, exception_str.str());
    }
    return ErrorInfo::OkStatus();
}

namespace {

std::string vitRoleAddress(const GenerateInput& input) {
    if (input.generate_config) {
        for (const auto& role_addr : input.generate_config->role_addrs) {
            if (role_addr.role == RoleType::VIT) {
                return role_addr.ip + ":" + std::to_string(role_addr.grpc_port);
            }
        }
    }
    return {};
}

}  // namespace

ErrorResult<MultimodalOutput> MultimodalProcessor::V41MultimodalEmbedding(
    const V41RequestInputs& inputs, const std::string&, int64_t timeout_ms, int64_t, grpc::ServerContext*) {
    py::gil_scoped_acquire acquire;
    if (mm_process_engine_.is_none()) {
        return ErrorInfo(ErrorCode::MM_EMPTY_ENGINE_ERROR, "V4.1 prepared images require a local ViT engine");
    }
    try {
        py::list images;
        for (const auto& image : inputs.images) {
            py::dict record;
            record["start"]              = image.start;
            record["n_vit_h"]            = image.n_vit_h;
            record["n_vit_w"]            = image.n_vit_w;
            record["patches"]            = image.patches;
            record["types"]              = image.types;
            record["content_sha256"]     = image.content_sha256;
            record["processor_identity"] = image.processor_identity;
            images.append(std::move(record));
        }
        auto             result = mm_process_engine_.attr("submit_v41")(images, timeout_ms);
        MultimodalOutput output;
        for (auto item : result.attr("embeddings")) {
            output.mm_features.push_back(py::cast<torch::Tensor>(item));
        }
        return output;
    } catch (const py::error_already_set& error) {
        return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, error.what());
    }
}

ErrorResult<ExpandedOutput>
MultimodalProcessor::useExpandedTokenLayout(const GenerateInput&                             input,
                                            const std::vector<torch::Tensor>&                features,
                                            const std::optional<std::vector<torch::Tensor>>& feature_hashes) {
    const auto& layout = input.multimodal_token_layout.value();
    const auto& ids    = input.input_ids;
    if (layout.spans.empty() || layout.spans.size() != features.size()
        || (feature_hashes.has_value() && feature_hashes->size() != features.size())) {
        return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "invalid frontend multimodal token layout");
    }
    if (!ids.defined() || !ids.device().is_cpu() || ids.scalar_type() != torch::kInt32 || ids.dim() != 1
        || !ids.is_contiguous()) {
        return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "invalid expanded multimodal token ids");
    }
    RETURN_IF_STATUS_ERROR(checkExpandLength(ExpandedOutput(ids)));
    int64_t                    previous_end   = 0;
    bool                       hashes_changed = false;
    std::vector<torch::Tensor> verified_hashes;
    verified_hashes.reserve(features.size());
    for (size_t i = 0; i < layout.spans.size(); ++i) {
        const auto [offset, length] = layout.spans[i];
        const int64_t end           = int64_t(offset) + length;
        if (offset < previous_end || length <= 0 || end > ids.numel() || !features[i].defined() || features[i].dim() < 1
            || features[i].size(0) != length) {
            return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "frontend multimodal spans do not match ViT output");
        }
        previous_end = end;
        // Metadata can outlive an embedding. Check the fetched result before
        // using frontend hashes as KV-cache keys; never publish mismatched data.
        torch::Tensor hashes;
        try {
            hashes = feature_hashes.has_value() ? feature_hashes->at(i) : getMultimodalFeatureHash(features[i]);
        } catch (const std::exception& error) {
            return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, error.what());
        }
        if (!hashes.defined() || !hashes.device().is_cpu() || hashes.scalar_type() != torch::kInt32 || hashes.dim() != 1
            || hashes.numel() != length) {
            return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "invalid ViT feature hashes for expanded input");
        }
        hashes               = hashes.contiguous();
        const auto* expected = ids.data_ptr<int32_t>() + offset;
        if (semantic_mm_token_id_ >= 0) {
            if (!input.multimodal_cache_spans || input.multimodal_cache_spans->size() != layout.spans.size())
                return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "missing frontend multimodal row hash sidecar");
            const auto& cache_span = input.multimodal_cache_spans->at(i);
            if (cache_span.offset != offset || cache_span.row_hashes.size() != static_cast<size_t>(length)
                || !std::all_of(
                    expected, expected + length, [this](int32_t id) { return id == semantic_mm_token_id_; }))
                return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "invalid semantic multimodal layout");
            expected = cache_span.row_hashes.data();
        }
        if (memcmp(expected, hashes.data_ptr<int32_t>(), hashes.nbytes()) != 0) {
            hashes_changed = true;
        }
        verified_hashes.push_back(std::move(hashes));
    }
    if (hashes_changed && input.generate_config && input.generate_config->pd_separation) {
        return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, "Prefill ViT feature hashes differ from the PD handoff tokens");
    }
    // Recomputed embeddings can differ after cache eviction. Preserve all text
    // and offsets, but make actual KV-cache keys describe the fetched values.
    // Validate every segment before making changes. Matching hashes reuse the
    // input storage; changed hashes copy it without rescanning or splicing.
    auto cache_spans  = semantic_mm_token_id_ >= 0 ? std::make_shared<MultimodalCacheSpans>() : nullptr;
    auto expanded_ids = hashes_changed && !cache_spans ? ids.clone() : ids;
    auto mask         = torch::ones({ids.numel()}, torch::kInt32);
    auto locs         = torch::empty({int64_t(layout.spans.size())}, torch::kInt32);
    for (size_t i = 0; i < layout.spans.size(); ++i) {
        const auto [offset, length] = layout.spans[i];
        locs.data_ptr<int32_t>()[i] = offset;
        std::fill(mask.data_ptr<int32_t>() + offset, mask.data_ptr<int32_t>() + offset + length, 0);
        if (cache_spans) {
            const auto* rows = verified_hashes[i].data_ptr<int32_t>();
            cache_spans->push_back({offset, std::vector<int32_t>(rows, rows + length)});
        } else if (hashes_changed) {
            memcpy(expanded_ids.data_ptr<int32_t>() + offset,
                   verified_hashes[i].data_ptr<int32_t>(),
                   verified_hashes[i].nbytes());
        }
    }
    return ExpandedOutput(std::move(expanded_ids), {}, std::move(mask), std::move(locs), std::move(cache_spans));
}

ErrorInfo MultimodalProcessor::updateMultimodalFeatures(std::shared_ptr<rtp_llm::GenerateInput>& input,
                                                        grpc::ServerContext*                     server_context) {
    if (input->generate_config && input->generate_config->calculate_loss) {
        return ErrorInfo(ErrorCode::MM_NOT_SUPPORTED_ERROR, "cannot calculate loss in multimodal query");
    }
    auto ip_port = vitRoleAddress(*input);
    if (input->v41_inputs) {
        const auto& prepared   = *input->v41_inputs;
        const auto  cpu_vector = [](const torch::Tensor& tensor, torch::ScalarType dtype) {
            return tensor.defined() && tensor.device().is_cpu() && tensor.dim() == 1 && tensor.scalar_type() == dtype;
        };
        if (!cpu_vector(input->input_ids, torch::kInt32) || !cpu_vector(prepared.token_types, torch::kInt32)
            || !cpu_vector(prepared.image_mask, torch::kBool)
            || prepared.token_types.numel() != input->input_ids.numel()
            || prepared.image_mask.numel() != input->input_ids.numel()) {
            return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "V4.1 token metadata must match the input token vector");
        }
        const int64_t token_count = input->input_ids.numel();
        if (token_count >= max_seq_len_) {
            return ErrorInfo(ErrorCode::MM_LONG_PROMPT_ERROR, "V4.1 prepared input exceeds the model sequence length");
        }
        auto        token_types = prepared.token_types.contiguous();
        auto        image_mask  = prepared.image_mask.contiguous();
        const auto* kinds       = token_types.data_ptr<int32_t>();
        const auto* mask        = image_mask.data_ptr<bool>();
        for (int64_t i = 0; i < token_count; ++i) {
            if (kinds[i] < -1 || kinds[i] > 3 || mask[i] != (kinds[i] != -1)) {
                return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "Invalid V4.1 token type or image mask");
            }
        }
        int64_t previous_end = 0;
        for (const auto& image : prepared.images) {
            if (!cpu_vector(image.types, torch::kInt32) || image.types.numel() == 0 || image.start < previous_end
                || image.types.numel() > token_count || image.start > token_count - image.types.numel()) {
                return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "Invalid V4.1 image span");
            }
            for (int64_t i = previous_end; i < image.start; ++i) {
                if (kinds[i] != -1) {
                    return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "V4.1 image spans do not cover token metadata");
                }
            }
            auto        image_types = image.types.contiguous();
            const auto* image_kinds = image_types.data_ptr<int32_t>();
            for (int64_t i = 0; i < image_types.numel(); ++i) {
                if (image_kinds[i] < 0 || image_kinds[i] != kinds[image.start + i]) {
                    return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "V4.1 image types do not match token metadata");
                }
            }
            previous_end = image.start + image_types.numel();
        }
        for (int64_t i = previous_end; i < token_count; ++i) {
            if (kinds[i] != -1) {
                return ErrorInfo(ErrorCode::MM_WRONG_FORMAT_ERROR, "V4.1 image spans do not cover token metadata");
            }
        }
        std::vector<torch::Tensor> features;
        if (!prepared.images.empty()) {
            int64_t timeout_ms = 0;
            if (input->generate_config && input->generate_config->timeout_ms > 0) {
                const auto elapsed_ms = input->begin_time_us > 0 ? (currentTimeUs() - input->begin_time_us) / 1000 : 0;
                timeout_ms            = input->generate_config->timeout_ms - elapsed_ms;
                if (timeout_ms <= 0) {
                    return ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "V4.1 ViT request deadline exceeded");
                }
            }
            CHECK_AND_RETURN_REF(
                result, V41MultimodalEmbedding(prepared, ip_port, timeout_ms, input->request_id, server_context));
            features = std::move(result.mm_features);
        }
        if (features.size() != prepared.images.size()) {
            return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, "V4.1 ViT returned an unexpected image count");
        }
        auto locs = torch::empty({static_cast<int64_t>(features.size())}, torch::kInt32);
        for (size_t index = 0; index < features.size(); ++index) {
            if (features[index].dim() != 2 || features[index].size(0) != prepared.images[index].types.numel()) {
                return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, "V4.1 ViT returned an invalid image span");
            }
            locs.data_ptr<int32_t>()[index] = prepared.images[index].start;
        }
        input->multimodal_features = std::move(features);
        input->text_tokens_mask    = prepared.image_mask.logical_not().to(torch::kInt32);
        input->mm_locs             = std::move(locs);
        input->mm_position_ids.reset();
        return ErrorInfo::OkStatus();
    }
    if (input->multimodal_request) {
        auto result = updateMultimodalFeatures(*input, *input->multimodal_request, server_context);
        // A retry still needs its media on error. On success only the types, tensors and hashes are needed.
        if (result.ok())
            input->multimodal_request.reset();
        return result;
    }
    if (!input->multimodal_inputs || input->multimodal_inputs->empty())
        return ErrorInfo::OkStatus();
#if USING_CUDA
    ScopedMultimodalCudaStream mm_stream;
#endif
    CHECK_AND_RETURN_REF(embedding,
                         MultimodalEmbedding(*input->multimodal_inputs, ip_port, input->request_id, server_context));
    return setGenerateMultimodalFeatures(*input, embedding);
}

ErrorResult<MultimodalOutput> MultimodalProcessor::MultimodalEmbeddingFromRpc(const MultimodalInputsPB& inputs,
                                                                              const std::string&        ip_port,
                                                                              int64_t                   request_id,
                                                                              grpc::ServerContext*      context) {
    std::optional<bool> inspect;
    if (inputs.has_input_inspection_enabled()) {
        inspect = inputs.input_inspection_enabled().value();
    }
    return MultimodalEmbedding(
        MultimodalPbConverter::mediaReferencesFromPb(inputs), ip_port, request_id, context, inspect);
}

ErrorInfo MultimodalProcessor::updateMultimodalFeatures(GenerateInput& input) {
    if (input.generate_config && input.generate_config->calculate_loss) {
        return ErrorInfo(ErrorCode::MM_NOT_SUPPORTED_ERROR, "cannot calculate loss in multimodal query");
    }
    if (!input.multimodal_inputs || input.multimodal_inputs->empty()) {
        return ErrorInfo::OkStatus();
    }
#if USING_CUDA
    ScopedMultimodalCudaStream mm_stream;
#endif
    CHECK_AND_RETURN_REF(
        mm_embedding_res,
        MultimodalEmbedding(*input.multimodal_inputs, vitRoleAddress(input), input.request_id, nullptr));
    return setGenerateMultimodalFeatures(input, mm_embedding_res);
}

ErrorInfo MultimodalProcessor::updateMultimodalFeatures(GenerateInput&            input,
                                                        const MultimodalInputsPB& mm_inputs,
                                                        grpc::ServerContext*      server_context) {
    if (input.generate_config && input.generate_config->calculate_loss) {
        return ErrorInfo(ErrorCode::MM_NOT_SUPPORTED_ERROR, "cannot calculate loss in multimodal query");
    }
    if (mm_inputs.multimodal_inputs().empty()) {
        return ErrorInfo::OkStatus();
    }
#if USING_CUDA
    // The receiving stream covers RDMA, feature hashing and the final D2H copy.
    ScopedMultimodalCudaStream mm_stream;
#endif
    CHECK_AND_RETURN_REF(
        mm_embedding_res,
        MultimodalEmbeddingFromRpc(mm_inputs, vitRoleAddress(input), input.request_id, server_context));
    return setGenerateMultimodalFeatures(input, mm_embedding_res);
}

ErrorInfo MultimodalProcessor::setGenerateMultimodalFeatures(GenerateInput& input, MultimodalOutput& mm_embedding_res) {
    auto                       mm_features = std::move(mm_embedding_res.mm_features);
    std::vector<torch::Tensor> mm_extra_input;
    if (mm_embedding_res.mm_extra_input.has_value()) {
        mm_extra_input = std::move(mm_embedding_res.mm_extra_input.value());
    }
    CHECK_AND_RETURN_REF(expanded_ids,
                         input.multimodal_token_layout ?
                             useExpandedTokenLayout(input, mm_features, mm_embedding_res.mm_feature_hashes) :
                             expandTokenIds(mm_features, input.input_ids, {}, mm_embedding_res.mm_feature_hashes));
    RETURN_IF_STATUS_ERROR(checkExpandLength(expanded_ids));
    RETURN_IF_STATUS_ERROR(pinMultimodalTensors(mm_features, "embedding"));
    RETURN_IF_STATUS_ERROR(pinMultimodalTensors(mm_extra_input, "extra input"));
    input.multimodal_features    = std::move(mm_features);
    input.mm_position_ids        = std::move(mm_embedding_res.mm_position_ids);
    input.mm_extra_input         = std::move(mm_extra_input);
    input.input_ids              = expanded_ids.expanded_ids;
    input.text_tokens_mask       = expanded_ids.text_tokens_mask;
    input.mm_locs                = expanded_ids.locs;
    input.multimodal_cache_spans = std::move(expanded_ids.cache_spans);
    return ErrorInfo::OkStatus();
}

ErrorInfo MultimodalProcessor::updateMultimodalFeatures(std::shared_ptr<rtp_llm::EmbeddingInput>&    input,
                                                        const std::vector<rtp_llm::MultimodalInput>& mm_inputs,
                                                        const std::string&                           vit_role_addr,
                                                        grpc::ServerContext*                         server_context) {
    CHECK_AND_RETURN_REF(mm_embedding_res,
                         MultimodalEmbedding(mm_inputs, vit_role_addr, input->request_id, server_context));
    MultimodalFeature mm_features;
    mm_features.features = std::move(mm_embedding_res.mm_features);
    CHECK_AND_RETURN_REF(
        expanded_ids,
        expandTokenIds(
            mm_features.features, input->token_ids, input->token_type_ids, mm_embedding_res.mm_feature_hashes));
    mm_features.expanded_ids     = expanded_ids.expanded_ids;
    mm_features.text_tokens_mask = expanded_ids.text_tokens_mask;
    mm_features.locs             = expanded_ids.locs;
    input->multimodal_features.emplace(mm_features);
    input->token_ids      = expanded_ids.expanded_ids;
    input->token_type_ids = expanded_ids.token_type_ids;
    if (input->input_lengths.numel() == 1 && expanded_ids.expanded_ids.defined()) {
        input->input_lengths.data_ptr<int32_t>()[0] = expanded_ids.expanded_ids.size(0);
        input->total_length                         = expanded_ids.expanded_ids.size(0);
    }
    return ErrorInfo::OkStatus();
}

ErrorResult<MultimodalFeature>
MultimodalProcessor::getMultimodalFeatures(const torch::Tensor&                         input_ids,
                                           const std::vector<rtp_llm::MultimodalInput>& mm_inputs) {
    MultimodalFeature mm_features;
    CHECK_AND_RETURN_REF(mm_embedding_res, MultimodalEmbedding(mm_inputs, {}, 0, nullptr));
    mm_features.features = std::move(mm_embedding_res.mm_features);
    CHECK_AND_RETURN_REF(expanded_ids,
                         expandTokenIds(mm_features.features, input_ids, {}, mm_embedding_res.mm_feature_hashes));
    mm_features.expanded_ids     = expanded_ids.expanded_ids;
    mm_features.text_tokens_mask = expanded_ids.text_tokens_mask;
    mm_features.locs             = expanded_ids.locs;
    return mm_features;
}

}  // namespace rtp_llm
