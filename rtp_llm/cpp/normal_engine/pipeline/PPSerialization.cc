#include "rtp_llm/cpp/normal_engine/pipeline/PPSerialization.h"

#include <cstring>
#include <limits>
#include <utility>

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm::pp_serialization {

namespace {

/** Payload fields have a fixed positional layout, so PP peers must use the same wire version. */
constexpr uint32_t kVersion = 1;

/** Stable wire values decouple PP messages from libtorch's internal ScalarType numbering. */
enum class WireDType : uint8_t {
    BOOL     = 1,
    INT32    = 2,
    INT64    = 3,
    FLOAT16  = 4,
    BFLOAT16 = 5,
    FLOAT32  = 6,
};

WireDType toWireDType(torch::ScalarType dtype) {
    switch (dtype) {
        case torch::kBool:
            return WireDType::BOOL;
        case torch::kInt32:
            return WireDType::INT32;
        case torch::kInt64:
            return WireDType::INT64;
        case torch::kFloat16:
            return WireDType::FLOAT16;
        case torch::kBFloat16:
            return WireDType::BFLOAT16;
        case torch::kFloat32:
            return WireDType::FLOAT32;
        default:
            RTP_LLM_FAIL("PP serialization does not support torch dtype=%d", static_cast<int>(dtype));
    }
}

torch::ScalarType fromWireDType(uint8_t raw_dtype) {
    switch (static_cast<WireDType>(raw_dtype)) {
        case WireDType::BOOL:
            return torch::kBool;
        case WireDType::INT32:
            return torch::kInt32;
        case WireDType::INT64:
            return torch::kInt64;
        case WireDType::FLOAT16:
            return torch::kFloat16;
        case WireDType::BFLOAT16:
            return torch::kBFloat16;
        case WireDType::FLOAT32:
            return torch::kFloat32;
        default:
            RTP_LLM_FAIL("PP serialization wire dtype=%u is invalid", static_cast<unsigned>(raw_dtype));
    }
}

size_t checkedTensorNumel(const std::vector<int64_t>& sizes) {
    size_t numel = 1;
    for (const auto dim : sizes) {
        RTP_LLM_CHECK_WITH_INFO(dim >= 0, "PP serialization tensor has negative dim=%ld", dim);
        const auto size = static_cast<size_t>(dim);
        RTP_LLM_CHECK_WITH_INFO(size == 0 || numel <= std::numeric_limits<size_t>::max() / size,
                                "PP serialization tensor numel overflow");
        numel *= size;
    }
    return numel;
}

size_t checkedTensorBytes(size_t numel, torch::ScalarType dtype) {
    const auto element_size = torch::elementSize(dtype);
    RTP_LLM_CHECK_WITH_INFO(numel <= std::numeric_limits<size_t>::max() / element_size,
                            "PP serialization tensor byte size overflow");
    return numel * element_size;
}

struct ByteEncoder {
    enum class Mode {
        CALCULATE_SIZE,
        WRITE,
    };

    explicit ByteEncoder(Mode mode, size_t size = 0): mode_(mode), capacity_(size) {
        if (mode_ == Mode::CALCULATE_SIZE) {
            return;
        }
        RTP_LLM_CHECK_WITH_INFO(size <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                                "PP serialization payload size=%zu exceeds torch int64 range",
                                size);
        output_ = torch::empty({static_cast<int64_t>(size)}, torch::TensorOptions().dtype(torch::kUInt8));
        data_   = static_cast<uint8_t*>(output_.data_ptr());
    }

    void raw(const void* p, size_t n) {
        RTP_LLM_CHECK_WITH_INFO(n <= std::numeric_limits<size_t>::max() - offset_,
                                "PP serialization payload size overflow");
        if (mode_ == Mode::WRITE) {
            RTP_LLM_CHECK_WITH_INFO(offset_ <= capacity_ && n <= capacity_ - offset_,
                                    "PP serialization size changed between sizing and writing");
            if (n > 0) {
                std::memcpy(data_ + offset_, p, n);
            }
        }
        offset_ += n;
    }

    template<typename T>
    void val(T v) {
        raw(&v, sizeof(T));
    }

    template<typename WireType, typename T>
    void scalar(const T& value) {
        val<WireType>(static_cast<WireType>(value));
    }

    template<typename T, typename Fields>
    void sequence(const std::vector<T>& values, Fields fields) {
        val<uint64_t>(values.size());
        for (const auto& value : values) {
            fields(*this, value);
        }
    }

    template<typename T, typename Fields>
    void optional(const std::optional<T>& value, Fields fields) {
        flag(value.has_value());
        if (value.has_value()) {
            fields(*this, *value);
        }
    }

    void flag(bool v) {
        val<uint8_t>(v ? 1 : 0);
    }

    void str(const std::string& s) {
        val<uint64_t>(s.size());
        raw(s.data(), s.size());
    }

    void strings(const std::vector<std::string>& values) {
        sequence(values, [](auto& io, auto& value) { io.str(value); });
    }

    void errors(const std::vector<ErrorInfo>& errors) {
        val<uint64_t>(errors.size());
        for (const auto& error : errors) {
            val<int32_t>(static_cast<int32_t>(error.code()));
            if (error.hasError()) {
                str(error.ToString());
            }
        }
    }

    void tensor(const torch::Tensor& t) {
        /** Encode definedness separately so empty tensors retain their dtype and shape. */
        flag(t.defined());
        if (!t.defined()) {
            return;
        }
        const auto wire_dtype = toWireDType(t.scalar_type());
        val<uint64_t>(static_cast<uint64_t>(t.dim()));
        for (int64_t d = 0; d < t.dim(); ++d) {
            val<int64_t>(t.size(d));
        }
        val<uint8_t>(static_cast<uint8_t>(wire_dtype));
        const auto nbytes = t.nbytes();
        val<uint64_t>(static_cast<uint64_t>(nbytes));
        if (nbytes > 0) {
            if (mode_ == Mode::WRITE) {
                const auto cpu = t.contiguous().cpu();
                raw(cpu.data_ptr(), cpu.nbytes());
            } else {
                raw(nullptr, nbytes);
            }
        }
    }

    size_t size() const {
        return offset_;
    }

    torch::Tensor finish() {
        RTP_LLM_CHECK_WITH_INFO(output_.defined() && offset_ == capacity_,
                                "PP serialization wrote size=%zu, expected=%zu",
                                offset_,
                                capacity_);
        return std::move(output_);
    }

private:
    Mode          mode_;
    torch::Tensor output_;
    uint8_t*      data_     = nullptr;
    size_t        capacity_ = 0;
    size_t        offset_   = 0;
};

template<typename Fields>
torch::Tensor encodeToTensor(Fields fields) {
    ByteEncoder sizer(ByteEncoder::Mode::CALCULATE_SIZE);
    fields(sizer);
    ByteEncoder writer(ByteEncoder::Mode::WRITE, sizer.size());
    fields(writer);
    return writer.finish();
}

struct ByteReader {
    explicit ByteReader(const torch::Tensor& buffer) {
        RTP_LLM_CHECK_WITH_INFO(buffer.defined() && buffer.device().is_cpu() && buffer.dtype() == torch::kUInt8
                                    && buffer.is_contiguous(),
                                "PP serialization payload must be a contiguous CPU uint8 tensor");
        cursor_     = static_cast<const uint8_t*>(buffer.data_ptr());
        buffer_end_ = cursor_ + static_cast<size_t>(buffer.numel());
    }

    void raw(void* out, size_t n) {
        RTP_LLM_CHECK_WITH_INFO(static_cast<size_t>(buffer_end_ - cursor_) >= n,
                                "PP serialization payload truncated");
        std::memcpy(out, cursor_, n);
        cursor_ += n;
    }

    template<typename T>
    T val() {
        T v;
        raw(&v, sizeof(T));
        return v;
    }

    template<typename WireType, typename T>
    void scalar(T& value) {
        value = static_cast<T>(val<WireType>());
    }

    template<typename T, typename Fields>
    void sequence(std::vector<T>& values, Fields fields) {
        values.resize(count());
        for (auto& value : values) {
            fields(*this, value);
        }
    }

    template<typename T, typename Fields>
    void optional(std::optional<T>& value, Fields fields) {
        if (flag()) {
            value.emplace();
            fields(*this, *value);
        } else {
            value.reset();
        }
    }

    bool flag() {
        const auto value = val<uint8_t>();
        RTP_LLM_CHECK_WITH_INFO(value <= 1,
                                "PP serialization boolean=%u is invalid",
                                static_cast<unsigned>(value));
        return value != 0;
    }

    void flag(bool& value) {
        value = flag();
    }

    std::string str() {
        const auto n = val<uint64_t>();
        RTP_LLM_CHECK_WITH_INFO(remaining() >= n, "PP serialization string truncated");
        std::string s(reinterpret_cast<const char*>(cursor_), static_cast<size_t>(n));
        cursor_ += static_cast<size_t>(n);
        return s;
    }

    void str(std::string& value) {
        value = str();
    }

    void strings(std::vector<std::string>& values) {
        values.resize(count());
        for (auto& value : values) {
            str(value);
        }
    }

    void errors(std::vector<ErrorInfo>& errors) {
        errors.resize(count());
        for (auto& error : errors) {
            const auto raw_code = val<int32_t>();
            const auto code = static_cast<ErrorCode>(raw_code);
            if (code != ErrorCode::NONE_ERROR) {
                error = ErrorInfo(code, str());
            }
        }
    }

    void tensor(torch::Tensor& out) {
        if (!flag()) {
            out = torch::Tensor();
            return;
        }
        const auto ndim = count();
        std::vector<int64_t> sizes(ndim);
        for (size_t d = 0; d < ndim; ++d) {
            sizes[d] = val<int64_t>();
        }
        const auto dtype           = fromWireDType(val<uint8_t>());
        const auto wire_nbytes     = val<uint64_t>();
        const auto expected_nbytes = checkedTensorBytes(checkedTensorNumel(sizes), dtype);
        RTP_LLM_CHECK_WITH_INFO(wire_nbytes == expected_nbytes,
                                "PP serialization tensor byte count mismatch: wire=%llu expected=%zu",
                                static_cast<unsigned long long>(wire_nbytes),
                                expected_nbytes);
        RTP_LLM_CHECK_WITH_INFO(remaining() >= expected_nbytes, "PP serialization tensor payload truncated");
        /** Plan execution uses fused H2D copies that require pinned source memory. */
        out = torch::empty(sizes, torch::TensorOptions().dtype(dtype).device(torch::kCPU).pinned_memory(true));
        if (expected_nbytes > 0) {
            raw(out.data_ptr(), expected_nbytes);
        }
    }

    size_t remaining() const {
        return static_cast<size_t>(buffer_end_ - cursor_);
    }

    size_t count() {
        const auto value = val<uint64_t>();
        RTP_LLM_CHECK_WITH_INFO(value <= remaining(),
                                "PP serialization count=%llu exceeds remaining payload=%zu",
                                static_cast<unsigned long long>(value),
                                remaining());
        return static_cast<size_t>(value);
    }

    void expectEnd() const {
        RTP_LLM_CHECK_WITH_INFO(cursor_ == buffer_end_,
                                "PP serialization payload has %zu trailing bytes",
                                remaining());
    }

private:
    const uint8_t* cursor_     = nullptr;
    const uint8_t* buffer_end_ = nullptr;
};

template<typename IO, typename List>
void optionalTensorListFields(IO& io, List& list) {
    io.optional(list, [](auto& io, auto& tensors) {
        io.sequence(tensors, [](auto& io, auto& tensor) { io.tensor(tensor); });
    });
}

/** One ordered field list serves const input on write and mutable output on read.
 * Shutdown stays in the plan header so metadata-only plans carry it too. */
template<typename IO, typename Input>
void modelInputFields(IO& io, Input& in) {
    io.tensor(in.combo_tokens);
    io.tensor(in.engram_token_windows);
    io.tensor(in.input_lengths);
    io.tensor(in.sequence_lengths);
    io.tensor(in.lm_output_indexes);
    io.tensor(in.custom_output_indexes);
    io.tensor(in.lm_output_lengths);
    io.tensor(in.prefix_lengths);
    io.tensor(in.sequence_lengths_plus_1);
    io.tensor(in.combo_tokens_type_ids);
    io.tensor(in.combo_position_ids);
    io.tensor(in.last_hidden_states);
    io.tensor(in.attention_mask);
    io.tensor(in.kv_cache_block_id);
    io.tensor(in.kv_cache_kernel_block_id);
    io.strings(in.kv_cache_group_tags);
    io.tensor(in.kv_cache_group_types);
    io.tensor(in.kv_cache_update_mapping);
    io.tensor(in.text_tokens_mask);
    io.tensor(in.mm_features_locs);
    io.tensor(in.mm_features_spans);
    io.tensor(in.input_embeddings_locs);
    io.tensor(in.request_id);
    io.tensor(in.request_pd_separation);
    io.tensor(in.cache_keys);
    optionalTensorListFields(io, in.multimodal_features);
    optionalTensorListFields(io, in.mm_extra_input);
    optionalTensorListFields(io, in.input_embeddings);
    io.template scalar<uint64_t>(in.kv_block_stride_bytes);
    io.template scalar<uint64_t>(in.kv_scale_stride_bytes);
    io.template scalar<uint64_t>(in.seq_size_per_block);
    io.template scalar<uint64_t>(in.kernel_seq_size_per_block);
    io.flag(in.pd_separation);
    io.flag(in.decode_entrance);
    io.flag(in.use_opaque_kv_cache_store);
    io.flag(in.need_all_logits);
    io.flag(in.skip_lm_head);
    io.flag(in.need_all_hidden_states);
    io.flag(in.need_moe_gating);
    io.flag(in.warmup);
    io.flag(in.skip_run);
    io.flag(in.is_fake_stream);
    io.flag(in.is_target_verify);
}

template<typename IO, typename Config>
void logitsProcessorConfigFields(IO& io, Config& cfg) {
    io.str(cfg.grammar_type);
    io.str(cfg.grammar_value);
    io.template scalar<int32_t>(cfg.combo_token_size);
    io.sequence(cfg.banned_combo_token_ids, [](auto& io, auto& banned) {
        io.sequence(banned, [](auto& io, auto& id) { io.template scalar<int32_t>(id); });
    });
    io.sequence(cfg.end_think_token_ids, [](auto& io, auto& id) { io.template scalar<int32_t>(id); });
    io.flag(cfg.enable_cross_sequence_ban);
    io.template scalar<int32_t>(cfg.cross_seq_diverge_start_combo);
}

template<typename IO, typename Sampling>
void samplingPlanFields(IO& io, Sampling& s) {
    io.sequence(s.random_seeds, [](auto& io, auto& seed) {
        io.optional(seed, [](auto& io, auto& value) { io.template scalar<int64_t>(value); });
    });
    io.sequence(s.logits_processor_configs, [](auto& io, auto& cfg) { logitsProcessorConfigFields(io, cfg); });
    io.sequence(s.num_return_sequences, [](auto& io, auto& count) { io.template scalar<int32_t>(count); });
    io.tensor(s.request_ids);
    io.tensor(s.token_ids);
    io.tensor(s.token_offsets);
    io.tensor(s.input_lengths);
    io.tensor(s.sequence_lengths);
    io.tensor(s.max_tokens);
    io.tensor(s.top_k);
    io.tensor(s.top_p);
    io.tensor(s.temperature);
    io.tensor(s.repetition_penalty);
    io.tensor(s.presence_penalty);
    io.tensor(s.frequency_penalty);
    io.tensor(s.no_repeat_ngram_size);
    io.tensor(s.do_sample);
    io.tensor(s.finished_mask);
    io.tensor(s.spec_do_sample);
    io.tensor(s.force_sp_accept);
}

template<typename IO, typename Request>
void promptLogitsRequestFields(IO& io, Request& request) {
    io.flag(request.enabled);
    io.template scalar<int32_t>(request.top_k);
    io.template scalar<int32_t>(request.start);
    io.template scalar<int32_t>(request.end);
    io.flag(request.return_target_logprob);
}

template<typename IO, typename Config>
void requestOutputConfigFields(IO& io, Config& config) {
    io.flag(config.return_logits);
    io.flag(config.return_softmax_probs);
    io.flag(config.return_cum_log_probs);
    io.flag(config.calculate_loss);
    io.flag(config.return_hidden_states);
    io.flag(config.return_all_hidden_states);
    io.flag(config.return_all_probs);
    promptLogitsRequestFields(io, config.prompt_logits);
}

template<typename IO, typename Config>
void outputConfigFields(IO& io, Config& output_config) {
    io.flag(output_config.return_logits);
    io.flag(output_config.return_softmax_probs);
    io.flag(output_config.return_cum_log_probs);
    io.flag(output_config.calculate_loss);
    io.flag(output_config.return_hidden_states);
    io.flag(output_config.return_all_hidden_states);
    io.template scalar<int32_t>(output_config.return_all_probs);
    io.sequence(output_config.requests, [](auto& io, auto& request) { requestOutputConfigFields(io, request); });
}

template<typename IO, typename Output>
void promptLogitsOutputFields(IO& io, Output& output) {
    io.tensor(output.topk_logprobs);
    io.tensor(output.topk_token_ids);
    io.tensor(output.target_logprobs);
    io.template scalar<int32_t>(output.start_pos);
    io.template scalar<int32_t>(output.end_pos);
}

template<typename IO, typename Output>
void requestOutputFields(IO& io, Output& output) {
    io.tensor(output.hidden_states);
    io.tensor(output.logits);
    io.tensor(output.softmax_probs);
    io.tensor(output.cum_log_probs);
    io.tensor(output.all_probs);
    io.tensor(output.loss);
    io.tensor(output.all_hidden_states);
    io.optional(output.prompt_logits,
                [](auto& io, auto& prompt_logits) { promptLogitsOutputFields(io, prompt_logits); });
}

template<typename IO, typename Result>
void executionResultFields(IO& io, Result& result) {
    io.template scalar<int64_t>(result.sample_input_us);
    io.tensor(result.request_ids);
    io.tensor(result.new_token_ids);
    io.tensor(result.new_token_lengths);
    io.tensor(result.propose_token_ids);
    io.tensor(result.propose_all_probs);
    io.tensor(result.propose_hidden_states);
    io.sequence(result.request_outputs, [](auto& io, auto& output) { requestOutputFields(io, output); });
    io.errors(result.request_errors);
}

}  // namespace

torch::Tensor serializePlan(const PPExecutionPlan& plan, bool metadata_only) {
    return encodeToTensor([&](ByteEncoder& w) {
        w.val<uint32_t>(kVersion);
        w.flag(metadata_only);
        w.flag(plan.model_input.shutdown);
        if (metadata_only) {
            /** TP peers need the upstream tag identities before synchronizing numeric cache rows. */
            w.strings(plan.model_input.kv_cache_group_tags);
        } else {
            modelInputFields(w, plan.model_input);
            samplingPlanFields(w, plan.sampling_plan);
            outputConfigFields(w, plan.output_config);
            w.flag(plan.is_decode);
            w.tensor(plan.draft_next_position_ids);
            w.val<uint64_t>(plan.finished_request_ids.size());
            for (const auto request_id : plan.finished_request_ids) {
                w.val<RequestId>(request_id);
            }
        }
    });
}

PPExecutionPlan deserializePlan(const torch::Tensor& buffer) {
    ByteReader r(buffer);
    RTP_LLM_CHECK_WITH_INFO(r.val<uint32_t>() == kVersion, "PP plan payload version mismatch");
    PPExecutionPlan plan;
    const bool      metadata_only = r.flag();
    plan.model_input.shutdown     = r.flag();
    if (metadata_only) {
        r.strings(plan.model_input.kv_cache_group_tags);
        r.expectEnd();
        return plan;
    }
    modelInputFields(r, plan.model_input);
    samplingPlanFields(r, plan.sampling_plan);
    outputConfigFields(r, plan.output_config);
    plan.is_decode = r.flag();
    r.tensor(plan.draft_next_position_ids);
    plan.finished_request_ids.resize(r.count());
    for (auto& request_id : plan.finished_request_ids) {
        request_id = r.val<RequestId>();
    }
    r.expectEnd();
    return plan;
}

torch::Tensor serializeExecutionResult(const PPExecutionResult& result) {
    return encodeToTensor([&](ByteEncoder& w) {
        w.val<uint32_t>(kVersion);
        executionResultFields(w, result);
    });
}

PPExecutionResult deserializeExecutionResult(const torch::Tensor& buffer) {
    ByteReader r(buffer);
    RTP_LLM_CHECK_WITH_INFO(r.val<uint32_t>() == kVersion, "PP execution-result payload version mismatch");

    PPExecutionResult result;
    executionResultFields(r, result);
    r.expectEnd();
    return result;
}

torch::Tensor serializeTensorsMetadata(const PPIntermediateTensors& tensors) {
    return encodeToTensor([&](ByteEncoder& w) {
        w.val<uint32_t>(kVersion);
        w.val<uint64_t>(tensors.tensors.size());
        for (const auto& [name, t] : tensors.tensors) {
            w.str(name);
            const bool present = t.defined() && t.numel() > 0;
            w.flag(present);
            if (!present) {
                continue;
            }
            w.val<uint64_t>(static_cast<uint64_t>(t.dim()));
            for (int64_t d = 0; d < t.dim(); ++d) {
                w.val<int64_t>(t.size(d));
            }
            w.val<uint8_t>(static_cast<uint8_t>(toWireDType(t.scalar_type())));
            w.flag(t.is_cuda());
        }
    });
}

PPIntermediateTensors deserializeTensorsMetadata(const torch::Tensor& metadata, PPIntermediateTensors& backing_buffers) {
    ByteReader r(metadata);
    RTP_LLM_CHECK_WITH_INFO(r.val<uint32_t>() == kVersion, "PP tensors-metadata payload version mismatch");
    PPIntermediateTensors out;
    const auto            n = r.count();
    for (size_t i = 0; i < n; ++i) {
        auto name = r.str();
        if (!r.flag()) {
            out.tensors.emplace(std::move(name), torch::Tensor{});
            continue;
        }
        const auto ndim = r.count();
        std::vector<int64_t> sizes(ndim);
        for (size_t d = 0; d < ndim; ++d) {
            sizes[d] = r.val<int64_t>();
        }
        const auto dtype   = fromWireDType(r.val<uint8_t>());
        const bool is_cuda = r.flag();
        const auto numel   = checkedTensorNumel(sizes);
        RTP_LLM_CHECK_WITH_INFO(numel <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                                "PP intermediate tensor numel=%zu exceeds torch int64 range",
                                numel);
        const auto tensor_numel = static_cast<int64_t>(numel);
        /** Batch token counts vary while tensor fields and types stay fixed for a stage boundary.
         * Reusable capacity avoids per-batch allocations; views retain the received shapes. */
        auto& buffer = backing_buffers.tensors[name];
        if (!buffer.defined() || buffer.numel() < tensor_numel) {
            buffer = torch::empty(
                {tensor_numel}, torch::TensorOptions().dtype(dtype).device(is_cuda ? torch::kCUDA : torch::kCPU));
        }
        out.tensors.emplace(std::move(name), buffer.narrow(0, 0, tensor_numel).view(sizes));
    }
    r.expectEnd();
    return out;
}

}  // namespace rtp_llm::pp_serialization
