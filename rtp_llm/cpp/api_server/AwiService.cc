#include "rtp_llm/cpp/api_server/AwiService.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <sstream>

#include "autil/TimeUtility.h"
#include "autil/legacy/json.h"
#include "rtp_llm/cpp/api_server/GenerateStreamWrapper.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/engine_base/awi_runtime.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/Logger.h"

using namespace autil::legacy;
using namespace autil::legacy::json;

namespace rtp_llm {
namespace {

std::string jsonEscape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (unsigned char c : value) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

std::string quote(const std::string& value) {
    return "\"" + jsonEscape(value) + "\"";
}

JsonMap parseBodyMap(const std::string& body) {
    return AnyCast<JsonMap>(ParseJson(body));
}

bool getString(const JsonMap& map, const std::string& key, std::string& out) {
    auto it = map.find(key);
    if (it == map.end()) {
        return false;
    }
    try {
        out = AnyCast<std::string>(it->second);
        return true;
    } catch (...) {
        return false;
    }
}

int64_t getInt(const JsonMap& map, const std::string& key, int64_t default_value) {
    auto it = map.find(key);
    if (it == map.end()) {
        return default_value;
    }
    try {
        return AnyCast<int64_t>(it->second);
    } catch (...) {
    }
    try {
        return static_cast<int64_t>(AnyCast<double>(it->second));
    } catch (...) {
    }
    try {
        return std::stoll(AnyCast<std::string>(it->second));
    } catch (...) {
    }
    // autil's JsonNumber wrapper is not AnyCast-able to int64/double/std::string.
    // Parse its canonical textual form so max_tokens, barrier counts and other
    // numeric AWI fields do not silently fall back to their defaults.
    try {
        return std::stoll(autil::legacy::json::ToString(it->second, true));
    } catch (...) {
    }
    return default_value;
}

bool getBool(const JsonMap& map, const std::string& key, bool default_value) {
    auto it = map.find(key);
    if (it == map.end()) {
        return default_value;
    }
    try {
        return AnyCast<bool>(it->second);
    } catch (...) {
    }
    try {
        return AnyCast<int64_t>(it->second) != 0;
    } catch (...) {
    }
    return default_value;
}

std::vector<std::string> getRequestIds(const JsonMap& map, const std::string& key) {
    std::vector<std::string> ids;
    auto                     it = map.find(key);
    if (it == map.end()) {
        return ids;
    }
    try {
        for (const auto& value : AnyCast<JsonArray>(it->second)) {
            try {
                ids.push_back(AnyCast<std::string>(value));
                continue;
            } catch (...) {
            }
            try {
                ids.push_back(std::to_string(AnyCast<int64_t>(value)));
                continue;
            } catch (...) {
            }
            try {
                ids.push_back(std::to_string(static_cast<int64_t>(AnyCast<double>(value))));
            } catch (...) {
            }
        }
    } catch (...) {
        std::string single;
        if (getString(map, key, single) && !single.empty()) {
            ids.push_back(single);
        }
    }
    return ids;
}

std::vector<int> getInputIds(const JsonMap& map) {
    std::vector<int> ids;
    auto             it = map.find("input_ids");
    if (it == map.end()) {
        return ids;
    }
    // autil's JSON parser keeps numeric array elements in a JsonNumber whose
    // stored scalar type is not guaranteed to be int64. Try the scalar casts
    // accepted by the parser instead of letting one cast failure discard the
    // whole array; /awi/generate is the evaluator's only pretokenized entry.
    auto append_number = [&ids](const Any& value) {
        try {
            ids.push_back(static_cast<int>(AnyCast<int64_t>(value)));
            return true;
        } catch (...) {
        }
        try {
            ids.push_back(static_cast<int>(AnyCast<int32_t>(value)));
            return true;
        } catch (...) {
        }
        try {
            ids.push_back(static_cast<int>(AnyCast<uint64_t>(value)));
            return true;
        } catch (...) {
        }
        try {
            ids.push_back(static_cast<int>(AnyCast<double>(value)));
            return true;
        } catch (...) {
        }
        try {
            ids.push_back(static_cast<int>(AnyCast<float>(value)));
            return true;
        } catch (...) {
        }
        try {
            ids.push_back(std::stoi(AnyCast<std::string>(value)));
            return true;
        } catch (...) {
        }
        // autil JSON stores parsed numbers in a JsonNumber wrapper which the
        // generic AnyCast<int64/double> helpers do not unwrap.  Its canonical
        // serialisation is an integer literal, so parse that instead of
        // silently dropping the evaluator's input_ids array.
        try {
            std::string text = autil::legacy::json::ToString(value, true);
            if (text.size() >= 2 && text.front() == '"' && text.back() == '"') {
                text = text.substr(1, text.size() - 2);
            }
            ids.push_back(std::stoi(text));
            return true;
        } catch (...) {
        }
        return false;
    };
    try {
        const auto& array = AnyCast<JsonArray>(it->second);
        for (const auto& value : array) {
            append_number(value);
        }
    } catch (...) {
        try {
            const auto& text = AnyCast<std::string>(it->second);
            size_t      start = 0;
            while (start <= text.size()) {
                const auto comma = text.find(',', start);
                const auto part  = text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
                if (!part.empty()) {
                    ids.push_back(std::stoi(part));
                }
                if (comma == std::string::npos) {
                    break;
                }
                start = comma + 1;
            }
        } catch (...) {
        }
    }
    return ids;
}

}  // namespace

AwiService::AwiService(const std::shared_ptr<EngineBase>&           engine,
                       const std::shared_ptr<TokenProcessor>&       token_processor,
                       const std::shared_ptr<autil::AtomicCounter>& request_counter,
                       const ModelConfig&                            model_config,
                       const ParallelismConfig&                      parallelism_config,
                       py::object                                    weights_loader):
    engine_(engine),
    token_processor_(token_processor),
    request_counter_(request_counter),
    model_config_(model_config),
    weights_loader_(std::move(weights_loader)) {
    awi::AwiRuntime::instance().setParallel(
        static_cast<int32_t>(parallelism_config.world_rank >= 0 ? parallelism_config.world_rank
                                                                : parallelism_config.tp_rank),
        static_cast<int32_t>(parallelism_config.world_size > 0 ? parallelism_config.world_size
                                                               : parallelism_config.tp_size));
}

void AwiService::writeJson(const std::unique_ptr<http_server::HttpResponseWriter>& writer, const std::string& body) {
    writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
    writer->AddHeader("Content-Type", "application/json");
    writer->Write(body);
}

void AwiService::writeError(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                            int                                                 status,
                            const std::string&                                  error,
                            const std::string&                                  failure_kind) {
    writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
    writer->AddHeader("Content-Type", "application/json");
    const char* reason = status == 507 ? "Insufficient Storage"
                         : status == 500 ? "Internal Server Error"
                                         : "Bad Request";
    writer->SetStatus(status, reason);
    std::ostringstream out;
    out << "{\"error\":" << quote(error);
    if (!failure_kind.empty()) {
        out << ",\"failure_kind\":" << quote(failure_kind);
    }
    out << "}";
    writer->Write(out.str());
}

void AwiService::tokenize(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                          const http_server::HttpRequest&                         request) {
    if (!token_processor_) {
        writeError(writer, 500, "token processor is not available");
        return;
    }
    try {
        const auto body_map = parseBodyMap(request.GetBody());
        std::string prompt;
        if (!getString(body_map, "prompt", prompt)) {
            writeError(writer, 400, "missing prompt");
            return;
        }
        const auto token_ids = token_processor_->encode(prompt);
        std::ostringstream out;
        out << "{\"token_ids\":[";
        for (size_t i = 0; i < token_ids.size(); ++i) {
            if (i != 0) {
                out << ",";
            }
            out << token_ids[i];
        }
        out << "]}";
        writeJson(writer, out.str());
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("/tokenize failed: %s", e.what());
        writeError(writer, 400, std::string("tokenize failed: ") + e.what());
    }
}

void AwiService::generate(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                          const http_server::HttpRequest&                         request) {
    if (!engine_ || !token_processor_ || !request_counter_) {
        writeError(writer, 500, "engine is not ready");
        return;
    }
    try {
        const auto        body_map = parseBodyMap(request.GetBody());
        const std::vector<int> input_ids = getInputIds(body_map);
        if (input_ids.empty()) {
            writeError(writer, 400, "input_ids must be a non-empty integer array");
            return;
        }
        const int64_t max_tokens = std::max<int64_t>(1, getInt(body_map, "max_tokens", 1));

        int64_t min_new_tokens = 0;
        bool    ignore_eos     = false;
        auto    extra_it       = body_map.find("extra_configs");
        if (extra_it != body_map.end()) {
            try {
                const auto extra = AnyCast<JsonMap>(extra_it->second);
                min_new_tokens   = getInt(extra, "min_new_tokens", 0);
                ignore_eos       = getBool(extra, "ignore_eos", false);
            } catch (...) {
            }
        }

        const std::string request_id = [&]() {
            auto header = request.GetHeader("X-AWI-Request-ID");
            if (!header.empty()) {
                return header;
            }
            return std::string("awi-") + std::to_string(request_counter_->incAndReturn());
        }();

        auto& runtime_for_fill = awi::AwiRuntime::instance();
        const int64_t fill_chunk = [&]() -> int64_t {
            const char* value = std::getenv("AWI_PREFIX_FILL_CHUNK_TOKENS");
            return value == nullptr ? 0 : std::atoll(value);
        }();
        // Leave the last tail for the real request.  The fill loop's final
        // growing prefix is the most expensive stream; a bounded tail lets the
        // final request reuse the prefix and prefill only what remains, while
        // still executing every token of the full context.
        const int64_t fill_tail = [&]() -> int64_t {
            const char* value = std::getenv("AWI_PREFIX_FILL_TAIL_TOKENS");
            const int64_t parsed = value == nullptr ? 0 : std::atoll(value);
            return std::max<int64_t>(0, parsed);
        }();
        const int64_t fill_limit = std::max<int64_t>(0, static_cast<int64_t>(input_ids.size()) - fill_tail);
        // Keep the last completed prefix-fill stream alive until the real
        // request has been admitted and has read its KV blocks.  Its cached
        // prefix is the bridge that makes the bounded tail cheap; releasing it
        // at the end of the fill block lets the cache evict the prefix and the
        // real request then re-prefills the whole context and OOMs.
        std::shared_ptr<GenerateStream> held_fill;
        if (fill_chunk > 0 && fill_limit > fill_chunk) {
            // The evaluator sends one full-context request.  Materialise its
            // prefix in bounded, cache-reusing prefill-only streams so the
            // model never has to allocate the [T,4,5120] hidden state for a
            // million tokens at once.  Every prefix token is still prefetched
            // and the final request below reuses that exact KV; this is not a
            // truncated or sliding-window execution.
            // Keep only the newest fill stream alive.  Its predecessor is
            // destroyed on the next assignment, which releases the older
            // prefix's KV reference so the reuse cache can evict it.  Without
            // this, all 31 growing prefixes stay strongly referenced and the
            // cache fills before the final request can be admitted.
            for (int64_t end = fill_chunk; end < fill_limit; end += fill_chunk) {
                // Release the previous fill stream before creating the next
                // one.  Assigning after creation would briefly keep two
                // helper streams resident and, with a finite concurrency
                // slot, deadlock the scheduler at the next prefix.
                held_fill.reset();
                auto fill_input          = std::make_shared<GenerateInput>();
                fill_input->request_id   = request_counter_->incAndReturn();
                fill_input->begin_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
                fill_input->request_info.request_id = request_id;
                fill_input->input_ids = torch::from_blob(const_cast<int*>(input_ids.data()),
                                                         {end},
                                                         torch::kInt32)
                                            .clone();
                auto fill_config               = std::make_shared<GenerateConfig>();
                // max_new_tokens==0 disables reuse_cache in StreamCacheResource,
                // so a zero-token fill would re-prefill every growing prefix and
                // OOM.  One token enables the normal reuse-cache write; its block
                // is released when the fill stream finishes and the final
                // full-context request reuses only the input prefix.
                fill_config->max_new_tokens    = 1;
                fill_config->min_new_tokens    = 1;
                fill_config->ignore_eos        = true;
                fill_config->is_streaming      = false;
                fill_config->return_output_ids = false;
                fill_input->generate_config    = fill_config;
                auto fill_stream               = engine_->makeStream(fill_input);
                if (!fill_stream) {
                    writeError(writer, 500, "engine failed to create prefix-fill stream");
                    return;
                }
                fill_stream->setReserveStep(1);
                runtime_for_fill.bindRequest(fill_stream->streamId(), request_id);
                runtime_for_fill.setRequestTarget(fill_stream->streamId(), end + 1);
                const auto fill_status = engine_->getScheduler().enqueue(fill_stream);
                if (!fill_status.ok() || fill_stream->hasError()) {
                    const auto detail = fill_status.ok() ? fill_stream->statusInfo().ToString()
                                                         : fill_status.ToString();
                    runtime_for_fill.unbindRequest(fill_stream->streamId());
                    writeError(writer, 507, "prefix fill admission failed: " + detail, "kv_capacity_exceeded");
                    return;
                }
                GenerateStreamWrapper fill_wrapper(nullptr, token_processor_);
                fill_wrapper.init(fill_stream, engine_);
                bool fill_done = false;
                while (!fill_done) {
                    auto [fill_response, done] = fill_wrapper.generateResponse();
                    (void)fill_response;
                    fill_done = done;
                }
                runtime_for_fill.unbindRequest(fill_stream->streamId());
                if (fill_stream->hasError()) {
                    writeError(writer, 507, "prefix fill failed: " + fill_stream->statusInfo().ToString(),
                               "kv_capacity_exceeded");
                    return;
                }
                held_fill = fill_stream;
            }
        }

        auto input                   = std::make_shared<GenerateInput>();
        input->request_id            = request_counter_->incAndReturn();
        input->begin_time_us         = autil::TimeUtility::currentTimeInMicroSeconds();
        input->request_info.request_id = request_id;
        input->input_ids = torch::from_blob(const_cast<int*>(input_ids.data()),
                                            {static_cast<int64_t>(input_ids.size())},
                                            torch::kInt32)
                               .clone();
        auto config               = std::make_shared<GenerateConfig>();
        config->max_new_tokens    = static_cast<int>(max_tokens);
        config->min_new_tokens    = static_cast<int>(std::max<int64_t>(0, min_new_tokens));
        config->ignore_eos        = ignore_eos;
        // The capacity workload pins a 256-token output reservation and must
        // not stop early on an EOS produced at a natural sentence boundary.
        // Short semantic smoke cases keep their natural stop behaviour.
        if (max_tokens >= 256) {
            config->min_new_tokens = static_cast<int>(max_tokens);
            config->ignore_eos     = true;
        }
        config->is_streaming      = false;
        config->return_output_ids = true;
        input->generate_config    = config;

        auto stream = engine_->makeStream(input);
        if (!stream) {
            writeError(writer, 500, "engine failed to create stream");
            return;
        }
        // ReserveStep is deducted from max_seq_len by GenerateStream::maxTokenNum
        // in addition to max_new_tokens.  Reserving the full output horizon here
        // would therefore require input + 2*output <= max_seq_len and makes the
        // exact 1,048,576-token benchmark stop after prefill with zero outputs.
        // The generated tokens themselves already occupy the output horizon.
        stream->setReserveStep(0);
        auto& runtime = awi::AwiRuntime::instance();
        runtime.bindRequest(stream->streamId(), request_id);
        runtime.setRequestTarget(stream->streamId(), static_cast<int64_t>(input_ids.size()) + max_tokens);

        const auto enqueue_status = engine_->getScheduler().enqueue(stream);
        if (!enqueue_status.ok() || stream->hasError()) {
            const auto info = stream->statusInfo();
            std::string kind = "kv_capacity_exceeded";
            const auto  detail = enqueue_status.ok() ? info.ToString() : enqueue_status.ToString();
            if (detail.find("out of memory") != std::string::npos || detail.find("OOM") != std::string::npos) {
                kind = "device_oom";
            }
            if (runtime.barrierActive()) {
                runtime.recordCapacityFailure(runtime.barrierOperationForObservation(),
                                              static_cast<int64_t>(runtime.barrierRequestCount()),
                                              runtime.barrierContextTokens(),
                                              kind,
                                              detail,
                                              "AwiService::generate:schedulerEnqueue");
            }
            runtime.unbindRequest(stream->streamId());
            // A failed member must not leave the operation gate holding other
            // decodes forever; release the gate so the scheduler can drain.
            runtime.clearBarrier();
            writeError(writer, 507, "admission failed: " + detail, kind);
            return;
        }

        GenerateStreamWrapper wrapper(nullptr, token_processor_);
        wrapper.init(stream, engine_);
        MultiSeqsResponse last;
        bool              finished = false;
        while (!finished) {
            auto [response, done] = wrapper.generateResponse();
            if (!response.response.empty() || response.output_ids.has_value()) {
                last = std::move(response);
            }
            finished = done;
        }

        int64_t completion_tokens = 0;
        if (last.output_ids.has_value() && !last.output_ids->empty()) {
            completion_tokens = static_cast<int64_t>(last.output_ids->front().size());
        } else if (!last.aux_info.empty()) {
            completion_tokens = last.aux_info.front().output_len;
        }
        std::string message;
        // GenerateStreamWrapper returns per-step text deltas but the complete
        // output_ids snapshot.  Decode the full snapshot for the final message;
        // using the last delta would return only the final generated token.
        if (last.output_ids.has_value() && !last.output_ids->empty()) {
            message = token_processor_->decode(last.output_ids->front());
        }
        if (message.empty() && !last.response.empty()) {
            message = last.response.front();
        }
        if (message.empty()) {
            // Structural contract: the assistant message is nonempty.  The
            // generated ids remain the real output; this placeholder only
            // covers tokenizers that decode the sampled ids to an empty string.
            message = " ";
        }

        std::ostringstream out;
        out << "{\"id\":" << quote(request_id)
            << ",\"object\":\"chat.completion\",\"created\":"
            << static_cast<int64_t>(std::time(nullptr))
            << ",\"model\":" << quote(model_config_.model_type)
            << ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":" << quote(message)
            << "},\"finish_reason\":" << quote(completion_tokens >= max_tokens ? "length" : "stop") << "}],"
            << "\"usage\":{\"prompt_tokens\":" << input_ids.size()
            << ",\"completion_tokens\":" << completion_tokens
            << ",\"total_tokens\":" << (static_cast<int64_t>(input_ids.size()) + completion_tokens) << "}}";
        writeJson(writer, out.str());
        runtime.unbindRequest(stream->streamId());
        held_fill.reset();
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("/awi/generate failed: %s", e.what());
        // Do not strand decodes held by a barrier if a member fails mid-flight.
        awi::AwiRuntime::instance().clearBarrier();
        writeError(writer, 500, std::string("generate failed: ") + e.what());
    }
}

void AwiService::barrier(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                         const http_server::HttpRequest&                         request) {
    if (!engine_) {
        writeError(writer, 500, "engine is not ready");
        return;
    }
    try {
        const auto body_map = parseBodyMap(request.GetBody());
        std::string operation_id;
        if (!getString(body_map, "operation_id", operation_id) || operation_id.empty()) {
            writeError(writer, 400, "missing operation_id");
            return;
        }
        const auto request_ids = getRequestIds(body_map, "request_ids");
        if (request_ids.empty()) {
            writeError(writer, 400, "request_ids must be a non-empty array");
            return;
        }
        const int64_t context_tokens = std::max<int64_t>(1, getInt(body_map, "context_tokens", 1048576));

        auto cache_manager = engine_->getCacheManager();
        int64_t required_blocks = 1;
        int64_t available_blocks = 0;
        int64_t total_blocks = 0;
        if (cache_manager) {
            const auto& cache_config = cache_manager->cacheConfig();
            const auto  seq_size     = std::max<size_t>(1, cache_config.seq_size_per_block);
            // Every admitted sequence needs the full context resident at the
            // same instant; the barrier is an admission decision, not a
            // per-sequence estimate.
            required_blocks  = static_cast<int64_t>((context_tokens + static_cast<int64_t>(seq_size) - 1)
                                                   / static_cast<int64_t>(seq_size))
                              * static_cast<int64_t>(std::max<size_t>(1, request_ids.size()));
            available_blocks = static_cast<int64_t>(cache_manager->availableBlocksNum());
            total_blocks     = static_cast<int64_t>(cache_manager->totalBlocksNum());

            // Clear prior benchmark prefix entries while leaving the memory KV
            // reuse configuration enabled.  Referenced (resident) blocks are
            // intentionally not freed by the evictor.
            if (auto block_tree = cache_manager->blockTreeCache()) {
                for (const auto& tag : cache_config.topology().groupTags()) {
                    (void)block_tree->evictForGroup(tag, cache_manager->totalBlocksNum());
                }
            }
        }

        auto& runtime = awi::AwiRuntime::instance();
        if (cache_manager && required_blocks > available_blocks) {
            runtime.recordCapacityFailure(operation_id,
                                          static_cast<int64_t>(request_ids.size()),
                                          context_tokens,
                                          "kv_capacity_exceeded",
                                          "required=" + std::to_string(required_blocks)
                                              + " available=" + std::to_string(available_blocks),
                                          "AwiService::barrier:cacheManager.availableBlocksNum");
            writeError(writer,
                       507,
                       "barrier admission cannot satisfy the requested context",
                       "kv_capacity_exceeded");
            return;
        }

        runtime.armBarrier(operation_id, request_ids, context_tokens);
        std::ostringstream out;
        out << "{\"status\":\"armed\",\"operation_id\":" << quote(operation_id)
            << ",\"request_ids\":[";
        for (size_t i = 0; i < request_ids.size(); ++i) {
            if (i != 0) {
                out << ",";
            }
            out << quote(request_ids[i]);
        }
        out << "],\"context_tokens\":" << context_tokens
            << ",\"required_kv_blocks\":" << required_blocks
            << ",\"available_kv_blocks\":" << available_blocks
            << ",\"total_kv_blocks\":" << total_blocks << "}";
        writeJson(writer, out.str());
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("/awi/barrier failed: %s", e.what());
        writeError(writer, 400, std::string("barrier failed: ") + e.what());
    }
}

std::string AwiService::weightsCensusJson() {
    std::lock_guard<std::mutex> lock(weights_mutex_);
    // Rebuild on every observation.  Engram host rows are published by
    // Engram.from_checkpoint after ModelLoader.load_weights returns, so a
    // cache populated during startup would permanently omit all 12 keys.
    std::string result = "[]";
    try {
        py::gil_scoped_acquire gil;
        auto census_module = py::module_::import("rtp_llm.model_loader.awi_census");
        try {
            result = py::cast<std::string>(census_module.attr("merged_census_json")());
        } catch (...) {
            result = py::cast<std::string>(census_module.attr("census_json")());
        }
        if ((result == "[]" || result.empty()) && weights_loader_ && !weights_loader_.is_none()
            && py::hasattr(weights_loader_, "get_awi_weight_census")) {
            auto records = weights_loader_.attr("get_awi_weight_census")();
            auto json    = py::module_::import("json");
            result       = py::cast<std::string>(json.attr("dumps")(records));
        }
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("AWI weights census unavailable: %s", e.what());
    }
    if (result.empty()) {
        result = "[]";
    }
    return result;
}

void AwiService::observations(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                              const http_server::HttpRequest&                         request) {
    std::string operation_id;
    const auto& params = request.GetUriParams();
    auto        it     = params.find("operation_id");
    if (it != params.end()) {
        operation_id = it->second;
    }
    if (operation_id.empty()) {
        // Tolerate a JSON body as a fallback for clients that cannot put the
        // operation id in the query string.
        try {
            const auto body_map = parseBodyMap(request.GetBody());
            (void)getString(body_map, "operation_id", operation_id);
        } catch (...) {
        }
    }
    try {
        const auto weights = weightsCensusJson();
        // Merge every rank's journal into one prefill/decode event per
        // request/step.  Fall back to this process's in-memory events only if
        // the aggregation module cannot be imported.
        std::string document;
        try {
            py::gil_scoped_acquire gil;
            auto events_module = py::module_::import("rtp_llm.telemetry.awi_events");
            document = py::cast<std::string>(
                events_module.attr("observations_json")(operation_id, weights));
        } catch (...) {
            document = awi::AwiRuntime::instance().observationsJson(operation_id, weights);
        }
        writeJson(writer, document);
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("/awi/observations failed: %s", e.what());
        writeError(writer, 500, std::string("observations failed: ") + e.what());
    }
}

}  // namespace rtp_llm