#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackendCpp.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"
#include "autil/EnvUtil.h"
#include <stdexcept>

using namespace autil;

namespace rtp_llm {

EngineBase::EngineBase(const EngineInitParams& params) {
    initRuntime(params);
    grammar_config_ = params.grammar_config;
    // Construct the native xgrammar backend from the Python-side
    // tokenizer_info_json (populated once at engine start). Empty json or
    // grammar_backend=="" / "none" ⇒ grammar disabled — leave backend null
    // and let GrammarManager short-circuit. Checking the backend string here
    // too (not only json emptiness) catches stale-state cases where a
    // restored config carries both fields. Construction failures are
    // non-fatal: they degrade the engine to "no grammar" rather than
    // aborting startup.
    const std::string& backend_name = grammar_config_.grammar_backend;
    const bool grammar_disabled =
        backend_name.empty() || backend_name == "none" || backend_name == "None";
    if (!grammar_disabled && !grammar_config_.tokenizer_info_json.empty()) {
        try {
            XGrammarBackendOptions opts;
            opts.any_whitespace        = !grammar_config_.constrained_json_disable_any_whitespace;
            opts.strict_mode           = true;
            opts.max_compiler_threads  = std::max(1, grammar_config_.num_workers);
            opts.enable_compiler_cache = true;
            opts.compiler_cache_bytes  = -1;  // unlimited
            if (grammar_config_.think_end_id >= 0) {
                opts.think_end_id = static_cast<int32_t>(grammar_config_.think_end_id);
            }
            if (!grammar_config_.override_stop_tokens.empty()) {
                opts.override_stop_tokens = grammar_config_.override_stop_tokens;
            }
            grammar_backend_ =
                std::make_shared<XGrammarBackendCpp>(grammar_config_.tokenizer_info_json, opts);
            RTP_LLM_LOG_INFO("XGrammarBackendCpp constructed: think_end_id=%ld, any_whitespace=%d, "
                             "override_stop_tokens=%zu, num_workers=%d",
                             grammar_config_.think_end_id,
                             opts.any_whitespace,
                             grammar_config_.override_stop_tokens.size(),
                             opts.max_compiler_threads);
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING(
                "XGrammarBackendCpp construction failed (%s); grammar disabled for this engine", e.what());
            grammar_backend_.reset();
        }
    }
}

EngineBase::~EngineBase() {}

std::vector<GenerateStreamPtr> EngineBase::batchEnqueue(const std::vector<std::shared_ptr<GenerateInput>>& inputs) {
    throw std::runtime_error("not implemeted");
}

std::shared_ptr<GenerateStream> EngineBase::makeStream(const std::shared_ptr<GenerateInput>& input) {
    throw std::runtime_error("not implemeted");
}

void EngineBase::initRuntime(const EngineInitParams& params) {
    const auto rank =
        params.parallelism_config.dp_rank * params.parallelism_config.tp_size + params.parallelism_config.tp_rank;
    Logger::getEngineLogger().setRank(rank);
    Logger::getEngineLogger().flush();
    size_t device_id = params.parallelism_config.world_rank % params.parallelism_config.local_world_size;
    mla_ops_type_    = rtp_llm::initRuntime(device_id,
                                         params.profiling_debug_logging_config.trace_memory,
                                         params.device_resource_config.enable_comm_overlap,
                                         params.model_config_.mla_ops_type);
    warmupNoBlockCopy();
}

std::shared_ptr<KVCacheManager> EngineBase::getCacheManager() const {
    return resource_context_.cache_manager;
}

}  // namespace rtp_llm
