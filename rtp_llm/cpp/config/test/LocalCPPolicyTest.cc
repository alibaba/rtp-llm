#include "rtp_llm/cpp/config/ConfigModules.h"

#include <iostream>
#include <stdexcept>
#include <string>

// Standalone CPU test of the actual production config (no copied policy or stubs).
// Also built as a Bazel cc_test; no CUDA initialization is required.
namespace {
int  checks = 0;
void check(bool value, const char* message) {
    ++checks;
    if (!value) {
        throw std::runtime_error(message);
    }
}
rtp_llm::ParallelismConfig makeCompat(bool cep) {
    rtp_llm::ParallelismConfig c;
    c.pp_size                  = cep ? 2 : 4;
    c.tp_size                  = cep ? 4 : 2;
    c.ffn_tp_size              = c.tp_size;
    c.tp_rank                  = 1;
    c.ffn_tp_rank              = 1;
    c.ep_size                  = cep ? 4 : 1;
    c.world_size               = 8;
    c.pp_ep_enabled            = cep;
    c.pp_ep_backend            = cep ? "fork_nccl_mxfp8" : "";
    c.prefill_cp_config.method = rtp_llm::CPRotateMethod::PREFILL_CP;
    return c;
}
void reject(rtp_llm::ParallelismConfig c,
            const std::string&         model       = "deepseek_v4",
            bool                       speculative = false,
            bool                       graph       = false,
            bool                       micro_batch = false) {
    bool rejected = false;
    try {
        c.resolve_local_cp(model, speculative, graph, micro_batch);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "unsupported profile did not fail closed");
    check(!c.dsv4_prefill_cp_compat, "failed resolution retained capability");
}
}  // namespace

int main() {
    using namespace rtp_llm;
    for (bool cep : {false, true}) {
        auto c = makeCompat(cep);
        check(!c.local_cp_enabled(), "raw PREFILL_CP is not a local capability");
        check(c.get_attn_tp_size() == c.tp_size, "unresolved weights must not become replicated");
        c.resolve_local_cp("deepseek_v4", false, false, false);
        check(c.local_cp_enabled(), "valid compatibility profile must engage");
        check(c.get_attn_tp_size() == 1 && c.get_attn_tp_rank() == 0,
              "attention weight partition differs from execution");
        check(c.get_ffn_tp_size() == 1 && c.get_ffn_tp_rank() == 0, "FFN weight partition differs from execution");
        check(c.prefill_cp_config.is_prefill_enabled() && !c.prefill_cp_config.is_enabled(),
              "upstream raw method semantics changed");
        c.role_type = RoleType::DECODE;
        check(!c.local_cp_enabled(), "stale compatibility leaked into decode");
        c.resolve_local_cp("deepseek_v4", false, false, false);
        check(!c.dsv4_prefill_cp_compat, "decode resolution retained local capability");
        check(c.get_attn_tp_size() == c.tp_size && c.get_ffn_tp_size() == c.ffn_tp_size,
              "decode weights incorrectly use remote CP geometry");
    }
    auto c = makeCompat(true);
    reject(c, "unrelated_model");
    reject(c, "deepseek_v4", true);
    reject(c, "deepseek_v4", false, true);
    reject(c, "deepseek_v4", false, false, true);
    c.role_type = RoleType::PREFILL;
    reject(c);
    c                                    = makeCompat(true);
    c.prefill_cp_config.kv_cache_sharded = true;
    reject(c);
    c                                   = makeCompat(true);
    c.prefill_cp_config.prefill_cp_size = 2;
    reject(c);
    c               = makeCompat(true);
    c.pp_ep_enabled = false;
    reject(c);
    c               = makeCompat(true);
    c.pp_ep_backend = "unknown";
    reject(c);
    c         = makeCompat(true);
    c.pp_size = 4;
    reject(c);
    c            = makeCompat(true);
    c.world_size = 16;
    reject(c);
    c           = makeCompat(true);
    c.enable_sp = true;
    reject(c);
    c             = makeCompat(true);
    c.use_ub_comm = true;
    reject(c);
    c                                                 = makeCompat(true);
    c.ffn_disaggregate_config.enable_ffn_disaggregate = true;
    reject(c);
    c         = makeCompat(true);
    c.tp_size = 1;
    c.resolve_local_cp("unrelated_model", false, false, false);
    check(!c.local_cp_enabled(), "one-rank metadata enabled local CP");
    for (auto method :
         {CPRotateMethod::ALL_GATHER, CPRotateMethod::ALL_GATHER_WITH_OVERLAP, CPRotateMethod::ALLTOALL}) {
        c                          = makeCompat(false);
        c.prefill_cp_config.method = method;
        c.role_type                = RoleType::PREFILL;
        c.resolve_local_cp("native_model", true, false, false);
        check(c.local_cp_enabled(), "native PREFILL CP semantics lost");
        check(!c.dsv4_prefill_cp_compat, "native mode misclassified as DSV4 compatibility");
        c.role_type = RoleType::PDFUSION;
        reject(c, "native_model");
        c.role_type = RoleType::DECODE;
        check(!c.local_cp_enabled(), "native method executed in decode");
    }
    std::cout << checks << " production local-CP policy checks passed\n";
}
