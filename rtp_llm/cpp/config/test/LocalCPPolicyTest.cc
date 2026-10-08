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
rtp_llm::ParallelismConfig makeProxy() {
    // CEP2PP2: the 4-rank local PD proxy (pp2 x dp1 x tp2 x ep2 x world4).
    rtp_llm::ParallelismConfig c;
    c.pp_size                  = 2;
    c.tp_size                  = 2;
    c.ffn_tp_size              = c.tp_size;
    c.tp_rank                  = 1;
    c.ffn_tp_rank              = 1;
    c.ep_size                  = 2;
    c.world_size               = 4;
    c.pp_ep_enabled            = true;
    c.pp_ep_backend            = "fork_nccl_mxfp8";
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
    // PREFILL (the PD prefill role) is admitted at the exact PP+EP shapes.
    c           = makeCompat(true);
    c.role_type = RoleType::PREFILL;
    c.resolve_local_cp("deepseek_v4", false, false, false);
    check(c.local_cp_enabled(), "PREFILL role must engage the qualified CEP4PP2 profile");
    check(c.get_attn_tp_size() == 1 && c.get_ffn_tp_size() == 1, "PREFILL role must use local CP geometry");
    c                                    = makeCompat(true);
    c.role_type                          = RoleType::PREFILL;
    c.prefill_cp_config.kv_cache_sharded = true;
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
    // CEP2PP2: the 4-rank local PD proxy profile must engage under both roles.
    for (auto role : {RoleType::PDFUSION, RoleType::PREFILL}) {
        c           = makeProxy();
        c.role_type = role;
        check(!c.local_cp_enabled(), "raw PREFILL_CP is not a local capability (proxy)");
        c.resolve_local_cp("deepseek_v4", false, false, false);
        check(c.local_cp_enabled(), "valid CEP2PP2 proxy profile must engage");
        check(c.get_attn_tp_size() == 1 && c.get_ffn_tp_size() == 1, "proxy weight partition differs from execution");
        c.role_type = RoleType::DECODE;
        check(!c.local_cp_enabled(), "stale proxy compatibility leaked into decode");
    }
    c           = makeProxy();
    c.role_type = RoleType::DECODE;
    c.resolve_local_cp("deepseek_v4", false, false, false);
    check(!c.dsv4_prefill_cp_compat, "decode resolution retained proxy capability");
    c                                    = makeProxy();
    c.prefill_cp_config.kv_cache_sharded = true;
    reject(c);
    c                                   = makeProxy();
    c.prefill_cp_config.prefill_cp_size = 4;
    reject(c);
    // The proxy is an exact profile, not a range: every near-miss stays rejected.
    c            = makeProxy();
    c.world_size = 8;
    reject(c);
    c         = makeProxy();
    c.tp_size = 4;
    reject(c);
    c         = makeProxy();
    c.ep_size = 4;
    reject(c);
    c         = makeProxy();
    c.pp_size = 1;
    reject(c);
    c         = makeProxy();
    c.dp_size = 2;
    reject(c);
    c               = makeProxy();
    c.pp_ep_backend = "unknown";
    reject(c);
    // The world8 target and the world4 proxy must not be confused for each other.
    c            = makeCompat(true);
    c.world_size = 4;
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
    // DSpARK allowance is explicit and PREFILL-only at the exact CEP profiles. All old
    // four-argument resolutions above still fail closed on speculation.
    c           = makeProxy();
    c.role_type = RoleType::PREFILL;
    c.resolve_local_cp("deepseek_v4", true, false, false, true);
    check(c.local_cp_enabled() && c.dsv4_dspark_prefill_compat, "DSpARK proxy capability missing");
    check(c.get_attn_tp_size() == 1, "DSpARK target and draft must retain CP weight semantics");
    c.resolve_local_cp("deepseek_v4", false, false, false);
    check(!c.dsv4_dspark_prefill_compat, "DSpARK capability survived ordinary re-resolution");

    auto reject_dspark = [](ParallelismConfig candidate, bool spec = true, bool graph = false, bool micro = false) {
        bool rejected = false;
        try {
            candidate.resolve_local_cp("deepseek_v4", spec, graph, micro, true);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected, "unsupported DSpARK prefill profile did not fail closed");
        check(!candidate.dsv4_dspark_prefill_compat && !candidate.dsv4_prefill_cp_compat,
              "failed DSpARK resolution retained a capability");
    };
    reject_dspark(c, false);
    reject_dspark(c, true, true);
    reject_dspark(c, true, false, true);
    for (auto role : {RoleType::PDFUSION, RoleType::DECODE}) {
        auto bad      = c;
        bad.role_type = role;
        reject_dspark(bad);
    }
    // Exercise both stages and every lane of the eight-rank serving profile.
    for (int rank = 0; rank < 8; ++rank) {
        auto target       = makeCompat(true);
        target.role_type  = RoleType::PREFILL;
        target.world_rank = rank;
        target.pp_rank    = rank / 4;
        target.tp_rank = target.ep_rank = target.ffn_tp_rank = rank % 4;
        target.prefill_cp_config.prefill_cp_size             = 4;
        target.resolve_local_cp("deepseek_v4", true, false, false, true);
        check(target.local_cp_enabled() && target.dsv4_dspark_prefill_compat, "DSpARK CEP4PP2 capability missing");
        check(target.get_attn_tp_size() == 1 && target.get_attn_tp_rank() == 0 && target.get_ffn_tp_size() == 1
                  && target.get_ffn_tp_rank() == 0,
              "DSpARK CEP4PP2 weights must retain local CP geometry");
        reject(target, "deepseek_v4", true);
        reject_dspark(target, false);
        reject_dspark(target, true, true);
        reject_dspark(target, true, false, true);
        for (auto role : {RoleType::PDFUSION, RoleType::DECODE}) {
            auto wrong_role      = target;
            wrong_role.role_type = role;
            reject_dspark(wrong_role);
        }
        target.resolve_local_cp("deepseek_v4", false, false, false);
        check(!target.dsv4_dspark_prefill_compat, "CEP4 MTP capability survived re-resolution");
    }
    for (auto profile : {makeProxy(), makeCompat(true)}) {
        profile.role_type = RoleType::PREFILL;
        auto bad          = profile;
        bad.tp_size = bad.ep_size = 3;
        bad.world_size            = 6;
        reject_dspark(bad);
        bad            = profile;
        bad.world_size = profile.world_size == 4 ? 8 : 4;
        reject_dspark(bad);
        bad         = profile;
        bad.dp_size = 2;
        reject_dspark(bad);
        bad         = profile;
        bad.pp_size = 4;
        reject_dspark(bad);
        bad               = profile;
        bad.pp_ep_backend = "unknown";
        reject_dspark(bad);
        bad                                    = profile;
        bad.prefill_cp_config.kv_cache_sharded = true;
        reject_dspark(bad);
        bad                                   = profile;
        bad.prefill_cp_config.prefill_cp_size = profile.tp_size == 2 ? 4 : 2;
        reject_dspark(bad);
    }
    auto bad                               = c;
    bad.prefill_cp_config.kv_cache_sharded = true;
    reject_dspark(bad);
    bad                          = c;
    bad.prefill_cp_config.method = CPRotateMethod::ALL_GATHER;
    reject_dspark(bad);
    bad           = c;
    bad.enable_sp = true;
    reject_dspark(bad);
    bad             = c;
    bad.ffn_sp_size = 2;
    reject_dspark(bad);
    bad             = c;
    bad.use_ub_comm = true;
    reject_dspark(bad);
    bad               = c;
    bad.pp_ep_enabled = false;
    reject_dspark(bad);
    bad                                   = c;
    bad.prefill_cp_config.prefill_cp_size = 4;
    reject_dspark(bad);
    std::cout << checks << " production local-CP policy checks passed\n";
}
