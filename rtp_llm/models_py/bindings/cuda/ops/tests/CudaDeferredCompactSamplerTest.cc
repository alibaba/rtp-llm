#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/models/Sampler.h"
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <gtest/gtest.h>
#include <pybind11/embed.h>

namespace rtp_llm {

TEST(CudaDeferredCompactSamplerTest, NativeAndEagerVerifyUseOriginalVocabAndSameRng) {
    namespace py = pybind11;
    py::scoped_interpreter python;
    py::dict               scope;
    py::exec(R"(
import os
import sys
from pathlib import Path
runfiles = Path(os.environ["TEST_SRCDIR"])
sys.path[:0] = [str(runfiles / os.environ["TEST_WORKSPACE"])] + [
    str(p) for p in sorted(runfiles.glob("pip*/site-packages"))
]
# The C++ binary also links BoringSSL through the remote KV client.
flags = sys.getdlopenflags()
try:
    sys.setdlopenflags(flags | os.RTLD_DEEPBIND)
    import hashlib
    import ssl
finally:
    sys.setdlopenflags(flags)
import torch
)",
             scope);
    constexpr int batch = 8, rows = 4, vocab = 4096;
    initializeSamplingRuntime(vocab);
    ASSERT_TRUE(supportsDeferredRejectionSampling());

    auto cuda = torch::TensorOptions().device(torch::kCUDA);
    auto ints = torch::TensorOptions().dtype(torch::kInt32);
    // The first four requests have exact matches through the bonus row. The
    // last four propose an absent token immediately and need a correction.
    auto target = torch::full({batch, rows, vocab}, -50.0f);
    auto drafts = torch::zeros({batch, rows - 1}, ints);
    for (int req = 0; req < batch; ++req) {
        for (int pos = 0; pos < rows; ++pos) {
            const int id         = 1900 + 61 * req + 7 * pos;
            target[req][pos][id] = 0.0f;
            if (pos == rows - 1 || req >= batch / 2) {
                target[req][pos][id + 1250] = -0.2f;
            }
            if (pos < rows - 1 && req < batch / 2) {
                drafts[req][pos] = id;
            }
        }
    }
    Sampler       sampler(SamplerInitParams{});
    SamplerInputs prep{};
    prep.batch_size           = batch * rows;
    prep.batch_size_out       = prep.batch_size;
    prep.vocab_size           = vocab;
    prep.logits               = target.view({batch * rows, vocab}).to(torch::kCUDA);
    prep.top_k                = torch::full({batch * rows}, 7, ints);
    prep.top_p                = torch::full({batch * rows}, 0.8f);
    prep.temperature          = torch::ones({batch * rows});
    prep.sequence_lengths     = torch::empty({0}, ints);
    prep.defer_token_sampling = true;
    prep.rejection_rows       = rows;
    for (int req = 0; req < batch; ++req) {
        auto generator = torch::make_generator<at::CUDAGeneratorImpl>();
        generator.set_current_seed(75223 + req);
        prep.rejection_generators.push_back(generator);
    }
    auto prepared = sampler.forward(prep);
    ASSERT_TRUE(prepared.deferred_token_sampling);
    ASSERT_TRUE(prepared.verify_candidate_ids.defined());
    ASSERT_TRUE(prepared.prepared_rejection_rng.defined());
    ASSERT_EQ(prepared.verify_candidate_ids.size(1), 1024);

    // History penalties apply to the sampling request only. Both start with A ahead
    // of B (4 vs 3); penalizing A swaps them (2 vs 3).
    constexpr int token_a = 3101, token_b = 1321;
    SamplerInputs mixed{};
    mixed.batch_size = mixed.batch_size_out = 2 * rows;
    mixed.vocab_size                        = vocab;
    mixed.rejection_rows                    = rows;
    mixed.step                              = 1;
    mixed.logits                            = torch::full({2 * rows, vocab}, -50.0f, cuda);
    mixed.logits.select(1, token_a).fill_(4.0f);
    mixed.logits.select(1, token_b).fill_(3.0f);
    mixed.input_lengths = mixed.sequence_lengths = torch::ones({2 * rows}, ints);
    mixed.token_ids                              = torch::full({2 * rows, 2}, token_a, ints);
    mixed.top_k                                  = torch::full({2 * rows}, 7, ints);
    mixed.top_p = mixed.temperature = torch::ones({2 * rows});
    mixed.repetition_penalty        = torch::full({2 * rows}, 2.0f);
    mixed.presence_penalty = mixed.frequency_penalty = torch::zeros({2 * rows});
    mixed.do_sample                                  = torch::zeros({2 * rows}, torch::kBool);
    mixed.do_sample.narrow(0, rows, rows).fill_(true);
    mixed.defer_token_sampling = true;
    auto filtered              = sampler.forward(mixed);
    ASSERT_TRUE(filtered.verify_candidate_ids.defined());
    EXPECT_TRUE(torch::equal(filtered.verify_do_sample.cpu(), torch::tensor({false, true}, torch::kBool)));
    auto best = filtered.verify_candidate_ids.gather(1, filtered.verify_logits.argmax(-1, true)).view({2, rows}).cpu();
    EXPECT_TRUE(best[0].eq(token_a).all().item<bool>());
    EXPECT_TRUE(best[1].eq(token_b).all().item<bool>());

    RejectionSamplingParams verify{};
    verify.draft_probs_point_mass      = true;
    verify.deferred_token_sampling     = true;
    verify.target_logits_d             = prepared.verify_logits.view({batch, rows, 1024});
    verify.target_candidate_ids_d      = prepared.verify_candidate_ids.view_as(verify.target_logits_d);
    verify.target_dense_fallback_d     = prepared.verify_dense_fallback.view({batch, rows, vocab});
    verify.target_fallback_rows_d      = prepared.verify_fallback_rows;
    verify.target_log_normalizer_d     = prepared.verify_log_normalizer;
    verify.prepared_rejection_rng      = prepared.prepared_rejection_rng;
    verify.draft_token_ids_d           = drafts.to(torch::kCUDA);
    verify.do_sample_d                 = prepared.verify_do_sample;
    verify.output_token_ids_d          = torch::empty({batch, rows}, cuda.dtype(torch::kInt32));
    verify.output_accepted_token_num_d = torch::empty({batch}, cuda.dtype(torch::kInt32));
    scope["verify"]                    = py::cpp_function([&verify] {
        execRejectionSampling(verify);
        return py::make_tuple(verify.output_token_ids_d.cpu(), verify.output_accepted_token_num_d.cpu());
    });
    py::exec(R"(
from unittest.mock import patch
from triton.compiler.errors import CompilationError
from rtp_llm.models_py.sampling import rejection_compact, rejection_graph, rejection_native

with patch.object(rejection_compact, "materialize", side_effect=AssertionError("native plan missed")):
    native_ids, native_counts = verify()
assert any(isinstance(plan, rejection_native._CompactPlan) for plan in rejection_native._plans.values())
# .cpu() in verify() has completed both output copies, so the plan can be evicted.
rejection_native._plans.clear()
error = CompilationError(None, None, "compact native compilation failed")
with patch.object(rejection_native, "_CompactPlan", side_effect=error) as compile_plan, \
     patch.object(rejection_graph, "try_replay", return_value=False), \
     patch.object(rejection_compact, "materialize", wraps=rejection_compact.materialize) as materialize:
    eager_ids, eager_counts = verify()
compile_plan.assert_called_once()
materialize.assert_called_once()
torch.testing.assert_close(native_ids, eager_ids, rtol=0, atol=0)
torch.testing.assert_close(native_counts, eager_counts, rtol=0, atol=0)
assert native_counts.tolist() == [4] * 4 + [1] * 4
for req, row in enumerate(native_ids.tolist()):
    kept = 3 if req < 4 else 0
    assert row[:kept] == [1900 + 61 * req + 7 * pos for pos in range(kept)]
    first = 1900 + 61 * req + 7 * kept
    assert row[kept] in (first, first + 1250)
    assert row[kept + 1:] == [0] * (3 - kept)
)",
             scope);
}

}  // namespace rtp_llm
