#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>
#include <torch/torch.h>
#include "test_api.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"

namespace rtp_llm {
BeamSearchOutput candidateSampleBeamSearch(BeamSearchParams);
BeamSearchOutput frozenSampleBeamSearch(BeamSearchParams);
}

namespace {
using torch::Tensor;
using namespace tensorrt_llm::kernels;

template <typename T> T* optionalPtr(Tensor const& t) {
    return t.numel() == 0 ? nullptr : static_cast<T*>(t.data_ptr());
}

Tensor addCum(Tensor values, Tensor cum, Tensor finished, Tensor ends, Tensor diversity, Tensor slots,
    int64_t stage1k, bool reference)
{
    TORCH_CHECK(values.is_cuda() && values.is_contiguous() && values.dim() == 2);
    TORCH_CHECK(cum.scalar_type() == torch::kFloat32 && cum.dim() == 2);
    TORCH_CHECK(stage1k > 0 && values.size(1) <= cum.size(1) * stage1k);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    int const threads = std::min<int64_t>(std::max<int64_t>((values.size(1) + 31) / 32 * 32, 32), 1024);
#define ADD(T)                                                                                           \
    do {                                                                                                 \
        auto f = reference ? &frozenLaunchAddCumLogProbs<T> : &launchAddCumLogProbs<T>;                    \
        f(static_cast<T*>(values.data_ptr()), cum.data_ptr<float>(), optionalPtr<FinishedState>(finished), \
          optionalPtr<int>(ends), optionalPtr<float>(diversity), optionalPtr<int>(slots),                  \
          values.size(0), cum.size(1), stage1k, values.size(1), threads, stream);                            \
    } while (0)
    if (values.scalar_type() == torch::kFloat32) { ADD(float); }
    else if (values.scalar_type() == torch::kFloat16) { ADD(half); }
    else { TORCH_CHECK(false, "test addCum only supports float/half"); }
#undef ADD
    return values;
}

std::vector<Tensor> sample(Tensor logits, Tensor ids, Tensor inputs, Tensor lengths, Tensor cum,
    int64_t outputWidth, bool reference)
{
    rtp_llm::BeamSearchParams p{logits, ids, inputs, lengths, cum, static_cast<size_t>(outputWidth)};
    auto out = reference ? rtp_llm::frozenSampleBeamSearch(p) : rtp_llm::candidateSampleBeamSearch(p);
    return {out.token_ids, out.input_lengths, out.sequence_lengths, out.cum_log_probs, out.beam_indices};
}

// All low-level state is supplied explicitly. This interface is test-only;
// production gains no toggle, environment variable or new public API.
std::vector<Tensor> stage3Prepared(std::vector<Tensor> state, std::vector<Tensor> buffers,
    int64_t outputWidth, int64_t vocab, bool alias,
    bool cba, bool tiled, bool v2, bool reference)
{
    TORCH_CHECK(state.size() == 8, "ids/scores/lengths/inputLengths/cum/finished/endIds/slots required");
    auto& ids = state[0]; auto& scores = state[1]; auto& lengths = state[2];
    auto& inputs = state[3]; auto& cum = state[4];
    int64_t const bs = ids.size(0), mbs = lengths.size(0), inWidth = lengths.size(1);
    TORCH_CHECK(!alias || inWidth == outputWidth);
    TORCH_CHECK(!cba || (inWidth == outputWidth && inWidth <= 8));
    TORCH_CHECK(!tiled || inWidth == outputWidth);
    TORCH_CHECK(buffers.size() == 14);
    auto& outIds = buffers[0]; auto& parents = buffers[1];
    Tensor outLengths = alias ? lengths : buffers[2];
    Tensor outInputs = alias ? inputs : buffers[3];
    Tensor outCum = alias ? cum : buffers[4];
    auto& tiledValues = buffers[5]; auto& numCba = buffers[6]; auto& minCba = buffers[7];
    auto& cbaIds = buffers[8]; auto& cbaLengths = buffers[9]; auto& cbaNorm = buffers[10];
    auto& cbaCum = buffers[11]; auto& dones = buffers[12]; auto& scratch = buffers[13];

    BeamHypotheses bh;
    bh.bVBWS = inWidth != outputWidth;
    bh.nBatchSize = bs; bh.nMaxBatchSize = mbs; bh.nBeamWidthIn = inWidth; bh.nBeamWidthOut = outputWidth;
    bh.nVocabSize = vocab; bh.nMaxSeqLen = 12; bh.nStage2TopK = ids.size(1);
    bh.sequenceLengthsIn = lengths.data_ptr<int>(); bh.inputLengthsIn = inputs.data_ptr<int>();
    bh.cumLogProbsIn = cum.data_ptr<float>(); bh.sequenceLengthsOut = outLengths.data_ptr<int>();
    bh.inputLengthsOut = outInputs.data_ptr<int>(); bh.cumLogProbsOut = outCum.data_ptr<float>();
    bh.outputIdsPtr = outIds.data_ptr<int>(); bh.parentIdsPtr = parents.data_ptr<int>();
    bh.finished = optionalPtr<FinishedState>(state[5]); bh.endIds = optionalPtr<int>(state[6]);
    bh.batchSlots = optionalPtr<int>(state[7]);
    if (tiled) { bh.logProbsTiled = tiledValues.data_ptr<float>(); }
    if (cba) {
        bh.numBeamsCBA = numCba.data_ptr<int>(); bh.minNormedScoresCBA = minCba.data_ptr<float>();
        bh.outputIdsCBA = cbaIds.data_ptr<int>(); bh.sequenceLengthsCBA = cbaLengths.data_ptr<int>();
        bh.normedScoresCBA = cbaNorm.data_ptr<float>(); bh.cumLogProbsCBA = cbaCum.data_ptr<float>();
        bh.batchDones = dones.data_ptr<bool>();
    }
    testLaunchStage3(ids.data_ptr(), scores.data_ptr(), scratch.data_ptr<float>(), bh,
        scores.scalar_type() == torch::kFloat16, v2, reference, at::cuda::getCurrentCUDAStream().stream());
    return {outIds, parents, outLengths, outInputs, outCum, state[5], tiledValues, numCba, minCba,
            cbaIds, cbaLengths, cbaNorm, cbaCum, dones};
}

std::vector<Tensor> stage3(std::vector<Tensor> state, int64_t outputWidth, int64_t vocab, bool alias,
    bool cba, bool tiled, bool v2, bool reference)
{
    auto iopts = state[0].options(); auto fopts = state[4].options();
    int64_t const bs = state[0].size(0), mbs = state[2].size(0), beams = state[2].size(1);
    std::vector<Tensor> buffers{
        torch::zeros({mbs, outputWidth}, iopts), torch::zeros({mbs, outputWidth}, iopts),
        torch::full({mbs, outputWidth}, -123, iopts), torch::full({mbs, outputWidth}, -123, iopts),
        torch::full({mbs, outputWidth}, -123.0, fopts), torch::full({12, mbs, outputWidth}, -123.0, fopts),
        torch::zeros({mbs}, iopts), torch::zeros({mbs}, fopts),
        torch::full({mbs, outputWidth * 2, 12}, -123, iopts),
        torch::full({mbs, outputWidth * 2}, -123, iopts),
        torch::full({mbs, outputWidth * 2}, -123.0, fopts),
        torch::full({mbs, outputWidth * 2}, -123.0, fopts),
        torch::zeros({mbs}, iopts.dtype(torch::kBool)), torch::empty({bs * beams * outputWidth * 2}, fopts)};
    return stage3Prepared(state, buffers, outputWidth, vocab, alias, cba, tiled, v2, reference);
}
} // namespace

TORCH_LIBRARY(rtp_beam_sampling_test, m) {
    m.def("add_cum(Tensor(a!) values, Tensor cum, Tensor finished, Tensor ends, Tensor diversity, Tensor slots, int stage1k, bool reference) -> Tensor(a!)", &addCum);
    m.def("sample(Tensor logits, Tensor ids, Tensor inputs, Tensor lengths, Tensor cum, int output_width, bool reference) -> Tensor[]", &sample);
    m.def("stage3(Tensor[] state, int output_width, int vocab, bool alias, bool cba, bool tiled, bool v2, bool reference) -> Tensor[]", &stage3);
    m.def("stage3_prepared(Tensor[] state, Tensor[] buffers, int output_width, int vocab, bool alias, bool cba, bool tiled, bool v2, bool reference) -> Tensor[]", &stage3Prepared);
}
