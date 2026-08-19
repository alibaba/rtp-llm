// Focused regression test for the V2 beam-search id-mapping math: the strides in
// addCumLogProbs and the divisors in gatherId decide output token ids and beam
// lineage, so a wrong stride must fail this test.
//
// Runs the V2 stage pipeline on small deterministic inputs:
//   A: invokeTopkLastDim per beam (nBS*nBMIn groups of nV)
//   B: launchAddCumLogProbs
//   C: invokeTopkLastDim merged per batch (nBS groups of nBMIn*nBMOut)
//   D: gatherId
// and compares stage-C output ids/values and the gathered final ids against a
// host reference element-wise.
//
// Two cases: symmetric (nBMIn == nBMOut == 4) and VBWS-asymmetric (nBMIn=3,
// nBMOut=5) — the strides mix nBMIn and nBMOut, so the asymmetric case is where
// a swapped divisor shows up. All candidate values are distinct (integer-exact
// in fp32), so ordering is unambiguous on every top-k path.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <vector>

#include "3rdparty/trt_beam_search/common.h"
#include "3rdparty/trt_beam_search/topkLastDim.h"
#include "3rdparty/trt_beam_search/beamSearchKernels.h"

namespace
{

using tensorrt_llm::kernels::invokeComputeTopkLastDimWorkspaceSize;
using tensorrt_llm::kernels::invokeTopkLastDim;
using tensorrt_llm::kernels::launchAddCumLogProbs;
using tensorrt_llm::kernels::gatherId;

char const* errString(cudaError_t err)
{
#if USING_ROCM
    return hipGetErrorString(err);
#else
    return cudaGetErrorString(err);
#endif
}

cudaError_t lastError()
{
#if USING_ROCM
    return hipGetLastError();
#else
    return cudaGetLastError();
#endif
}

#define CHECK_CUDA(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t const err_ = (call);                                                                               \
        if (err_ != cudaSuccess)                                                                                       \
        {                                                                                                              \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", errString(err_), __FILE__, __LINE__);                     \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

template <typename T>
T* toDevice(std::vector<T> const& host)
{
    T* dev = nullptr;
    CHECK_CUDA(cudaMalloc(&dev, host.size() * sizeof(T)));
    CHECK_CUDA(cudaMemcpy(dev, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return dev;
}

template <typename T>
std::vector<T> fromDevice(T const* dev, size_t n)
{
    std::vector<T> host(n);
    CHECK_CUDA(cudaMemcpy(host.data(), dev, n * sizeof(T), cudaMemcpyDeviceToHost));
    return host;
}

// logProbs are multiples of 2^-4 (exact in fp32), strictly decreasing in token index
// within each row, and crafted so merged stage-C winners interleave beams. cumLogProbs
// are multiples of 2^-5. The merged metric of candidate (m, v) in batch b is
// proportional to 3m + 2*nBMOut*v + (32+nBMIn)*b, all distinct.
int runCase(int nBS, int nBMIn, int nBMOut, int nV, char const* label)
{
    int const nCand = nBMIn * nBMOut; // candidates per batch after stage A

    std::vector<float> logProbs(nBS * nBMIn * nV);
    for (int b = 0; b < nBS; ++b)
    {
        for (int m = 0; m < nBMIn; ++m)
        {
            for (int v = 0; v < nV; ++v)
            {
                float const val = (v < nBMOut) ? -(m + nBMOut * v + 16 * b) * 0.0625f
                                               : -(1000.f + static_cast<float>(v)) * 0.0625f;
                logProbs[(b * nBMIn + m) * nV + v] = val;
            }
        }
    }
    std::vector<float> cumLogProbs(nBS * nBMIn);
    for (int i = 0; i < nBS * nBMIn; ++i)
    {
        cumLogProbs[i] = -static_cast<float>(i) * 0.03125f;
    }

    // Host reference: per-beam top-nBMOut (value desc, index asc), add cumLogProbs,
    // merged top-nBMOut per batch (value desc, flat stage-1 index asc), then the
    // gatherId mapping final = beam * nV + token.
    std::vector<int> refStage2Ids(nBS * nBMOut); // flat stage-1 indices, value-desc
    std::vector<float> refStage2Vals(nBS * nBMOut);
    std::vector<int> refFinalIds(nBS * nBMOut); // beam * nV + token
    for (int b = 0; b < nBS; ++b)
    {
        struct Cand
        {
            float val;
            int flat; // m * nBMOut + v: position in this batch's stage-1 candidate block
        };
        std::vector<Cand> cands;
        for (int m = 0; m < nBMIn; ++m)
        {
            for (int v = 0; v < nBMOut; ++v)
            {
                cands.push_back({logProbs[(b * nBMIn + m) * nV + v] + cumLogProbs[b * nBMIn + m], m * nBMOut + v});
            }
        }
        std::sort(cands.begin(), cands.end(), [](Cand const& x, Cand const& y) {
            return x.val > y.val || (x.val == y.val && x.flat < y.flat);
        });
        for (int j = 0; j < nBMOut; ++j)
        {
            refStage2Ids[b * nBMOut + j] = cands[j].flat;
            refStage2Vals[b * nBMOut + j] = cands[j].val;
            int const beam = cands[j].flat / nBMOut;
            int const tok = cands[j].flat % nBMOut; // position == token: in-row values decrease in token index
            refFinalIds[b * nBMOut + j] = beam * nV + tok;
        }
    }

    float* dLogProbs = toDevice(logProbs);
    float* dCumLogProbs = toDevice(cumLogProbs);
    std::optional<float> mask(-std::numeric_limits<float>::infinity());

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Stage A: top-nBMOut per beam.
    size_t const wsABytes = invokeComputeTopkLastDimWorkspaceSize<float>(nBS * nBMIn, nV, nBMOut, true, 0);
    void* wsA = nullptr;
    CHECK_CUDA(cudaMalloc(&wsA, wsABytes));
    float* dStage1Vals = nullptr;
    int* dStage1Ids = nullptr;
    CHECK_CUDA(cudaMalloc(&dStage1Vals, sizeof(float) * nBS * nCand));
    CHECK_CUDA(cudaMalloc(&dStage1Ids, sizeof(int) * nBS * nCand));
    invokeTopkLastDim<float>(nBS * nBMIn, nV, nBMOut, true, mask, dLogProbs, dStage1Vals, dStage1Ids, wsA, stream,
        /*sorted=*/false, /*force_path=*/0);

    // Stage B: add cumLogProbs in place.
    launchAddCumLogProbs<float>(dStage1Vals, dCumLogProbs, /*finished=*/nullptr, /*endIds=*/nullptr,
        /*diversityRates=*/nullptr, /*batchSlots=*/nullptr, nBS, nBMIn, nBMOut, /*nThread=*/32, stream);

    // Stage C: merged top-nBMOut per batch.
    size_t const wsCBytes = invokeComputeTopkLastDimWorkspaceSize<float>(nBS, nCand, nBMOut, true, 0);
    void* wsC = nullptr;
    CHECK_CUDA(cudaMalloc(&wsC, wsCBytes));
    float* dStage2Vals = nullptr;
    int* dStage2Ids = nullptr;
    CHECK_CUDA(cudaMalloc(&dStage2Vals, sizeof(float) * nBS * nBMOut));
    CHECK_CUDA(cudaMalloc(&dStage2Ids, sizeof(int) * nBS * nBMOut));
    invokeTopkLastDim<float>(nBS, nCand, nBMOut, true, mask, dStage1Vals, dStage2Vals, dStage2Ids, wsC, stream,
        /*sorted=*/true, /*force_path=*/0);

    auto devStage2Ids = fromDevice(dStage2Ids, nBS * nBMOut);
    auto devStage2Vals = fromDevice(dStage2Vals, nBS * nBMOut);

    // Stage D: gatherId rewrites stage-2 ids into final logProbs indices.
    gatherId<<<nBS, 32, 0, stream>>>(dStage1Ids, dStage2Ids, nBS, nBMIn, nBMOut, nV);
    CHECK_CUDA(lastError());
    auto devFinalIds = fromDevice(dStage2Ids, nBS * nBMOut);

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(wsA));
    CHECK_CUDA(cudaFree(wsC));
    CHECK_CUDA(cudaFree(dStage1Vals));
    CHECK_CUDA(cudaFree(dStage1Ids));
    CHECK_CUDA(cudaFree(dStage2Vals));
    CHECK_CUDA(cudaFree(dStage2Ids));
    CHECK_CUDA(cudaFree(dLogProbs));
    CHECK_CUDA(cudaFree(dCumLogProbs));

    int failures = 0;
    for (int i = 0; i < nBS * nBMOut; ++i)
    {
        if (devStage2Ids[i] != refStage2Ids[i] || devStage2Vals[i] != refStage2Vals[i])
        {
            std::printf("case[%s] stage2 mismatch at %d: dev(id=%d val=%f) ref(id=%d val=%f)\n", label, i,
                devStage2Ids[i], devStage2Vals[i], refStage2Ids[i], refStage2Vals[i]);
            ++failures;
        }
        if (devFinalIds[i] != refFinalIds[i])
        {
            std::printf("case[%s] gatherId mismatch at %d: dev=%d ref=%d\n", label, i, devFinalIds[i], refFinalIds[i]);
            ++failures;
        }
    }
    std::printf("case[%s]: %s (nBS=%d nBMIn=%d nBMOut=%d nV=%d)\n", label, failures == 0 ? "PASS" : "FAIL", nBS,
        nBMIn, nBMOut, nV);
    return failures;
}

} // namespace

int main()
{
    int failures = 0;
    failures += runCase(2, 4, 4, 32, "symmetric");
    failures += runCase(2, 3, 5, 32, "vbws-asymmetric");
    if (failures == 0)
    {
        std::printf("beam_mapping_test: PASS\n");
        return 0;
    }
    std::printf("beam_mapping_test: %d mismatches\n", failures);
    return 1;
}
