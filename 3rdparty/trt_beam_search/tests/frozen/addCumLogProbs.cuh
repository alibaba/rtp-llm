// See beamSearchKernels.h comment: host wrappers below keep the kernel launch
// in the same TU as the kernel definition so nvcc 13's local-linkage
// instantiations can be resolved.
namespace {
template <typename T>
__global__ void addCumLogProbsKernel(
    T* __restrict pStage1LogProbs, float const* __restrict cumLogProbs,
    FinishedState const* finished, int const* endIds, float const* diversityRates,
    runtime::SizeType32 const* batchSlots, size_t const nBS, size_t const nBMIn, size_t const nStage1TopK,
    size_t const nStage2InputLen)
{
    int const bid = blockIdx.x; // Index of request in batch
    runtime::SizeType32 const slot = batchSlots ? batchSlots[bid] : bid;
    float const diversityRate{diversityRates == nullptr ? kBeamSearchDiversity : diversityRates[slot]};
    T* pLocalLogProbs = pStage1LogProbs + bid * nStage2InputLen;

    for (int i = threadIdx.x; i < nStage2InputLen; i += blockDim.x)
    {
        int const iBMIn = i / nStage1TopK;
        if (finished && finished[slot * nBMIn + iBMIn].isFinished())
        {
            pLocalLogProbs[i] += endIds && (i == endIds[slot]) ? T(1.0f) : T(0.0f);
        }
        else
        {
            pLocalLogProbs[i] += cumLogProbs[slot * nBMIn + iBMIn] + diversityRate * iBMIn;
        }
    }
    return;
}

} // namespace

template <typename T>
void launchAddCumLogProbs(
    T* pStage1LogProbs, float const* cumLogProbs, FinishedState const* finished,
    int const* endIds, float const* diversityRates,
    runtime::SizeType32 const* batchSlots, size_t nBS, size_t nBMIn, size_t nStage1TopK,
    size_t nStage2InputLen,
    int nThread, cudaStream_t stream)
{
    addCumLogProbsKernel<T><<<nBS, nThread, 0, stream>>>(
        pStage1LogProbs, cumLogProbs, finished, endIds, diversityRates, batchSlots, nBS, nBMIn, nStage1TopK,
        nStage2InputLen);
    check_cuda_error();
}

template void launchAddCumLogProbs<float>(
    float*, float const*, FinishedState const*, int const*, float const*,
    runtime::SizeType32 const*, size_t, size_t, size_t, size_t, int, cudaStream_t);

template void launchAddCumLogProbs<half>(
    half*, float const*, FinishedState const*, int const*, float const*,
    runtime::SizeType32 const*, size_t, size_t, size_t, size_t, int, cudaStream_t);
