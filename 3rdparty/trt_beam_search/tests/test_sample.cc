#include "test_api.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "rtp_llm/models_py/bindings/core/torch_utils/TypeConvert.h"
#include "3rdparty/trt_beam_search/beamSearch.h"

// Both sides use the byte-exact production wrapper from Round 2. The candidate
// calls the live production launcher; the reference calls its frozen launcher.
#define sampleBeamSearch candidateSampleBeamSearch
#include "frozen/CudaBeamSearchOp.cc"
#undef sampleBeamSearch

namespace tensorrt_llm {
template<typename T>
BeamSearchConfig configureFrozenBeamSearch(runtime::SizeType32 batch,
                                           runtime::SizeType32 in,
                                           runtime::SizeType32 out,
                                           runtime::SizeType32 vocab) {
    // Frozen launchers always use auto top-k, even when the candidate is forced.
    return configureBeamSearch<T>(batch, in, out, vocab, 0);
}
}  // namespace tensorrt_llm
#define configureBeamSearch configureFrozenBeamSearch
#define sampleBeamSearch frozenSampleBeamSearch
#define invokeTopkBeamSearch frozenInvokeTopkBeamSearch
#include "frozen/CudaBeamSearchOp.cc"
#undef configureBeamSearch
#undef invokeTopkBeamSearch
#undef sampleBeamSearch
