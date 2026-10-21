#include "test_api.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "rtp_llm/models_py/bindings/core/torch_utils/TypeConvert.h"
#include "3rdparty/trt_beam_search/beamSearch.h"

// Both sides use the byte-exact production wrapper from Round 2. The candidate
// calls the live production launcher; the reference calls its frozen launcher.
#define sampleBeamSearch candidateSampleBeamSearch
#include "frozen/CudaBeamSearchOp.cc"
#undef sampleBeamSearch

#define sampleBeamSearch frozenSampleBeamSearch
#define invokeTopkBeamSearch frozenInvokeTopkBeamSearch
#include "frozen/CudaBeamSearchOp.cc"
#undef invokeTopkBeamSearch
#undef sampleBeamSearch
