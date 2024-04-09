#include "src/fastertransformer/devices/testing/TestBase.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "maga_transformer/cpp/models/Sampler.h"
#include "maga_transformer/cpp/test/ModelTestUtil.h"

using namespace std;
using namespace rtp_llm;

// TODO: make this test device-independent
class SamplerTest : public DeviceTestBase<DeviceType::Cuda> {
};
