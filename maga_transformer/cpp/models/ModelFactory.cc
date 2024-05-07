#include "maga_transformer/cpp/models/ModelFactory.h"

using namespace std;

namespace rtp_llm {

unique_ptr<GptModel> createGptModel(const GptModelInitParams& params) {
    // TODO(yitian team): create own model implementation and return.
    if (params.device->getDeviceProperties().type == ft::DeviceType::Yitian) {
        return make_unique<GptModel>(params);
    }
    return make_unique<GptModel>(params);
}

} // namespace rtp_llm
