#include "rtp_llm/platforms/MtpPolicy.h"

#if USE_PPU
#include "rtp_llm/platforms/ppu/MtpPolicy.h"
#endif

namespace rtp_llm::platforms {

MtpPolicy mtpPolicy(const std::string& model_type, bool is_dspark, int64_t hidden_size, int64_t num_layers) {
    if (is_dspark) {
        return {};
    }
#if USE_PPU
    return ppu::mtpPolicy(model_type, hidden_size, num_layers);
#else
    (void)model_type;
    (void)hidden_size;
    (void)num_layers;
    return {};
#endif
}

}  // namespace rtp_llm::platforms
