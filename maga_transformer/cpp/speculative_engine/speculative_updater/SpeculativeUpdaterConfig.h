#include <memory>
#include "maga_transformer/cpp/dataclass/EngineInitParameter.h"

namespace rtp_llm {

struct SpeculativeUpdaterConfig {
    bool propose_compact_kv_cache = false;
    bool score_compact_kv_cache   = false;
    bool save_score_last_state    = false;
};

SpeculativeUpdaterConfig
createSpeculativeUpdaterConfig(std::unique_ptr<ProposeModelEngineInitParams>& propose_model_engine_init_params);

}  // namespace rtp_llm